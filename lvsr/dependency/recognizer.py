import numpy
import theano
import logging
from theano import tensor
import re

from blocks.bricks import (
    Bias, Identity, Initializable, MLP, Tanh, NDimensionalSoftmax)
from blocks.bricks.base import application
from blocks.bricks.recurrent import (
    BaseRecurrent, RecurrentStack)
from blocks.bricks.sequence_generators import Readout 
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.search import CandidateNotFoundError
from blocks.serialization import load_parameter_values

from lvsr.bricks import Encoder, InitializableSequence, SoftmaxMultiEmitter
from lvsr.utils import global_push_initialization_config, SpeechModel, MultiGet,\
    rename, resizeArray

from lvsr.dependency.attention import AttendedFeedback, \
    ParsingAttentionRecurrent, ParsingAttention

from lvsr.dependency.generator import Generator
from blocks.roles import OUTPUT

from lvsr.dependency import get_var_path

import numpy as np

from lvsr.dependency import edmonts
from blocks.utils import dict_union

from blocks.filter import get_brick
from itertools import chain
from blocks.select import Selector
from theano.gradient import DisconnectedType

logger = logging.getLogger(__name__)

class Bottom(Initializable):
    """
    A bottom class that mergers possibly many input sources into one
    sequence.

    The bottom is responsible for allocating variables for single and
    multiple sequences in a batch.

    In speech recognition this will typically be the identity transformation
    ro a small MLP.

    Parameters
    ----------

    input_sources: list
        list of source names meaningful to the chosen Bottom class,
        such as "recordings" or "ivectors".

    """
    def __init__(self, input_sources,
                 **kwargs):
        super(Bottom, self).__init__(**kwargs)
        self.input_sources = input_sources

def dictionaryCopy(dictionary):
    output = {}
    for d,v in dictionary.iteritems():
        if isinstance(v, dict):
            v = dictionaryCopy(v)
        output[d] = v
    return output

def nestedDictionaryValues(dictionary):
    values = dictionary.values()
    for i,v in enumerate(values):
        if isinstance(v, dict):
            values = values[:i] + nestedDictionaryValues(v) + values[i+1:]
    return values

def get_bricks_children(bricks):
    queue = list(bricks)
    visited = set()
    while queue:
        brick = queue.pop()
        if brick not in visited:
            queue = list(brick.children) + queue
            visited.add(brick)
    return visited

def get_bricks_parents(bricks):
    queue = list(bricks)
    visited = set()
    while queue:
        brick = queue.pop()
        if brick not in visited:
            queue = list(brick.parents) + queue
            visited.add(brick)
    return visited

class MultilangDependencyRecognizer(Initializable):
    def __init__(self, langs, info_data, postfix_manager, parameter_unifications_include,
                 parameter_unifications_exclude, **net_config):
        super(MultilangDependencyRecognizer, self).__init__(name='recognizer')
        self.langs = langs
        self.info_data = info_data
        self.postfix_manager = postfix_manager
        self.parameter_unifications_include = [re.compile(unification)
                                               for unification in parameter_unifications_include]
        self.parameter_unifications_exclude = [re.compile(unification)
                                               for unification in parameter_unifications_exclude]
        self.init_recognizers(**net_config)
        self.selector = Selector(self)
        self.child_postfix_regexp = [re.compile('.*'+chld.names_postfix+'($|_.*)')
                                     for chld in self.children]


    def init_recognizers(self, **orig_net_config):
        for lang in self.langs:
            #net_config = copy.deepcopy(orig_net_config)
            net_config = dictionaryCopy(orig_net_config)
            orig_lang = lang
            lang = self.postfix_manager.get_lang_postfix(lang) 
            
            addidional_sources = ['labels']
            if 'additional_sources' in net_config:
                addidional_sources += net_config['additional_sources']

            net_config['bottom']['lang_postfix'] = lang
            net_config['input_sources_dims'] = {}
            for src in net_config['input_sources']:
                net_config['input_sources_dims'][src+lang] = self.info_data.num_features(src)
            net_config['additional_sources_dims'] = {}
            for src in addidional_sources:
                net_config['additional_sources_dims'][src+lang] = self.info_data.num_features(self.info_data.sources_map[src])

            net_config['input_sources'] = [source + lang for source in
                                           net_config['input_sources']]
            net_config['additional_sources'] = [source + lang for source in
                                                      net_config['additional_sources']]
            recognizer = DependencyRecognizer(
                eos_label=self.info_data.eos_label,
                num_phonemes=self.info_data.num_characters,
                name='recognizer_'+orig_lang,
                character_map=self.info_data.char2num,
                names_postfix=lang,
                **net_config)
            self.children += [recognizer]
            
    def child_id_from_postfix(self, name):
        empty_postfix = None
        found_chld = -1
        for i in xrange(len(self.children)):
            if self.children[i].names_postfix == '':
                if empty_postfix is not None:
                    raise ValueError('Only one child can have empty postfix')
                empty_postfix = i
                continue
            if self.child_postfix_regexp[i].match(name):
                if found_chld != -1:
                    raise ValueError('Ambigious postfix in '+name)
                found_chld = i
        if found_chld == -1:
            return empty_postfix
        else:
            return found_chld

    def activate_masks(self, mask_dict):
        for child in self.children:
            child.mask_dict = mask_dict
    
    @application
    def cost(self, **kwargs):
        cost_matrix = 0
        split_kwargs = self.pop_dict_by_postfix(kwargs,
                            [chld.names_postfix for chld in self.children
                             if len(chld.names_postfix) > 0])
        for chld in self.children:
            if chld.names_postfix in split_kwargs:
                chldkwargs = split_kwargs[chld.names_postfix]
            else:
                chldkwargs = kwargs
            cost_matrix += chld.cost(**chldkwargs)
        return cost_matrix
    
    def pop_dict_by_postfix(self, dictionary, postfixes):
        output = {}
        for postfix in postfixes:
            output[postfix] = {}
            for k in dictionary.keys():
                if k.endswith(postfix):
                    output[postfix][k] = dictionary.pop(k)
        return output

    @application
    def generate(self, application_call, **kwargs):
        main = None
        for i in xrange(len(self.langs)):
            args = dictionaryCopy(kwargs)
            if 'inputs_mask' in args:
                args['inputs_mask'] = args['inputs_mask'][i]
            bottom_input = args['bottom_inputs'][i]
            del args['bottom_inputs']
            args = dict_union(args, bottom_input)
            args['generate_pos'] = False
            gen = self.children[i].generate(**args)
            if i == 0:
                main = gen
            else:
                for k in main.keys():
                    main[k] = main[k] + gen[k]
        if i == 0:
            for k in main.keys():
                main[k] = main[k] + 0
        for k in main.keys():
            application_call.add_auxiliary_variable(main[k], name=k)
        return main

    def load_params(self, path):
        graphs = [self.get_cost_graph().outputs[0],
                  ComputationGraph(self.get_generate_graph()['outputs'])]
        param_values = load_parameter_values(path)
        for graph in graphs:
            SpeechModel(graph).set_parameter_values(param_values)

    def get_generate_graph(self, use_mask=True, n_steps=None):
        inputs_mask = None
        if use_mask:
            inputs_mask = [chld.inputs_mask for chld in self.children]
        bottom_inputs = [chld.inputs for chld in self.children]
        return self.generate(n_steps=n_steps,
                             inputs_mask=inputs_mask,
                             bottom_inputs=bottom_inputs)
        
    def get_cost_graph(self, batch=True):
        params_dict = {}
        for chld in self.children:    
            if batch:
                inputs = chld.inputs
                inputs_mask = chld.inputs_mask
                labels = chld.labels
                labels_mask = chld.labels_mask
            else:
                inputs, inputs_mask = chld.bottom.single_to_batch_inputs(
                    chld.single_inputs)
                labels = chld.single_labels[:, None]
                labels_mask = None
            params_dict = dict_union(params_dict, inputs)
            params_dict['additional_sources'+chld.names_postfix] = dict(chld.additional_sources)
            params_dict['inputs_mask'+chld.names_postfix] = inputs_mask
            params_dict['labels'+chld.names_postfix] = labels
            params_dict['labels_mask'+chld.names_postfix] = labels_mask

        cost = self.cost(**params_dict)
        cost_cg = ComputationGraph(cost)

        return cost_cg

    def get_top_brick(self, param):
        brick = get_brick(param)
        while len(brick.parents) > 0 and not isinstance(brick, DependencyRecognizer):
            brick = brick.parents[0]
        return brick
    
    def replace_parameter(self, path, value):
        path = path.split('.')
        param_name = path[1]
        path = path[0]
        
        brick = self.selector.select(path).bricks

        if len(brick) != 1:
            raise ValueError('Cannot replace parameter from path {}. \
                              Wrong number of bricks ({})'.format(path, len(brick)))
            
        brick = brick[0]
        for i in xrange(len(brick.parameters)):
            if brick.parameters[i].name == param_name:
                orig_val = brick.parameters[i]
                brick.parameters[i] = value.copy(name=param_name)
                brick.parameters[i].tag.annotations = orig_val.tag.annotations
                brick.parameters[i].tag.roles = orig_val.tag.roles
    
    def unify_parameters(self, source_id, dest_id):
        source = self.children[source_id]
        source_name = self.children[source_id].name
        source_prefix = '/'+source_name+'/'
        dest_name = self.children[dest_id].name
        dest_prefix = '/'+self.name+'/'+dest_name+'/'
        
        source_params = Selector(source).get_parameters()
        
        replaced = []

        self.unified_parameters = []

        for param, var in source_params.iteritems():
            if not param.startswith(source_prefix):
                continue
            source_param = '/'+self.name+param
            param = param[len(source_prefix):]
            for unification in self.parameter_unifications_include:
                if unification.match(param):
                    exclude = False
                    for ex_unification in self.parameter_unifications_exclude:
                        if ex_unification.match(param):
                            exclude = True
                            break
                    if exclude:
                        continue
                    self.replace_parameter(dest_prefix+param,
                                           var)
                    replaced += [dest_prefix+param]
                    self.unified_parameters += [source_param]
        self.unified_parameters = self.convert_names_to_bricks(
                                    set(self.unified_parameters)|set(replaced))
        return replaced

    def convert_names_to_bricks(self, names):
        bricks = []
        for name in names:
            if '.' in name:
                name = name[:name.rindex('.')]
            bricks += self.selector.select(name).bricks
        return bricks

    def find_params(self, brick, path):
        path = path + '/' + brick.name
        params = ", ".join([param.__str__() for param in brick.parameters])
        print path, '->', params
        for chld in brick.children:
            self.find_params(chld, path)
        
    def get_bricks_children(self, cg):
        bricks = [get_brick(var) for var
                  in cg.variables + cg.scan_variables if get_brick(var)]
        children = set(chain(*(brick.children for brick in bricks)))
        return bricks, children

    def init_beam_search(self, lang_id, beam_size):
        self.children[lang_id].init_beam_search(beam_size)

    def beam_search(self, lang_id, *args, **kwargs):
        return self.children[lang_id].beam_search(*args, **kwargs)

    def all_children(self):
        return MultiGet(self.children)

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ['_analyze', '_beam_search']:
            state.pop(attr, None)
        return state

class DependencyRecognizer(Initializable):
    def __init__(self,
                 input_sources,
                 input_sources_dims,
                 eos_label,
                 num_phonemes,
                 dim_dec, dims_bidir,
                 enc_transition, dec_transition,
                 use_states_for_readout,
                 attention_type,
                 criterion,
                 bottom,
                 enc_transition_params={}, dec_transition_params={},
                 names_postfix='',
                 lm=None, character_map=None,
                 bidir=True,
                 bidir_aggregation='concat',
                 subsample=None,
                 dims_top=None,
                 prior=None, conv_n=None,
                 post_merge_activation=None,
                 post_merge_dims=None,
                 dim_matcher=None,
                 embed_outputs=True,
                 dim_output_embedding=None,
                 dec_stack=1,
                 conv_num_filters=1,
                 data_prepend_eos=False,
                 # softmax is the default set in SequenceContentAndConvAttention
                 energy_normalizer=None,
                 # for speech this is the approximate phoneme duration in frames
                 max_decoded_length_scale=3,
                 use_dependent_words_for_labels=False,
                 use_dependent_words_for_attention=False,
                 reproduce_rec_weight_init_bug=True,
                 pointers_weight=0.5,
                 tags_weight=1.0,
                 tag_layer=-1,  # -1 is last, 0 is after first bidir layer
                 dependency_type='recurrent_soft',
                 **kwargs):

        if post_merge_activation is None:
            post_merge_activation = Tanh()

        self.regularization_bricks = []
        possible_regularization_bricks = []
            
        self.names_postfix = names_postfix

        self.mask_dict = {}
        
        self.pointers_name = 'pointers'+names_postfix

        self.additional_sources = kwargs.pop('additional_sources')
        self.additional_sources_dims = kwargs.pop('additional_sources_dims')

        self.pointer_weight = pointers_weight
        self.soft_pointer_val = kwargs.pop('pointers_soften', 0.0)
        self.soft_pointer = self.soft_pointer_val > 0.0

        self.tags_weight = tags_weight
        self.tag_layer = tag_layer
        self.train_tags = True
        if self.tags_weight < 0 or len(self.additional_sources) <= 1:
            self.train_tags = False

        self.dependency_type = dependency_type

        super(DependencyRecognizer, self).__init__(**kwargs)

        self.reproduce_rec_weight_init_bug = reproduce_rec_weight_init_bug

        self.eos_label = eos_label
        self.data_prepend_eos = data_prepend_eos

        self.rec_weights_init = None
        self.initial_states_init = None

        self.enc_transition = enc_transition
        self.dec_transition = dec_transition
        self.dec_stack = dec_stack

        self.criterion = criterion

        self.max_decoded_length_scale = max_decoded_length_scale

        self.post_merge_activation = post_merge_activation

        if dim_matcher is None:
            dim_matcher = dim_dec

        # The bottom part, before BiRNN
        bottom_class = bottom.pop('bottom_class')
        bottom = bottom_class(input_sources=input_sources,
                              input_sources_dims=input_sources_dims,
                              name='bottom', pointers_soften=self.soft_pointer,
                              additional_sources=self.additional_sources,
                              **bottom)

        # BiRNN
        if not subsample:
            subsample = [1] * len(dims_bidir)
        encoder = Encoder(self.enc_transition, dims_bidir,
                          bottom.output_dim,
                          subsample, bidir=bidir,
                          bidir_aggregation=bidir_aggregation,
                          enc_transition_params=enc_transition_params)
        possible_regularization_bricks += encoder.enc_transitions
        dim_encoded = encoder.get_dim(encoder.apply.outputs[0])

        # The top part, on top of BiRNN but before the attention
        if dims_top:
            top = MLP([Tanh()],
                      [dim_encoded] + dims_top + [dim_encoded], name="top")
        else:
            top = Identity(name='top')

        self.additional_sources_mlp = {}
        ndim_softmax = NDimensionalSoftmax()
        ndim_softmax._extra_ndim = 1
        for source in self.additional_sources:
            if source != self.pointers_name:
                if len(self.names_postfix) > 0:
                    source_glob_name = source[:-len(self.names_postfix)]
                else:
                    source_glob_name = source
                self.additional_sources_mlp[source] = \
                    MLP([ndim_softmax], [dim_encoded, self.additional_sources_dims[source]],
                        name='additional_'+source_glob_name)

        if dec_stack == 1:
            transition = self.dec_transition(
                dim=dim_dec, activation=Tanh(), name="transition", **dec_transition_params)
            possible_regularization_bricks += [transition]
        else:
            transitions = [self.dec_transition(dim=dim_dec,
                                               activation=Tanh(),
                                               name="transition_{}".format(trans_level),
                                               **dec_transition_params)
                           for trans_level in xrange(dec_stack)]
            possible_regularization_bricks += transitions
            transition = RecurrentStack(transitions=transitions,
                                        skip_connections=True)
        # Choose attention mechanism according to the configuration
        attention_class = ParsingAttention
        attention_kwargs = {}
        transition_with_att_class = ParsingAttentionRecurrent

        if self.dependency_type == "recurrent_soft":
            attention_kwargs['use_pointers'] = None
        elif self.dependency_type == "recurrent_hard":
            attention_kwargs['use_pointers'] = 'hard'
        elif self.dependency_type == "recurrent_semihard":
            attention_kwargs['use_pointers'] = 'semihard'
        else:
            raise ValueError("Unknown dependency type {}"
                             .format(self.dependency_type))

        if attention_type == "content":
            pass
        elif attention_type == "content_hard":
            attention_kwargs['hard_attention'] = True
        else:
            raise ValueError("Unknown attention type {}"
                             .format(attention_type))

        if use_dependent_words_for_attention:
            attention_kwargs['use_word_annotations'] = True
            attention_kwargs['word_annontation_dim'] = dim_encoded

        attention = attention_class(
            state_names=transition.apply.states,
            attended_dim=dim_encoded, match_dim=dim_matcher,
            name="cont_att", **attention_kwargs)

        feedback = AttendedFeedback(num_phonemes + 1, dim_encoded)
        if criterion['name'] == 'log_likelihood':
            emitter = SoftmaxMultiEmitter(initial_output=num_phonemes, name="emitter")
        else:
            raise ValueError("Unknown criterion {}".format(criterion['name']))
        readout_source_names = (
            transition.apply.states if use_states_for_readout else []
            ) + [attention.take_glimpses.outputs[0]]

        if use_dependent_words_for_labels:
            readout_source_names.append('attended')

        readout_config = dict(
            readout_dim=num_phonemes,
            source_names=readout_source_names,
            emitter=emitter,
            feedback_brick=feedback,
            name="readout")
        if post_merge_dims:
            readout_config['merged_dim'] = post_merge_dims[0]
            readout_config['post_merge'] = InitializableSequence([
                Bias(post_merge_dims[0]).apply,
                post_merge_activation.apply,
                MLP([post_merge_activation] * (len(post_merge_dims) - 1) + [Identity()],
                    # MLP was designed to support Maxout is activation
                    # (because Maxout in a way is not one). However
                    # a single layer Maxout network works with the trick below.
                    # For deeper Maxout network one has to use the
                    # Sequence brick.
                    [d//getattr(post_merge_activation, 'num_pieces', 1)
                     for d in post_merge_dims] + [num_phonemes]).apply,
            ],
                name='post_merge')
        readout = Readout(**readout_config)

        generator = Generator(
            readout=readout, transition=transition, attention=attention,
            dim_dec=dim_dec, pointer_weight=self.pointer_weight,
            transition_with_att_class=transition_with_att_class, name="generator")

        for brick in possible_regularization_bricks:
            if 'regularize' in dir(brick):
                self.regularization_bricks += [brick]

        logger.info("Regularization bricks: {}".format(str(self.regularization_bricks))) 

        # Remember child bricks
        self.encoder = encoder
        self.bottom = bottom
        self.top = top
        self.generator = generator
        self.children = [encoder, top, bottom, generator]
        self.children.extend(self.additional_sources_mlp.values())

        # Create input variables
        self.inputs = self.bottom.get_batch_inputs()
        self.inputs_mask = self.bottom.get_mask()

        self.additional_sources = self.bottom.get_batch_additional_sources()

        self.labels = tensor.lmatrix('labels'+names_postfix)
        self.labels_mask = tensor.matrix('labels'+names_postfix+'_mask')
        #self.labels_mask = tensor.matrix('labels_mask'+names_postfix)

        self.single_inputs = self.bottom.get_single_sequence_inputs()
        self.single_labels = tensor.lvector('labels'+names_postfix)
        self.single_additional_sources = self.bottom.get_single_additional_sources()
        self.n_steps = tensor.lscalar('n_steps'+names_postfix)

    def push_initialization_config(self):
        super(DependencyRecognizer, self).push_initialization_config()
        if self.rec_weights_init:
            rec_weights_config = {'weights_init': self.weights_init,
                                  'recurrent_weights_init': self.rec_weights_init}
            if self.reproduce_rec_weight_init_bug:
                rec_weights_config['weights_init'] = self.rec_weights_init
            global_push_initialization_config(self,
                                              rec_weights_config,
                                              BaseRecurrent)
        if self.initial_states_init:
            global_push_initialization_config(self,
                                              {'initial_states_init': self.initial_states_init})

    def activate_masks(self, cg):
        if self.mask_dict is None:
            return {}
        outputs = VariableFilter(roles=[OUTPUT])(cg)
        replace_masks = {}
        for mask_name, mask_value in self.mask_dict.iteritems():
            if mask_name.startswith('recognizer/recognizer_'):
                mask_name = mask_name[24:]
            for output in outputs:
                if get_var_path(output).endswith(mask_name):
                    value = (np.float32(1.0) - mask_value).astype(output.dtype)
                    replace_masks[output] = output*value
        return cg.replace(replace_masks)

    def set_regularization_bricks(self, val):
        logger.info("Setting regularization bricks to: {}".format(val))
        for regularization_brick in self.regularization_bricks:
            regularization_brick.regularize = val

    @application
    def cost(self, application_call, **kwargs):
        self.set_regularization_bricks(True)
        # pop inputs we know about
        inputs_mask = kwargs.pop('inputs_mask'+self.names_postfix)
        labels = kwargs.pop('labels'+self.names_postfix)
        labels_mask = kwargs.pop('labels_mask'+self.names_postfix)
        additional_sources = kwargs.pop('additional_sources'+self.names_postfix)
        pointers = additional_sources.pop(self.pointers_name)

        # the rest is for bottom
        bottom_processed = self.bottom.apply(**kwargs)
        encoded, encoded_mask = self.encoder.apply(
            input_=bottom_processed,
            mask=inputs_mask)
        encoded = self.top.apply(encoded)
        cost_matrix = self.generator.cost_matrix(
            [labels, pointers], labels_mask,
            attended=encoded, attended_mask=encoded_mask)
        if self.train_tags:
            if self.tag_layer == -1:
                tag_cost = self.addTagCost(encoded, encoded_mask,
                                           **additional_sources)
            else:
                selected_encoded, = VariableFilter(
                    applications=[self.encoder.children[self.tag_layer].apply],
                    roles=[OUTPUT])(ComputationGraph(encoded))
                tag_cost = self.addTagCost(selected_encoded, encoded_mask,
                                           **additional_sources)
        else:
            tag_cost = 0.0
        if 'nodeps' in self.name:
            return self.tags_weight*tag_cost
        else:
            return cost_matrix + self.tags_weight*tag_cost

    @application(outputs=['output'])
    def addTagCost(self, application_call, encoded, encoded_mask, **additional_sources):
        tag_costs = 0.0
        # Tagger
        for name, tvar in additional_sources.iteritems():
            result = self.additional_sources_mlp[name].apply(encoded)
            tvar_flat = tvar.flatten()
            tvar_flat += tensor.arange(tvar_flat.shape[0]) * result.shape[-1]
            tvar_cost = -tensor.log(
                    result.flatten()[tvar_flat
                                     ].reshape(tvar.shape, tvar.ndim) + 1e-18)
            tvar_cost *= encoded_mask
            # tvar_cost = theano.tensor.switch( theano.tensor.eq(tvar_cost.shape[0], 23),
                    # tvar_cost / 0.0,
                    # tvar_cost)
            application_call.add_auxiliary_variable(tvar_cost.copy(),
                    name='tag_'+name+'_nll')
            tag_costs += tvar_cost / len(additional_sources)

        return tag_costs

    @application
    def generate(self, **kwargs):
        self.set_regularization_bricks(False)
        inputs_mask = kwargs.pop('inputs_mask')
        n_steps = kwargs.pop('n_steps')
        generate_pos = kwargs.pop('generate_pos', True)

        encoded, encoded_mask = self.encoder.apply(
            input_=self.bottom.apply(**kwargs),
            mask=inputs_mask)
        encoded = self.top.apply(encoded)
        pos_gens = []
        generated = self.generator.generate(
            n_steps=n_steps if n_steps is not None else self.n_steps,
            batch_size=encoded.shape[1],
            attended=encoded,
            attended_mask=encoded_mask,
            as_dict=True)
        if generate_pos:
            selected_encoded, = VariableFilter(
                applications=[self.encoder.children[self.tag_layer].apply],
                roles=[OUTPUT])(ComputationGraph(encoded))
            pos_gens = {name: rename(mlp.apply(selected_encoded), name) for name, mlp in self.additional_sources_mlp.iteritems()}
            generated['pos'] = pos_gens
        return generated

    def load_params(self, path):
        generated = self.get_generate_graph()
        param_values = load_parameter_values(path)
        SpeechModel(generated['outputs']).set_parameter_values(param_values)

    def get_generate_graph(self, use_mask=True, n_steps=None, **kwargs):
        inputs_mask = None
        if use_mask:
            inputs_mask = self.inputs_mask
        bottom_inputs = self.inputs.copy()
        bottom_inputs.update(kwargs)
        return self.generate(n_steps=n_steps,
                             inputs_mask=inputs_mask,
                             **bottom_inputs)

    def get_cost_graph(self, batch=True,
                       prediction=None, prediction_mask=None):

        if batch:
            inputs = self.inputs
            inputs_mask = self.inputs_mask
            groundtruth = self.labels
            groundtruth_mask = self.labels_mask
        else:
            inputs, inputs_mask = self.bottom.single_to_batch_inputs(
                self.single_inputs)
            groundtruth = self.single_labels[:, None]
            groundtruth_mask = None

        if not prediction:
            prediction = groundtruth
        if not prediction_mask:
            prediction_mask = groundtruth_mask

        kwargs = dict(inputs_mask=inputs_mask,
                         labels=prediction,
                         labels_mask=prediction_mask,
                         additional_sources=dict(self.additional_sources))
        kwargs = {(k+self.names_postfix): v for k,v in kwargs.iteritems()}
        kwargs = dict_union(kwargs, inputs)

        cost = self.cost(**kwargs)
        cost_cg = ComputationGraph(cost)

        return cost_cg

    def analyze(self, inputs, groundtruth, prediction=None):
        """Compute cost and aligment."""

        input_values_dict = dict(inputs)
        input_values_dict['groundtruth'] = groundtruth
        if prediction is not None:
            input_values_dict['prediction'] = prediction
        if not hasattr(self, "_analyze"):
            input_variables = list(self.single_inputs.values())
            input_variables.append(self.single_labels.copy(name='groundtruth'))

            prediction_variable = tensor.lvector('prediction')
            if prediction is not None:
                input_variables.append(prediction_variable)
                cg = self.get_cost_graph(
                    batch=False, prediction=prediction_variable[:, None])
            else:
                cg = self.get_cost_graph(batch=False)
            cost = cg.outputs[0]

            weights, = VariableFilter(
                bricks=[self.generator], name="weights")(cg)

            energies = VariableFilter(
                bricks=[self.generator], name="energies")(cg)
            energies_output = [energies[0][:, 0, :] if energies
                               else tensor.zeros_like(weights)]

            self._analyze = theano.function(
                input_variables,
                [cost[:, 0], weights[:, 0, :]] + energies_output,
                on_unused_input='warn')
        return self._analyze(**input_values_dict)

    def init_beam_search(self, beam_size):
        """Compile beam search and set the beam size.

        See Blocks issue #500.

        """
        if hasattr(self, 'search_function'):
            # Only recompile if the user wants a different beam size
            return

        generated = self.get_generate_graph(use_mask=True)
        cg = ComputationGraph(nestedDictionaryValues(generated))
        cg = self.activate_masks(cg)
        self.search_function_inputs = [x.name for x in  cg.inputs]
        self.search_function_pos_outputs = [x.name for x in cg.outputs[3:]]
        self.search_function = cg.get_theano_function()

    def is_subbrick(self, brick, var):
        if var.tag.annotations == []:
            return False

        any_subbrick = False
        for parent in var.tag.annotations:
            if parent == brick:
                return True
            else:
                any_subbrick = any_subbrick or self.is_subbrick(brick, parent)

    @application
    def generate_grad_stats(self, **kwargs):
        generated = self.get_generate_graph(use_mask=True, generate_pos=False)
        cg = ComputationGraph(generated.values())
        grad = theano.grad(generated['costs'].sum(), cg.parameters,
                           disconnected_inputs='ignore',
                           return_disconnected='Disconnected')
        out_grad = []
        grad_names = []
        for i,v in enumerate(cg.parameters):
            if not isinstance(grad[i].type, DisconnectedType):
                grad_names.append(get_var_path(v))
                out_grad.append(grad[i])
        return grad_names, out_grad

    @application
    def generate_activation_stats(self, watched_bricks=[], **kwargs):
        generated = self.get_generate_graph(use_mask=True)
        cg = ComputationGraph(nestedDictionaryValues(generated))
        watched_affected = get_bricks_parents(watched_bricks)
        watched_outputs = VariableFilter(roles=[OUTPUT],
                                         bricks=watched_affected)(cg)
        return [get_var_path(w) for w in watched_outputs], watched_outputs
        
    def init_stats_computer(self, what, **kwargs):
        if what == 'grad':
            names, generator = self.generate_grad_stats(**kwargs)
        elif what == 'activation':
            names, generator = self.generate_activation_stats(**kwargs)
        else:
            raise Exception('Unknown stats computer {}'.format(what))
        cg = ComputationGraph(generator)
        self.stat_functs = cg.get_theano_function()
        self.stat_functs_inputs = [inp.name for inp in cg.inputs]
        self.stat_names = names
    
    def prepare_input_multi(self, inputs):
        search_inputs = {}
        inp_shapes = None

        input_vals = inputs.values()
        max_ndim_id = np.argmax([iv[0].ndim for iv in input_vals])

        vinput = input_vals[max_ndim_id]
        inp_shapes = [inp.shape for inp in vinput]
        sizes = np.array(inp_shapes)
        new_size = sizes.max(axis=0)

        for var in self.inputs.values():
            vinputs = inputs.pop(var.name)
            search_inputs[var.name+'_mask'] = np.stack([resizeArray(np.ones((inp.shape[0],), dtype=np.float32), (new_size[0],))
                                                for inp in vinputs], axis=1)
            search_inputs[var.name] = np.stack([resizeArray(inp, new_size)
                                                for inp in vinputs], axis=1)
            
        if inputs:
            raise Exception(
                'Unknown inputs passed to beam search: {}'.format(
                    inputs.keys()))
        return search_inputs, inp_shapes
    
    def prepare_input_single(self, inputs):
        search_inputs = {}
        inp_shapes = None
        for var in self.inputs.values():
            vinputs = inputs.pop(var.name)
            if inp_shapes is None:
                inp_shapes = [vinputs.shape]
            search_inputs[var.name] = vinputs[:, numpy.newaxis, ...]
            search_inputs[var.name+'_mask'] = np.ones((vinputs.shape[0], 1), dtype=np.float32)
        if inputs:
            raise Exception(
                'Unknown inputs passed to beam search: {}'.format(
                    inputs.keys()))
        return search_inputs, inp_shapes
    
    def prepare_input(self, inputs):
        if isinstance(inputs.values()[0], list):
            return self.prepare_input_multi(inputs)
        else:
            return self.prepare_input_single(inputs)
    
    def compute_stats(self, raw_inputs):
        if not hasattr(self, 'stat_functs'):
            return {}
        raw_inputs = dict(raw_inputs)
        inputs, _ = self.prepare_input(raw_inputs)
        inputs_svals = []
        for input_name in self.stat_functs_inputs:
            inputs_svals.append(inputs[input_name])
        raw_result = self.stat_functs(*inputs_svals)
        result = {}
        for i,name in enumerate(self.stat_names):
            result[name] = raw_result[i]
        return result

    def beam_search_multi(self, inputs, full_pointers=False, decoder_type=None,
                          unpack_if_one=False, **kwargs):
        inputs = dict(inputs)
        search_inputs, inp_shapes = self.prepare_input(inputs)
        search_inputs_svals = []
        for input_name in self.search_function_inputs:
            search_inputs_svals.append(search_inputs[input_name])
        all_result = self.search_function(*search_inputs_svals)
        all_cost, all_outputs, all_pointers_data = all_result[:3]
        pos = [ [ {} for word in xrange(shape[0]) ] for shape in inp_shapes ]
        if len(all_result) > 3:
            for i, name in enumerate(self.search_function_pos_outputs):
                pos_result = all_result[3+i].argmax(axis=2)
                for sentence_num, sentence_shape in enumerate(inp_shapes):
                    for j in xrange(sentence_shape[0]):
                        pos[sentence_num][j][name] = pos_result[j][sentence_num]
        all_pointers = []
        decoder_is_global = False
        if isinstance(decoder_type, list):
            decoder_is_global = True
        for i in xrange(all_pointers_data.shape[1]):
            pointers_data = all_pointers_data[:,i,:]
            pointers_data = pointers_data[:inp_shapes[i][0], :inp_shapes[i][0]]
            decoder = decoder_type[i] if decoder_is_global else decoder_type
            if decoder == 'nonproj':
                pointers = self.find_tree_all(pointers_data)
            else:
                pointers = pointers_data.argmax(axis=1).ravel()
            all_pointers.append(pointers)
        all_output = []
        for i in xrange(all_outputs.shape[1]):
            output = all_outputs[:,i]
            output = output[:inp_shapes[i][0]]
            all_output.append(output)
        
        if unpack_if_one and len(all_output) == 1:
            all_output = all_output[0]
            all_pointers = all_pointers[0]

        if full_pointers:
            return (all_output, all_pointers, all_pointers_data), all_cost, pos
        else:
            return (all_output, all_pointers), all_cost, pos

    def beam_search(self, *args, **kwargs):
        kwargs['unpack_if_one'] = True
        return self.beam_search_multi(*args, **kwargs)
        
    def find_tree_all(self, pointers):
        pointers = np.squeeze(pointers)
        G = {}
        for x in xrange(0, pointers.shape[0] - 1):
            G[x] = {}
            for y in xrange(pointers.shape[0] - 1):
                G[x][y] = 1.0 - pointers[y][x]
        mst = edmonts.edmonts(0, G)
        output = np.zeros((pointers.shape[0],), dtype=np.int)
        for e in mst:
            for v in mst[e]:
                output[v] = e
        output[-1] = pointers.shape[0] - 1
        return output

    def init_generate(self):
        generated = self.get_generate_graph(use_mask=False)
        inputs = [v.copy(name=n) for (n, v) in self.inputs.items()]
        inputs.append(self.n_steps.copy(name='n_steps'))
        self._do_generate = theano.function(inputs, generated)

    def sample(self, inputs, n_steps=None):
        if not hasattr(self, '_do_generate'):
            self.init_generate()
        batch, unused_mask = self.bottom.single_to_batch_inputs(inputs)
        batch['n_steps'] = n_steps if n_steps is not None \
            else int(self.bottom.num_time_steps(**batch) /
                     self.max_decoded_length_scale)
        return self._do_generate(**batch)

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ['_analyze', '_beam_search']:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # To use bricks used on a GPU first on a CPU later
        try:
            emitter = self.generator.readout.emitter
            del emitter._theano_rng
        except:
            pass
