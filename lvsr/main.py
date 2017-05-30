from __future__ import print_function
import time
import logging
import pprint
import math
import os
import cPickle as pickle
import sys
import shutil

import numpy
import matplotlib
from lvsr.algorithms import BurnIn
from blocks_extras.extensions.embed_ipython import EmbedIPython
matplotlib.use('Agg')
import theano
from theano import tensor
from blocks.bricks.lookup import LookupTable
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from blocks.algorithms import (GradientDescent,
                               StepClipping, CompositeRule,
                               Momentum, RemoveNotFinite, AdaDelta,
                               Restrict, VariableClipping, Adam)
from blocks.monitoring import aggregation
from blocks.theano_expressions import l2_norm
from blocks.extensions import (
    FinishAfter, Printing, Timing, ProgressBar, SimpleExtension,
    TrainingExtension, PrintingFilterList)
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.monitoring import (
    TrainingDataMonitoring, DataStreamMonitoring)
from blocks_extras.extensions.plot import Plot
from blocks.extensions.training import TrackTheBest
from blocks.extensions.predicates import OnLogRecord
from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter, get_brick
from blocks.roles import WEIGHT
from blocks.utils import reraise_as, dict_subset
from blocks.search import CandidateNotFoundError
from blocks.select import Selector

from lvsr.datasets import MultilangData
from lvsr.expressions import entropy
from lvsr.extensions import CGStatistics, AdaptiveClipping, Patience
from lvsr.graph import apply_adaptive_noise
from lvsr.utils import SpeechModel, rename
from blocks.serialization import load_parameter_values
from lvsr.log_backends import NDarrayLog

floatX = theano.config.floatX
logger = logging.getLogger(__name__)

default_languages = ['default']

data_params_valid = {}
data_params_train = {}

from lvsr.dependency.recognizer import MultilangDependencyRecognizer
from lvsr.dependency.bricks import DependencyErrorRate, count_errors,\
    AuxiliaryErrorRates


def _gradient_norm_is_none(log):
    return math.isnan(log.current_row.get('total_gradient_norm', 0))


class SwitchOffLengthFilter(SimpleExtension):
    def __init__(self, length_filter, **kwargs):
        self.length_filter = length_filter
        super(SwitchOffLengthFilter, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        self.length_filter.max_length = None
        self.main_loop.log.current_row['length_filter_switched'] = True

class EarlyTermination(FinishAfter):
    def __init__(self, param_name, min_performance_by_epoch, **kwargs):
        kwargs.setdefault('after_epoch', True)
        kwargs.setdefault('before_first_epoch', True)
        self.param_name = param_name
        self.min_performance_by_epoch = min_performance_by_epoch
        logger.info("Will early stop if performance criteria not met: %s %s" % (
                    str(self.min_performance_by_epoch),
                    str(self.param_name)))
        super(EarlyTermination, self).__init__(**kwargs)

    def do(self, *args, **kwargs):

        cur_performance = self.main_loop.log.status[self.param_name]
        if not self.min_performance_by_epoch:
            return
        termination_epoch = 0
        for min_perf, max_epochs in self.min_performance_by_epoch:
            if cur_performance <= min_perf:
                termination_epoch = max(max_epochs, termination_epoch)
        logger.info("Early Term: criterion is %f, will wait for %d epochs" % (
            cur_performance, termination_epoch, ))
        if self.main_loop.log.status['epochs_done'] > termination_epoch:
            self.main_loop.log.current_row['training_finish_requested'] = True

class LoadLog(TrainingExtension):
    """Loads a the log from the checkoint.

    Makes a `LOADED_FROM` record in the log with the dump path.

    Parameters
    ----------
    path : str
        The path to the folder with dump.

    """
    def __init__(self, path, **kwargs):
        super(LoadLog, self).__init__(**kwargs)
        self.path = path[:-4] + '_log.zip'

    def load_to(self, main_loop):

        with open(self.path, "rb") as source:
            loaded_log = pickle.load(source)
            #TODO: remove and fix the printing issue!
            loaded_log.status['resumed_from'] = None
            #make sure that we start a new epoch
            if loaded_log.status.get('epoch_started'):
                logger.warn('Loading a snaphot taken during an epoch. '
                            'Iteration information will be destroyed!')
                loaded_log.status['epoch_started'] = False
        main_loop.log = loaded_log

    def before_training(self):
        if not os.path.exists(self.path):
            logger.warning("No log dump found")
            return
        logger.info("loading log from {}".format(self.path))
        try:
            self.load_to(self.main_loop)
            #self.main_loop.log.current_row[saveload.LOADED_FROM] = self.path
        except Exception:
            reraise_as("Failed to load the state")

def create_recognizer(config, net_config, langs, info_dataset,
              postfix_manager, load_path=None, mask_path=None):
    if 'dependency' in net_config:
        net_config.pop('dependency')
    unification_include = []
    unification_exclude = []
    if 'unification_rules' in net_config:
        ur = net_config.pop('unification_rules')
        unification_include = ur.get('include', [])
        unification_exclude = ur.get('exclude', [])

        
    recognizer = MultilangDependencyRecognizer(langs, info_dataset, postfix_manager, unification_include, unification_exclude, **net_config)

    if recognizer.children[0].soft_pointer:
        global data_params_valid
        global data_params_train
        data_params_valid = {'soften_distributions': {'pointers': (0.0, None)}}
        data_params_train = {'soften_distributions':
                                {'pointers':
                                    (recognizer.children[0].soft_pointer_val,
                                     None)}}

    if load_path:
        recognizer.load_params(load_path)
        unifications = []
        for dest_id in xrange(1, len(recognizer.children)):
            unifications += recognizer.unify_parameters(0, dest_id)
        logger.info("Unified parameters: \n"+
                    pprint.pformat(unifications))
    else:
        for brick_path, attribute_dict in sorted(
                config['initialization'].items(),
                key=lambda (k, v): k.count('/')):
            for attribute, value in attribute_dict.items():
                brick, = Selector(recognizer).select(brick_path).bricks
                setattr(brick, attribute, value)
                brick.push_initialization_config()
        recognizer.initialize()
        unifications = []
        for dest_id in xrange(1, len(recognizer.children)):
            unifications += recognizer.unify_parameters(0, dest_id)
        logger.info("Unified parameters: \n"+
                    pprint.pformat(unifications))
    if mask_path:
        with open(mask_path, 'r') as f:
            mask_dict = pickle.load(f)
            recognizer.activate_masks(mask_dict)


    return recognizer

def create_model(config, data, langs,
                 load_path=None,
                 test_tag=False):
    """
    Build the main brick and initialize or load all parameters.

    Parameters
    ----------

    config : dict
        the configuration dict

    data : object of class Data
        the dataset creation object

    load_path : str or None
        if given a string, it will be used to load model parameters. Else,
        the parameters will be randomly initalized by calling
        recognizer.initialize()

    test_tag : bool
        if true, will add tag the input variables with test values

    """
    # First tell the recognizer about required data sources
    net_config = dict(config["net"])
    addidional_sources = ['labels']
    if 'additional_sources' in net_config:
        addidional_sources += net_config['additional_sources']
    data.default_sources = net_config['input_sources'] + addidional_sources
    net_config['input_sources_dims'] = {}
    for src in net_config['input_sources']:
        net_config['input_sources_dims'][src] = data.num_features(src)
    net_config['additional_sources_dims'] = {}
    for src in addidional_sources:
        net_config['additional_sources_dims'][src] = data.num_features(data.sources_map[src])

    recognizer = create_recognizer(config, net_config, langs, data.info_dataset, data.postfix_manager, load_path)

    if test_tag:
        # fails with newest theano
        # tensor.TensorVariable.__str__ = tensor.TensorVariable.__repr__
        __stream = data.get_stream("train")
        __data = next(__stream.get_epoch_iterator(as_dict=True))
        for __var in recognizer.inputs.values() + [
                recognizer.inputs_mask, recognizer.labels, recognizer.labels_mask
                ] + recognizer.additional_sources.values():
            __var.tag.test_value = __data[__var.name]
        theano.config.compute_test_value = 'warn'
    return recognizer

def initialize_all(config, test_tag, save_path, bokeh_name,
                   params, bokeh_server, bokeh, use_load_ext,
                   load_log, fast_start):
    langs = default_languages
    if 'input_languages' in config['data']:
        langs = config['data'].pop('input_languages')
    data = MultilangData(langs, **config['data'])
    recognizer = create_model(config, data, test_tag=test_tag, langs=langs)
    regularized = initialize_graph(recognizer, data, config, params)
    return initialaze_algorithm(config, save_path, bokeh_name, params, bokeh_server,
                                bokeh, use_load_ext, load_log, fast_start, **regularized)
    
def initialize_graph(recognizer, data, config, params):
    # Separate attention_params to be handled differently
    # when regularization is applied
    attentions = recognizer.all_children().generator.transition.attention.get()
    attention_params = [Selector(attention).get_parameters().values()
                        for attention in attentions]

    logger.info(
        "Initialization schemes for all bricks.\n"
        "Works well only in my branch with __repr__ added to all them,\n"
        "there is an issue #463 in Blocks to do that properly.")

    def show_init_scheme(cur):
        result = dict()
        for attr in dir(cur):
            if attr.endswith('_init'):
                result[attr] = getattr(cur, attr)
        for child in cur.children:
            result[child.name] = show_init_scheme(child)
        return result
    logger.info(pprint.pformat(show_init_scheme(recognizer)))

    observables = []  # monitored each batch
    cg = recognizer.get_cost_graph(batch=True)
    labels = []
    labels_mask = []
    for chld in recognizer.children:
        lbls = VariableFilter(applications=[chld.cost], name='labels'+chld.names_postfix)(cg)
        lbls_mask = VariableFilter(applications=[chld.cost], name='labels_mask'+chld.names_postfix)(cg)
        if len(lbls) == 1:
            labels += lbls
            labels_mask += lbls_mask

    batch_cost = cg.outputs[0].sum()
    batch_size = rename(labels[0].shape[1], "batch_size")
    # Assumes constant batch size. `aggregation.mean` is not used because
    # of Blocks #514.
    cost = batch_cost / batch_size

    cost.name = "sequence_total_cost"
    logger.info("Cost graph is built")

    # Fetch variables useful for debugging.
    # It is important not to use any aggregation schemes here,
    # as it's currently impossible to spread the effect of
    # regularization on their variables, see Blocks #514.
    cost_cg = ComputationGraph(cost)
    
    bottom_output = VariableFilter(
        # We need name_regex instead of name because LookupTable calls itsoutput output_0
        applications=recognizer.all_children().bottom.apply.get(), name_regex="output")(
            cost_cg)
    
    attended = VariableFilter(
        applications=recognizer.all_children().generator.transition.apply.get(), name="attended")(
            cost_cg)
    attended_mask = VariableFilter(
        applications=recognizer.all_children().generator.transition.apply.get(), name="attended_mask")(
            cost_cg)
    weights = VariableFilter(
        applications=recognizer.all_children().generator.evaluate.get(), name="weights")(
            cost_cg)
    
    def get_renamed_list(rlist, elem_func, elem_name):
        return [rename(elem_func(elem), elem_name+chld.names_postfix)
                    for elem,chld in zip(rlist, recognizer.children)]
        
    max_sentence_lengths = get_renamed_list(bottom_output,
                                            lambda e: e.shape[0],
                                            "max_sentence_length")
    max_attended_mask_lengths = get_renamed_list(attended_mask,
                                            lambda e: e.shape[0],
                                            "max_attended_mask_length")
    max_attended_lengths = get_renamed_list(attended,
                                            lambda e: e.shape[0],
                                            "max_attended_length")
    max_num_characters = get_renamed_list(labels,
                                            lambda e: e.shape[0],
                                            "max_num_characters")
    
    mean_attended = get_renamed_list(attended,
                                            lambda e: abs(e).mean(),
                                            "mean_attended")
    mean_bottom_output = get_renamed_list(bottom_output,
                                            lambda e: abs(e).mean(),
                                            "mean_bottom_output")
    
    mask_density = get_renamed_list(labels_mask,
                                            lambda e: e.mean(),
                                            "mask_density")
    weights_entropy = [rename(entropy(w, lm),
                             "weights_entropy"+chld.names_postfix)
                       for w, lm, chld in zip(weights, labels_mask, recognizer.children)]

    observables += max_attended_lengths + max_attended_mask_lengths + max_sentence_lengths
    #
    # Monitoring of cost terms is tricky because of Blocks #514 - since the
    # costs are annotations that are not part of the original output graph,
    # they are unaffected by replacements such as dropout!!
    #
    cost_terms = []
    for chld in recognizer.children:
        chld_cost_terms = VariableFilter(applications=[chld.generator.evaluate],
                                name_regex='.*_nll')(cost_cg)
        chld_cost_terms = [rename(var, var.name[:-4] + chld.names_postfix + '_nll')
                           for var in chld_cost_terms]
        cost_terms += chld_cost_terms
        
    cg = ComputationGraph([cost, batch_size] +
        weights_entropy + mean_attended +
        mean_bottom_output + max_num_characters +
        mask_density + cost_terms)

    # Regularization. It is applied explicitly to all variables
    # of interest, it could not be applied to the cost only as it
    # would not have effect on auxiliary variables, see Blocks #514.
    reg_config = config['regularization']
    regularized_cg = cg

    if reg_config.get('dropout'):
        drop_conf = reg_config['dropout']
        bot_drop = drop_conf.get('bottom', 0.0)
        if bot_drop:
            logger.info('apply bottom dropout')
            regularized_cg = apply_dropout(regularized_cg,
                                           bottom_output, bot_drop)
        enc_drop = drop_conf.get('encoder', 0.0)
        if enc_drop:
            logger.info('apply encoder dropout')
            enc_bricks = reduce(lambda acc,x: acc+list(x), recognizer.all_children().encoder.children.get(), [])
            enc_states = VariableFilter(bricks=enc_bricks,
                                        name_regex='states')(regularized_cg)
            regularized_cg = apply_dropout(regularized_cg,
                                           enc_states,
                                           enc_drop)
        post_merge_drop = drop_conf.get('post_merge', 0.0)
        if post_merge_drop:
            logger.info('apply post_merge dropout')
            pm_bricks = []
            for chld in recognizer.children:
                cpm_bricks = list(chld.generator.readout.post_merge.children)
                cpm_bricks += cpm_bricks[-1].children
                cpm_bricks = [b for b in cpm_bricks if
                             isinstance(b, type(chld.post_merge_activation))]
                pm_bricks += cpm_bricks
            regularized_cg = apply_dropout(
                regularized_cg,
                VariableFilter(bricks=pm_bricks, name='output')(regularized_cg),
                post_merge_drop)

    if reg_config.get('noise'):
        logger.info('apply noise')
        noise_subjects = [p for p in cg.parameters if p not in attention_params]
        regularized_cg = apply_noise(cg, noise_subjects, reg_config['noise'])

    train_cost = regularized_cg.outputs[0]

    if reg_config.get("penalty_coof", .0) > 0:
        # big warning!!!
        # here we assume that:
        # regularized_weights_penalty = regularized_cg.outputs[1]
        train_cost = (train_cost +
                      reg_config.get("penalty_coof", .0) *
                      regularized_cg.outputs[1] / batch_size)
        
    if reg_config.get("decay", .0) > 0:
        train_cost = (train_cost + reg_config.get("decay", .0) *
                      l2_norm(VariableFilter(roles=[WEIGHT])(cg.parameters)) ** 2)

    train_cost = train_cost.copy(name='train_cost')

    gradients = None
    if reg_config.get('adaptive_noise'):
        logger.info('apply adaptive noise')
        if ((reg_config.get("penalty_coof", .0) > 0) or
                (reg_config.get("decay", .0) > 0)):
            logger.error('using  adaptive noise with alignment weight panalty '
                         'or weight decay is probably stupid')
        train_cost, regularized_cg, gradients, noise_brick = apply_adaptive_noise(
            cg, cg.outputs[0],
            variables=cg.parameters,
            num_examples=data.get_dataset('train').num_examples,
            parameters=SpeechModel(regularized_cg.outputs[0]
                                   ).get_parameter_dict().values(),
            **reg_config.get('adaptive_noise')
        )
        train_cost.name = 'train_cost'
        adapt_noise_cg = ComputationGraph(train_cost)
        model_prior_mean = rename(
            VariableFilter(applications=[noise_brick.apply],
                           name='model_prior_mean')(adapt_noise_cg)[0],
            'model_prior_mean')
        model_cost = rename(
            VariableFilter(applications=[noise_brick.apply],
                           name='model_cost')(adapt_noise_cg)[0],
            'model_cost')
        model_prior_variance = rename(
            VariableFilter(applications=[noise_brick.apply],
                           name='model_prior_variance')(adapt_noise_cg)[0],
            'model_prior_variance')
        regularized_cg = ComputationGraph(
            [train_cost, model_cost] +
            regularized_cg.outputs +
            [model_prior_mean, model_prior_variance])
        observables += [
            regularized_cg.outputs[1],  # model cost
            regularized_cg.outputs[2],  # task cost
            regularized_cg.outputs[-2],  # model prior mean
            regularized_cg.outputs[-1]]  # model prior variance

    if len(cost_terms):
        # Please note - the aggragation (mean) is done in
        # "attach_aggregation_schemes"
        ct_names = [v.name for v in cost_terms]
        for v in regularized_cg.outputs:
            if v.name in ct_names:
                observables.append(rename(v.sum()/batch_size,
                                                  v.name))
    for chld in recognizer.children:
        if chld.train_tags:
            tags_cost = VariableFilter(applications=[chld.addTagCost],
                                       name='output')(regularized_cg)[0]
            observables += [rename(tags_cost.sum()/batch_size, 'tags_nll'+chld.names_postfix)]

    # Model is weird class, we spend lots of time arguing with Bart
    # what it should be. However it can already nice things, e.g.
    # one extract all the parameters from the computation graphs
    # and give them hierahical names. This help to notice when a
    # because of some bug a parameter is not in the computation
    # graph.
    model = SpeechModel(train_cost)
    if params:
        logger.info("Load parameters from " + params)
        # please note: we cannot use recognizer.load_params
        # as it builds a new computation graph that dies not have
        # shapred variables added by adaptive weight noise
        param_values = load_parameter_values(params)
        model.set_parameter_values(param_values)

    parameters = model.get_parameter_dict()

    logger.info("Parameters:\n" +
                pprint.pformat(
                    [(key, parameters[key].get_value().shape) for key
                     in sorted(parameters.keys())],
                    width=120))
    max_norm_rules = []
    if reg_config.get('max_norm', False) > 0:
        logger.info("Apply MaxNorm")
        maxnorm_subjects = VariableFilter(roles=[WEIGHT])(cg.parameters)
        if reg_config.get('max_norm_exclude_lookup', False):
            maxnorm_subjects = [v for v in maxnorm_subjects
                                if not isinstance(get_brick(v), LookupTable)]
        logger.info("Parameters covered by MaxNorm:\n"
                    + pprint.pformat([name for name, p in parameters.items()
                                      if p in maxnorm_subjects]))
        logger.info("Parameters NOT covered by MaxNorm:\n"
                    + pprint.pformat([name for name, p in parameters.items()
                                      if not p in maxnorm_subjects]))
        max_norm_rules = [
            Restrict(VariableClipping(reg_config['max_norm'], axis=0),
                     maxnorm_subjects)]

        
    return { 'observables': observables, 'max_norm_rules': max_norm_rules,
             'cg': cg, 'regularized_cg' : regularized_cg, 'train_cost' : train_cost,
             'cost' : cost, 'batch_size' : batch_size, 'batch_cost' : batch_cost,
             'parameters' : parameters, 'gradients': gradients, 
             'model' : model, 'data' : data, 'recognizer' : recognizer,
             'weights_entropy' : weights_entropy,
             'labels_mask' : labels_mask, 'labels' : labels }

def initialaze_algorithm(config, save_path, bokeh_name, params, bokeh_server,
                         bokeh, use_load_ext, load_log, fast_start, 
                         recognizer, data, model, cg, regularized_cg,
                         cost, train_cost, parameters, 
                         max_norm_rules, observables,
                         batch_size, batch_cost, weights_entropy, 
                         labels_mask, labels,  gradients=None):
    primary_observables = observables
    secondary_observables = []
    validation_observables = []
    root_path, extension = os.path.splitext(save_path)
    train_conf = config['training']
    # Define the training algorithm.
    clipping = StepClipping(train_conf['gradient_threshold'])
    clipping.threshold.name = "gradient_norm_threshold"
    rule_names = train_conf.get('rules', ['momentum'])
    core_rules = []
    if 'momentum' in rule_names:
        logger.info("Using scaling and momentum for training")
        core_rules.append(Momentum(train_conf['scale'], train_conf['momentum']))
    if 'adadelta' in rule_names:
        logger.info("Using AdaDelta for training")
        core_rules.append(AdaDelta(train_conf['decay_rate'], train_conf['epsilon']))
    if 'adam' in rule_names:
        assert len(rule_names) == 1
        logger.info("Using Adam for training")
        core_rules.append(
            Adam(learning_rate=train_conf.get('scale', 0.002),
                 beta1=train_conf.get('beta1', 0.1),
                 beta2=train_conf.get('beta2', 0.001),
                 epsilon=train_conf.get('epsilon', 1e-8),
                 decay_factor=train_conf.get('decay_rate', (1 - 1e-8))))
    burn_in = []
    if train_conf.get('burn_in_steps', 0):
        burn_in.append(
            BurnIn(num_steps=train_conf['burn_in_steps']))
    algorithm = GradientDescent(
        cost=train_cost,
        parameters=parameters.values(),
        gradients=gradients,
        step_rule=CompositeRule(
            [clipping] + core_rules + max_norm_rules +
            # Parameters are not changed at all
            # when nans are encountered.
            [RemoveNotFinite(0.0)] + burn_in),
        on_unused_sources='warn')
        #theano_func_kwargs={'mode':NanGuardMode(nan_is_error=True)})

    logger.debug("Scan Ops in the gradients")
    gradient_cg = ComputationGraph(algorithm.gradients.values())
    for op in ComputationGraph(gradient_cg).scans:
        logger.debug(op)

    # More variables for debugging: some of them can be added only
    # after the `algorithm` object is created.
    secondary_observables += list(regularized_cg.outputs)
    if not 'train_cost' in [v.name for v in secondary_observables]:
        secondary_observables += [train_cost]
    secondary_observables += [
        algorithm.total_step_norm, algorithm.total_gradient_norm,
        clipping.threshold]
    for name, param in parameters.items():
        num_elements = numpy.product(param.get_value().shape)
        norm = param.norm(2) / num_elements ** 0.5
        grad_norm = algorithm.gradients[param].norm(2) / num_elements ** 0.5
        step_norm = algorithm.steps[param].norm(2) / num_elements ** 0.5
        stats = tensor.stack(norm, grad_norm, step_norm, step_norm / grad_norm)
        stats.name = name + '_stats'
        secondary_observables.append(stats)

    primary_observables += [
        train_cost,
        algorithm.total_gradient_norm,
        algorithm.total_step_norm, clipping.threshold]

    validation_observables += [
        rename(aggregation.mean(batch_cost, batch_size), cost.name),
        rename(aggregation.sum_(batch_size), 'num_utterances')] + weights_entropy


    def attach_aggregation_schemes(variables):
        # Aggregation specification has to be factored out as a separate
        # function as it has to be applied at the very last stage
        # separately to training and validation observables.
        result = []
        for var in variables:
            if var.name.startswith('weights_entropy'):
                chld_id = recognizer.child_id_from_postfix(var.name)
                result.append(rename(aggregation.mean(var, labels_mask[chld_id].sum()),
                                     'weights_entropy_per_label'+
                                     recognizer.children[chld_id].names_postfix))
            elif var.name.endswith('_nll'):
                chld_id = recognizer.child_id_from_postfix(var.name)
                result.append(rename(aggregation.mean(var.sum(),
                                                      labels_mask[chld_id].sum()),
                                     var.name+'_per_label'))
            else:
                result.append(var)
        return result

    mon_conf = config['monitoring']
    # Build main loop.
    logger.info("Initialize extensions")
    extensions = []
    if use_load_ext and params:
        extensions.append(Load(params, load_iteration_state=True, load_log=True))
    if load_log and params:
        extensions.append(LoadLog(params))
    extensions += [
        Timing(after_batch=True),
        CGStatistics(),
        #CodeVersion(['lvsr']),
    ]
    extensions.append(TrainingDataMonitoring(
        primary_observables, after_batch=True))
    average_monitoring = TrainingDataMonitoring(
        attach_aggregation_schemes(secondary_observables),
        prefix="average", every_n_batches=10)
    extensions.append(average_monitoring)
    validation = DataStreamMonitoring(
        attach_aggregation_schemes(validation_observables),
        data.get_stream("valid", shuffle=False, **data_params_valid), prefix="valid").set_conditions(
            before_first_epoch=not fast_start,
            every_n_epochs=mon_conf['validate_every_epochs'],
            every_n_batches=mon_conf['validate_every_batches'],
            after_training=False)
    extensions.append(validation)

    additional_patience_notifiers = []
    uas = DependencyErrorRate(recognizer.children[0], data,
                              **config['monitoring']['search'])
    las = AuxiliaryErrorRates(uas, name='LAS')
    lab = AuxiliaryErrorRates(uas, name='LAB')
    per_monitoring = DataStreamMonitoring(
        [uas, las, lab], data.get_one_stream("valid", data.langs[0], batches=False, shuffle=False, **data_params_valid)[0],
        prefix="valid").set_conditions(
                before_first_epoch=not fast_start,
                every_n_epochs=mon_conf['search_every_epochs'],
                every_n_batches=mon_conf['search_every_batches'],
                after_training=False)
    extensions.append(per_monitoring)
    track_the_best_uas = TrackTheBest(
        per_monitoring.record_name(uas)).set_conditions(
            before_first_epoch=True, after_epoch=True)
    track_the_best_las = TrackTheBest(
        per_monitoring.record_name(las)).set_conditions(
            before_first_epoch=True, after_epoch=True)
    track_the_best_lab = TrackTheBest(
        per_monitoring.record_name(lab)).set_conditions(
            before_first_epoch=True, after_epoch=True)
    extensions += [track_the_best_uas,
                   track_the_best_las,
                   track_the_best_lab,
                   ]
    per = uas
    track_the_best_per = track_the_best_uas
    additional_patience_notifiers = [track_the_best_lab,
                                     track_the_best_las]
    track_the_best_cost = TrackTheBest(
        validation.record_name(cost)).set_conditions(
            before_first_epoch=True, after_epoch=True)
    extensions += [track_the_best_cost]
    extensions.append(AdaptiveClipping(
        algorithm.total_gradient_norm.name,
        clipping, train_conf['gradient_threshold'],
        decay_rate=0.998, burnin_period=500,
        num_stds=train_conf.get('clip_stds', 1.0)))
    extensions += [
        SwitchOffLengthFilter(
            data.length_filter,
            after_n_batches=train_conf.get('stop_filtering')),
        FinishAfter(after_n_batches=train_conf['num_batches'],
                    after_n_epochs=train_conf['num_epochs']),
            # .add_condition(["after_batch"], _gradient_norm_is_none),
    ]
    main_postfix = recognizer.children[0].names_postfix
    channels = [
        # Plot 1: training and validation costs
        [average_monitoring.record_name(train_cost),
         validation.record_name(cost)],
        # Plot 2: gradient norm,
        [average_monitoring.record_name(algorithm.total_gradient_norm),
         average_monitoring.record_name(clipping.threshold)],
        # Plot 3: phoneme error rate
        [per_monitoring.record_name(per)],
        # Plot 4: training and validation mean weight entropy
        [average_monitoring._record_name('weights_entropy_per_label'+main_postfix),
         validation._record_name('weights_entropy_per_label'+main_postfix)],
        # Plot 5: training and validation monotonicity penalty
        [average_monitoring._record_name('weights_penalty_per_recording'+main_postfix),
         validation._record_name('weights_penalty_per_recording'+main_postfix)]]
    if bokeh:
        extensions += [
            Plot(bokeh_name if bokeh_name
                 else os.path.basename(save_path),
                 channels,
                 every_n_batches=10,
                 server_url=bokeh_server),]
    extensions += [
        Checkpoint(save_path,
                   before_first_epoch=not fast_start, after_epoch=True,
                   every_n_batches=train_conf.get('save_every_n_batches'),
                   save_separately=["model", "log"],
                   use_cpickle=True)
        .add_condition(
            ['after_epoch'],
            OnLogRecord(track_the_best_per.notification_name),
            (root_path + "_best" + extension,))
        .add_condition(
            ['after_epoch'],
            OnLogRecord(track_the_best_cost.notification_name),
            (root_path + "_best_ll" + extension,)),
        ProgressBar()]
    extensions.append(EmbedIPython(use_main_loop_run_caller_env=True))

    if train_conf.get('patience'):
        patience_conf = train_conf['patience']
        if not patience_conf.get('notification_names'):
            # setdefault will not work for empty list
            patience_conf['notification_names'] = [
                track_the_best_per.notification_name,
                track_the_best_cost.notification_name] + additional_patience_notifiers
        extensions.append(Patience(**patience_conf))

    if train_conf.get('min_performance_stops'):
        extensions.append(EarlyTermination(
            param_name=track_the_best_per.best_name,
            min_performance_by_epoch=train_conf['min_performance_stops']))

    extensions.append(Printing(every_n_batches=1,
                               attribute_filter=PrintingFilterList()))

    return model, algorithm, data, extensions


def train(config, save_path, bokeh_name,
          params, bokeh_server, bokeh, test_tag, use_load_ext,
          load_log, fast_start):

    conf_dump = pickle.dumps(config, protocol=0)

    model, algorithm, data, extensions = initialize_all(
        config, test_tag, save_path, bokeh_name,
        params, bokeh_server, bokeh, use_load_ext,
        load_log, fast_start)
    data.get_stream("train", **data_params_train)

    dataset_dump = pickle.dumps(data.info_dataset, protocol=0)
    postfix_dump = pickle.dumps(data.postfix_manager, protocol=0) 

    # Save the config into the status
    log = NDarrayLog()
    log.status['_config'] = repr(config)
    log.status['_config_pickle'] = repr(conf_dump)
    log.status['_dataset_pickle'] = repr(dataset_dump)
    log.status['_postfix_pickle'] = repr(postfix_dump)

    main_loop = MainLoop(
        model=model, log=log, algorithm=algorithm,
        data_stream=data.get_stream("train", **data_params_train),
        extensions=extensions)
    main_loop.conf_pickle_shared = theano.shared(
        numpy.frombuffer(conf_dump, numpy.byte),
        name='_config_pickle')
    main_loop.data_pickle_shared = theano.shared(
        numpy.frombuffer(dataset_dump, numpy.byte),
        name='_dataset_pickle')
    main_loop.post_pickle_shared = theano.shared(
        numpy.frombuffer(postfix_dump, numpy.byte),
        name='_postfix_pickle')
    main_loop.run()
    return main_loop

def search(config, params, load_path, part, decode_only, report,
           decoded_save, nll_only, seed, decoder_type, lang=None,
           collect_stats='none', **kwargs):
    matplotlib.use("Agg")
    if load_path == 'None':
        load_path = None
    langs = default_languages
        
    if 'input_languages' in config['data']:
        langs = config['data'].pop('input_languages')
        
    if lang is None:
        lang_id = 0
    else:
        assert lang in langs
        lang_id = langs.index(lang)

    data = MultilangData(langs, **config['data'])
    add_sources = ()
    logger.info("Recognizer initialization started")
    global_recognizer = create_model(config, data, langs, load_path)
    dataset = data.get_dataset(part, langs[lang_id], add_sources)
    stream,_ = data.get_one_stream(part, batches=False,
                             lang=langs[lang_id],
                             shuffle=part == 'train',
                             add_sources=add_sources,
                             num_examples=500 if part == 'train' else None,
                             seed=seed, only_stream=True)

    search_conf = config['monitoring']['search']
    recognizer = global_recognizer.children[lang_id]
    recognizer.init_beam_search(search_conf['beam_size'])
    logger.info("Recognizer is initialized")
    if collect_stats != 'none':
        logger.info("Stats collecter initialization started")
        recognizer.init_stats_computer(collect_stats,
                                       watched_bricks=global_recognizer.unified_parameters)
        logger.info("Stats collecter is initialized")

    it = stream.get_epoch_iterator(as_dict=True)
    if decode_only is not None:
        decode_only = eval(decode_only)

    print_to = sys.stdout

    decoded_file = None
    if decoded_save:
        decoded_file = open(decoded_save, 'w')

    num_examples = .0
    total_errors = .0
    total_errors_labels = .0
    total_errors_pointers = .0
    total_errors_pos = .0
    total_length = .0
    not_recognized_count = 0
    recognized_count = 0

    report_data = []

    if config.get('vocabulary'):
        with open(os.path.expandvars(config['vocabulary'])) as f:
            vocabulary = dict(line.split() for line in f.readlines())

        def to_words(chars):
            words = chars.split()
            words = [vocabulary[word] if word in vocabulary
                     else vocabulary['<UNK>'] for word in words]
            return words

    for number, example in enumerate(it):
        if decode_only and number not in decode_only:
            continue
        uttids = example.pop('uttids', None)
        example = {(k+recognizer.names_postfix):v for k,v in example.iteritems()}
        raw_groundtruth = example.pop('labels'+recognizer.names_postfix)
        required_inputs = dict_subset(example, recognizer.inputs.keys())

        print("Utterance {} ({})".format(number, uttids), file=print_to)

        groundtruth = dataset.decode(raw_groundtruth)
        groundtruth_pointers = example.pop('pointers'+recognizer.names_postfix)[1:-1]
        groundtruth_text = dataset.print_text(required_inputs)
        num_examples += 1
        print("Groundtruth:", groundtruth, file=print_to)
        print("Groundtruth Pointers:", repr(groundtruth_pointers.tolist()), file=print_to)
        print("Groundtruth text:", groundtruth_text, file=print_to)

        if nll_only:
            print_to.flush()
            continue

        before = time.time()
        try:
            outputs, search_cost, pos_output = recognizer.beam_search(
                required_inputs,
                char_discount=search_conf['char_discount'],
                round_to_inf=search_conf['round_to_inf'],
                stop_on=search_conf['stop_on'],
                full_pointers = True, decoder_type=decoder_type,
                validate_solution_function=getattr(data.info_dataset,
                                                   'validate_solution',
                                                   None))
            stats = recognizer.compute_stats(required_inputs)
        except CandidateNotFoundError:
            outputs = ()
            pos_output = []
            search_cost = numpy.NaN

        took = time.time() - before
        if outputs != ():
            recognized = dataset.decode(outputs[0])
            recognized_pointers = outputs[1][1:-1]
            recognized_text = dataset.pretty_print(outputs[0], example)
        else:
            recognized = None
            recognized_pointers = None
            recognized_text = None

        if recognized:
            error_labels = count_errors(groundtruth, recognized)
            error_pointers = count_errors(groundtruth_pointers, recognized_pointers)
            error = count_errors(zip(groundtruth,groundtruth_pointers), zip(recognized,recognized_pointers))
            pos_good = 0
            pos_bad = 0
            for j, word in enumerate(pos_output[0]):
                for pos_name,pos_value in word.iteritems():
                    if recognizer.names_postfix != '':
                        pos_clean_name = pos_name[:-len(recognizer.names_postfix)]
                    else:
                        pos_clean_name = pos_name
                    vmap = dataset.value_maps[pos_clean_name].num2tok
                    if vmap[example[pos_name][j]] == 'UNK' and \
                       vmap[pos_value] == 'UNK':
                        continue
                    if pos_value == example[pos_name][j]:
                        pos_good += 1
                    else:
                        pos_bad += 1
            if pos_good + pos_bad > 0:
                error_pos = pos_bad*1.0/(pos_good+pos_bad)
            else:
                error_pos = 1.0
            #error = (error_labels+error_pointers)/2.0
        else:
            error_labels = 1.0
            error_pointers = 1.0
            error = 1.0
            error_pos = 1.0
        total_errors += len(groundtruth) * error
        total_errors_labels += len(groundtruth) * error_labels
        total_errors_pointers += len(groundtruth) * error_pointers
        total_errors_pos += len(groundtruth) * error_pos
        total_length += len(groundtruth)

        if recognized_pointers is not None:
            report_data.append( {
                          'groundtruth': groundtruth,
                          'groundtruth_pointers' : groundtruth_pointers,
                          'recognized' : recognized,
                          'recognized_pointers' : recognized_pointers.tolist(),
                          'recognized_pointers_full' : outputs[2] if outputs != () else None,
                          'error_labels' : error_labels,
                          'error_pointers' : error_pointers,
                          'search_cost' : search_cost,
                          'stats' : stats
                          } )

        if decoded_file is not None:
            print("{} {}".format(uttids, ' '.join(recognized)),
                  file=decoded_file)

        print("Decoding took:", took, file=print_to)
        if recognized:
            recognized_count += 1
            print("Recognized:", repr(recognized_text), file=print_to)
            print("Recognized (pointers):", repr(recognized_pointers.tolist()), file=print_to)
        else:
            not_recognized_count += 1
            print("Didn't recognize")
        print("CER:", error, file=print_to)
        print("CER(labels):", error_labels, file=print_to)
        print("CER(pointers):", error_pointers, file=print_to)
        print("CER(pos):", error_pos, file=print_to)
        print("Average CER (LAS):", (1.0 - total_errors / total_length)*100.0, file=print_to)
        print("Average CER(labels):", (1.0 - total_errors_labels / total_length)*100.0, file=print_to)
        print("Average CER(pointers) (UAS):", (1.0 - total_errors_pointers / total_length)*100.0, file=print_to)
        print("Average CER(pos):", (1.0 - total_errors_pos / total_length)*100.0, file=print_to)
        print("Recognized "+str(recognized_count*100.0/(recognized_count+not_recognized_count)) +"%")
        print_to.flush()

    if report:
        with open(report, 'w') as f:
            pickle.dump(report_data, f)

def sample(config, params, load_path, part):
    #data = Data(**config['data'])
    data = MultilangData(**config['data'])
    recognizer = create_model(config, data, load_path)

    dataset = data.get_dataset(part, add_sources=('uttids',))
    stream = data.get_stream(part, batches=False, shuffle=False,
                             add_sources=('uttids',))
    it = stream.get_epoch_iterator(as_dict=True)

    print_to = sys.stdout
    for number, data in enumerate(it):
        uttids = data.pop('uttids', None)
        print("Utterance {} ({})".format(number, uttids),
              file=print_to)
        raw_groundtruth = data.pop('labels')
        groundtruth_text = dataset.pretty_print(raw_groundtruth)
        print("Groundtruth:", groundtruth_text, file=print_to)
        sample = recognizer.sample(data)['outputs'][:, 0]
        recognized_text = dataset.pretty_print(sample)
        print("Recognized:", recognized_text, file=print_to)

def train_multistage(config, save_path, bokeh_name, params, start_stage, **kwargs):
    """Run multiple stages of the training procedure."""
    override_file = kwargs.pop('override_file')
    if config.multi_stage:
        if not start_stage:
            if override_file and os.path.exists(save_path):
                shutil.rmtree(save_path)
            os.mkdir(save_path)
        start_stage = (list(config.ordered_stages).index(start_stage)
                       if start_stage else 0)
        stages = list(config.ordered_stages.items())
        for number in range(start_stage, len(stages)):
            stage_name, stage_config = stages[number]
            logging.info("Stage \"{}\" config:\n".format(stage_name)
                         + pprint.pformat(stage_config, width=120))
            stage_save_path = '{}/{}.zip'.format(save_path, stage_name)
            stage_bokeh_name = '{}_{}'.format(save_path, stage_name)
            if number and not params:
                stage_params = '{}/{}{}.zip'.format(
                    save_path, stages[number - 1][0],
                    stage_config['training'].get('restart_from', ''))
            else:
                stage_params = params
                # Avoid loading the params twice
                params = None

            ret = train(stage_config, stage_save_path, stage_bokeh_name,
                        stage_params, **kwargs)
        return ret
    else:
        return train(config, save_path, bokeh_name, params, **kwargs)

def show_config(config, *args, **kwargs):
    pprint.pprint(config)

def test(config, **kwargs):
    raise NotImplementedError()
