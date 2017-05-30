from collections import OrderedDict
from theano import tensor

from blocks.bricks import Initializable, Tanh
from blocks.bricks.base import application, lazy
from blocks.bricks.parallel import Fork
from blocks.utils import dict_union, dict_subset
from blocks.roles import add_role, COST

from blocks.bricks.sequence_generators import FakeAttentionRecurrent, SoftmaxEmitter
from blocks.bricks.sequences import MLP
from lvsr.dependency.debug import debugTheanoVar
from blocks.bricks.simple import Linear
from blocks.bricks.recurrent import BaseRecurrent, recurrent


class Generator(Initializable):
    @lazy()
    def __init__(self, readout, transition, dim_dec, attention=None,
                 add_contexts=True, pointer_weight=0.5,
                 transition_with_att_class=None,
                 use_word_annotations=False, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.inputs = [name for name in transition.apply.sequences
                       if 'mask' not in name]
        self.dim_dec = dim_dec
        self.pointer_weight = pointer_weight
        fork = Fork(self.inputs)
        kwargs.setdefault('fork', fork)
        if attention:
            transition = transition_with_att_class(
                transition, attention,
                add_contexts=add_contexts, name="att_trans")
        else:
            transition = FakeAttentionRecurrent(transition,
                                                name="with_fake_attention")
        self.readout = readout
        self.transition = transition
        self.fork = fork
        self.children = [self.readout, self.fork, self.transition]

        self.use_word_annotations = use_word_annotations
        if use_word_annotations:
            self.word_annotation_preprocessor = Linear(
                name='input_attention_preprocessor', bias=False)
            self.children.append(self.word_annotation_preprocessor)

    def _push_allocation_config(self):
        # Configure readout. That involves `get_dim` requests
        # to the transition. To make sure that it answers
        # correctly we should finish its configuration first.
        self.transition.push_allocation_config()
        transition_sources = (self._state_names + self._context_names +
                              self._glimpse_names)
        self.readout.source_dims = []
        for name in self.readout.source_names:
            if name in transition_sources:
                self.readout.source_dims.append(self.transition.get_dim(name))
            elif self.language_model and name in self._lm_state_names:
                self.readout.source_dims.append(
                    self.get_dim(name))
            else:
                self.readout.get_dim(name)

        # Configure fork. For similar reasons as outlined above,
        # first push `readout` configuration.
        self.readout.push_allocation_config()
        feedback_name, = self.readout.feedback.outputs
        self.fork.input_dim = self.readout.get_dim(feedback_name)
        self.fork.output_dims = self.transition.get_dims(
            self.fork.apply.outputs)

    @property
    def _state_names(self):
        return self.transition.compute_states.outputs

    @property
    def _context_names(self):
        return self.transition.apply.contexts

    @property
    def _glimpse_names(self):
        return self.transition.take_glimpses.outputs

    @application
    def evaluate(self, application_call, outputs, mask=None, **kwargs):
        pointers = outputs[1]
        outputs = outputs[0]
        # We assume the data has axes (time, batch, features, ...)
        #batch_size = outputs.shape[1]
        states = dict_subset(kwargs, self._state_names, must_have=False)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)

        inputs = self.fork.apply(contexts['attended'], as_dict=True)
        additional_inputs = {}
        if 'wordannotations'in self.transition.do_apply.sequences:
            additional_inputs['wordannotations'] = contexts['attended']
        if 'pointers' in self.transition.do_apply.sequences:
            if pointers.ndim == 3:
                additional_inputs['pointers'] = pointers
            else:
                additional_inputs['pointers'] = tensor.eye(
                        pointers.shape[0], pointers.shape[0])[pointers]

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts, additional_inputs))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = OrderedDict((name, results[name][:-1]) for name in self._state_names)
        glimpses = OrderedDict((name, results[name][1:]) for name in self._glimpse_names)

        # Compute the cost
        readouts = self.readout.readout(
            **dict_union(states, glimpses, contexts))

        costs = self.readout.cost(readouts, outputs)
        if mask is not None:
            costs *= mask

        weights = glimpses['weights']
        ######
        # weights has shape: (time=elem, batch, elem)
        # pointers has shape: (elem, batch)
        ######

        if pointers.ndim == 3:
            wtp = (weights*pointers).sum(axis=-1)
            if mask is not None:
                wtp += -(mask - 1) #prevent 0 from goint to inf
            weights_cost = -tensor.log(wtp)
        else:
            pointers_flat = pointers.flatten()
            pointers_flat += tensor.arange(pointers_flat.shape[0]) * weights.shape[-1]
            weights_cost = -tensor.log(
                weights.flatten()[pointers_flat
                                  ].reshape(pointers.shape, pointers.ndim) + 1e-18)
        if mask is not None:
            weights_cost *= mask

        application_call.add_auxiliary_variable(weights_cost.copy(),
                                                name='pointer_nll')
        application_call.add_auxiliary_variable(costs.copy(),
                                                name='label_nll')
        costs = ((1 - self.pointer_weight) * costs +
                 self.pointer_weight * weights_cost)

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)


        return [costs] + states.values() + glimpses.values()

    @evaluate.property('outputs')
    def evaluate_outputs(self):
        return ['costs'] + self._state_names + self._glimpse_names

    @application
    def generate(self, mask=None, **kwargs):
        #pointers = outputs[1]['pointers']
        #outputs = outputs[0]
        # We assume the data has axes (time, batch, features, ...)
        #batch_size = outputs.shape[1]

        states = dict_subset(kwargs, self._state_names, must_have=False)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)

        inputs = self.fork.apply(contexts['attended'], as_dict=True)

        additional_inputs = {}
        if 'wordannotations'in self.transition.do_apply.sequences:
            additional_inputs['wordannotations'] = contexts['attended']
        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts, additional_inputs))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = OrderedDict((name, results[name][:-1]) for name in self._state_names)
        glimpses = OrderedDict((name, results[name][1:]) for name in self._glimpse_names)

        # Compute the cost
        readouts = self.readout.readout(
            **dict_union(states, glimpses, contexts))

        outputs = self.readout.emit(readouts)

        costs = self.readout.cost(readouts, outputs)
        if mask is not None:
            costs *= mask

        weights = glimpses['weights']

        return costs, outputs, weights
        #return [costs] + states.values() + glimpses.values()

    @generate.property('outputs')
    def generate_outputs(self):
        return ['costs', 'outputs', 'weights']

    @application
    def cost(self, application_call, outputs, mask=None, **kwargs):
        # Compute the sum of costs
        costs = self.cost_matrix(outputs, mask=mask, **kwargs)
        cost = tensor.mean(costs.sum(axis=0))
        add_role(cost, COST)

        # Add auxiliary variable for per sequence element cost
        application_call.add_auxiliary_variable(
            (costs.sum() / mask.sum()) if mask is not None else costs.mean(),
            name='per_sequence_element')
        return cost

    @application
    def cost_matrix(self, outputs, mask=None, **kwargs):
        """Returns generation costs for output sequences.

        See Also
        --------
        :meth:`cost` : Scalar cost.

        """
        return self.evaluate(outputs, mask=mask, **kwargs)[0]

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        state_dict = dict(
            self.transition.initial_states(
                batch_size, as_dict=True, *args, **kwargs),
            outputs=self.readout.initial_outputs(batch_size))
        if self.language_model:
            lm_initial_states = self.language_model.initial_states(
                batch_size, as_dict=True, *args, **kwargs)
            state_dict = dict_union(state_dict,
                                    {"lm_" + name: state for name, state
                                     in lm_initial_states.items()})
        return [state_dict[state_name]
                for state_name in self.generate.states]

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.generate.states
