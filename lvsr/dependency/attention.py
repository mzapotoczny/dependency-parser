'''
Created on Mar 20, 2016
'''

from blocks.bricks.base import application
from blocks.bricks import Initializable
from blocks.bricks.recurrent import recurrent
from blocks.bricks.sequence_generators import AbstractFeedback
from blocks.bricks.attention import SequenceContentAttention,\
                                    AttentionRecurrent
from theano import tensor
from blocks.utils import dict_subset, dict_union


class AttendedFeedback(AbstractFeedback, Initializable):
    def __init__(self, num_outputs=None, feedback_dim=None, **kwargs):
        super(AttendedFeedback, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.feedback_dim = feedback_dim

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0
        return outputs

    def get_dim(self, name):
        if name == 'feedback':
            return self.feedback_dim
        return super(AttendedFeedback, self).get_dim(name)


class ParsingAttention(SequenceContentAttention):
    def __init__(self, hard_attention=None,
                 use_pointers=None,
                 use_word_annotations=False,
                 word_annontation_dim=None,
                 **kwargs):

        if hard_attention is None:
            hard_attention = (use_pointers == 'hard')
        self.hard_attention = hard_attention
        self.add_sequences = []
        self.use_pointers = use_pointers
        if use_pointers:
            self.add_sequences.append('pointers')
        #
        # Warning: this is inefficient!
        # The computation should be factored out of the scan loop
        #
        self.extra_dims = {}
        if use_word_annotations:
            self.add_sequences.append('wordannotations')
            kwargs['state_names'] = kwargs.get('state_names', []) + ['wordannotations']
            self.extra_dims['wordannotations'] = word_annontation_dim

        super(ParsingAttention, self).__init__(**kwargs)

    @application(outputs=['weighted_averages', 'weights'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, **states):
        pointers = None
        if 'pointers' in states:
            pointers = states.pop('pointers')
        energies = self.compute_energies(attended, preprocessed_attended,
                                         states)
        weights = self.compute_weights(energies, attended_mask)
        transformed_weights = weights
        if pointers:
            if self.use_pointers == 'hard':
                transformed_weights = pointers.T
            elif self.use_pointers == 'semihard':
                transformed_weights = (pointers.T + weights) * 0.5
            else:
                raise Exception('Unknown use_pointers value')
        elif self.hard_attention:
            transformed_weights = tensor.eye(weights.shape[0])[
                weights.argmax(axis=0)].T
        weighted_averages = self.compute_weighted_averages(
            transformed_weights, attended)
        return weighted_averages, weights.T

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        return (['attended', 'preprocessed_attended', 'attended_mask'] +
                self.state_names + self.add_sequences)


class ParsingAttentionRecurrent(AttentionRecurrent):
    def __init__(self, *args, **kwargs):
        super(ParsingAttentionRecurrent, self).__init__(*args, **kwargs)
        assert isinstance(self.attention, ParsingAttention)
        self.add_sequences = list(self.attention.add_sequences)

    def _push_allocation_config(self):
        self.attention.state_dims = []
        for sn in self.attention.state_names:
            if sn in self.attention.extra_dims:
                self.attention.state_dims.append(self.attention.extra_dims[sn])
            else:
                self.attention.state_dims.append(self.transition.get_dim(sn))

        self.attention.attended_dim = self.get_dim(self.attended_name)
        self.distribute.source_dim = self.attention.get_dim(
            self.distribute.source_name)
        self.distribute.target_dims = self.transition.get_dims(
            self.distribute.target_names)

    @application
    def take_glimpses(self, **kwargs):
        states = dict_subset(kwargs, self._state_names, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
        glimpses_needed = dict_subset(glimpses, self.previous_glimpses_needed)

        add_seqs = dict_subset(kwargs, self.add_sequences, pop=True,
                               must_have=False)

        result = self.attention.take_glimpses(
            kwargs.pop(self.attended_name),
            kwargs.pop(self.preprocessed_attended_name, None),
            kwargs.pop(self.attended_mask_name, None),
            **dict_union(states, glimpses_needed, add_seqs))
        # At this point kwargs may contain additional items.
        # e.g. AttentionRecurrent.transition.apply.contexts
        return result

    @take_glimpses.property('outputs')
    def take_glimpses_outputs(self):
        return self._glimpse_names

    @recurrent
    def do_apply(self, **kwargs):
        attended = kwargs[self.attended_name]
        preprocessed_attended = kwargs.pop(self.preprocessed_attended_name)
        attended_mask = kwargs.get(self.attended_mask_name)
        sequences = dict_subset(kwargs, self._sequence_names, pop=True,
                                must_have=False)
        states = dict_subset(kwargs, self._state_names, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
        add_seqs = dict_subset(kwargs, self.add_sequences, pop=True,
                               must_have=False)

        current_glimpses = self.take_glimpses(
            as_dict=True,
            **dict_union(
                states, glimpses,
                {self.attended_name: attended,
                 self.attended_mask_name: attended_mask,
                 self.preprocessed_attended_name: preprocessed_attended
                 },
                add_seqs
                 ))
        current_states = self.compute_states(
            as_list=True,
            **dict_union(sequences, states, current_glimpses, kwargs))
        return current_states + list(current_glimpses.values())

    @do_apply.property('sequences')
    def do_apply_sequences(self):
        return self._sequence_names + self.add_sequences

    @do_apply.property('contexts')
    def do_apply_contexts(self):
        return self._context_names + [self.preprocessed_attended_name]

    @do_apply.property('states')
    def do_apply_states(self):
        return self._state_names + self._glimpse_names

    @do_apply.property('outputs')
    def do_apply_outputs(self):
        return self._state_names + self._glimpse_names
