import logging
from blocks.bricks.recurrent import recurrent,GatedRecurrent, LSTM
from blocks.config import config
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor

logger = logging.getLogger(__name__)

class ZoneOutGatedRecurrent(GatedRecurrent):
    def __init__(self, zone_rng=None, zone_seed=None, state_stay_prob=0.0,
            *args, **kwargs):
        if not zone_rng and not zone_seed:
            zone_seed = config.default_seed
        if not zone_rng:
            zone_rng = MRG_RandomStreams(zone_seed)

        self.zone_rng = zone_rng
        self.state_stay_prob = state_stay_prob
        self.regularize = True

        super(ZoneOutGatedRecurrent, self).__init__(*args, **kwargs)

    @recurrent(sequences=['mask', 'inputs', 'gate_inputs'],
               states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, gate_inputs, states, mask=None):
        gate_values = self.gate_activation.apply(
            states.dot(self.state_to_gates) + gate_inputs)
        update_values = gate_values[:, :self.dim]
        reset_values = gate_values[:, self.dim:]
        states_reset = states * reset_values
        next_states = self.activation.apply(
            states_reset.dot(self.state_to_state) + inputs)
        next_states = (next_states * update_values +
                       states * (1 - update_values))
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        if not self.regularize:
            return next_states
        stay_mask = self.zone_rng.binomial((self.dim,), p=self.state_stay_prob,
                                      dtype=states.dtype)
        return stay_mask*states + (1 - stay_mask)*next_states

class ZoneOutLSTM(LSTM):
    def __init__(self, zone_rng=None, zone_seed=None, state_stay_prob=0.0,
                 cell_stay_prob=0.0, *args, **kwargs):
        if not zone_rng and not zone_seed:
            zone_seed = config.default_seed
        if not zone_rng:
            zone_rng = MRG_RandomStreams(zone_seed)
        self.regularize = True

        self.zone_rng = zone_rng
        self.state_stay_prob = state_stay_prob
        self.cell_stay_prob = cell_stay_prob

        super(ZoneOutLSTM, self).__init__(*args, **kwargs)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        def slice_last(x, no):
            return x[:, no*self.dim: (no+1)*self.dim]

        nonlinearity = self.children[0].apply

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0) +
                                      cells * self.W_cell_to_in)
        forget_gate = tensor.nnet.sigmoid(slice_last(activation, 1) +
                                          cells * self.W_cell_to_forget +
                                          self.initial_forget_bias)
        next_cells = (forget_gate * cells +
                      in_gate * nonlinearity(slice_last(activation, 2)))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 3) +
                                       next_cells * self.W_cell_to_out)
        next_states = out_gate * nonlinearity(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)
        if not self.regularize:
            return next_states, next_cells
        state_stay_mask = self.zone_rng.binomial((self.dim,), p=self.state_stay_prob,
                                      dtype=states.dtype)
        cell_stay_mask = self.zone_rng.binomial((self.dim,), p=self.cell_stay_prob,
                                      dtype=states.dtype)

        return state_stay_mask*states + (1 - state_stay_mask)*next_states,\
               cell_stay_mask*cells + (1 - cell_stay_mask)*next_cells
