import logging
import numpy
import theano
from theano import tensor

from blocks.roles import VariableRole, add_role, WEIGHT, INITIAL_STATE
from blocks.bricks import (
    Initializable, Linear, Sequence, Tanh)
from blocks.bricks.base import lazy, application, _Brick
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import Bidirectional, BaseRecurrent, recurrent
from blocks.bricks.sequence_generators import (
    AbstractFeedback, LookupFeedback, AbstractEmitter, SoftmaxEmitter)
from blocks.utils import dict_union, check_theano_variable,\
    shared_floatx_nans, shared_floatx_zeros

import copy
from picklable_itertools.extras import equizip
from blocks.bricks.simple import Rectifier, Logistic
from six import add_metaclass
from blocks.bricks.interfaces import Feedforward

logger = logging.getLogger(__name__)


class RecurrentWithFork(Initializable):

    @lazy(allocation=['input_dim'])
    def __init__(self, recurrent, input_dim, **kwargs):
        super(RecurrentWithFork, self).__init__(**kwargs)
        self.recurrent = recurrent
        self.input_dim = input_dim
        self.fork = Fork(
            [name for name in self.recurrent.sequences
             if name != 'mask'],
             prototype=Linear())
        self.children = [recurrent.brick, self.fork]

    def _push_allocation_config(self):
        self.fork.input_dim = self.input_dim
        self.fork.output_dims = [self.recurrent.brick.get_dim(name)
                                 for name in self.fork.output_names]

    @application(inputs=['input_', 'mask'])
    def apply(self, input_, mask=None, **kwargs):
        return self.recurrent(
            mask=mask, **dict_union(self.fork.apply(input_, as_dict=True),
                                    kwargs))

    @apply.property('outputs')
    def apply_outputs(self):
        return self.recurrent.states


class InitializableSequence(Sequence, Initializable):
    pass


class FakeRecurrent(BaseRecurrent, Initializable):
    """The traditional recurrent transition.

    The most well-known recurrent transition: a matrix multiplication,
    optionally followed by a non-linearity.

    Parameters
    ----------
    dim : int
        The dimension of the hidden state
    activation : :class:`.Brick`
        The brick to apply as activation.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, **kwargs):
        super(FakeRecurrent, self).__init__(**kwargs)
        self.dim = dim
        self.children = [activation]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (FakeRecurrent.apply.sequences +
                    FakeRecurrent.apply.states):
            return self.dim
        return super(FakeRecurrent, self).get_dim(name)

    def _allocate(self):
        pass

    def _initialize(self):
        pass

    @recurrent(sequences=['mask'], states=[],
               outputs=[], contexts=[])
    def apply(self, mask=None):
        """Apply the simple transition.

        Parameters
        ----------
        inputs : :class:`~tensor.TensorVariable`
            The 2D inputs, in the shape (batch, features).
        states : :class:`~tensor.TensorVariable`
            The 2D states, in the shape (batch, features).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if
            there is data available, 0 if not. Assumed to be 1-s
            only if not given.

        """
        return ()

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return ()


class BidirectionalWithAdd(Initializable):
    """Bidirectional network.

    A bidirectional network is a combination of forward and backward
    recurrent networks which process inputs in different order.

    Parameters
    ----------
    prototype : instance of :class:`BaseRecurrent`
        A prototype brick from which the forward and backward bricks are
        cloned.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    has_bias = False

    @lazy()
    def __init__(self, prototype, bidir_aggregation,
                 **kwargs):
        super(BidirectionalWithAdd, self).__init__(**kwargs)
        self.prototype = prototype
        self.bidir_aggregation = bidir_aggregation
        self.children = [copy.deepcopy(prototype) for _ in range(2)]
        self.children[0].name = 'forward'
        self.children[1].name = 'backward'

    @application
    def apply(self, *args, **kwargs):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, *args, **kwargs)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           *args, **kwargs)]
        if self.bidir_aggregation == 'concat':
            return [tensor.concatenate([f, b], axis=2)
                    for f, b in equizip(forward, backward)]
        elif self.bidir_aggregation == 'add':
            return [f + b
                    for f, b in equizip(forward, backward)]
        elif self.bidir_aggregation == 'mean':
            return [(f + b)/2.0
                    for f, b in equizip(forward, backward)]
        else:
            raise Exception("Unknown bidir_aggregation: %s" % (
                self.bidir_aggregation, ))

    @apply.delegate
    def apply_delegate(self):
        return self.children[0].apply

    def get_dim(self, name):
        if name in self.apply.outputs:
            if self.bidir_aggregation == 'concat':
                return self.prototype.get_dim(name) * 2
            elif self.bidir_aggregation in ['add', 'mean']:
                return self.prototype.get_dim(name)
            else:
                raise Exception("Unknown bidir_aggregation: %s" % (
                    self.bidir_aggregation, ))
        return self.prototype.get_dim(name)


class Encoder(Initializable):
    def __init__(self, enc_transition, dims, dim_input, subsample, bidir,
                 bidir_aggregation='concat', enc_transition_params={},
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.subsample = subsample

        self.enc_transitions = []

        assert bidir_aggregation in ['concat', 'add']

        dims_under = [dim_input] + list(
            (2 if (bidir and bidir_aggregation == 'concat') else 1
             ) * numpy.array(dims))
        for layer_num, (dim_under, dim) in enumerate(zip(dims_under, dims)):
            self.enc_transitions += [enc_transition(dim=dim, activation=Tanh(),
                                                    **enc_transition_params)]
            layer = RecurrentWithFork(
                    self.enc_transitions[-1].apply,
                    dim_under,
                    name='with_fork{}'.format(layer_num))
            if bidir:
                layer = BidirectionalWithAdd(
                    layer, name='bidir{}'.format(layer_num),
                    bidir_aggregation=bidir_aggregation)
            self.children.append(layer)
        self.dim_encoded = (2 if (bidir and bidir_aggregation == 'concat')
                            else 1) * dims[-1]

    @application(outputs=['encoded', 'encoded_mask'])
    def apply(self, input_, mask=None):
        for layer, take_each in zip(self.children, self.subsample):
            input_ = layer.apply(input_, mask)
            if isinstance(input_, list):
                input_, = [ii for ii in input_ if ii.tag.name == 'states']
            input_ = input_[::take_each]
            if mask:
                mask = mask[::take_each]
        return input_, (mask if mask else tensor.ones_like(input_[:, :, 0]))

    def get_dim(self, name):
        if name == self.apply.outputs[0]:
            return self.dim_encoded
        return super(Encoder, self).get_dim(name)


class OneOfNFeedback(AbstractFeedback, Initializable):
    """A feedback brick for the case when readout are integers.

    Stores and retrieves distributed representations of integers.

    """
    def __init__(self, num_outputs=None, feedback_dim=None, **kwargs):
        super(OneOfNFeedback, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.feedback_dim = num_outputs

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0
        eye = tensor.eye(self.num_outputs)
        check_theano_variable(outputs, None, "int")
        output_shape = [outputs.shape[i]
                        for i in range(outputs.ndim)] + [self.feedback_dim]
        return eye[outputs.flatten()].reshape(output_shape)

    def get_dim(self, name):
        if name == 'feedback':
            return self.feedback_dim
        return super(LookupFeedback, self).get_dim(name)


class OtherLoss(VariableRole):
    pass


OTHER_LOSS = OtherLoss()


class LinearActivationDocumentation(_Brick):
    """Dynamically adds documentation to LinearActivation descendants.
    Notes
    -----
    See http://bugs.python.org/issue12773.
    """
    def __new__(cls, name, bases, classdict):
        classdict['__doc__'] = \
            """Linear transformation followed by elementwise application of \
            {0} function.""".format(name[6:].lower())
        if 'apply' in classdict:
            classdict['apply'].__doc__ = \
                """Apply the linear transformation, then {0} activation.
                Parameters
                ----------
                input_ : :class:`~tensor.TensorVariable`
                    Input tensor.
                Returns
                -------
                output : :class:`~tensor.TensorVariable`
                    Liearly transformetd and activated input.
                """.format(name[6:].lower())
        return super(LinearActivationDocumentation, cls).__new__(cls, name,
                                                                 bases,
                                                                 classdict)


@add_metaclass(LinearActivationDocumentation)
class LinearActivation(Initializable, Feedforward):
    """Base class that adds documentation and has all the logic."""
    @lazy(allocation=['input_dim', 'output_dim'])
    def __init__(self, input_dim, output_dim, activation, **kwargs):
        super(LinearActivation, self).__init__(**kwargs)
        self.linear = Linear()
        self.activation = activation
        self.children = [self.linear,
                         self.activation]

        self.input_dim = input_dim
        self.output_dim = output_dim

    @property
    def input_dim(self):
        return self.linear.input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.linear.input_dim = value

    @property
    def output_dim(self):
        return self.linear.output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.linear.output_dim = value

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        pre_activation = self.linear.apply(input_)
        output = self.activation.apply(pre_activation)
        return output


class LinearTanh(LinearActivation):
    def __init__(self, **kwargs):
        super(LinearTanh, self).__init__(activation=Tanh(), **kwargs)


class LinearRectifier(LinearActivation):
    def __init__(self, **kwargs):
        super(LinearRectifier, self).__init__(activation=Rectifier(), **kwargs)


class LinearLogistic(LinearActivation):
    def __init__(self, **kwargs):
        super(LinearLogistic, self).__init__(activation=Logistic(), **kwargs)
        
class SoftmaxMultiEmitter(SoftmaxEmitter):
    @application
    def emit(self, readouts):
        return readouts.argmax(axis=-1)
