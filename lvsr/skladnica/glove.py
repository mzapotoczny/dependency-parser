#!/usr/bin/env python

import logging
import codecs
from blocks.bricks.simple import Identity, LinearMaxout, Maxout

try:
    from common import config
    from common.utils import attach_test_values, LoadableModel
    from blocks.roles import ALGORITHM_BUFFER, SPARSE_SELECTION
except ImportError:
    pass

import pprint
from blocks.serialization import load_parameter_values
import theano
logger = logging.getLogger(__name__)

from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale, RMSProp, Adam,\
    CompositeRule, Restrict, VariableClipping, Momentum, AdaGrad, StepRule,\
    AdaDelta
from blocks.bricks import MLP, Tanh, Softmax, Initializable,\
    Feedforward, Linear, Logistic, Rectifier
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant, Uniform
from fuel.streams import DataStream
from fuel.transformers import Flatten, ScaleAndShift, Cast, Batch
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme, ConstantScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop

import copy
from blocks.bricks.base import lazy, application
from blocks.extensions.training import SharedVariableModifier
import numpy
from blocks.utils import shared_floatx_nans, shared_floatx
from blocks.roles import add_role, BIAS, has_roles, PARAMETER, WEIGHT
from fuel.datasets.base import Dataset
from collections import OrderedDict
import os
from theano.tensor.subtensor import inc_subtensor

try:
    from blocks.extras.extensions.plot import Plot
    BLOCKS_EXTRAS_AVAILABLE = True
except:
    BLOCKS_EXTRAS_AVAILABLE = False


class LookupEmbeddings(Initializable):
    @lazy(allocation=['embedding_dim', 'vocabulary_size'])
    def __init__(self, embedding_dim, vocabulary_size,
                 untied=True, **kwargs):
        kwargs.setdefault('weights_init',
                          Uniform(width=1.0/embedding_dim))
        kwargs.setdefault('biases_init',
                          Uniform(width=1.0/embedding_dim))
        super(LookupEmbeddings, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.vocabulary_size = vocabulary_size
        self.untied = untied

    @property
    def W(self):
        return self.parameters[0]

    @property
    def b(self):
        return self.parameters[1]

    @property
    def Wc(self):
        return self.parameters[0 + 2*self.untied]

    @property
    def bc(self):
        return self.parameters[1 + 2*self.untied]

    def _allocate(self):
        W = shared_floatx_nans((self.vocabulary_size, self.embedding_dim),
                               name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)
        if self.use_bias:
            b = shared_floatx_nans((self.vocabulary_size,), name='b')
            add_role(b, BIAS)
            self.parameters.append(b)
        if self.untied:
            Wc = shared_floatx_nans((self.vocabulary_size, self.embedding_dim),
                                    name='Wc')
            add_role(Wc, WEIGHT)
            self.parameters.append(Wc)
            if self.use_bias:
                bc = shared_floatx_nans((self.vocabulary_size,), name='bc')
                add_role(bc, BIAS)
                self.parameters.append(bc)

    def _initialize(self):
        W, b = self.parameters[:2]
        self.weights_init.initialize(W, self.rng)
        self.biases_init.initialize(b, self.rng)
        if self.untied:
            Wc, bc = self.parameters[2:]
            self.weights_init.initialize(Wc, self.rng)
            self.biases_init.initialize(bc, self.rng)

    @application
    def apply(self, ws, wcs):
        ret = self.W[ws], self.b[ws], self.Wc[wcs], self.bc[wcs]
        for r in ret:
            add_role(r, SPARSE_SELECTION)
        return ret


class GloveCost(Initializable):
    def __init__(self, count_max=100, alpha=0.75,
                 **kwargs):
        super(GloveCost, self).__init__(**kwargs)
        self.count_max = count_max
        self.alpha = alpha

    @application
    def cost(self, counts, W, b, Wc, bc):
        # cooccurence weighting
        f = tensor.minimum(counts/float(self.count_max), 1.0)**self.alpha
        diffs = (W*Wc).sum(1) + b + bc - tensor.log(counts)
        cost = (0.5 * f * tensor.sqr(diffs)).mean()
        return cost


class Glove(Initializable):
    def __init__(self, vocabulary_size, embedder_conf={}, glove_conf={},
                 **kwargs):
        super(Glove, self).__init__(**kwargs)
        self.embedder = LookupEmbeddings(vocabulary_size=vocabulary_size,
                                         **embedder_conf)
        self.children.append(self.embedder)

        self.glove_cost = GloveCost(**glove_conf)
        self.children.append(self.glove_cost)

    @application
    def cost(self, ws, wcs, counts):
        W, b, Wc, bc = self.embedder.apply(ws, wcs)
        return self.glove_cost.cost(counts, W, b, Wc, bc)

    def get_cost_graph(self, test_batch=None):
        Ws = tensor.ivector('ws')
        Wcs = tensor.ivector('wcs')
        C = tensor.vector('counts')

        attach_test_values(locals(), test_batch)

        return self.cost(Ws, Wcs, C)

    def get_sources(self):
        return ('ws', 'wcs', 'counts')

    def embed_vocab(self, data, vocab_file_in, vocab_file_out):
        # get the vector for unknown
        W = self.embedder.W.get_value()
        Wc = self.embedder.Wc.get_value()
        # just the mean of the 100 least frequent words
        # as in the original GloVe
        unk = W[-100:].mean(0)
        unk_c = Wc[-100:].mean(0)

        with codecs.open(vocab_file_in, 'rt', 'utf8') as f_in, \
                codecs.open(vocab_file_out, 'wt', 'utf8') as f_out:
            for line in f_in:
                word = line.split()[0].strip()
                word_id = data.word2tok.get(word)
                if word_id:
                    embedding = W[word_id] + Wc[word_id]
                else:
                    embedding = unk + unk_c
                f_out.write("%s %s\n" % (
                    word, ' '.join("%lf" % (e,) for e in embedding)))

            word = "<unk>"
            embedding = unk + unk_c
            f_out.write("%s %s\n" % (
                word, ' '.join("%lf" % (e,) for e in embedding)))


class Highway(Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, **kwargs):
        super(Highway, self).__init__(**kwargs)
        self.linear = Linear(name='linear')
        self.activation = activation
        self.activation.name = 'activation'
        self.gate = Linear(name='gate')
        self.gate_activ = Logistic(name='gate_activation')
        self.children = [self.linear,
                         self.activation,
                         self.gate,
                         self.gate_activ]
        self.highway_biases_init = Constant(-2.0)
        self.dim = dim

    @property
    def dim(self):
        return self.linear.input_dim

    @dim.setter
    def dim(self, value):
        self.linear.input_dim = value
        # Ugly, the real solution are the LinearActivation classes...
        if isinstance(self.activation, Maxout):
            self.linear.output_dim = value * self.activation.num_pieces
        else:
            self.linear.output_dim = value
        self.gate.input_dim = value
        self.gate.output_dim = value

    def _push_initialization_config(self):
        super(Highway, self)._push_initialization_config()
        self.gate.biases_init = self.highway_biases_init

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        output = self.activation.apply(self.linear.apply(input_))
        gate = self.gate_activ.apply(self.gate.apply(input_))
        return gate * output + (1.0 - gate) * input_
    
class ActivatedLinear(Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, **kwargs):
        super(ActivatedLinear, self).__init__(**kwargs)
        self.linear = Linear(name='linear')
        self.activation = activation
        self.activation.name = 'activation'
        self.children = [self.linear,
                         self.activation]
        self.dim = dim

    @property
    def dim(self):
        return self.linear.input_dim

    @dim.setter
    def dim(self, value):
        self.linear.input_dim = value
        # Ugly, the real solution are the LinearActivation classes...
        if isinstance(self.activation, Maxout):
            self.linear.output_dim = value * self.activation.num_pieces
        else:
            self.linear.output_dim = value

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return self.activation.apply(self.linear.apply(input_))


class CharEmbedder(Initializable):
    @lazy(allocation=['embedding_dim', 'character_count'])
    def __init__(self, embedding_dim, character_count,
                 **kwargs):
        super(CharEmbedder, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.character_count = character_count

    @property
    def W(self):
        return self.parameters[0]

    def _allocate(self):
        W = shared_floatx_nans((self.character_count, self.embedding_dim),
                               name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)

    def _initialize(self):
        W, = self.parameters
        self.weights_init.initialize(W, self.rng)

    @application
    def apply(self, cs):
        embedded = self.W[cs]
        # mask padding characters
        embedded = embedded * (tensor.shape_padright(cs > 0))
        return embedded


class CharConvFilter(Initializable):
    @lazy(allocation=['filter_width', 'num_filters', 'embedding_dim'])
    def __init__(self, filter_width, num_filters, embedding_dim,
                 filter_activation=Tanh(),
                 **kwargs):
        super(CharConvFilter, self).__init__(**kwargs)
        self.filter_width = filter_width
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        self.filter_activation = filter_activation
        self.filter_activation.name = 'activation'
        self.children.append(self.filter_activation)

    @property
    def F(self):
        return self.parameters[0]

    @property
    def b(self):
        return self.parameters[1]

    def _allocate(self):
        F = shared_floatx_nans((self.num_filters, 1,
                                self.filter_width, self.embedding_dim),
                               name='F')
        add_role(F, WEIGHT)
        self.parameters.append(F)

        b = shared_floatx_nans((self.num_filters, ),
                               name='b')
        add_role(b, BIAS)
        self.parameters.append(b)

    def _initialize(self):
        F, b = self.parameters
        self.weights_init.initialize(F, self.rng)
        self.biases_init.initialize(b, self.rng)

    @application
    def apply(self, C, mask=None):
        f = tensor.nnet.conv.conv2d(C, self.F, border_mode='valid')
        if mask:
            f = f - 1e20*(1.0 - mask.dimshuffle(0, 'x', 1, 'x'))
        f = f.max(axis=2).reshape(f.shape[:2])
        f = f + tensor.shape_padleft(self.b)
        f = self.filter_activation.apply(f)
        return f


class CharacetrToWordEmbeddings(Initializable):
    def __init__(self,
                 character_count,
                 filters=[[2, 50]],
                 character_embedding_dim=15,
                 filter_activation=Tanh(),
                 num_highway_layers=1,
                 project_dim=None,
                 highway_activation=Rectifier(),
                 mask_padding_characters=False,
                 highway_ignore_gate=False,
                 **kwargs):
        super(CharacetrToWordEmbeddings, self).__init__(**kwargs)
        self.mask_padding_characters = mask_padding_characters

        self.character_embeder = CharEmbedder(character_embedding_dim,
                                              character_count)
        if isinstance(filter_activation, str):
            filter_activation = eval(filter_activation)
        if isinstance(highway_activation, str):
            filter_activation = eval(highway_activation)

        self.children.append(self.character_embeder)
        self.max_filter_width = 0
        self.filters = []
        self.output_dim = 0
        for filter_width, num_filters in filters:
            self.max_filter_width = max(self.max_filter_width, filter_width)
            self.filters.append(
                CharConvFilter(
                    filter_width, num_filters,
                    character_embedding_dim,
                    filter_activation=copy.deepcopy(filter_activation),
                    name='filter_%d' % (filter_width,)))
            self.output_dim += num_filters
        self.highway_layers = []
        self.children.extend(self.filters)

        if project_dim is None:
            self.projection = Identity(name='proj')
        else:
            self.projection = Linear(input_dim=self.output_dim,
                                     output_dim=project_dim,
                                     name='proj')
            self.output_dim = project_dim
        self.children.append(self.projection)
        
        highway_class = Highway
        
        if highway_ignore_gate:
            highway_class = ActivatedLinear

        for hl in xrange(num_highway_layers):
            self.highway_layers.append(
                highway_class(dim=self.output_dim,
                        activation=copy.deepcopy(highway_activation),
                        name='hw_%d' % (hl, )))
        self.children.extend(self.highway_layers)

    @application(inputs=['chars'], outputs=['output'])
    def apply(self, chars):
        C = self.character_embeder.apply(chars)
        if self.mask_padding_characters:
            C_padded = C.dimshuffle(0, 'x', 1, 2)
            char_mask = (chars > 0).astype(theano.config.floatX)
        else:
            padding = self.max_filter_width-1
            C_padded = tensor.zeros(
                (C.shape[0],  # batch size
                 1,  # stack height
                 C.shape[1] + 2*padding,  # rows - just one
                 C.shape[2],  # cols - embedding dimension
                 ))
            C_padded = tensor.set_subtensor(
                C_padded[:, 0, padding:-padding, :],
                C,
                # inplace=True, tolerate_inplace_aliasing=True
                )
        f_outs = []
        for filter_ in self.filters:
            mask = None
            if self.mask_padding_characters:
                mask = char_mask[
                    :, :char_mask.shape[1] - filter_.filter_width + 1]
            f_outs.append(filter_.apply(C_padded, mask=mask))
        embedding = tensor.concatenate(f_outs, axis=1)
        embedding = self.projection.apply(embedding)
        for hl in self.highway_layers:
            embedding = hl.apply(embedding)
        return embedding

class FeaturesToWordEmbeddings(Initializable):
    def __init__(self,
                 features_count,
                 num_highway_layers=1,
                 project_dim=None,
                 highway_activation=Rectifier(),
                 highway_ignore_gate=False,
                 **kwargs):
        super(FeaturesToWordEmbeddings, self).__init__(**kwargs)

        self.highway_layers = []
        self.features_count = features_count

        self.output_dim = features_count

        if project_dim is None:
            self.projection = Identity(name='proj')
        else:
            self.projection = Linear(input_dim=self.output_dim,
                                     output_dim=project_dim,
                                     name='proj')
            self.output_dim = project_dim
        self.children.append(self.projection)
        
        highway_class = Highway
        
        if highway_ignore_gate:
            highway_class = ActivatedLinear

        for hl in xrange(num_highway_layers):
            self.highway_layers.append(
                highway_class(dim=self.output_dim,
                        activation=copy.deepcopy(highway_activation),
                        name='hw_%d' % (hl, )))
        self.children.extend(self.highway_layers)

    @application(inputs=['features'], outputs=['output'])
    def apply(self, features):
        embedding = self.projection.apply(features.astype(theano.config.floatX))
        for hl in self.highway_layers:
            embedding = hl.apply(embedding)
        return embedding

class CharacterGlove(Initializable):
    def __init__(self,
                 character_count,
                 untied=True,
                 embedder_conf={},
                 glove_conf={},
                 **kwargs):
        kwargs.setdefault('weights_init', IsotropicGaussian(0.03))
        kwargs.setdefault('biases_init', Constant(0))
        super(CharacterGlove, self).__init__(**kwargs)
        self.untied = untied
        self.word_embedder = CharacetrToWordEmbeddings(character_count,
                                                       name='word_embedder',
                                                       **embedder_conf)
        self.bias_computer = Linear(input_dim=self.word_embedder.output_dim,
                                    output_dim=1,
                                    name='bias_computer')
        self.children.extend([self.word_embedder, self.bias_computer])
        if self.untied:
            self.word_c_embedder = CharacetrToWordEmbeddings(
                character_count, name='word_c_embedder', **embedder_conf)
            self.bias_c_computer = Linear(
                input_dim=self.word_embedder.output_dim, output_dim=1,
                name='bias_c_computer')
            self.children.extend([self.word_c_embedder, self.bias_c_computer])
        else:
            self.word_c_embedder = self.word_embedder
            self.bias_c_computer = self.bias_computer

        self.glove_cost = GloveCost(**glove_conf)
        self.children.append(self.glove_cost)

    @application
    def cost(self, chars, chars_c, counts):
        # embed the caharacters
        W = self.word_embedder.apply(chars)
        b = self.bias_computer.apply(W).ravel()
        Wc = self.word_c_embedder.apply(chars_c)
        bc = self.bias_c_computer.apply(Wc).ravel()
        return self.glove_cost.cost(counts, W, b, Wc, bc)

    def get_cost_graph(self, test_batch=None):
        chars = tensor.imatrix('chars')
        chars_c = tensor.imatrix('chars_c')
        C = tensor.vector('counts')

        attach_test_values(locals(), test_batch)
        return self.cost(chars, chars_c, C)

    def get_sources(self):
        return ('chars', 'chars_c', 'counts')

    def embed_vocab(self, data, vocab_file_in, vocab_file_out):
        if not hasattr(self, '_embedding_fun'):
            chars = tensor.imatrix('chars')
            W = self.word_embedder.apply(chars)
            Wc = self.word_c_embedder.apply(chars)
            self._embedding_fun = theano.function([chars],
                                                  [W, Wc])

        with codecs.open(vocab_file_in, 'rt', 'utf8') as f_in, \
                codecs.open(vocab_file_out, 'wt', 'utf8') as f_out:
            for line in f_in:
                word = line.split()[0].strip()
                chars = data.word2chars(word)
                chars += [0] * (self.word_embedder.max_filter_width -
                                len(chars))
                chars = numpy.array(chars
                                    ).reshape(1, -1).astype(numpy.int32)
                W, Wc = self._embedding_fun(chars)
                embedding = (W+Wc).ravel()
                f_out.write("%s %s\n" % (
                    word, ' '.join("%lf" % (e,) for e in embedding)))


class AdaGradGlove(StepRule):
    """Implements the AdaGrad learning rule.

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.0002.
    epsilon : float, optional
        Stabilizing constant for one over root of sum of squares.
        Defaults to 1e-6.

    Notes
    -----
    For more information, see [ADAGRAD]_.

    .. [ADADGRAD] Duchi J, Hazan E, Singer Y.,
       *Adaptive subgradient methods for online learning and
        stochastic optimization*,
       http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

    """
    def __init__(self, learning_rate=1.0, initial=1.0):
        self.learning_rate = learning_rate
        self.initial = initial

    def compute_step(self, parameter, previous_step):
        name = 'adagrad_sqs'

        selector = None
        if has_roles(parameter, [SPARSE_SELECTION]):
            parameter, selector = split_sparse_selection(parameter)

        if parameter.name:
            name += '_' + parameter.name
        ssq = shared_floatx(parameter.get_value() * 0. + self.initial,
                            name=name)
        add_role(ssq, ALGORITHM_BUFFER)

        if selector:
            ssq_sel = ssq[selector]
            updates = [(ssq,
                        inc_subtensor(ssq_sel, tensor.sqr(previous_step)))]
            ssq = ssq_sel
        else:
            ssq_t = (tensor.sqr(previous_step) + ssq)
            updates = [(ssq, ssq_t)]

        step = (self.learning_rate * previous_step /
                (tensor.sqrt(ssq)))

        return step, updates


class CoccurenceDataset(Dataset):
    example_iteration_scheme = None

    DTYPE = numpy.dtype([('ws', '<i4'), ('wcs', '<i4'),
                         ('counts', '<f8')])

    def __init__(self, cooccurence_file, vocab_file=None,
                 word_padding=None, **kwargs):
        self.cooccurence_file = cooccurence_file
        self.num_examples = (os.path.getsize(cooccurence_file) /
                             self.DTYPE.itemsize)
        self.provides_sources = ('ws', 'wcs', 'counts')
        self.axis_labels = OrderedDict([('ws', ('batch',)),
                                        ('wcs', ('batch',)),
                                        ('counts', ('batch',))])
        if vocab_file is not None:
            self.prep_vocabulary(vocab_file, word_padding)
        else:
            self.vocabulary_size = self._get_vocabulary_dim()
        super(CoccurenceDataset, self).__init__(**kwargs)

    def _get_vocabulary_dim(self):
        maxw = 0
        for w, wc, _ in self.get_batched_stream(10000000).get_epoch_iterator():
            maxw = max(maxw, w.max(), wc.max())
        return maxw+1

    def word2chars(self, word):
        return ([self.char2tok['<w>']] +
                [self.char2tok[c] for c in word] +
                [self.char2tok['</w>']])

    def prep_vocabulary(self, vocab_file, word_padding):
        if not word_padding:
            word_padding = (0, 0)
        self.provides_sources = self.provides_sources + ('chars', 'chars_c')
        self.axis_labels['chars'] = ('batch', 'character')
        self.axis_labels['chars_c'] = ('batch', 'character')
        characters = set()
        self.word2tok = {}
        with codecs.open(vocab_file, 'rt', 'utf8') as vocab_file:
            word_num = 0
            for line in vocab_file:
                line = line.strip()
                if not line:
                    continue
                word, unused_count = line.split()
                self.word2tok[word] = word_num
                word_num += 1
                characters |= set(word)
            self.characters = [''] + sorted(characters) + ['<w>', '</w>']

            self.char2tok = {c: n for (n, c) in enumerate(self.characters)}

            vocab_file.seek(0)
            vocab = []
            max_w_len = 0
            for line in vocab_file:
                word, unused_count = line.strip().split()
                vocab.append(self.word2chars(word))
                max_w_len = max(max_w_len, len(vocab[-1]))

        self.vocabulary_size = len(vocab)
        self.vocab = numpy.zeros((self.vocabulary_size,
                                  max_w_len + sum(word_padding)),
                                 dtype='int32')
        for vi, word in enumerate(vocab):
            self.vocab[vi, word_padding[0]:word_padding[0]+len(word)] = word

    def open(self):
        return (open(self.cooccurence_file, 'rb'),)

    def close(self, state):
        state[0].close()

    def reset(self, state):
        state[0].seek(0)
        return state

    def get_data(self, state=None, request=None):
        arr = numpy.fromfile(state[0],
                             dtype=self.DTYPE,
                             count=request)
        if arr.shape[0] == 0:
            raise StopIteration

        data = {'ws': arr['ws']-1,
                'wcs': arr['wcs']-1,
                'counts': arr['counts'].astype(numpy.float32)}
        if ('chars') in self.sources:
            data['chars'] = self.vocab[data['ws']]
            data['chars_c'] = self.vocab[data['wcs']]
        return tuple(data[s] for s in self.sources)

    def get_batched_stream(self, batch_size, times=None):
        return DataStream(self,
                          iteration_scheme=ConstantScheme(batch_size,
                                                          times=times))


def get_sparse_selections(cg):
    return [var for var in cg.variables
            if has_roles(var, [SPARSE_SELECTION])]


def split_sparse_selection(var):
    param, selection = var.owner.inputs
    param, = ComputationGraph(param).parameters
    assert has_roles(param, [PARAMETER])
    return param, selection


def get_sparse_parameters(cg):
    parameters = set(cg.parameters)
    sparse_sels = get_sparse_selections(cg)
    for var in sparse_sels:
        param, _ = split_sparse_selection(var)
        parameters.remove(param)
    return list(parameters) + sparse_sels


def load_glove_bin(file_, model):
    logger.info("Loading model from: %s", file_)
    WW = numpy.fromfile(file_, dtype='float64')
    P = dict((p.name, p) for p in model.parameters)
    num_words, dim = P['W'].get_value(borrow=True).shape
    assert WW.shape[0] == 2 * num_words * (dim+1)
    WW = WW.reshape(2, num_words, dim+1).astype('float32')
    P['W'].set_value(WW[0, :, :dim])
    P['b'].set_value(WW[0, :, dim])
    P['Wc'].set_value(WW[1, :, :dim])
    P['bc'].set_value(WW[1, :, dim])


def save_glove_bin(file_, model):
    P = dict((p.name, p) for p in model.parameters)
    num_words, dim = P['W'].get_value(borrow=True).shape
    WW = numpy.zeros(2, num_words, dim+1, dtype='float64')
    WW[0, :, :dim] = P['W'].get_value(borrow=True)
    WW[0, :, dim] = P['b'].get_value(borrow=True)
    WW[1, :, :dim] = P['Wc'].get_value(borrow=True)
    WW[1, :, dim] = P['bc'].get_value(borrow=True)
    WW.ravel().tofile(file_, )


def get_data(args):
    return CoccurenceDataset(args.coccurrences,
                             vocab_file=args.vocab)


def get_cost_and_model(args, conf, data):
    net_conf = conf['net']
    train_conf = conf['training']

    test_batch = None
    if train_conf.get('debug', False):
        test_batch = next(data.get_batched_stream(3).get_epoch_iterator())
        test_batch = dict(zip(data.sources, test_batch))

    embedding_type = net_conf.pop('embedding_type', 'lookup')
    if embedding_type == 'lookup':
        glove = Glove(vocabulary_size=data.vocabulary_size,
                      **net_conf)
    elif embedding_type == 'characters':
        glove = CharacterGlove(len(data.characters),
                               **net_conf)
    else:
        raise Exception("Unknown embedding_type: %s" % (embedding_type,))
    data.sources = glove.get_sources()
    glove.initialize()
    cost = glove.get_cost_graph(test_batch=test_batch)

    # cg = ComputationGraph([cost])

    model = LoadableModel(cost)

    if args.load_glove is not None:
        load_glove_bin(args.load_glove, model)

    if args.load_params is not None:
        logger.info("Load parameters from " + args.load_params)
        param_values = load_parameter_values(args.load_params)
        model.set_parameter_values(param_values)

    parameters = model.get_parameter_dict()
    logger.info("Parameters:\n" +
                pprint.pformat(
                    [(key, parameters[key].get_value().shape) for key
                        in sorted(parameters.keys())],
                    width=120))

    return glove, cost, model


def embed(args, conf):
    data = get_data(args)

    glove, cost, model = get_cost_and_model(args, conf, data)
    if args.embed_vocab_in is None:
        args.embed_vocab_in = args.vocab
    glove.embed_vocab(data,
                      args.embed_vocab_in,
                      args.embed_vocab_out)


def train(args, conf):
    data = get_data(args)

    glove, cost, model = get_cost_and_model(args, conf, data)

    logger.info("Initialization schemes for all bricks.")

    def show_init_scheme(cur):
        result = dict()
        for attr in dir(cur):
            if attr.endswith('_init'):
                result[attr] = getattr(cur, attr)
        for child in cur.children:
            result[child.name] = show_init_scheme(child)
        return result
    logger.info(pprint.pformat(show_init_scheme(glove)))

    cg = ComputationGraph([cost])

    train_conf = conf['training']
    train_rules = []
    if not train_conf.get('debug', False):
        for lrule in train_conf['lrules']:
            train_rules.append(eval(lrule))
    else:
        train_rules.append(Scale(0.0))

    logger.info("Train rules in use:\n" + pprint.pformat(train_rules))

    if train_conf.get('use_sparse_updates', True):
        # this is faster, but much less tested
        parameters = get_sparse_parameters(cg)
    else:
        parameters = cg.parameters

    algorithm = GradientDescent(
        cost=cost, parameters=parameters,
        step_rule=CompositeRule(train_rules))

    if not train_conf.get('debug', False):
        monitoring_frequency = train_conf.get('monitoring_frequency', 100)
    else:
        monitoring_frequency = 1000 / train_conf['batch_size']
    extensions = [
        Timing(every_n_batches=monitoring_frequency),
        FinishAfter(after_n_epochs=train_conf['num_epochs']),
        TrainingDataMonitoring(
            [aggregation.mean(cost),
             aggregation.mean(algorithm.total_gradient_norm)
             ],
            prefix="train",
            every_n_batches=monitoring_frequency,
            ),
        Checkpoint(args.save_to,
                   save_separately=['log', 'model'],
                   use_cpickle=True,
                   after_epoch=True,
                   ),
        Printing(every_n_batches=monitoring_frequency),
        ]

    main_loop = MainLoop(
        algorithm,
        data.get_batched_stream(train_conf['batch_size']),
        model=model,
        extensions=extensions)

    main_loop.run()
    # import IPython; IPython.embed()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("Train GloVe vectors")
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--save-to", default="glove.pkl",
        help=("Destination to save the state of the training process."))

    embed_parser = subparsers.add_parser("embed")
    embed_parser.add_argument('--embed-vocab-in')
    embed_parser.add_argument('--embed-vocab-out')

    for subparser in [train_parser, embed_parser]:
        subparser.add_argument("--vocab", default=None,
                               help=("GloVe vocabulary file"))
        subparser.add_argument("coccurrences",
                               help=("Coccurrences file to use"))
        subparser.add_argument("--load-glove", default=None,
                               help=("Load embeddings form binary vectors "
                                     "from GloVe"))
        subparser.add_argument("--load-params", default=None,
                               help=("Load parameters from the pickle"))
        config.add_args(subparser)

    train_parser.set_defaults(func=train)
    embed_parser.set_defaults(func=embed)
    args = parser.parse_args()
    args.func(args, config.get_conf(args))
