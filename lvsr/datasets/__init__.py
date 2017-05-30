import cPickle
import functools
import os

import fuel
import numpy
from fuel.schemes import (
    ConstantScheme, ShuffledExampleScheme, SequentialExampleScheme, IndexScheme)
from fuel.streams import DataStream, AbstractDataStream
from fuel.transformers import (
    SortMapping, Padding, ForceFloatX, Batch, Mapping, Unpack, Filter,
    FilterSources, Transformer, Rename, Merge)
from fuel import config

from lvsr.datasets.h5py import H5PYAudioDataset
from blocks.utils import dict_subset

import numpy as np


import logging
from picklable_itertools.iter_dispatch import iter_
from numpy.distutils.misc_util import dict_append

from lvsr.utils import resizeArray
logger = logging.getLogger(__name__)


def switch_first_two_axes(batch):
    result = []
    for array in batch:
        if array.ndim == 2:
            result.append(array.transpose(1, 0))
        else:
            result.append(array.transpose(1, 0, 2))
    return tuple(result)


class _Length(object):
    def __init__(self, index):
        self.index = index

    def __call__(self, example):
        return len(example[self.index])


class _AddLabel(object):

    def __init__(self, label, index, append=True, times=1):
        self.label = label
        self.append = append
        self.times = times
        self.index = index

    def __call__(self, example):
        example = list(example)
        i = self.index
        if self.append:
            # Not using `list.append` to avoid having weird mutable
            # example objects.
            example[i] = numpy.hstack([example[i], self.times * [self.label]])
        else:
            example[i] = numpy.hstack([self.times * [self.label], example[1]])
        return example


class _LengthFilter(object):

    def __init__(self, index, max_length):
        self.index = index
        self.max_length = max_length

    def __call__(self, example):
        if self.max_length:
            return 0 < len(example[self.index]) <= self.max_length
        return True

class SoftenResult(object):
    def __init__(self, data_labels, soften_distributions):
        self.soften_distributions = []
        for label,soften_distribution in soften_distributions.iteritems():
            self.soften_distributions += \
                [ (data_labels.index(label),
                   soften_distribution[0],
                   soften_distribution[1]) ]

    def __call__(self, example):
        example = list(example)
        for label_id, soften_factor, distribution in self.soften_distributions:
            data = example[label_id]
            assert data.ndim == 1, "data to soften must be 1-dimensional"
            if distribution is not None:
                mask = np.eye(distribution.shape[0])[data]
                tile = np.tile(distribution, (data.shape[0], 1))
                distribution_matrix = \
                (tile + distribution.T[data][:, np.newaxis]/(distribution.shape[0]-1))*soften_factor
            elif soften_factor >= 0:
                mask = np.eye(data.shape[0])[data]
                distribution_matrix = np.tile(1.0/(data.shape[0]-1), (data.shape[0], data.shape[0]))*soften_factor
            else:
                mask = np.zeros((data.shape[0], data.shape[0]))
                distribution_matrix = np.tile(1.0/(data.shape[0]), (data.shape[0], data.shape[0]))
            result = distribution_matrix * -(mask - 1) + (1-soften_factor)*mask
            result = result.astype(np.float32)
            example[label_id] = result
        example = tuple(example)
        return example

class ConvertToMask(object):
    def __init__(self, data_labels, source, mask_size):
        self.source_id = data_labels.index(source)
        self.mask_size = mask_size

    def __call__(self, example):
        sample = example[self.source_id]
        def convert_single(x):
            out = np.zeros((self.mask_size,), dtype=np.int32)
            out[ x[x>-1] ] = 1
            return out
        sample = np.apply_along_axis(convert_single, 1, sample)
        return example[:self.source_id] + (sample,) + example[(self.source_id+1):]

class ForceCContiguous(Transformer):
    """Force all floating point numpy arrays to be floatX."""
    def __init__(self, data_stream):
        super(ForceCContiguous, self).__init__(
            data_stream, axis_labels=data_stream.axis_labels)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        result = []
        for piece in data:
            if isinstance(piece, numpy.ndarray):
                result.append(numpy.ascontiguousarray(piece))
            else:
                result.append(piece)
        return tuple(result)

class GlobalPadding(Transformer):
    def __init__(self, data_stream, mask_sources=None, mask_dtype=None,
                 **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches of '
                             'examples, not examples')
        super(GlobalPadding, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        if mask_sources is None:
            mask_sources = self.data_stream.sources
        self.mask_sources = mask_sources
        if mask_dtype is None:
            self.mask_dtype = config.floatX
        else:
            self.mask_dtype = mask_dtype

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source in self.mask_sources:
                sources.append(source + '_mask')
        return tuple(sources)

    def transform_batch(self, batch):
        batch_with_masks = []
        max_length = 0
        for data in batch:
            max_sequence_length = max([numpy.asarray(sample).shape[0] for sample in data])
            max_length = max(max_length, max_sequence_length)
            
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source not in self.mask_sources:
                batch_with_masks.append(source_batch)
                continue

            shapes = [numpy.asarray(sample).shape for sample in source_batch]
            lengths = [shape[0] for shape in shapes]
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                assert all([shape[0] == shape[1] for shape in shapes]),\
                    "Arrays must be quadratic"
                dtype = numpy.asarray(source_batch[0]).dtype

                padded_batch = numpy.zeros(
                    (len(source_batch), max_length, max_length),
                    dtype=dtype)
                for i, sample in enumerate(source_batch):
                    padded_batch[i, :len(sample), :len(sample)] = sample
            else:
                dtype = numpy.asarray(source_batch[0]).dtype

                padded_batch = numpy.zeros(
                    (len(source_batch), max_length) + rest_shape,
                    dtype=dtype)
                for i, sample in enumerate(source_batch):
                    padded_batch[i, :len(sample)] = sample
            batch_with_masks.append(padded_batch)

            mask = numpy.zeros((len(source_batch), max_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            batch_with_masks.append(mask)
        return tuple(batch_with_masks)

class Data(object):
    """Dataset manager.

    This class is in charge of accessing different datasets
    and building preprocessing pipelines.

    Parameters
    ----------
    dataset_filename : str
        Dataset file name.
    name_mapping : dict
        A map from conceptual split names (train, test) into concrete split
        names (e.g. 93eval).
    sources_map: dict
        A map from conceptual source names, such as "labels" or "recordings"
        into names of dataset entries.
    batch_size : int
        Batch size.
    validation_batch_size : int
        Batch size used for validation.
    sort_k_batches : int
    max_length : int
        Maximum length of input, longer sequences will be filtered.
    normalization : str
        Normalization file name to use.
    add_eos : bool
        Add end of sequence symbol.
    add_bos : int
        Add this many beginning-of-sequence tokens.
    eos_label : int
        Label to use for eos symbol.
    default_sources : list
        Default sources to include in created datasets
    dataset_class : object
        Class for this particulat dataset kind (WSJ, TIMIT)
    """
    def __init__(self, dataset_filename, name_mapping, sources_map,
                 batch_size, validation_batch_size=None,
                 sort_k_batches=None,
                 max_length=None, normalization=None,
                 add_eos=True, eos_label=None,
                 add_bos=0, prepend_eos=False,
                 default_sources=None,
                 dataset_class=H5PYAudioDataset):
        assert not prepend_eos

        if normalization:
            with open(normalization, "rb") as src:
                normalization = cPickle.load(src)

        self.dataset_filename = dataset_filename
        self.dataset_class = dataset_class
        self.name_mapping = name_mapping
        self.sources_map = sources_map
        if default_sources is None:
            logger.warn(
                "The Data class was provided with no default_sources.\n"
                "All instantiated Datasets or Datastreams will use all "
                "available sources.\n")
            self.default_sources = sources_map.keys()

        self.normalization = normalization
        self.batch_size = batch_size
        if validation_batch_size is None:
            validation_batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.sort_k_batches = sort_k_batches
        self.max_length = max_length
        self.add_eos = add_eos
        self.prepend_eos = prepend_eos
        self._eos_label = eos_label
        self.add_bos = add_bos
        self.dataset_cache = {}
        #
        # Hardcode the number of source for length at 0
        # this typixcally works, as main.get_net_config
        # will properly set default_sources, such that the label is last
        # Unfortunately, we cannot query for a source name, as the
        # list of sources will differ....
        #
        self.length_filter = _LengthFilter(
            index=0,
            max_length=self.max_length)

    @property
    def info_dataset(self):
        return self.get_dataset("train")

    @property
    def num_labels(self):
        return self.info_dataset.num_characters

    @property
    def character_map(self):
        return self.info_dataset.char2num

    def num_features(self, feature_name):
        return self.info_dataset.num_features(feature_name)

    @property
    def eos_label(self):
        if self._eos_label:
            return self._eos_label
        return self.info_dataset.eos_label

    @property
    def bos_label(self):
        return self.info_dataset.bos_label

    def decode(self, labels):
        return self.info_dataset.decode(labels)

    def pretty_print(self, labels, example):
        return self.info_dataset.pretty_print(labels, example)

    def get_dataset(self, part, add_sources=()):
        """Returns dataset from the cache or creates a new one"""
        sources = []
        for src in self.default_sources + list(add_sources):
            sources.append(self.sources_map[src])
        sources = tuple(sources)
        key = (part, sources)
        if key not in self.dataset_cache:
            self.dataset_cache[key] = self.dataset_class(
                file_or_path=os.path.join(fuel.config.data_path[0],
                                          self.dataset_filename),
                which_sets=(self.name_mapping.get(part, part), ),
                sources_map=self.sources_map,
                sources=sources)
        return self.dataset_cache[key]

    def get_stream(self, part, batches=True, shuffle=True, add_sources=(),
                   num_examples=None, rng=None, seed=None):
        dataset = self.get_dataset(part, add_sources=add_sources)
        if num_examples is None:
            num_examples = dataset.num_examples

        if shuffle:
            iteration_scheme = ShuffledExampleScheme(num_examples, rng=rng)
        else:
            iteration_scheme = SequentialExampleScheme(num_examples)

        stream = DataStream(
            dataset, iteration_scheme=iteration_scheme)

        if self.add_eos:
            stream = Mapping(stream, _AddLabel(
                self.eos_label,
                index=stream.sources.index(self.sources_map['labels'])))
        if self.add_bos:
            if self.bos_label is None:
                raise Exception('No bos label given')
            stream = Mapping(stream, _AddLabel(
                self.bos_label, append=False, times=self.add_bos,
                index=stream.sources.index(self.sources_map['labels'])))

        if self.max_length:
            stream = Filter(stream, self.length_filter)

        if self.sort_k_batches and batches:
            stream = Batch(stream,
                           iteration_scheme=ConstantScheme(
                               self.batch_size * self.sort_k_batches))
            #
            # Hardcode 0 for source on which to sort. This will be good, as
            # most source lengths are correlated and, furthermore, the
            # labels will typically be the last source, thus in a single-input
            # case this sorts on input lengths
            #
            stream = Mapping(stream, SortMapping(_Length(
                index=0)))
            stream = Unpack(stream)

        if self.normalization:
            stream = self.normalization.wrap_stream(stream)
        stream = ForceFloatX(stream)
        stream = Rename(stream,
                        names=dict_subset({v: k for (k, v)
                                           in self.sources_map.items()},
                                          stream.sources,
                                          must_have=False))
        if not batches:
            return stream

        stream = Batch(
            stream,
            iteration_scheme=ConstantScheme(self.batch_size if part == 'train'
                                            else self.validation_batch_size))
        stream = Padding(stream)
        stream = Mapping(stream, switch_first_two_axes)
        stream = ForceCContiguous(stream)
        stream._produces_examples = False
        return stream

class PostfixManager:
    def __init__(self, languages, name_mapping):
        self.langs = languages
        self.name_mapping = name_mapping
    
    def combine_part_lang(self, part, lang):
        part = self.name_mapping.get(part, part)
        if lang != self.langs[0]:
            return part+'_'+lang
        else:
            return part
        
    def get_lang_postfix(self, lang):
        assert lang in self.langs
        if lang == self.langs[0]:
            return ''
        else:
            return '_'+lang
    
    def embed_lang_in_source(self, source, lang):
        postfix = self.get_lang_postfix(lang)
        if source.endswith('_mask'):
            return source[:-5]+postfix+'_mask'
        else:
            return source+postfix

class MultilangData(Data):
    _binary_convertable_data = ['features_per_word']
    def __init__(self, languages, *args, **kwargs):
        assert len(languages) > 0
        self.langs = languages
        super(MultilangData, self).__init__(*args, **kwargs)
        self.postfix_manager = PostfixManager(languages, self.name_mapping)
    
    def combine_part_lang(self, part, lang):
        return self.postfix_manager.combine_part_lang(part, lang)
        
    def get_lang_postfix(self, lang):
        return self.postfix_manager.get_lang_postfix(lang)
    
    def embed_lang_in_source(self, source, lang):
        return self.postfix_manager.embed_lang_in_source(source, lang)

    @property
    def info_dataset(self):
        return self.get_dataset("train", self.langs[0])

    def get_dataset(self, part, lang, add_sources=()):
        """Returns dataset from the cache or creates a new one"""
        part = self.combine_part_lang(part, lang)
        sources = []
        for src in self.default_sources + list(add_sources):
            sources.append(self.sources_map[src])
        sources = tuple(sources)
        key = (part, sources)
        if key not in self.dataset_cache:
            self.dataset_cache[key] = self.dataset_class(
                file_or_path=os.path.join(fuel.config.data_path[0],
                                          self.dataset_filename),
                which_sets=(self.name_mapping.get(part, part), ),
                sources_map=self.sources_map,
                sources=sources)
        return self.dataset_cache[key]

    def get_one_stream(self, part, lang=None, batches=True, shuffle=True, add_sources=(),
                   num_examples=None, rng=None, seed=None, num_result=None,
                   soften_distributions=None, only_stream=False):
        assert lang in self.langs
        dataset = self.get_dataset(part, lang, add_sources=add_sources)
        if num_examples is None:
            num_examples = dataset.num_examples

        if shuffle:
            iteration_scheme = ShuffledExampleScheme(num_examples, rng=rng)
        else:
            iteration_scheme = SequentialExampleScheme(num_examples)

        if num_result is None:
            num_result = num_examples

        if lang != self.langs[0] and not only_stream:
            iteration_scheme = RandomExampleScheme(num_examples, num_result=num_result, rng=rng)

        stream = DataStream(
            dataset, iteration_scheme=iteration_scheme)

        if soften_distributions:
            stream = Mapping(stream, SoftenResult(self.default_sources, soften_distributions))

        for bconv in self._binary_convertable_data:
            if bconv in self.default_sources:
                stream = Mapping(stream, ConvertToMask(self.default_sources,
                                                       bconv,
                                                       self.num_features(bconv)))

        if self.add_eos:
            stream = Mapping(stream, _AddLabel(
                self.eos_label,
                index=stream.sources.index(self.sources_map['labels'])))
        if self.add_bos:
            if self.bos_label is None:
                raise Exception('No bos label given')
            stream = Mapping(stream, _AddLabel(
                self.bos_label, append=False, times=self.add_bos,
                index=stream.sources.index(self.sources_map['labels'])))

        if self.max_length:
            stream = Filter(stream, self.length_filter)

        if self.sort_k_batches and batches:
            stream = Batch(stream,
                           iteration_scheme=ConstantScheme(
                               self.batch_size * self.sort_k_batches))
            #
            # Hardcode 0 for source on which to sort. This will be good, as
            # most source lengths are correlated and, furthermore, the
            # labels will typically be the last source, thus in a single-input
            # case this sorts on input lengths
            #
            stream = Mapping(stream, SortMapping(_Length(
                index=0)))
            stream = Unpack(stream)

        if self.normalization:
            stream = self.normalization.wrap_stream(stream)
        stream = ForceFloatX(stream)
        stream = Rename(stream,
                        names=dict_subset({v: k for (k, v)
                                           in self.sources_map.items()},
                                          stream.sources,
                                          must_have=False))
        if not batches:
            return stream, num_examples

        stream = Batch(
            stream,
            iteration_scheme=ConstantScheme(self.batch_size if part == 'train'
                                            else self.validation_batch_size))

        stream._produces_examples = False
        return stream, num_examples

    def get_stream(self, *args, **kwargs):
        lang_streams = []
        sources = []
        num_result=None
        
        for lang in self.langs:
            kwargs['lang'] = lang
            stream,num_examples = self.get_one_stream(*args, num_result=num_result, **kwargs)
            lang_streams += [stream]
            if lang == self.langs[0]:
                num_result = num_examples
            sources += [self.embed_lang_in_source(source, lang)
                        for source in lang_streams[-1].sources]

        if kwargs.get('batches', True):
            stream = Merge(lang_streams, sources)
            stream._produces_examples = False
            stream = GlobalPadding(stream)
            stream = Mapping(stream, switch_first_two_axes)
            stream = ForceCContiguous(stream)
        else:
            stream = Merge(lang_streams, sources)

        return stream
    
class RandomExampleScheme(IndexScheme):
    """Shuffled examples iterator.

    Returns examples in random order.

    """
    def __init__(self, *args, **kwargs):
        self.rng = kwargs.pop('rng', None)
        self.num_result = kwargs.pop('num_result')
        if self.rng is None:
            self.rng = numpy.random.RandomState()
        super(RandomExampleScheme, self).__init__(*args, **kwargs)

    def get_request_iterator(self):
        indices = list(self.indices)
        #self.rng.shuffle(indices)
        if indices != []:
            indices = self.rng.choice(indices, self.num_result)
        return iter_(indices)
