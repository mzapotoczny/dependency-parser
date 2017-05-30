'''
Created on Mar 20, 2016
'''

import numpy
import numbers
from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import Subset

class VMap(object):
    def __init__(self, vmap_dset):
        dtype = vmap_dset.dtype
        dtype_unicode = [('key', 'U%d' % (dtype[0].itemsize/4,)),
                         ('val', dtype[1])]
        vmap_dset = numpy.array(vmap_dset).view(dtype=dtype_unicode)
        self.tok2num = dict(vmap_dset)
        self.num2tok = {num: char for char, num in self.tok2num.iteritems()}
        
class VMapIdentity(VMap):
    def __init__(self, dset):
        dset = numpy.array(dset)
        self.tok2num = {}
        for sample in dset:
            for num in sample:
                self.tok2num[num] = num;
        self.num2tok = self.tok2num
        #self.num2tok = {num: char for char, num in self.tok2num.iteritems()}

class H5PYTextDataset(H5PYDataset):
    _clean_sources = ['deps', 'features_per_word']

    def __init__(self, sources_map, **kwargs):
        self.sources_map = sources_map
        super(H5PYTextDataset, self).__init__(**kwargs)
        self.open()
        self.value_maps = {}
        for source in self._file_handle:
            if '_value_map' not in source and source not in self._clean_sources:
                continue
            if source not in self._clean_sources:
                source = source[:-len('_value_map')]
            try:
                vmap_dset = self._file_handle[
                    self._file_handle[source].attrs['value_map']]
                self.value_maps[source] = VMap(vmap_dset)
            except KeyError:
                self.value_maps[source] = VMapIdentity(self._file_handle[source])

        self.char2num = self.value_maps[self.sources_map['labels']].tok2num
        self.num2char = self.value_maps[self.sources_map['labels']].num2tok
        self.num_characters = len(self.num2char)
        self.eos_label = 0
        self.bos_label = 0

    def _parse_dataset_info(self):
        self._out_of_memory_open()
        handle = self._file_handle
        available_splits = self.get_all_splits(handle)
        which_sets = self.which_sets
        provides_sources = None
        for split in which_sets:
            if split not in available_splits:
                raise ValueError(
                    "'{}' split is not provided by this ".format(split) +
                    "dataset. Available splits are " +
                    "{}.".format(available_splits))
            split_provides_sources = set(
                self.get_provided_sources(handle, split))
            if provides_sources:
                provides_sources &= split_provides_sources
            else:
                provides_sources = split_provides_sources
        if 'additionals' in provides_sources:
            sources = map(lambda x: x[:-len('_value_map')],
                           filter(lambda x: '_value_map'  in x, handle))
            self.additional_sources = sorted(list(set(sources) - provides_sources))
            provides_sources |= set(sources)
        else:
            self.additional_sources = []
        self.provides_sources = tuple(sorted(provides_sources))
        self.vlen_sources = self.get_vlen_sources(handle)
        self.default_axis_labels = self.get_axis_labels(handle)
        self._out_of_memory_close()

    @staticmethod
    def get_subsets(h5file, splits, sources):
        split_sources = set([r['source'] for r in  h5file.attrs['split']])
        if 'additionals' in split_sources:
            sources = [s if s in split_sources else 'additionals' for s in sources]
        return H5PYDataset.get_subsets(h5file, splits, sources)

    def _in_memory_get_data(self, state=None, request=None):
        raise NotImplemented()

    def _out_of_memory_get_data(self, state=None, request=None):
        if not isinstance(request, (numbers.Integral, slice, list)):
            raise ValueError()
        data = []
        shapes = []
        handle = self._file_handle
        
        integral_request = isinstance(request, numbers.Integral)
        
        if self.additional_sources != []:
            additional_name = [s for s in self.additional_sources if s in self.sources]
            if len(additional_name) > 0:
                additional_name = additional_name[0]
                asubset = self.subsets[self.sources.index(additional_name)]
                additionals_shapes = asubset.index_within_subset(
                                    handle['additionals'].dims[0]['shapes'], request,
                                    sort_indices=self.sort_indices)
                additionals_data = asubset.index_within_subset(
                                handle['additionals'], request,
                                sort_indices=self.sort_indices)
        for source_name, subset in zip(self.sources, self.subsets):
            # Process the data request within the context of the data source
            # subset
            if source_name in self.additional_sources:
                source_index = self.additional_sources.index(source_name)
                if integral_request:
                    data.append(
                        additionals_data[source_index::additionals_shapes[1]])
                    shapes.append(additionals_shapes[:1])
                else:
                    data.append(
                        numpy.array([x[source_index::additionals_shapes[i][1]]
                                     for i,x in enumerate(additionals_data)])
                                )
                    shapes.append(additionals_shapes[:,:1])
            else:
                data.append(
                    subset.index_within_subset(
                        handle[source_name], request,
                        sort_indices=self.sort_indices))
                # If this source has variable length, get the shapes as well
                if source_name in self.vlen_sources:
                    shapes.append(
                        subset.index_within_subset(
                            handle[source_name].dims[0]['shapes'], request,
                            sort_indices=self.sort_indices))

                else:
                    shapes.append(None)
        #from IPython import embed; embed()
        return data, shapes

    def num_features(self, feature_name):
        if feature_name in self._clean_sources and 'per_word' in feature_name:
            return self._file_handle.attrs[feature_name+'_length']
        else:
            return len(self.value_maps[feature_name].tok2num)

    def decode(self, labels, keep_eos=False):
        return map(lambda x: self.num2char[x], labels[1:-1])

    def pretty_print(self, labels, example):
        labels = self.decode(labels)
        return labels

    def get_value_map_name(self, source):
        if source == '':
            return source

        if source in self.value_maps:
            return source
        else:
            return self.get_value_map_name(source[:source.rindex('_')])
    
    def print_text(self, words):
        key = words.keys()[0]
        vkey = self.get_value_map_name(key)
        value = words[key]
        sentence = []
        vmap = self.value_maps[vkey].num2tok
        try:
            if value.ndim == 2: 
                max_char = self.value_maps[vkey].tok2num['UNK']
                for i in xrange(value.shape[0]):
                    word = []
                    for vi,v in enumerate(value[i]):
                        word += [vmap[v]]
                        if vi > 0 and v > max_char:
                            break
                    sentence += [''.join(word[1:-1])]
        except:
            return "UNKNOWN TEXT"
        return ' '.join(sentence[1:-1])
                
    def validate_solution(self, inp, candidate_out):
        num_words = len(inp.values()[0])
        deps = candidate_out[0]
        pointers = candidate_out[1]
        if deps.shape[0] != pointers.shape[0] or \
           deps.shape[0] != num_words:
            return False

        return True
