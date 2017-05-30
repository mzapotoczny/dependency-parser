'''
Created on Jan 20, 2016

@author: jch
'''

import numpy
from fuel.datasets.hdf5 import H5PYDataset


class VMap(object):
    def __init__(self, vmap_dset):
        dtype = vmap_dset.dtype
        dtype_unicode = [('key', 'U%d' % (dtype[0].itemsize/4,)),
                         ('val', dtype[1])]
        vmap_dset = numpy.array(vmap_dset).view(dtype=dtype_unicode)
        self.tok2num = dict(vmap_dset)
        self.num2tok = {num: char for char, num in self.tok2num.iteritems()}


class H5PYTextDataset(H5PYDataset):
    def __init__(self, sources_map, **kwargs):
        self.sources_map = sources_map
        super(H5PYTextDataset, self).__init__(**kwargs)
        self.open()
        self.value_maps = {}
        for source in self.sources:
            vmap_dset = self._file_handle[
                self._file_handle[source].attrs['value_map']]
            self.value_maps[source] = VMap(vmap_dset)

        self.char2num = self.value_maps[self.sources_map['labels']].tok2num
        self.num2char = self.value_maps[self.sources_map['labels']].num2tok
        self.num_characters = len(self.num2char)
        self.eos_label = self.char2num['<eol>']
        self.bos_label = self.char2num.get('<bol>')

        self.matching_parens = []
        for isym in self.char2num.keys():
            if isym[0] != '(':
                continue
            beg = self.char2num[isym]
            end = self.char2num[isym[1:] + ')']
            self.matching_parens.append((beg, end))

    def num_features(self, feature_name):
        return len(self.value_maps[feature_name].tok2num)

    def decode(self, labels, keep_eos=False):
        ret = []
        for label in labels:
            if label in [self.eos_label, self.bos_label]:
                continue
            ph = self.num2char[label]
            ret.append(ph)
        return ret

    def pretty_print(self, labels, example):
        labels = self.decode(labels)

        num2tok = self.value_maps['chars_per_word'].num2tok

        words = []
        for encoded_w in example['chars_per_word'][1:-1]:
            chars = encoded_w[encoded_w > 0]
            words.append(u''.join(num2tok[c] for c in chars[1:-1]))

        ret = []
        ii = 0
        li = 0
        while li < len(labels) and ii < len(words):
            if labels[li] == 'slowo':
                ret.append(words[ii])
                li += 1
                ii += 1
            else:
                ret.append(labels[li])
                li += 1
        if li < len(labels):
            ret.extend(labels[li:])
        if ii < len(words):
            ret.append('missing:')
            ret.extend(words[ii:])
        labels = ' '.join(ret)
        return labels

    def validate_solution(self, inp, candidate_out):
        num_words = None
        for sname, sval in inp.iteritems():
            # Handle theano variables
            sname = getattr(sname, 'name', sname)
            vmap = self.value_maps[self.sources_map[sname]].tok2num
            if sname in ['words', 'base_forms', 'tags']:
                assert sval[0] == vmap['<s>']
                assert sval[-1] == vmap['</s>']
                num_words = len(sval) - 2  # subtract bos & eos
                break
            elif sname == 'chars_per_word':
                assert sval[0, 0, 1] == vmap['{']
                assert sval[-1, 0, 1] == vmap['}']
                num_words = len(sval) - 2  # subtract bos & eos
                break
            elif sname == 'characters':
                assert sval[0] == vmap['{']
                assert sval[-1] == vmap['}']
                num_words = (sval == vmap[' ']).sum() + 1
                break

        candidate_out = numpy.asarray(candidate_out)
        if candidate_out[-1] != self.eos_label:
            return False
        if ((candidate_out == self.char2num['slowo']
             ).sum() != num_words):
            return False
        for beg, end in self.matching_parens:
            no = numpy.cumsum(0.0 +  # avoid bool-bool
                              (candidate_out == beg) -
                              (candidate_out == end))
            if no[-1] != 0:
                return False
            if numpy.any(no < 0):
                return False
        return True
