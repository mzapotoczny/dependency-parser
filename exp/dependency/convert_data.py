#!/usr/bin/env python

import logging
logger = logging.getLogger(__name__)

import numpy as np
import h5py
import codecs
from fuel.datasets.hdf5 import H5PYDataset
from collections import OrderedDict

from generate_word_pieces import WordPiecesModel

from feature_detectors import FeatureDetectors

def analyse_line(line, max_word_length, words=None, baseForms=None,
                 pos_tags=None, dep_types=None, dep_nums=None, split=False):
    num, word, baseWord, pos, _, posTag, depNum, depType, _, _ = line.split('\t')
    num = int(num)
    word = word.lower()
    baseWord = baseWord.lower()
    if len(word) >= max_word_length:
        logger.info("Truncating word {}".format(word))
        word = word[:max_word_length]
    if len(baseWord) >= max_word_length:
        logger.info("Truncating base {}".format(baseWord))
        baseWord = baseWord[:max_word_length]
    if split:
        pos = {'pos': pos.lower()}
        if posTag != '_':
            for tag in posTag.split('|'):
                tags = tag.split('=')
                pos[tags[0].lower()] = tags[1].lower()
    else:
        pos = {'poses':pos+'|'+posTag}
    if words is not None:
        words[word] = words.get(word, 0) + 1
        baseForms[baseWord] = baseForms.get(baseWord, 0) + 1
        for k,v in pos.iteritems():
            if k in pos_tags:
                pos_tags[k].add(v)
            else:
                pos_tags[k] = set([v])
        dep_types.add(depType)
    
    return word, baseWord, pos, depNum, depType

def create_and_insert_vlen_int32_dataset(h5file, name, value_map=None, values=None,
                              extra_dims=[] ):
    shapes = h5file.create_dataset("%s_shapes" % (name,),
                                   (0, 1 + len(extra_dims)),
                                   dtype='int32',
                                   maxshape=(None, 1 + len(extra_dims)))
    shape_labels = h5file.create_dataset("%s_shape_labels" % (name,),
                                         (1 + len(extra_dims),),
                                         dtype='S7')
    shape_labels[...] = extra_dims + ['token'.encode('utf8')]

    dataset = h5file.create_dataset(name, (0,),
                                    dtype=h5py.special_dtype(vlen=np.int32),
                                    maxshape=(None,))
    dataset.dims[0].label = 'batch'
    dataset.dims.create_scale(shapes, 'shapes')
    dataset.dims[0].attach_scale(shapes)

    dataset.dims.create_scale(shape_labels, 'shape_labels')
    dataset.dims[0].attach_scale(shape_labels)

    if value_map is not None:
        max_k_len = max(len(k) for k in value_map.iterkeys())
        value_map_arr = np.fromiter(value_map.iteritems(),
                                    dtype=[('key', 'U{}'.format(max_k_len)),
                                           ('val', 'int32')])
        value_map_arr_s = value_map_arr.view(
            dtype=[('key', 'S{}'.format(4*max_k_len)), ('val', 'int32')])
        value_map_data = h5file.create_dataset(
            "%s_value_map" % (name,), data=value_map_arr_s)
        dataset.attrs['value_map'] = value_map_data.ref

    if values is not None:
        size = len(values)
        if dataset.shape[0] < size:
            dataset.resize((size,) + dataset.shape[1:])
        if shapes.shape[0] < size:
            shapes.resize((size,) + shapes.shape[1:])
        if isinstance(values[0], np.ndarray):
            for i, v in enumerate(values):
                shapes[i, :] = v.shape
                dataset[i] = v.ravel()
        else:
            for i, v in enumerate(values):
                shapes[i, :] = (len(v),)
                dataset[i] = v


def convert_files(h5file_name, files, curriculum_sort, split=False,
                  printsources=False, generate_pieces=None, max_sent_len=None,
                  max_word_length=None, infrequent_del_prob=None,
                  feature_detectors=None):
    if max_sent_len <= 0:
        max_sent_len = 1e8
    if max_word_length <= 0:
        max_word_length = 1e8 
    words={}
    baseForms={}
    pos_tags={}
    dep_types=set()
    dep_nums=set()
    sentences = []
    splits = []
    last_split = 0
    max_sentence_len = 0
    max_fid = 0
    for name, file_paths in files.iteritems():
        file_sentences = []
        for fid, file_path in file_paths:
            max_fid = max(max_fid, fid)
            with codecs.open(file_path, 'r', encoding='utf8') as f:
                sentence = []
                sentence_base = []
                sentence_pos = []
                sentence_dep = []
                sentence_dep_type = []
                for line in f:
                    if line.strip() == '' or line[0] == '#':
                        if sentence:
                            if len(sentence) <= max_sent_len:
                                file_sentences.append((fid, sentence, sentence_base, sentence_pos, sentence_dep, sentence_dep_type))
                                max_sentence_len = max(max_sentence_len, len(sentence) + 2)
                            else:
                                logger.info(u"Sentence too long ({}, max: {}): {}".format(len(sentence), max_sent_len, " ".join(sentence)))
                        sentence = []
                        sentence_base = []
                        sentence_pos = []
                        sentence_dep = []
                        sentence_dep_type = []
                        continue
                    try:
                        word, baseWord, posTag, depNum, depType = \
                            analyse_line(line, max_word_length, words, baseForms, pos_tags, dep_types, dep_nums, split=split)
                        sentence.append(word)
                        sentence_base.append(baseWord)
                        sentence_pos.append(posTag)
                        sentence_dep_type.append(depType)
                        sentence_dep.append(depNum)
                    except ValueError:
                        pass
        if curriculum_sort:
            #print("Sorting...")
            sentences.extend( sorted(file_sentences, key=lambda sent: len(sent[1])) )
        else:
            sentences.extend(file_sentences)
        splits.append( (name, last_split, len(sentences)-1) )
        last_split = len(sentences)


    #from IPython import embed; embed()
    characters = set()
    for word in words.iterkeys():
        characters |= set(word)
    characters.add('<')
    characters.add('>')
    characters.add('/')
    for i in xrange(max_fid):
        characters.add(str(i))

    max_word_len = len(max(words.iterkeys(), key=len)) + 2

    if infrequent_del_prob:
        infreq_words = [word for word,occurrence_num in words.iteritems() if occurrence_num == 1]
        chosen = np.random.choice(np.arange(len(infreq_words)),
                                  size=infrequent_del_prob*len(infreq_words),
                                  replace=False)
        for infreq_i in chosen:
            chosen_word = infreq_words[infreq_i]
            del words[chosen_word]
            logger.info(u"Deleting infrequent word {}".format(chosen_word))

    if generate_pieces is None:
        generate_pieces = False
    else:
        generate_pieces_count = int(generate_pieces)
        generate_pieces = True

    word_pieces = []
    if generate_pieces:
        for var,name,var_name in [(words, "words", "pieces_per_word"), (baseForms, "base words", "pieces_per_base")]:
            wpm = WordPiecesModel(var)
            logger.info(u"Generating word pieces for: {}".format(name))
            wpm.generate_wp(generate_pieces_count)
            chopped = wpm.chop_words()
            max_len = len(max(chopped.itervalues(), key=len)) + 2
            word_pieces += [(var_name, max_len, chopped)]
    
    characters_out = OrderedDict(((tok, num) for num, tok 
                            in enumerate(sorted(list(characters)))))
    words_out = OrderedDict(((tok, num) for num, tok
                        in enumerate(sorted(words.iterkeys()))))
    baseForms_out = OrderedDict(((tok, num) for num, tok
                        in enumerate(sorted(baseForms.iterkeys()))))
    dep_types_out = OrderedDict(((tok, num) for num, tok
                        in enumerate(sorted(dep_types))))
    word_pieces_out = []
    for _, _, word_dict in word_pieces:
        pieces_set = set()
        for pieces in word_dict.itervalues():
            pieces_set |= set(pieces)
        out = OrderedDict(((tok, num) for num, tok
                        in enumerate(sorted(list(pieces_set)))))
        word_pieces_out += [out]
    
    def fid_to_char(fid, closing=False):
        return ('</' if closing else '<') + str(fid) + '>'
    
    def addInfoTypes(dictionary):
        dictionary.setdefault('{', len(dictionary))
        dictionary.setdefault('}', len(dictionary))
        dictionary['UNK'] = len(dictionary)
        for i in xrange(max_fid+1):
            dictionary[fid_to_char(i)] = len(dictionary)
            dictionary[fid_to_char(i, True)] = len(dictionary)

    pos_tags_out = {}
    for key in pos_tags.keys():
        pos_tags_out[key] = OrderedDict(((tok, num) for num, tok
                                in enumerate(sorted(pos_tags[key]))))
        addInfoTypes(pos_tags_out[key])
    pos_tags_keys = sorted(pos_tags_out.keys())
    
    addInfoTypes(words_out)
    addInfoTypes(baseForms_out)
    addInfoTypes(dep_types_out)
    addInfoTypes(characters_out)
    for piece_out in word_pieces_out:
        addInfoTypes(piece_out)
    
    def sentence_to_chars(sentence, out_dict, max_len, fid, pieces_dict={}):
        output = []
        st = fid_to_char(fid)
        en = fid_to_char(fid, True)
        for word in sentence:
            word_list = pieces_dict.get(word, list(word))
            word = [st] + word_list + [en]
            chars = np.zeros((max_len,), dtype=np.int32)
            i = 0
            for i in xrange(len(word)):
                chars[i] = out_dict[word[i]]
            output += [chars.ravel()]
        return np.vstack(output)

    def mapPOS(poses, fid):
        st = fid_to_char(fid)
        en = fid_to_char(fid, True)
        out = []
        out.append([pos_tags_out[tag][st] for tag in pos_tags_keys])
        for i,pos in enumerate(poses):
            out.append([pos_tags_out[tag][pos[tag]]
                        if tag in pos else pos_tags_out[tag]['UNK']
                        for tag in pos_tags_keys])
        out.append([pos_tags_out[tag][en] for tag in pos_tags_keys])
        return np.array(out, dtype=np.int32)

    def constructFeatures(sentence):
        if feature_detectors:
            return feature_detectors(['{'] + sentence[1] + ['}'])
        else:
            return None
    
    def remapSentence(sentence):
        fid,sent,base,pos,dep,dep_type = sentence
        st = [fid_to_char(fid)]
        en = [fid_to_char(fid, True)]
        pieces_sent = []
        for i, (name, max_len, pieces_dict) in enumerate(word_pieces):
            data_source = sent if name.endswith('word') else base
            pieces_sent += [sentence_to_chars(['{'] + data_source + ['}'], 
                    word_pieces_out[i], max_len, fid, pieces_dict)]
        return \
            map(lambda x: words_out.get(x, words_out['UNK']), st + sent + en),\
            map(lambda x: baseForms_out[x],  st + base + en),\
            mapPOS(pos, fid),\
            sentence_to_chars(['{'] + sent + ['}'], characters_out, max_word_len, fid),\
            map(lambda x: x, [0] + dep + [len(dep)+1]),\
            map(lambda x: dep_types_out[x], st + dep_type + en), \
            pieces_sent, \
            constructFeatures(sentence)
    sentences_out = [remapSentence(sent) for sent in sentences]
    
    with h5py.File(h5file_name, 'w') as h5file:
        _sents,_bases,_poses,_chars_per_word,_deps,_dep_types,_pieces_list,_feats_list=zip(*sentences_out)
        _pieces_list = zip(*_pieces_list)
        create_and_insert_vlen_int32_dataset(h5file, 'sentences', words_out, _sents)
        create_and_insert_vlen_int32_dataset(h5file, 'deps', None, _deps)
        create_and_insert_vlen_int32_dataset(h5file, 'deps_types', dep_types_out, _dep_types)
        create_and_insert_vlen_int32_dataset(h5file, 'bases', baseForms_out, _bases)
        create_and_insert_vlen_int32_dataset(h5file, 'chars_per_word', characters_out,
                                             _chars_per_word, extra_dims=['word'])
        if feature_detectors is not None:
            create_and_insert_vlen_int32_dataset(h5file, 'features_per_word', None,
                                                 _feats_list, extra_dims=['word'])
            h5file.attrs['features_per_word_length'] = feature_detectors.length()
        for i, (name, _, _) in enumerate(word_pieces):
            create_and_insert_vlen_int32_dataset(h5file, name, word_pieces_out[i],
                                            _pieces_list[i], extra_dims=['word'])

        for tag in pos_tags_keys:
            create_and_insert_vlen_int32_dataset(h5file, tag, pos_tags_out[tag], None)
        create_and_insert_vlen_int32_dataset(h5file, 'additionals', None, _poses,
                                             extra_dims=['poses'])
        split_dict = {}
        for split_name, split_start, split_end in splits:
            se = (split_start,split_end)
            split_dict[split_name] = {'sentences': se,
                                     'deps_types': se,
                                     'deps' : se,
                                     'bases': se,
                                     'chars_per_word': se,
                                     'additionals' : se}
            for name, _, _ in word_pieces:
                split_dict[split_name][name] = se
            if feature_detectors is not None:
                split_dict[split_name]['features_per_word'] = se
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    if printsources:
        print("    sources_map:")
        for tname in pos_tags_keys:
            print("        {}: {}".format(tname, tname))
        print("net:")
        print("    additional_sources:")
        print("        - pointers")
        for tname in pos_tags_keys:
            print("        - {}".format(tname))
    else:
        print("Available grammar tags: \n{}".format("\n".join(pos_tags_keys)))

"""
    --enumerate-tags Choose wheter we want to have different bos, eos for every
                     language. Different languages in one split are in form
                     x=file x=file2 or x=file x_lang=file2. The order matters
                     i.e. order of languages for each split sould be the same.
"""

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(
        usage='./convert_data.py [--sort] [--split-tags] [--print-sources] [--enumerate-langs] [--generate-pieces PIECES_NUM] [--max-sentence-length MAX_LEN] dependency.h5 train=data/UD_Polish/pl-ud-train.conllu dev=data/UD_Polish/pl-ud-dev.conllu test=data/UD_Polish/pl-ud-test.conllu')
    parser.add_argument('--sort', action='store_true')
    parser.add_argument('--split-tags', action='store_true')
    parser.add_argument('--print-sources', action='store_true')
    parser.add_argument('--enumerate-langs', action='store_true')
    parser.add_argument('--generate-pieces', action='store')
    parser.add_argument('--max-sentence-length', action='store')
    parser.add_argument('--max-word-length', action='store')
    parser.add_argument('--infrequent-del-prob', action='store')
    parser.add_argument('--feature-detectors', nargs='*')
    parser.add_argument('h5file')
    parser.add_argument('files', nargs='*')

    args = parser.parse_args()
    curriculum_sort = args.sort
    split = args.split_tags
    h5file = args.h5file
    printsources = args.print_sources
    enumerate_langs = args.enumerate_langs
    generate_pieces = args.generate_pieces
    max_len = args.max_sentence_length
    max_word_length = args.max_word_length
    infrequent_del_prob = args.infrequent_del_prob
    feature_detectors = args.feature_detectors
    files = [s.split('=') for s in args.files]
    files_dict = {}
    if max_len is not None:
        max_len = int(max_len)
    if infrequent_del_prob is not None:
        infrequent_del_prob = float(infrequent_del_prob)
    if max_word_length is not None:
        max_word_length = int(max_word_length)
    for name,path in files:
        dict_len = len(files_dict[name]) if name in files_dict else 0
        fid = dict_len
        if '_' in name and enumerate_langs:
            orig_name = name[:name.rindex('_')]
            if orig_name in files_dict:
                fid = len(files_dict[orig_name])
        if not enumerate_langs:
            fid = 0
        if dict_len > 0:
            files_dict[name] += [(fid, path)]
        else:
            files_dict[name] = [(fid, path)]

    if feature_detectors:
        feature_detectors = FeatureDetectors(feature_detectors)

    convert_files(h5file, files_dict, curriculum_sort, split, printsources,
                  generate_pieces, max_len, max_word_length, infrequent_del_prob,
                  feature_detectors)
