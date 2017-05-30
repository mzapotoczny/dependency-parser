import logging
from lvsr.datasets import MultilangData
import numpy
from lvsr.main import create_recognizer 
from lvsr.config import Configuration
import cPickle as pickle
from blocks.search import CandidateNotFoundError
from segtok import tokenizer

logger = logging.getLogger(__name__)

DEFAULT_TAG_CHAR='s'

word_pieces = []

def list_to_ndarray(input_list):
    max_size = max(map(lambda x: len(x), input_list))
    return numpy.array( [ numpy.array(x + [0]*(max_size-len(x)),
                                dtype=numpy.int32) for x in input_list ] )

def chop_word(word, all_pieces=[]):
    pieces = []
    word_postfix = word
    while word_postfix != "":
        found = False
        for piece in all_pieces:
            if word_postfix.startswith(piece):
                pieces += [piece]
                word_postfix = word_postfix[len(piece):]
                found = True
                break
        if not found:
            pieces += word_postfix[:1]
            word_postfix = word_postfix[1:]
    assert u''.join(pieces) == word, u"Pieces {} do not equal word {}".format(str(pieces), word)
    return pieces

def remap_per_word(sentence, value_map, bos, eos):
    unk_id = value_map['UNK']

    # Some legacy dependencies
    if bos[0] in value_map and eos[0] in value_map:
        bos_num = [value_map[bos[0]]]
        eos_num = [value_map[eos[0]]]
    else:
        bos_num = [value_map[k] for k in bos[0]]
        eos_num = [value_map[k] for k in eos[0]]

    if '{' in value_map and '}' in value_map:
        wbos_num = [value_map['{']]
        weos_num = [value_map['}']]
    else:
        wbos_num = [value_map[k] for k in bos[0]]
        weos_num = [value_map[k] for k in eos[0]]

    output_value = []
    output_value += [ bos_num + wbos_num + eos_num ]
    for word in sentence:
        output_value += [map( lambda x: value_map[x]
                                if x in value_map
                                else value_map['UNK'],
                                bos+word+eos)]
    output_value += [ bos_num + weos_num + eos_num ]
    return list_to_ndarray(output_value)

def split_sentence(sentence_str):
    try:
        return tokenizer.web_tokenizer(sentence_str)
    except:
        return sentence_str.split()

def unify_format(sentences):
    # v1: only text
    # v2: dict
    example = sentences[0]
    if isinstance(example, basestring):
        return [{'text' : split_sentence(sentence)} for sentence in sentences]
    else:
        return sentences

def remap_sentences(sentences, inputs, dataset, postfix, tag_char='s'):
    sentences = unify_format(sentences)
    out = {}
    bos = ['<{}>'.format(tag_char)]
    eos = ['</{}>'.format(tag_char)]
    global word_pieces
    for input_type in inputs:
        orig_input_type = input_type
        if postfix != '':
            input_type = input_type[:-len(postfix)]

        output_values = []
        pieces = []
        if input_type.startswith('pieces_per_word') and word_pieces == []:
            word_pieces = sorted(dataset.value_maps['pieces_per_word'].tok2num.keys(),
                            key=len, reverse=True)
        for sentence in sentences:
            output_value = []
            if input_type.startswith('chars_per_word'):
                sentence = [word.lower() for word in sentence['text']]
                sentence = [list(word) for word in sentence]
                value_map = dataset.value_maps['chars_per_word'].tok2num
                output_value = remap_per_word(sentence, value_map, bos, eos)
            elif input_type.startswith('pieces_per_word'):
                sentence = [word.lower() for word in sentence['text']]
                sentence = [chop_word(word, word_pieces) for word in sentence]
                value_map = dataset.value_maps['pieces_per_word'].tok2num
                output_value = remap_per_word(sentence, value_map, bos, eos)
            elif input_type == 'sentences':
                value_map = dataset.value_maps[input_type].tok2num
                unk_id = value_map['UNK']
                output_value = [value_map.get(word.lower(), unk_id)
                            for word in bos + sentence['text'] + eos]
                output_value = numpy.array(output_value).T
            elif input_type in dataset.value_maps: #pos tag, words etc
                pos_sentence = [ sentence.get(input_type, {}).get(i, 'UNK')
                                      for i in xrange(len(sentence['text'])) ]
                value_map = dataset.value_maps[input_type].tok2num
                output_value = [value_map.get(val, value_map['UNK'])
                                    for val in bos + pos_sentence + eos]
                output_value = numpy.array(output_value).T
            else:
                raise ValueError('Unknown input type {}'.format(input_type))
            output_values.append(output_value)
        out[orig_input_type] = output_values
    return out

def output_conll(info_dataset, postfix_manager, sentence, lang, labels, pointers, pos_tags):
    sentence = unify_format([sentence])[0]
    sentence = sentence['text']
    labels_mapper =  info_dataset.value_maps['deps_types'].num2tok
    labels = [labels_mapper[x] for x in labels]
    assert len(sentence) == len(labels)
    output = u""
    pos_postfix_len = len(postfix_manager.get_lang_postfix(lang))
    for i in xrange(len(sentence)):
        pos_data = {}
        for n,v in pos_tags[i].iteritems():
            if pos_postfix_len > 0:
                n = n[:-pos_postfix_len]
            if v < info_dataset.value_maps[n].tok2num['UNK']:
                pos_data[n] = info_dataset.value_maps[n].num2tok[v]
        pos_tag = pos_data.pop('pos', '_').upper()
        pos_string = "|".join(["{}={}".format(n,v) for n,v in pos_data.iteritems()])
        if pos_string == '':
            pos_string = '_'
        output += u"{}\t{}\t_\t{}\t_\t{}\t{}\t{}\t_\t_\n".format(
                          i+1, sentence[i], pos_tag, pos_string, pointers[i], labels[i] 
                          )
    output += u"\n"
    return output
    
def extract_from_data(data):
    return pickle.loads("".join(numpy.frombuffer(data, 'S1')))

def get_parser(load_path, decoder_type, lang, tag_char=None, mask_path=None,
                **params):
    logger.info("Loading config from piclke!")
    v = numpy.load(load_path)
    config_path = extract_from_data(v['_config_pickle'])
    
    config = Configuration(
        config_path,
        '$LVSR/lvsr/configs/schema.yaml',
        {}
    )

    if 'input_languages' in config['data']:
        langs = config['data'].pop('input_languages')
    else:
        langs = ['default']

    if tag_char is not None:
        langs_tags = {k: tag_char for k in langs}
    else:
        langs_tags = {k: str(i) for i,k in enumerate(langs)}

    if lang is None:
        lang_id = 0
    else:
        if lang not in langs:
            raise ValueError('Wrong language {}. Available are: {}'.format(lang, repr(langs)))
        lang_id = langs.index(lang)
    lang = langs[lang_id]
      
    if '_dataset_pickle' in v and '_postfix_pickle' in v:
        info_dataset = extract_from_data(v['_dataset_pickle'])
        postfix_manager = extract_from_data(v['_postfix_pickle'])
    else:
        data = MultilangData(langs, **config['data'])
        net_config = dict(config["net"])
        addidional_sources = ['labels']
        if 'additional_sources' in net_config:
            addidional_sources += net_config['additional_sources']
        data.default_sources = net_config['input_sources'] + addidional_sources
        info_dataset = data.info_dataset
        postfix_manager = data.postfix_manager

    logger.info("Recognizer initialization started")
    multi_recognizer = create_recognizer(config, config['net'], langs, info_dataset,
                                         postfix_manager, load_path, mask_path)
    recognizer = multi_recognizer.children[lang_id]
    recognizer.init_beam_search(0)
    logger.info("Recognizer is initialized")

    required_inputs = recognizer.inputs.keys()
    def parse_sentences(sentences, decoder_type=decoder_type):
        remapped_sentences = remap_sentences(sentences, required_inputs, info_dataset,
                                             postfix_manager.get_lang_postfix(lang), langs_tags[lang])
        try:
            outputs, cost, pos_tags = recognizer.beam_search_multi(
                remapped_sentences, decoder_type=decoder_type,
                full_pointers = False,
                validate_solution_function=getattr(info_dataset,
                                                   'validate_solution',
                                                   None))
        except CandidateNotFoundError:
            outputs = ()
        return [output_conll(info_dataset, postfix_manager, sentence, lang,
                     outputs[0][i][1:-1],
                     outputs[1][i][1:-1],
                     pos_tags[i][1:-1]) for i, sentence in enumerate(sentences)]
    return parse_sentences
