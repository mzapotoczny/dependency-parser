import logging
logger = logging.getLogger(__name__)

from collections import defaultdict
from operator import itemgetter
import cPickle as pickle
import codecs
import re

def parse_file(filename):
    words = defaultdict(int)
    splitter = re.compile('\w+', re.UNICODE)
    with codecs.open(filename, 'r', 'utf8') as f:
        for line in f:
            for word in splitter.findall(line):
                words[word] += 1
    return words

class WordPiecesModel:
    def __init__(self, words_freq):
        self.words = words_freq

    def get_bigrams(self):
        bigrams = defaultdict(int)
        for skey, count in self.words.iteritems():
            key = self.words_translated[skey]
            for i in xrange(len(key)-1):
                bigrams[key[i], key[i+1]] += count
        return bigrams

    def convert_to_wp(self):
        converted = {}
        keys = sorted(self.pieces_dict.iterkeys(), reverse=True, key=len)
        for word in self.words.iterkeys():
            cword = word
            word_pieces = []
            while cword != "":
                for key in keys:
                    if cword.startswith(key):
                        word_pieces += key
                        cword = cword[len(key):]
                        break
            converted[word] = word_pieces
        return converted

    def generate_wp(self, iter_count=1):
        characters = set()
        for word in self.words.keys():
            characters |= set(word)
        self.pieces_dict = dict( ((v,k) for k,v in enumerate(sorted(characters))) )
        self.pieces_dict_r = dict( (v,k) for k,v in self.pieces_dict.iteritems() )
        self.words_translated = {}
        for k in self.words.keys():
            k_translated = tuple(map(lambda c: self.pieces_dict[c], k))
            self.words_translated[k] = k_translated 

        self.bigrams = self.get_bigrams()
        for iter in xrange(iter_count):
            max_bigram = max(self.bigrams.iteritems(), key=itemgetter(1))[0]

            bigram_name = self.pieces_dict_r[max_bigram[0]] + self.pieces_dict_r[max_bigram[1]]
            bigram_num = len(self.pieces_dict)

            self.pieces_dict[bigram_name] = bigram_num
            self.pieces_dict_r[bigram_num] = bigram_name

            self.update_dict(max_bigram, bigram_num)
            logger.info(u"New piece: {}".format(bigram_name))
        self.word_pieces = sorted(self.pieces_dict.iterkeys(), reverse=True,
                                  key=len)

    def chop_words(self, word_list=None, greedy=True):
        if word_list is None:
            word_list = self.words.keys()
        word_dict = {}
        for word in word_list:
            word = word.lower()
            pieces = []
            if greedy:
                word_postfix = word
                while word_postfix != "":
                    found = False
                    for piece in self.word_pieces:
                        if word_postfix.startswith(piece):
                            pieces += [piece]
                            word_postfix = word_postfix[len(piece):]
                            found = True
                            break
                    if not found:
                        pieces += word_postfix[:1]
                        word_postfix = word_postfix[1:]
                assert u''.join(pieces) == word, u"Pieces {} do not equal word {}".format(str(pieces), word)
            else:
                pieces = [self.pieces_dict_r[piece_num] for piece_num in self.words_translated[word]]
            word_dict[word] = pieces
        return word_dict

    def update_dict(self, bigram, value):
        for word, translation in self.words_translated.iteritems():
            word_count = self.words[word]
            i = 0
            while i < len(translation) - 1:
                current_bigram = (translation[i], translation[i+1])
                if current_bigram == bigram:
                    if i > 0:
                        self.bigrams[translation[i-1], translation[i]] -= word_count
                        self.bigrams[translation[i-1], value] += word_count
                        logger.debug( u"Left removed {} {}; added {} {}".format(
                                            translation[i-1],
                                            translation[i],
                                            translation[i-1],
                                            value) )
                    if i < len(translation) - 2:
                        self.bigrams[translation[i+1], translation[i+2]] -= word_count
                        self.bigrams[value, translation[i+2]] += word_count
                        logger.debug( u"Right removed {} {}; added {} {}".format(
                                            translation[i+1],
                                            translation[i+2],
                                            value,
                                            translation[i+2]) )
                    old_trans = translation
                    translation = translation[:i] + (value,) + translation[i+2:]
                    logger.debug(u"Found bigram {}{} in {}. \
                                  Old translation {}; new one {}".format(
                                      bigram[0], bigram[1],
                                      word,
                                      old_trans,
                                      translation) )
                i += 1
            self.words_translated[word] = translation
        del self.bigrams[bigram]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(
        usage='./generate_word_pieces.py iterations_count input_filename output_filename')
    parser.add_argument('iters', type=int)
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()

    word_dict = parse_file(args.infile)

    WPM = WordPiecesModel(word_dict)
    pieces_dict = WPM.generate_wp(args.iters)
    pieces_list = sorted(pieces_dict.keys(), key=len, reverse=True)

    with open(args.outfile, 'w') as fout:
        pickle.dump(pieces_list, fout)
