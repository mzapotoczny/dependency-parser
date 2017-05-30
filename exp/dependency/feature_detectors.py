import re
import codecs
import numpy as np


class AbstractFeatureDetector(object):
    def detect(self, sentence):
        result = []
        for word in sentence:
            result += [self.detectWord(word)]
        return result

    def detectWord(self, word):
        raise NotImplemented("Use derived class instead of AbstractFeatureDetector")

    def length(self):
        raise NotImplemented("Use derived class instead of AbstractFeatureDetector")

class BosEosFeatureDetector(AbstractFeatureDetector):
    _bos = '{'
    _eos = '}'

    def detect(self, sentence):
        result = [[0]] + [[] for _ in xrange(len(sentence)-2)] + [[1]]
        return result

    def length(self):
        return 2

class RegexFeatureDetector(AbstractFeatureDetector):
    _regexes = []
    _special_chars = ['?', '.']

    def __init__(self, filename):
        with codecs.open(filename, 'r', 'utf8') as f:
            for line in f:
                regex = line.strip()
                for sc in self._special_chars:
                    regex = regex.replace(sc, "\\"+sc)
                try:
                    self._regexes += [ re.compile(regex, re.UNICODE) ]
                except:
                    print "Cannot compile {}".format(regex)
                    raise

    def detectWord(self, word):
        out = []
        for i,matcher in enumerate(self._regexes):
            if matcher.search(word) is not None:
                out += [i]
        return out

    def length(self):
        return len(self._regexes)

class NumberFeatureDetector(AbstractFeatureDetector):
    def detectWord(self, word):
        try:
            float(word)
            return [0]
        except:
            return []

    def length(self):
        return 1

class PosFeatureDetector(AbstractFeatureDetector):
    _words = {}
    _tags = {}

    def __init__(self, filename):
        possible_tags = set()
        words = {}
        with codecs.open(filename, 'r', 'utf8') as f:
            for line in f:
                l = line.split()
                possible_tags |= set(l[1:])
                words[l[0]] = l[1:]

        self._tags = {tok: num for num, tok in enumerate(possible_tags)}
        for word, tags in words.iteritems():
            self._words[word] = map(self._tags.get, tags)

    def detectWord(self, word):
        return self._words.get(word, [])

    def length(self):
        return len(self._tags.keys())

class FeatureDetectors(object):
    _detectors = [BosEosFeatureDetector()]

    def __init__(self, detectors):
        for detector in detectors:
            detector = detector.split('=')
            dname = detector[0]
            args = detector[1:]
            self._detectors += [ globals()[dname](*args) ]

    def __call__(self, param):
        results = [[] for _ in xrange(len(param))]
        curr_length = 0
        for detector in self._detectors:
            curr_results = detector.detect(param)
            for wid, word in enumerate(curr_results):
                results[wid].extend( [res+curr_length for res in word] )
            curr_length += detector.length()
        max_len = max([len(wflt) for wflt in results])
        for i in xrange(len(results)):
            if len(results[i]) < max_len:
                results[i].extend([-1]*(max_len-len(results[i])))
        # if len(results) > 5:
            # from IPython import embed; embed()
        return np.array(results)

    def length(self):
        return sum([det.length() for det in self._detectors]) + 2

