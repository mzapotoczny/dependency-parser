#!/usr/bin/env python

import logging
import codecs
import numpy as np
logger = logging.getLogger(__name__)

def get_and_shuffle(ifname, ofname, count):
    sentences = []
    with codecs.open(ifname, 'r', encoding='utf8') as f:
        current_sentence = []
        for line in f:
            if line.strip() == '' or line[0] == '#':
                if current_sentence != []:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence += [line]
        if current_sentence != []:
            sentences.append(current_sentence)
            current_sentence = []
    indicies = np.random.choice( len(sentences), count, replace=False)
    with codecs.open(ofname, 'w', encoding='utf8') as f:
        for i in indicies:
            f.writelines(sentences[i])
            f.write("\n")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(
        usage='./shuffle_connlu_data.py data_file output_file examples_count')
    parser.add_argument('file')
    parser.add_argument('output')
    parser.add_argument('count')

    args = parser.parse_args()
    ifname = args.file
    ofname = args.output
    count = int(args.count)
    get_and_shuffle(ifname, ofname, count)
