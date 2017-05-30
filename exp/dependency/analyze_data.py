#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Available data:
        report_data.append( {
                      'groundtruth': groundtruth,
                      'groundtruth_pointers' : groundtruth_pointers,
                      'recognized' : recognized,
                      'recognized_pointers' : recognized_pointers.tolist(),
                      'recognized_pointers_full' : outputs[2] if outputs != () else None,
                      'error_labels' : error_labels,
                      'error_pointers' : error_pointers
                      } )
"""

import argparse
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import operator

def save_confidence_fig(X, Y, C, file_name, \
        title="Confidence of choice", ylabel="confidence of pointer", xlabel="word number",
        subplots=[]):
    plt.figure(figsize=(30,20))
    plt.title(title)
    if not subplots:
        subplots = [(np.array([True] * X.shape[0]), "")]
    for segment, label in subplots:
        plt.scatter(X[segment], Y[segment], color=C[segment], alpha=0.5, label=label)
    plt.axis([-X.max()*0.01, X.max()*1.01, -0.01, 1.01])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if len(subplots) > 1:
        plt.legend(loc='lower right')
    plt.savefig(file_name)
    plt.close()

def analyze_confidence(data, output_dir):
    Y = []
    C = []
    count = 0
    for elem in data:
        gpointers = elem['groundtruth_pointers']
        rpointers = elem['recognized_pointers']
        full_pointers = elem['recognized_pointers_full'].squeeze()
        for word in xrange(len(gpointers)):
            if gpointers[word] != rpointers[word]:
                C.append('r')
            else:
                C.append('g')
            count+=1
            min_val = 1. / len(gpointers+2) # +2 because of bos and eos
            value = full_pointers[word+1][rpointers[word]]
            Y.append( (value - min_val) / (1 - min_val) )
    X = np.arange(count)
    C = np.array(C)
    Y = np.array(Y)

    bads = (C == 'r')
    goods = (C == 'g')
    
    save_confidence_fig(X[goods], Y[goods], C[goods], os.path.join(output_dir, "goods.png"),
            title="Confidence of choice (proper pointer)")
    save_confidence_fig(X[bads], Y[bads], C[bads], os.path.join(output_dir, "bads.png"),
            title="Confidence of choice (wrong pointer)")
    save_confidence_fig(X, Y, C, os.path.join(output_dir, "all.png"),
            subplots=[(goods, "Proper pointer"), (bads, "Wrong pointer")])

def analyze_confidence_with_label(data, output_dir):
    Y = []
    C = []
    X = []
    count = 0
    for elem in data:
        orig_labels = elem['groundtruth']
        reco_labels = elem['recognized']
        orig_pointers = elem['groundtruth_pointers']
        reco_pointers = elem['recognized_pointers']
        full_pointers = elem['recognized_pointers_full'].squeeze()
        for word in xrange(len(orig_pointers)):
            count += 1
            if orig_pointers[word] != reco_pointers[word]:
                if orig_labels[word] != reco_labels[word]:
                    C.append('r')
                else:
                    C.append('g')
                X.append(count)
                min_val = 1. / len(orig_pointers+2) # +2 because of bos and eos
                value = full_pointers[word+1][reco_pointers[word]]
                Y.append( (value - min_val) / (1 - min_val) )
    X = np.array(X)
    C = np.array(C)
    Y = np.array(Y)

    bads = (C == 'r')
    goods = (C == 'g')
    
    save_confidence_fig(X, Y, C, os.path.join(output_dir, "bad_labels.png"),
            title="Confidence of choice for wrong pointers",
            subplots=[(bads, "Wrong label"),(goods, "Proper label")])
    
def analyze_swaps(data):
    swaps = defaultdict(lambda: 0)
    frequency = defaultdict(lambda: 0)
    frequency_count = 0
    for elem in data:
        orig_labels = elem['groundtruth']
        reco_labels = elem['recognized']
        for i in xrange(len(orig_labels)):
            frequency[orig_labels[i]] += 1
            frequency_count += 1
            if orig_labels[i] != reco_labels[i]:
                swaps[(orig_labels[i], reco_labels[i])] += 1
    print("Frequency:")
    frequency = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
    for (label,count) in frequency:
        print("\t{}: {:.3f}".format(label, count / float(frequency_count)))

    print("Swaps:")
    swaps = sorted(swaps.items(), key=operator.itemgetter(1), reverse=True)
    for swap in swaps:
        print("\t{} -> {} : {}".format(swap[0][0], swap[0][1], swap[1]))

def analyze_correlation(data):
    pointer_with_label = 0
    wrong_pointer_count = 0
    label_with_pointer = 0
    wrong_label_count = 0
    for elem in data:
        orig_labels = elem['groundtruth']
        reco_labels = elem['recognized']
        orig_pointers = elem['groundtruth_pointers']
        reco_pointers = elem['recognized_pointers']

        for i in xrange(len(orig_labels)):
            pointer_diff = orig_pointers[i] != reco_pointers[i]
            label_diff   = orig_labels[i] != reco_labels[i]
            if pointer_diff:
                wrong_pointer_count += 1
                pointer_with_label += 1 if label_diff else 0

            if label_diff:
                wrong_label_count += 1
                label_with_pointer += 1 if pointer_diff else 0

    print("Correlations:")
    pointer_with_label_per = pointer_with_label*100.0/wrong_pointer_count
    print("\tWrongly classified pointers with wrong labels: {:.3f}%".format(pointer_with_label_per))
    print("\tWrongly classified pointers with proper labels: {:.3f}%".format(100 - pointer_with_label_per))

    label_with_pointer_per = label_with_pointer*100.0/wrong_label_count
    print("\tWrongly classified labels with wrong pointers: {:.3f}%".format(label_with_pointer_per))
    print("\tWrongly classified labels with proper pointers: {:.3f}%".format(100 - label_with_pointer_per))

def analyze_LAS_score(data):
    """
        Based on: Stacking or Supertagging for Dependency Parsing – What’s the Difference?:
        LAS = 
        The ratio of tokens with a correct head and label to the total number of tokens in the test data.
    """
    correct = 0
    count = 0
    for elem in data:
        orig_labels = elem['groundtruth']
        reco_labels = elem['recognized']
        orig_pointers = elem['groundtruth_pointers']
        reco_pointers = elem['recognized_pointers']
        for i in xrange(len(orig_labels)):
            count += 1
            if orig_labels[i] == reco_labels[i] and \
               orig_pointers[i] == reco_pointers[i]:
                correct += 1
    print("LAS score: {:.3f}".format(correct*100.0/count))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='./analyze_data.py report_file.pickle output_dir')
    parser.add_argument('report_file')
    parser.add_argument('output_dir')

    args = parser.parse_args()
    report_file = args.report_file
    outpt_dir = args.output_dir
    with open(report_file, 'r') as f:
        data = pickle.load(f)
        analyze_swaps(data)
        analyze_correlation(data)
        analyze_LAS_score(data)
        #analyze_confidence(data, outpt_dir)
        #analyze_confidence_with_label(data, outpt_dir)
