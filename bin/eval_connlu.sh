#!/bin/bash

USAGE_STR="USAGE: $0 MODEL INPUT_CONLLU_FILE [--lang LANG]"

function usage {
    echo $USAGE_STR
    exit 1
}

function check_file {
    if [ ! -f "$1" ] || [ "$1" == "" ]; then
        usage
    fi
}

MODEL=$1
INPUT_CONLLU_FILE=$2
MALTEVALCONFIG="exp/dependency/malteval/evalpl_comparison/01_overall.xml"
ADDITIONAL_PARAMS=$3

PARSED_FILE="/tmp/parsed_$$.conll"
GROUNDTRUTH_FILE="/tmp/groundtruth_$$.conll"

check_file $MODEL
check_file $INPUT_CONLLU_FILE
check_file $MALTEVALCONFIG

#export THEANO_FLAGS=device=gpu
#export FUEL_DATA_PATH=`pwd`/exp/dependency 

python bin/parse.py --input-is-conllu $ADDITIONAL_PARAMS $MODEL $INPUT_CONLLU_FILE $PARSED_FILE
grep -v -G "^#" $INPUT_CONLLU_FILE | grep -v -G "^[0-9]*-[0-9]*" > $GROUNDTRUTH_FILE
java -jar exp/dependency/malteval/lib/MaltEval.jar -e $MALTEVALCONFIG -s $PARSED_FILE -g $GROUNDTRUTH_FILE
echo "Output saved as $PARSED_FILE"
