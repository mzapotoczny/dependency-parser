Preparing dataset
=====
This program uses HDF5 file format for its input. We have included a converter
that can read conllu files. In the following subsection we show how you can
prepare dataset in different cases. All commands should be invoked inside
this directory.

## Single language
To prepare dataset for single language invoke
`python convert_data.py [OPTIONS] [OUTFILE.h5] train=[TRAIN_SUBSET.conllu] dev=[DEV_SUBSET.conllu] test=[TEST_SUBSET.conllu]`

## Multiple languages
For multiple languages we can write parameters manually in format:
`train=[FST_LANG] train_XX=[SND_LANG] train_YY=[THIRD_LANG]`
where `XX` and `YY` are shortcuts for language name (pl, cs, en etc).
The other option is to use script that will prepare the parameters for us:
`python convert_data_params.py UD_TREEBANKS_FOLDER FST_LANGUAGE_FOLDER SND_LANGUAGE_FOLDER ...`
So for example the command for creating polish-czech dataset will be
`python convert_data.py [OPTIONS] [OUTFILE.h5] `python convert_data_params.py data/ UD_Polish UD_Czech` `

## Converter options

Option | Description
------------ | -------------
--sort | Sort each split (train, dev, ...) by number of words
--split-tags | Treat each tag category as individual feature 
--print-sources | Print name_mapping and additional_sources options that can be used in config file
--enumerate-langs | Make different bos and eos for each language
--generate-pieces NUM | Generate word pieces and add them to the dataset
--max-sentence-length NUM | Only use sentences that <= number of words than NUM
--max-word-length NUM | Truncate words longer than NUM

## Examples:
#### Generate sorted polish dataset with additional pieces_per_word source (50 pieces)
    python convert_data.py --sort --split-tags --print-sources --generate-pieces 50 polish.h5 train=data/UD_Polish/pl-ud-train.conllu dev=data/UD_Polish/pl-ud-dev.conllu test=data/UD_Polish/pl-ud-test.conllu
#### Generate polish-czech dataset with enumerated langs (beggining-of-string tags: <0> for polish, <1> for czech)
    python convert_data.py --sort --split-tags --print-sources --enumerate-langs polish_czech.h5 `python convert_data_params.py data/ UD_Polish UD_Czech`

Training
=====
Config files for single and multiple languages learning are located respectively
in `configs/` and `configs/multi/` subfolders. To start the training process on
gpu run:
     `THEANO_FLAGS=device=gpu,mode=FAST_RUN,floatX=float32 FUEL_DATA_PATH=**H5_FILES_FOLDER** bin/run.py train --bokeh=0 **OUTPUT_DIR** **CONFIG_FILE_PATH** `
Now just wait from few hours to few days, depending on chosen config and your hardware.

For more options check `bin/run.py --help`.

Parsing sentences
=====
To interactively parse sentences invoke `python bin/parse.py MODEL INPUT_FILE OUTPUT_FILE`.
Both INPUT_FILE and OUTPUT_FILE may be set to - to indicate respectively stdin and stdout.

Evaluating with script
=====
Run: `bin/eval.sh MODEL.ZIP TEST_CONLLU_FILE`

Example:
    `bin/eval.sh local_storage2/optimized_tags/pretraining_best.zip exp/dependency/data/UD_Polish/pl-ud-test.conllu`

Manual evaluation 
=====
1. Run command `python bin/conll_to_sentences.py TEST_CONLLU_FILE.conllu | THEANO_FLAGS=device=gpu FUEL_DATA_PATH=`pwd`/exp/dependency python bin/parse.py MODEL_FILE - OUTPUT_FILE`

2. If there are comments in TEST_CONLLU_FILE, remove them by invoking 
    `grep -v -G "^#" INPUT_CONLLU_FILE > OUTPUT_FILE`

3. Invoke MaltParser
   `java -jar exp/dependency/malteval/lib/MaltEval.jar -e MALTEVALCONFIG -s PARSED_FILE -g GROUNDTRUTH_FILE`

Server
=====
We have included a simple json server. You can run it by invoking:
     `THEANO_FLAGS=device=gpu,mode=FAST_RUN,floatX=float32 bin/server.py [--port (defult 8888)] [--lang (default first)] MODEL_FILE`
It works by accepting a json POST request with the following schema:
```python
schema = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "minLength": 1},
        "decoder": {"type": "string", "enum" : ["greedy", "nonproj"]},
    },
    "required": ["text"],
    "additionalProperties": False
}
```
The server will queue the incoming request to run the neural network with certain
batch size (which will be smaller than desired when timeout will occur).
Checkout `site/` subfolder for example site.
