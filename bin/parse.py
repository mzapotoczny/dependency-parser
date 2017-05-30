# Parse sentences
import logging
import sys
import codecs
from lvsr.parser import get_parser 
from load_xml_tags import XMLTagConverter
from conllu import ConlluConverter

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--decoder-type', default=None)
    parser.add_argument('--lang', default=None)
    parser.add_argument('--tag-char', default=None)
    parser.add_argument('--mask-path', default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--input-tags', default=None)
    parser.add_argument('--input-is-conllu', default=False, action="store_true")
    parser.add_argument('inputfile', nargs='?', default='-')
    parser.add_argument('outputfile', nargs='?', default='-')

    args = parser.parse_args()
    model = args.model
    input_file = args.inputfile
    output_file = args.outputfile
    decoder_type = args.decoder_type
    tag_char = args.tag_char
    batch_size = args.batch_size
    mask_path = args.mask_path
    input_is_conllu = args.input_is_conllu
    lang = args.lang
    input_tags = args.input_tags

    tag_converter = None

    if input_tags is not None:
        tag_converter = XMLTagConverter(input_tags)
        if input_file == '-':
            input_file = sys.stdin
        else:
            input_file = open(input_file, 'r')
    else:
        if input_file == '-':
            input_file = codecs.getreader('utf-8')(sys.stdin) 
        else:
            input_file = codecs.open(input_file, 'r', 'utf-8')

    if input_is_conllu:
        tag_converter = ConlluConverter()

    if output_file == '-':
        output_file = codecs.getwriter('utf-8')(sys.stdout)
    else:
        output_file = codecs.open(output_file, 'w', 'utf-8')

    try:
        parser = get_parser(model, decoder_type, lang, tag_char, mask_path=mask_path)

        sentences = []
        def parseAndWrite():
            global sentences
            outputs = parser(sentences)
            for output in outputs:
                output_file.write(output)
                sentences = []

        if tag_converter is not None:
            all_sentences = tag_converter.load(input_file)
            for sentence in all_sentences:
                sentences.append(sentence)
                if len(sentences) >= batch_size:
                    parseAndWrite()
            parseAndWrite()
        elif input_file.isatty():
            while True:
                sentence = raw_input("> ").decode('utf-8')
                sentences.append(sentence)
                parseAndWrite()
            parseAndWrite()
        else:
            for sentence in input_file:
                sentences.append(sentence)
                if len(sentences) >= batch_size:
                    parseAndWrite()
            parseAndWrite()
    finally:
        input_file.close()
        output_file.close()
