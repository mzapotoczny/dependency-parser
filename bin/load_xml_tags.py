from xml.etree.ElementTree import ElementTree
import cPickle as pickle

class PolishXMLTagConverter:
    def __init__(self, pos_convert_table):
        self.tree = ElementTree()
        self.pos_convert_table = {}
        with open(pos_convert_table, 'r') as f:
            for line in f:
                original, pos, tags = line.split()
                pos_tags = {'Pos' : pos}
                if tags != '_':
                    for pos_tag in tags.split('|'):
                        tag,value = pos_tag.split('=')
                        pos_tags[tag] = value
                self.pos_convert_table[original] = pos_tags

    def load(self, fileobj):
        self.tree.parse(fileobj)
        self.sentences = []
        for xmlsentence in self.tree.iterfind("chunk/sentence"):
            sentence = {'text' : []}
            i = -1
            for tok in xmlsentence:
                if tok.tag != 'tok':
                    continue
                i += 1
                value = tok.find('orth').text
                ctag  = tok.find('lex/ctag').text
                sentence['text'] += [value]
                for tag, val in self.convertTag(ctag).iteritems():
                    tag = tag.lower()
                    val = val.lower()
                    if not tag in sentence:
                        sentence[tag] = {}
                    sentence[tag][i] = val
            if sentence['text'] == ['.']:
                print(sentence['text'])
                continue
            self.sentences += [sentence]
        return self.sentences

    def convertTag(self, tag):
        return self.pos_convert_table.get(tag, {'Pos':'X'})

    def tagSentence(self, words):
        pass

class XMLTagConverter(PolishXMLTagConverter):
    pass


if __name__ == "__main__":
    tagconv = PolishXMLTagConverter('convert_table.txt')
    with open('pl-test-tags.xml', 'r') as f:
        tagconv.load(f)
