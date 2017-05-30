
class ConlluConverter:
    def __init__(self):
        pass

    def splitTags(self, pos, posTag):
        pos = {'pos': pos.lower()}
        if posTag != '_':
            for tag in posTag.split('|'):
                tags = tag.split('=')
                pos[tags[0].lower()] = tags[1].lower()
        return pos

    def load(self, conllufile):
        sentences = []
        current_sentence = {}
        for line in conllufile:
            if line.strip() == '' or line[0] == '#':
                if len(current_sentence.get('text', [])) > 0:
                    sentences += [current_sentence]
                    current_sentence = {}
            else:
                num, word, baseWord, pos, _, posTag, depNum, depType, _, _ = line.split('\t')
                try:
                    num = int(num)
                except:
                    print "Ignoring {}".format(num)
                    continue # multi word token
                current_sentence['text'] = current_sentence.get('text', []) + [word]
                for tag, value in self.splitTags(pos, posTag).iteritems():
                    if tag not in current_sentence:
                        current_sentence[tag] = {}
                    current_sentence[tag][num-1] = value
        if len(current_sentence.get('text', [])) > 0:
            sentences += [current_sentence]
            current_sentence = {}
        return sentences
