import sys 
import codecs

def printFile(inFile, outFile, doSpaces):
    buff = []
    for line in inFile:
        if line.strip() == '':
            if buff != []:
                outFile.write(u" ".join(buff))
                outFile.write("\n")
                buff = []
            continue
        try:
            if line[0] != '#':
                line_splitted = line.split('\t')
                num = int(line_splitted[0])
                if line_splitted[3] == 'PUNCT' and buff != [] and doSpaces:
                    buff[-1] += line_splitted[1]
                else:
                    buff.append(line_splitted[1])
        except: # when we have multi-token word
            pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--do-spaces', default=False, action="store_true")
    parser.add_argument('conllfile')

    args = parser.parse_args()
    conllfile = args.conllfile

    with codecs.open(conllfile, 'r', 'utf8') as infile:
        with codecs.getwriter('utf-8')(sys.stdout) as outFile:
            printFile(infile, outFile, args.do_spaces)
