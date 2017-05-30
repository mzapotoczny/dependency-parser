#!/bin/bash

set -x 

########## Konfiguracja

MP="java -jar maltparser-1.8.1/maltparser-1.8.1.jar"
ME="java -jar malteval/dist-20141005/lib/MaltEval.jar"
UD_PATH=universal-dependencies-1.2/UD_Polish


##########
# Kroki instalacyjne:
# 1. sciagnalem maltparser z http://maltparser.org/dist/maltparser-1.8.1.tar.gz i wypakowalem
# 2. sciagnalem i wypakowalem malteval z http://www.maltparser.org/malteval.html https://doc-0c-4g-docs.googleusercontent.com/docs/securesc/7nusdasqvnn4ppaorvdrdgj5tghsp00v/q9tn213ra2go5vvr679nr3i8q9085v64/1461571200000/12546850436938328080/10585014011597262938/0B1KaZVnBJE8_QnhqNE52T2FZWVE?e=download&nonce=dml9anvk22ecs&user=10585014011597262938&hash=cfdpmcq19a2l50ilh8khusd0t2pvkvgj
# 3. sanity check: `java -jar maltparser-1.8.1/maltparser-1.8.1.jar` i wypakowalem

##########
# Preprocessing

function eval_parser {
    for f in pl-ud-test.conllu pl-ud-dev.conllu #pl-ud-train.conllu 
    do 
	$MP -c $1 -m parse -i pl_data/$f -o parsed/${1}-$f
	echo "${1}-$f" >> RESULTS
	$ME --Metric 'LAS;UAS' -g pl_data/$f -s parsed/${1}-$f | tee -a RESULTS
    done
}

# Konwersja danych - usuwamy komentarze
mkdir -p pl_data
for f in pl-ud-train.conllu pl-ud-test.conllu pl-ud-dev.conllu
do 
    cat $UD_PATH/$f | grep -v '#' | sed -e 's/root/ROOT/g' > pl_data/$f
done

mkdir -p parsed

##########
# Modele

# model 00 - wszystko na defaultach

$MP -c 00-defaults -m learn -i pl_data/pl-ud-train.conllu 
eval_parser 00-defaults

# model 01 - nivre standard arc parser

cat > 01-nivrestd.xml <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<experiment>
        <optioncontainer>
                <optiongroup groupname="config">
                        <option name="name" value="01-nivrestd"/>
                        <option name="flowchart" value="learn"/>
                </optiongroup>
                <optiongroup groupname="singlemalt">
                        <option name="parsing_algorithm" value="nivrestandard"/>
                </optiongroup>
                <optiongroup groupname="input">
                        <option name="infile" value="pl_data/pl-ud-train.conllu"/>
                </optiongroup>
        </optioncontainer>
</experiment>
EOF

$MP -f 01-nivrestd.xml
eval_parser 01-nivrestd

# model 02 - cechy jak u Wroblewskiej

cat > 02-wroblewska-feats-baseline.xml <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<featuremodels>
	<featuremodel name="nivreeager">
		<feature>InputColumn(FORM, Stack[0])</feature>
		<feature>InputColumn(POSTAG, Stack[0])</feature>
                <feature>InputColumn(POSTAG, Stack[1])</feature>
		<feature>InputColumn(CPOSTAG, Stack[0])</feature>
                <feature>InputColumn(CPOSTAG, Stack[1])</feature>

		<feature>InputColumn(FORM, Input[0])</feature>
		<feature>InputColumn(FORM, Input[1])</feature>
		<feature>InputColumn(POSTAG, Input[0])</feature>
		<feature>InputColumn(POSTAG, Input[1])</feature>
		<feature>InputColumn(POSTAG, Input[2])</feature>
		<feature>InputColumn(POSTAG, Input[3])</feature>
		<feature>InputColumn(CPOSTAG, Input[0])</feature>
		<feature>InputColumn(CPOSTAG, Input[1])</feature>
		<feature>InputColumn(CPOSTAG, Input[2])</feature>
		<feature>InputColumn(CPOSTAG, Input[3])</feature>

		<feature>InputColumn(FORM, head(Stack[0]))</feature>

		<feature>OutputColumn(DEPREL, Stack[0])</feature>
		<feature>OutputColumn(DEPREL, ldep(Stack[0]))</feature>
		<feature>OutputColumn(DEPREL, rdep(Stack[0]))</feature>
		<feature>OutputColumn(DEPREL, ldep(Input[0]))</feature>
	</featuremodel>
</featuremodels>
EOF

cat > 02-wroblewska-baseline.xml <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<experiment>
        <optioncontainer>
                <optiongroup groupname="config">
                        <option name="name" value="02-wroblewska-baseline"/>
                        <option name="flowchart" value="learn"/>
                </optiongroup>
                <optiongroup groupname="singlemalt">
                        <option name="parsing_algorithm" value="nivrestandard"/>
                </optiongroup>
                <optiongroup groupname="guide">
                        <option name="features" value="02-wroblewska-feats-baseline.xml"/>
                </optiongroup>

                <optiongroup groupname="input">
                        <option name="infile" value="pl_data/pl-ud-train.conllu"/>
                </optiongroup>
        </optioncontainer>
</experiment>
EOF

$MP -f 02-wroblewska-baseline.xml
eval_parser 02-wroblewska-baseline

# model 03 - cechy zoptymalizowane jak u Wroblewskiej

cat > 03-wroblewska-feats.xml <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<featuremodels>
	<featuremodel name="nivreeager">
		<feature>InputColumn(FORM, Stack[0])</feature>
		<feature>InputColumn(POSTAG, Stack[0])</feature>
                <feature>InputColumn(POSTAG, Stack[1])</feature>
		<feature>InputColumn(CPOSTAG, Stack[0])</feature>
                <feature>InputColumn(CPOSTAG, Stack[1])</feature>
		<feature>InputColumn(LEMMA, Stack[0])</feature>
		<feature>InputColumn(FEATS, Stack[0])</feature>

		<feature>InputColumn(FORM, Input[0])</feature>
		<feature>InputColumn(FORM, Input[1])</feature>
		<feature>InputColumn(POSTAG, Input[0])</feature>
		<feature>InputColumn(POSTAG, Input[1])</feature>
		<feature>InputColumn(POSTAG, Input[2])</feature>
		<feature>InputColumn(POSTAG, Input[3])</feature>
		<feature>InputColumn(CPOSTAG, Input[0])</feature>
		<feature>InputColumn(CPOSTAG, Input[1])</feature>
		<feature>InputColumn(CPOSTAG, Input[2])</feature>
		<feature>InputColumn(CPOSTAG, Input[3])</feature>
		<feature>InputColumn(LEMMA, Input[0])</feature>
		<feature>InputColumn(LEMMA, Input[1])</feature>
		<feature>InputColumn(FEATS, Input[0])</feature>
		<feature>InputColumn(FEATS, Input[1])</feature>

		<feature>InputColumn(FORM, head(Stack[0]))</feature>
		<feature>InputColumn(LEMMA, head(Stack[0]))</feature>

		<feature>OutputColumn(DEPREL, Stack[0])</feature>
		<feature>OutputColumn(DEPREL, ldep(Stack[0]))</feature>
		<feature>OutputColumn(DEPREL, rdep(Stack[0]))</feature>
		<feature>OutputColumn(DEPREL, ldep(Input[0]))</feature>
	</featuremodel>
</featuremodels>
EOF

cat > 03-wroblewska.xml <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<experiment>
        <optioncontainer>
                <optiongroup groupname="config">
                        <option name="name" value="03-wroblewska"/>
                        <option name="flowchart" value="learn"/>
                </optiongroup>
                <optiongroup groupname="singlemalt">
                        <option name="parsing_algorithm" value="nivrestandard"/>
                </optiongroup>
                <optiongroup groupname="guide">
                        <option name="features" value="03-wroblewska-feats.xml"/>
                </optiongroup>

                <optiongroup groupname="input">
                        <option name="infile" value="pl_data/pl-ud-train.conllu"/>
                </optiongroup>
        </optioncontainer>
</experiment>
EOF

$MP -f 03-wroblewska.xml
eval_parser 03-wroblewska


# model 04 - cechy zoptymalizowane jak u Wroblewskiej, ale ograniczone do tych wzietych przez Zapotoczengo do sieci

cat > 04-zapotoczny-feats.xml <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<featuremodels>
	<featuremodel name="nivreeager">
		<feature>InputColumn(LEMMA, Stack[0])</feature>
		<feature>Merge(InputColumn(CPOSTAG, Stack[0]), InputColumn(FEATS, Stack[0]))</feature>
                <feature>Merge(InputColumn(CPOSTAG, Stack[1]), InputColumn(FEATS, Stack[1]))</feature>

		<feature>InputColumn(LEMMA, Input[0])</feature>
		<feature>InputColumn(LEMMA, Input[1])</feature>
		<feature>Merge(InputColumn(CPOSTAG, Input[0]), InputColumn(FEATS, Input[0]))</feature>
		<feature>Merge(InputColumn(CPOSTAG, Input[1]), InputColumn(FEATS, Input[1]))</feature>
		<feature>Merge(InputColumn(CPOSTAG, Input[2]), InputColumn(FEATS, Input[2]))</feature>
		<feature>Merge(InputColumn(CPOSTAG, Input[3]), InputColumn(FEATS, Input[3]))</feature>

		<feature>InputColumn(LEMMA, head(Stack[0]))</feature>
		<feature>Merge(InputColumn(CPOSTAG, head(Stack[0])), InputColumn(FEATS, head(Stack[0])))</feature>      

		<feature>OutputColumn(DEPREL, Stack[0])</feature>
		<feature>OutputColumn(DEPREL, ldep(Stack[0]))</feature>
		<feature>OutputColumn(DEPREL, rdep(Stack[0]))</feature>
		<feature>OutputColumn(DEPREL, ldep(Input[0]))</feature>
	</featuremodel>
</featuremodels>
EOF

cat > 04-zapotoczny.xml <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<experiment>
        <optioncontainer>
                <optiongroup groupname="config">
                        <option name="name" value="04-zapotoczny"/>
                        <option name="flowchart" value="learn"/>
                </optiongroup>
                <optiongroup groupname="singlemalt">
                        <option name="parsing_algorithm" value="nivrestandard"/>
                </optiongroup>
                <optiongroup groupname="guide">
                        <option name="features" value="04-zapotoczny-feats.xml"/>
                </optiongroup>

                <optiongroup groupname="input">
                        <option name="infile" value="pl_data/pl-ud-train.conllu"/>
                </optiongroup>
        </optioncontainer>
</experiment>
EOF

$MP -f 04-zapotoczny.xml
eval_parser 04-zapotoczny
