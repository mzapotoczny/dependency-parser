# Fully neural dependency parser

The reference implementation for the papers

Read, Tag, and Parse All at Once, or Fully-neural Dependency Parsing
_Jan Chorowski, Michal Zapotoczny, Pawel Rychlikowski_
([arxiv draft](https://arxiv.org/abs/1609.03441))

and

On Multilingual Training of Neural Dependency Parsers
_Michal Zapotoczny, Pawel Rychlikowski, Jan Chorowski_
([arxiv draft](http://arxiv.org/abs/1705.10209), accepted into the TSD 2017).

### How to use

- install all the dependencies (see the list below)
- set your environment variables by calling `source env.sh`
- download [pre-trained models](http://goo.gl/H6UQdo)
- run one of the models e.g. `python bin/parse.py universal/baseline/universal_pl.zip` will
run an interactive parser for polish model trained on [Universal Dependencies](http://universaldependencies.org/)
v1.3 data.

For more details please proceed to [`exp/dependency`](exp/dependency/README.md).
### Online demo

An online demo is available at [http://zapotoczny.pl/parser](http://zapotoczny.pl/parser)

### Dependencies

- Python packages: pykwalify, toposort, pyyaml, numpy, pandas, progressbar, picklable-itertools, segtok

### License

MIT

This code is based on heavily modified [End-to-End Attention-Based Large Vocabulary Speech Recognition](https://github.com/rizar/attention-lvcsr)
