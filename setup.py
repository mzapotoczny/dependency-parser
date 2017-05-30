from os import path
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

setup(
    name='neuro-parser',
    description='Fully Neural Dependency Parser',
    url='https://github.com/mzapotoczny/neuro-parser',
    author='Michal Zapotoczny',
    license='MIT',
    packages=find_packages(exclude=['examples', 'docs', 'tests']),
    zip_safe=False,
    install_requires=['numpy', 'pykwalify', 'pyyaml', 'progressbar',
                      'picklable-itertools', 'pandas', 'segtok']
)
