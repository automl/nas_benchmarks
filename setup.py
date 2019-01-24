import os
import sys
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))

requires = [
    'h5py',
    'numpy',
    'ConfigSpace'
    ]

setup(name='profet',
      version='0.0.1',
      description='tabular benchmarks for feed forward neural networks',
      author='Aaron Klein',
      author_email='kleinaa@cs.uni-freiburg.de',
      keywords='hyperparameter optimization',
      packages=find_packages(),
      install_requires=requires)
