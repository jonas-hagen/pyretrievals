#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='retrievals',
      version='0.2',
      description='Retrievals in python for the wind radiometers.',
      author='Jonas Hagen',
      author_email='jonas.hagen@iap.unibe.ch',
      url='',
      packages=find_packages(include='retrievals*'),
      install_requires=[
          'numpy',
          'numba',
          'scipy',
          'pandas',
          'xarray',
          'dask',
          'toolz',
          'python-dotenv',
          'docopt',
          'typhon',
      ],
      )
