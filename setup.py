#!/usr/bin/env python

from distutils.core import setup

setup(name='retrievals',
      version='0.1',
      description='Retrievals in python for the wind radiometers.',
      author='Jonas Hagen',
      author_email='jonas.hagen@iap.unibe.ch',
      url='',
      packages=['retrievals', ],
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
