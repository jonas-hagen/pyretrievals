# pyretrievals

Building on the great projects [ARTS](http://radiativetransfer.org) and 
[Typhon](http://www.radiativetransfer.org/misc/typhon/doc/index.html), this library
provides some tools I use to run simulations and retrievals for the WIRA-C interface.

Currently this projects is probably only usefult for very specific applications in ground based microwave measurements, namely the WIRA-C
instrument and similar.
But, possibly with minor adjustments, it might also be useful for other things!

## Install

This package requires Python 3.6 and a recent version of [ARTS](http://radiativetransfer.org).

Create an environment using your favourite tool (conda, virtualenv, ...) and activate it. Then first install the
developement version of [Typhon](http://www.radiativetransfer.org/misc/typhon/doc/index.html) 
and then this package:

```bash
$ git clone <this repo>
$ cd pyretrievals
$ pip install git+https://github.com/atmtools/typhon.git
$ pip install -e .
```

To setup the full environment for testing and development, run:

```bash
pip install -r requirements.txt
```

## Environment setup

Copy the ``dot-env-example`` file to ``.env`` and edit the values therein.
After that run ``source load_env.sh`` to export all the variables into your shell.
If you do not export the values, make sure to run all scripts and tests from the root directory
of the project, where the ``.env`` file is located.

## Compiling ARTS

Compiling [ARTS](http://radiativetransfer.org) needs the following packages installed on a fresh
Ubuntu system:

* ``subversion``
* ``build-essential``
* ``cmake``
* ``zlib1g``
* ``zlib1g-dev``
* ``libblas3``
* ``libblas2-dev``
* ``liblapack``
* ``liblapack-dev``

## Examples

The ``examples`` directory contains a ``test_oem.py`` file which is similar to the ARTS cfile
``controlfiles/artscomponents/test_oem.arts``. 
It simulates the ozone line at 110 GHz and retrieves ozone VMR, frequency shift and a polynomial baseline.

