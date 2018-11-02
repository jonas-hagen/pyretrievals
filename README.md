# pyretrievals

Building on the great projects [ARTS](http://radiativetransfer.org) and 
[Typhon](http://www.radiativetransfer.org/misc/typhon/doc/index.html), this library
provides some tools I use to run simulations and retrievals for the WIRA-C instrument.

Currently this projects is probably only usefult for very specific applications in ground based microwave measurements, namely the WIRA-C
instrument and similar.
But, possibly with minor adjustments, it might also be useful for other things!

See also the [**Documentation**](http://www.iapmw.unibe.ch/research/projects/pyretrievals/index.html).

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

Note: `pygrib` can be a hassle to install.

## Environment setup

The variables `$ARTS_BUILD_PATH` and `$ARTS_DATA_PATH` should be exported in your shell.
Example:

```bash
ARTS_DATA_PATH=/opt/arts-dev/arts-xml-data/
ARTS_SRC_PATH=/opt/arts-dev/arts/
```

## Examples

The ``examples`` directory contains a ``test_oem.py`` file which is similar to the ARTS cfile
``controlfiles/artscomponents/test_oem.arts``. 
It simulates the ozone line at 110 GHz and retrieves ozone VMR, frequency shift and a polynomial baseline.
