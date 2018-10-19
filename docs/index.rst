.. pyretrievals documentation master file, created by
   sphinx-quickstart on Wed Oct 10 10:17:26 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyretrievals's documentation!
========================================

Introduction
------------

ARTS_ is the Atmospheric Radiative Transfer Simulator. Thanks to the Typhon_ project, we have useful tools for
atmospheric sciences and also a python interface to ARTS itself, via `typhon.arts.workspace`.
Modules in this package build on these projects and create an interface for simple simulations and retrievals.

Currently this is limited to very specific applications in ground based microwave measurements, namely the WIRA-C
instrument.
But -- with none or minor adjustments -- it might also be useful for other things, let me know!

.. _ARTS: http://radiativetransfer.org
.. _Typhon: http://www.radiativetransfer.org/misc/typhon/doc/index.html


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    overview
    examples
    atmospheric_data
    scripts
    reference
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
