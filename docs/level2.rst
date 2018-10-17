Level2 data processing
======================

.. automodule:: retrievals.level2


Summary
-------

Averaging kernels
'''''''''''''''''

The rows of an Averaging Kernel Matrix `avkm` are called Averaging Kernels (`avk`'s).
Most metrics regarding AVKMs need an associated coordinate `x`, that is usually the altitude coordinate in meters.

.. autosummary::

    avkm_mr
    avkm_fwhm
    avkm_peak
    avkm_offset


Reference
---------

.. autofunction:: avkm_mr
.. autofunction:: avkm_fwhm
.. autofunction:: avkm_peak
.. autofunction:: avkm_offset

