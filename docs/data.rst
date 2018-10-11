Atmospheric data
================

.. automodule:: retrievals.data


General utils
-------------

.. py:module:: retrievals.data

.. autofunction:: date_glob


ECMWF Specific
--------------

.. py:module:: retrievals.data.ecmwf
.. automodule:: retrievals.data.ecmwf

.. py:data:: hybrid_level

    :py:class:`xarray.Dataset` with all `ECMWF hybrid level definition`_ parameters for all levels.

.. autoclass:: ECMWFLocationFileStore
    :members:



.. _ECMWF hybrid level definition: https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels