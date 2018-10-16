Atmospheric data
================

.. automodule:: retrievals.data

.. contents::
    :local:


General utils
-------------

.. py:module:: retrievals.data

.. autofunction:: date_glob

.. autofunction:: nc_index

.. autofunction:: interpolate

.. autofunction:: p_interpolate


ECMWF Specific
--------------

.. py:module:: retrievals.data.ecmwf
.. automodule:: retrievals.data.ecmwf

.. py:data:: hybrid_level

    :py:class:`xarray.Dataset` with all `ECMWF hybrid level definition`_ parameters for all levels.

.. autoclass:: ECMWFLocationFileStore
    :members:

.. _ECMWF hybrid level definition: https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels


WACCM Specific
--------------

.. py:module:: retrievals.data.waccm
.. automodule:: retrievals.data.waccm

.. autoclass:: WaccmLocationSingleFileStore
    :members:


NASA Earthdata
--------------

.. py:module:: retrievals.data.earthdata
.. automodule:: retrievals.data.earthdata

.. autofunction:: open_dataset

.. autofunction:: download

.. autofunction:: load_credentials

.. autofunction:: setup_session