Atmospheric data
================

.. automodule:: retrievals.data

.. contents::
    :local:


General utils
-------------

.. autofunction:: date_glob

.. autofunction:: nc_index

.. autofunction:: interpolate

.. autofunction:: p_interpolate


ECMWF Specific
--------------

.. automodule:: retrievals.data.ecmwf

.. py:data:: hybrid_level

    :py:class:`xarray.Dataset` with all `ECMWF hybrid level definition`_ parameters for all levels.

.. autoclass:: ECMWFLocationFileStore
    :members:

.. _ECMWF hybrid level definition: https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels


WACCM Specific
--------------

.. automodule:: retrievals.data.waccm

.. autoclass:: WaccmLocationSingleFileStore
    :members:


NASA Earthdata
--------------

.. automodule:: retrievals.data.earthdata

.. autofunction:: open_dataset

.. autofunction:: download

.. autofunction:: load_credentials

.. autofunction:: setup_session