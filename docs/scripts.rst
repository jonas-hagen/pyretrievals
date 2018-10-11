Scripts
=======

.. _script-ecmwf-extract-locations:

ecmwf_extract_locations.py
--------------------------

This script is used to extract smaller netCDF files from global GRIB files of ECMWF data.


.. literalinclude:: ../scripts/ecmwf_extract_locations.py
    :language: none
    :start-after: """
    :end-before: """


The locations are identified using a locations file. The attributes `lat` and `lon` are required.


.. code-block:: json

    {
        "BERN": {
            "lat": 46.94790,
            "lon": 7.444600,
            "anything": "anyvalue"
        },
        "NAME": {
            "lat": 0,
            "lon": 0
        }
    }

