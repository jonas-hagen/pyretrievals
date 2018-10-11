Atmospheric Data
================

Dealing with atmospheric data from different sources is difficult.


ECMWF
-----

We download GRIB1 files from the ECMWF `ecaccess` servers and store them locally.
These Files contain global data, and the whole database is quite large.
The following workflow helps to keep things practical:

1. Download huge amounts of global GRIB1 files.
2. Use the script :ref:`script-ecmwf-extract-locations` to extract the data regarding locations of interest.
3. Use the :py:class:`retrievals.data.ecmwf.ECMWFLocationFileStore` to select the data you need.
