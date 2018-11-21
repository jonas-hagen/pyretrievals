from retrievals.data import dtutils
from retrievals.data.ecmwf import levels
import pandas as pd
import xarray as xr


class ECMWFLocationFileStore:
    """
    This data store assumes that the files are organised in the following way:

    * One day per file
    * One location `(lat, lon)` per file.
    * All data is along a `loc` coordinate and `lat`, `lon` are along this coordinate.
    * The variable holding the logarithm of surface pressure is `logarithm_of_surface_pressure`
    * The variable holding the level is called `level`

    """
    def __init__(self, path, fmt):
        """
        Build a store given the path and format.
        If the files are organized as `/path/to/folder/2018/ecmwf_2018-01-01.nc`, one can build the store with:

        >>> es = ECMWFLocationFileStore('/path/to/folder', '%Y/ecmwf_%Y-%m-%d.nc')

        and then ask for desired data:

        >>> es.select_time('2018-01-01 12:30', '2018-01-02 16:45')

        :param path: The base path to the files.
        :param fmt: The format string for the file names as used by :py:meth:`datetime.datetime.strftime`
        """
        self._path = path
        self._fmt = fmt
        self._files = dict(dtutils.date_glob(path, fmt))

    def select_time(self, t1, t2, **kwargs):
        """
        Select all data within time interval `(t1, t2)`

        :param t1: Start time
        :type t1: Anything understood by :py:func:`pandas.to_datetime`
        :param t2: End time
        :type t2: Anything understood by :py:func:`pandas.to_datetime`
        :param kwargs: Additional arguments to :py:func:`xarray.open_mfdataset`
        :return: A dataset that has been normalized by :py:meth:`normalize`
        :rtype: xarray.Dataset
        """
        ts1 = pd.to_datetime(t1)
        ts2 = pd.to_datetime(t2)
        days = pd.date_range(ts1.floor('D'), ts2.floor('D'), freq='D')
        try:
            files = sorted(self._files[d] for d in days)
        except KeyError as e:
            raise KeyError('No ECMWF data found for {}'.format(e.args[0])) from e

        ds = xr.open_mfdataset(files, **kwargs)
        ds = ds.sel(time=slice(ts1, ts2))
        return self.normalize(ds)

    def select_hours(self, t1, t2, hour1, hour2):
        """
        Select all data for certain hours within time interval `[t1, t2]` (inclusive).

        :param t1: Start time
        :type t1: Anything understood by :py:func:`pandas.to_datetime`
        :param t2: End time
        :type t2: Anything understood by :py:func:`pandas.to_datetime`
        :param int hour1: First hour
        :param int hour2: Last hour (might be smaller than `hour1` if range spans midnight)
        :return: A dataset that has been normalized by :py:meth:`normalize`
        :rtype: xarray.Dataset
        """
        ds = self.select_time(t1, t2)
        rel_hours = (ds['time.hour'] - hour1) % 24
        rel_hour2 = (hour2 - hour1) % 24

        ds = ds.where(rel_hours <= rel_hour2, drop=True)
        return ds


    @staticmethod
    def normalize(ds):
        """
        Normalize an ECMWF dataset in the following way:

        * Strip the (single) location
        * Add a pressure field
        * Sort by increasing altitude (decreasing level)
        * Sort by time

        """
        # We only have one location
        ds = ds.isel(loc=0)

        # Add Pressure
        a = levels.hybrid_level['a']
        b = levels.hybrid_level['b']
        sp = xr.ufuncs.exp(ds['logarithm_of_surface_pressure'])
        ds['pressure'] = a + b * sp

        # Sort it from surface to top of atmosphere
        ds = ds.sortby('level', ascending=False)
        ds = ds.sortby('time')

        return ds

    @property
    def path(self):
        return self._path

    @property
    def fmt(self):
        return self._fmt

    @property
    def file_index(self):
        return self._files

    @property
    def file_names(self):
        return sorted(set(self._files.values()))

    @property
    def location(self):
        ds = xr.open_dataset(self.file_names[0])
        name = ds['loc'].values[0]
        lat = ds['lat'].values[0]
        lon = ds['lon'].values[0]
        return name, lat, lon
