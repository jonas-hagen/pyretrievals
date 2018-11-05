import xarray as xr
import pandas as pd
from retrievals.data import dtutils


class WaccmLocationSingleFileStore:
    """
    This data store assumes that there exists a single file for the model year 2000:

    * One location `(lat, lon)` per file.
    * Coordinate `time` is fractional zero-based day-of-year, so going from 0 to 356.
    * Attributes `loc_[name,lat,lon]` exist.
    * The variables `hyam`, `hybm`, `PS` and `P0` are present for pressure calculations.
    * The variable holding the level is called `lev`
    """
    def __init__(self, file, **kwargs):
        """
        :param file: File name.
        :param kwargs: Additional arguments to :py:func:`xarray.open_mfdataset`
        """
        self._file = file
        self._ds = None
        kwargs['decode_times'] = False
        self._open_kwargs = kwargs

    def select_time(self, t1, t2):
        """
        Select all data within time interval `[t1, t2]` (inclusive).

        :param t1: Start time
        :type t1: Anything understood by :py:func:`pandas.to_datetime`
        :param t2: End time
        :type t2: Anything understood by :py:func:`pandas.to_datetime`
        :return: A dataset that has been normalized by :py:meth:`normalize`
        :rtype: xarray.Dataset
        """
        ts1 = pd.to_datetime(t1)
        ts2 = pd.to_datetime(t2)

        if abs(ts2 - ts1) > pd.Timedelta(1, 'Y'):
            # Times are in day-of-year for one year only
            raise ValueError('Time ranges of more than 1 year not supported.')

        # Data is stored with fractional zero-based doy as time index
        doy1 = dtutils.fz_dayofyear(ts1)
        doy2 = dtutils.fz_dayofyear(ts2)

        if ts1.is_leap_year:
            if doy1 >= 60:
                doy1 -= 1
        if ts2.is_leap_year:
            if doy2 >= 61:
                doy2 -= 1

        if doy1 < doy2:
            ds = self.ds.sel(time=slice(doy1, doy2))
            ref_time = dtutils.year_start(ts1)
        else:
            ds1 = self.ds.sel(time=slice(doy1, 366))
            ds2 = self.ds.sel(time=slice(0, doy2))
            ds1 = ds1.where(ds1['time'] < 365, drop=True)
            ds1['time'] -= 365
            ds = xr.concat([ds1, ds2], dim='time')
            ref_time = dtutils.year_start(ts2)

        return self.normalize(ds, ref_time)

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
    def normalize(ds, ref_time=None):
        ds = ds.sortby('time')
        ds = ds.sortby('lev')  # sigma pressure levels

        # Pressure field
        pressure = ds['hyam'] * ds['P0'] + ds['hybm'] * ds['PS']
        ds['pressure'] = (('lev', 'time'), pressure)

        # Set "correct" dates
        if ref_time is not None:
            ref_time = pd.to_datetime(ref_time)
            ds['time'] = (ref_time + pd.to_timedelta(ds['time'].values, unit='D')).round('H')

        return ds

    @property
    def ds(self):
        if self._ds is None:
            self._ds = xr.open_mfdataset(self._file, **self._open_kwargs)
        return self._ds

    @property
    def location(self):
        if 'loc_name' in self.ds.attrs:
            name = self.ds.attrs['loc_name']
            lat = self.ds.attrs['loc_lat']
            lon = self.ds.attrs['loc_lon']
            return name, lat, lon
        else:
            raise KeyError('No location information found in file ' + self.file)

    @property
    def file(self):
        return self._file
