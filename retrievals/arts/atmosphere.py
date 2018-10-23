from glob import glob
import os
import re
from typing import Dict

import numpy as np
import xarray as xr

from typhon.arts import xml
from typhon.arts.griddedfield import GriddedField3

# Default names of variables
P_NAME = 'p'  # Pressure
Z_NAME = 'z'  # Geometric altitude
T_NAME = 't'  # Temperature
U_NAME = 'u'  # Zonal wind
V_NAME = 'v'  # Meridional Wind
W_NAME = 'w'  # Vertical wind
LAT_NAME = 'lat'
LON_NAME = 'lon'

DEFAULT_TO_ARTS = {
    'p': 'Pressure',
    'lat': 'Latitude',
    'lon': 'Longitude',
}

ARTS_TO_DEFAULT = {v: k for k, v in DEFAULT_TO_ARTS.items()}


def z2p_simple(z):
    return 10 ** (5 - z / 16e3)


def p2z_simple(p):
    return 16e3 * (5 - np.log10(p))


class Atmosphere:
    """
    Represents the atmospheric state, including absorbing species, wind and temperature and altitude fields.
    """

    def __init__(self, data: Dict[str, xr.DataArray]= None):
        if data is None:
            self.data = {}
        else:
            self.data = {name: self.check_and_normalize_field(name, da) for name, da in data.items()}

    def __getitem__(self, item):
        """Return the raw data for a quantity."""
        return self.data[item]

    def __setitem__(self, key, value):
        """Set the data for a quantity."""
        self.data[key] = self.check_and_normalize_field(key, value)

    def __contains__(self, item):
        return item in self.data

    @classmethod
    def check_and_normalize_field(cls, name, da: xr.DataArray):
        if not set(da.dims) == {P_NAME, LAT_NAME, LON_NAME}:
            # This is more restrictive than needed
            raise ValueError('Dimensions of `{}` must be p, lat, lon. Got: {}'.format(name, ', '.join(da.dims)))
        da = da.transpose(P_NAME, LAT_NAME, LON_NAME)
        da = da.sortby(P_NAME, ascending=False)
        da.name = name
        return da

    @property
    def t_field(self):
        """
        The temperature field.

        :type: typhon.arts.griddedfield.GriddedField3
        """
        return self._gf_from_xarray(self.data[T_NAME].rename(DEFAULT_TO_ARTS))

    @property
    def z_field(self):
        """
        The geometric altitude field.

        :type: typhon.arts.griddedfield.GriddedField3
        """
        return self._gf_from_xarray(self.data[Z_NAME].rename(DEFAULT_TO_ARTS))

    def wind_field(self, component):
        """
        Gives a specific component of the wind field.

        :param component: One of `u`, `v`, `w`
        :rtype: typhon.arts.griddedfield.GriddedField3
        """
        component = str.lower(component)
        component_to_name = {'u': U_NAME, 'v': V_NAME, 'w': W_NAME}
        var_name = component_to_name[component]
        if var_name not in self.data:
            raise KeyError('No data for wind component ' + var_name)
        return self._gf_from_xarray(self.data[var_name].rename(DEFAULT_TO_ARTS))

    def vmr_field(self, species):
        """
        Gives the VMR field for a species.

        :param species: Species identifier, e.g. o3 (lowercase O3, for ozone)
        :rtype: typhon.arts.griddedfield.GriddedField3
        """
        species = str.lower(species)
        if species not in self.data:
            raise KeyError('No data for absorption species ' + species)
        da = self.data[species].rename(DEFAULT_TO_ARTS)
        return self._gf_from_xarray(da)

    def _set_field(self, name, p, x, lat=None, lon=None):
        lat = np.array([0]) if lat is None else lat
        lon = np.array([0]) if lon is None else lon
        name = str.lower(name)

        if p.ndim != 1:
            raise ValueError('Pressure must be 1d.')
        if len(lat) != 1 or len(lon) != 1:
            raise NotImplementedError('Multi dimensional fields not supported.')
        if x.shape != p.shape:
            raise ValueError('`p` and `x` must have same shape.')

        data = x[:, np.newaxis, np.newaxis]
        da = xr.DataArray(data, coords=[p, lat, lon], dims=[P_NAME, LAT_NAME, LON_NAME], name=name)
        self.data[name] = self.check_and_normalize_field(name, da)

    def set_t_field(self, p, x, lat=None, lon=None):
        """Set the Temperature field. Only 1d supported."""
        self._set_field(T_NAME, p, x, lat, lon)

    def set_z_field(self, p, x, lat=None, lon=None):
        """Set the Altitude field. Only 1d supported."""
        self._set_field(Z_NAME, p, x, lat, lon)

    def set_wind_field(self, component, p, x, lat=None, lon=None):
        """Set the wind field for a `component`. Only 1d supported."""
        component_to_name = {'u': U_NAME, 'v': V_NAME, 'w': W_NAME}
        self._set_field(component_to_name[component], p, x, lat, lon)

    def set_vmr_field(self, species, p, x, lat=None, lon=None):
        """Set the VMR field for a `species`. Only 1d supported."""
        self._set_field(species, p, x, lat, lon)

    @classmethod
    def from_dataset(cls, ds):
        """
        Create atmosphere from a :py:class:`xarray.Dataset`.

        :param ds: The dataset must have the coordinates `pressure`, `lat`, `lon` and can have
                   variables `t`, `z` and absorbing species like `o3` (all lowercase).
        :type ds: xarray.Dataset
        :rtype: Atmosphere
        """
        ds = ds.copy()
        data = {name: da for name, da in ds.data_vars.items()}
        return cls(data)

    @classmethod
    def from_arts_xml(cls, prefix, ext='.xml'):
        """
        Read from ARTS XML data, all files matching 'prefix*.ext' are read.

        :param str prefix: Prefix (`/path/something`)
        :param str ext: File extension, mostly '.xml' or '.xml.gz. Default: '.xml'
        :rtype: Atmosphere
        """
        path = os.path.dirname(prefix)
        basename = os.path.basename(prefix)
        pattern = re.escape(basename) + '(.*)' + re.escape(ext)
        p = re.compile(pattern)
        files = {p.match(fn).group(1): fn for fn in os.listdir(path) if p.match(fn)}

        if not files:
            raise FileNotFoundError('No files found for matching ' + prefix + '*' + ext)

        def load(fn):
            da = xml.load(os.path.join(path, fn)).to_xarray()
            da = da.rename(ARTS_TO_DEFAULT)
            return da

        data = {str.lower(var_name): load(fn) for var_name, fn in files.items()}
        return cls(data)

    @staticmethod
    def _gf_from_xarray(da):
        gf = GriddedField3.from_xarray(da)
        return gf
