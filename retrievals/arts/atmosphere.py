from glob import glob
import os
import re

import numpy as np
import xarray as xr

from typhon.arts import xml
from typhon.arts.griddedfield import GriddedField, GriddedField3

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


def z2p_simple(z):
    return 10 ** (5 - z / 16e3)


def p2z_simple(p):
    return 16e3 * (5 - np.log10(p))


class Atmosphere:
    """
    Represents the atmospheric state, including absorbing species, wind and temperature and altitude fields.
    """

    def __init__(self, data):
        self.data = self.check_and_normalize_data(data)

    @classmethod
    def check_and_normalize_data(cls, ds):
        if not set(ds.dims) == {P_NAME, LAT_NAME, LON_NAME}:
            # This is more restrictive than needed
            raise ValueError('Dimensions of dataset must be p, lat, lon. Got: ' + ', '.join(ds.dims))
        if T_NAME not in ds:
            raise ValueError(f'Temperature field "{T_NAME}" is missing. Got: ' + ', '.join(ds.data_vars))
        if Z_NAME not in ds:
            raise ValueError(f'Altitude field "{T_NAME}" is missing. Got: ' + ', '.join(ds.data_vars))

        ds = ds.transpose(P_NAME, LAT_NAME, LON_NAME)
        return ds

    @property
    def lat(self):
        """
        The latitude grid.

        :type: numpy.ndarray
        """
        return self.data[LAT_NAME].values

    @property
    def lon(self):
        """
        The longitude grid.

        :type: numpy.ndarray
        """
        return self.data[LON_NAME].values

    @property
    def dimensions(self):
        """Dimensionality is 1, if only one (lat, lon) coordinate present, 3 otherwise."""
        if len(self.lat) == 1 and len(self.lon) == 1:
            return 1
        if len(self.lat) > 1 or len(self.lon) > 1:
            return 3

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
        return self._gf_from_xarray(self.data[species].rename(DEFAULT_TO_ARTS))

    @classmethod
    def from_dataset(cls, ds):
        """
        Create atmosphere from a :py:class:`xarray.Dataset`.

        :param ds: The dataset must have the variables `t`, `z`, `pressure`, `lat`, `lon` and can have
                   absorbing species like `o3` (all lowercase).
        :type ds: xarray.Dataset
        :rtype: Atmosphere
        """
        ds = ds.copy()
        return cls(ds)

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

        dss = []
        for var_name, fn in files.items():
            ds = xml.load(os.path.join(path, fn)).to_xarray()
            ds.name = str.lower(var_name)
            dss.append(ds)
        ds = xr.merge(dss).rename({v: k for k, v in DEFAULT_TO_ARTS.items()})
        return cls(ds)

    @staticmethod
    def _gf_from_xarray(da):
        gf = GriddedField.from_xarray(da)
        gf.__class__ = GriddedField3
        return gf
