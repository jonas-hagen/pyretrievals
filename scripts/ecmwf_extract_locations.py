"""
Extract ecmwf data for certain locations from global GRIB1 files.

Usage:
  ecmwf_extract_locations.py extract <locations_file> <output_prefix> <grib_file>...
  ecmwf_extract_locations.py list_params <grib_file>

Options:
  -h --help     Show this screen.
  --version     Show version.
"""
from retrievals.data.ecmwf import grib as ecmwf_grib
from retrievals.data.ecmwf import levels as ecmwf_levels
from glob import glob
import xarray as xr
from docopt import docopt
import json
import logging
import pygrib
import numpy as np
import pandas as pd


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)


ATTRS = {
    'lat': {
        'standard_name': 'latitude',
        'long_name': 'Latitude',
        'units': 'degree_north',
        'axis': 'Y',
    },
    'lon': {
        'standard_name': 'longitude',
        'long_name': 'Longitude',
        'units': 'degree_east',
        'axis': 'X',
    },
    'time': {
        'standard_name': 'time',
        'long_name': 'Time',
    },
    'level': {
        'long_name': 'ECMWF model level'
    },
    'pressure': {
        'long_name': 'Pressure',
        'standard_name': 'air_pressure',
        'units': 'Pa',
        'axis': 'Z',
        'positive': 'down',
    }
}


def find_files(grib_files):
    if len(grib_files) > 1:
        return sorted(grib_files)
    return sorted(glob(grib_files[0]))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def extract(grib_files, locations, output_prefix):
    parameters, dims = get_dims(grib_files[0])
    n_times = len(dims['time'])
    n_levels = len(dims['level'])

    coerced_coords = dict()
    data = dict()
    for loc_name, loc in locations.items():
        # find coerced coords
        lat = find_nearest(dims['lat'], loc['lat'])
        lon = find_nearest(dims['lon'], loc['lon'])
        i_lat = np.where(dims['lat'] == lat)
        i_lon = np.where(dims['lon'] == lon)
        coerced_coords[loc_name] = (lat, lon, i_lat, i_lon)

        # initialize arrays
        data[loc_name] = dict()
        for p in parameters:
            if p in ['Logarithm of surface pressure', 'Geopotential']:
                a = np.ndarray((1, n_times), dtype=np.float32)
            else:
                a = np.ndarray((n_levels, n_times), dtype=np.float32)
            data[loc_name][p] = a

    # Read every file
    for file in grib_files:
        logging.debug('Clear arrays')
        for d in data.values():
            for a in d.values():
                a.fill(np.nan)

        dims['time'] = []

        logging.info('Open ' + file)
        grbs = pygrib.open(file)
        for msg in grbs:
            if msg.analDate not in dims['time']:
                dims['time'].append(msg.analDate)
            i_time = dims['time'].index(msg.analDate)
            i_level = dims['level'].index(msg.level)
            values = msg.values

            # extract data for each location
            for loc_name, ps in data.items():
                lat, lon, i_lat, i_lon = coerced_coords[loc_name]
                v = values[i_lat, i_lon]
                ps[msg.parameterName][i_level, i_time] = v


        # Complie data and store
        for loc_name, xs in data.items():
            lat, lon, _, _ = coerced_coords[loc_name]
            data_vars = dict()
            for p, a in xs.items():
                slug = p.lower().replace(' ', '_')
                if p in ['Logarithm of surface pressure', 'Geopotential']:
                    data_vars[slug] = (('loc', 'time',),
                                       a[0, :][np.newaxis, :],
                                       {'grib_name': p})
                else:
                    data_vars[slug] = (('loc', 'level', 'time'),
                                       a[np.newaxis, :, :],
                                       {'grib_name': p})
            # calculate pressure
            pressure = ecmwf_levels.pressure_levels(xs['Logarithm of surface pressure'])
            data_vars['pressure'] = (('loc', 'level', 'time'),
                                     pressure,
                                     ATTRS['pressure'])

            coords = {
                'time': dims['time'],
                'level': dims['level'],
                'lat': ('loc', [lat], ATTRS['lat']),
                'lon': ('loc', [lon], ATTRS['lon']),
                'loc': ('loc', [loc_name], {'long_name': 'Location identifier'}),
            }
            attrs = {
                'grib_file': file,
            }
            ds = xr.Dataset(data_vars, coords, attrs=attrs)
            ds['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00'

            day_str = pd.Timestamp(ds['time'].values[0]).strftime('%Y%m%d')
            out_fn = output_prefix + loc_name + '_' + day_str + '.nc'
            logging.info('Write ' + out_fn)
            ds.to_netcdf(out_fn, unlimited_dims=['time',])


def get_dims(grib_file):
    ge = ecmwf_grib.GribECMWF(grib_file)

    msg = ge.grbs[1]
    lons = np.linspace(-float(msg['longitudeOfFirstGridPointInDegrees']),
                       float(msg['longitudeOfLastGridPointInDegrees']),
                       int(msg['Ni']))
    lats = np.linspace(float(msg['latitudeOfFirstGridPointInDegrees']),
                       float(msg['latitudeOfLastGridPointInDegrees']),
                       int(msg['Nj']))

    parameters = sorted(ge.index.sel_values('parameterName'))
    dims = {
        'level': sorted(ge.index.sel_values('level')),
        'time': sorted(ge.index.sel_values('analDate')),
        'lat': lats,
        'lon': lons,
    }

    return parameters, dims


def list_params(grib_file):
    logging.info('Create index for ' + str(grib_file))
    ge = ecmwf_grib.GribECMWF(grib_file)
    parameters = ge.index.sel_values('parameterName')
    print()
    for p in parameters:
        n_levels = len(ge.index.sel(parameterName=p)) / 4
        print(p, int(n_levels), 'levels')


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Extract ECMWF 1.0')

    if arguments['extract']:
        with open(arguments['<locations_file>']) as f:
            locations = json.load(f)
        extract(find_files(arguments['<grib_file>']),
                locations,
                arguments['<output_prefix>'])
    elif arguments['list_params']:
        list_params(arguments['<grib_file>'][0])
