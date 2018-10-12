"""
Extract waccm data for certain locations from global NetCDF files.

Usage:
  waccm_extract_locations.py extract <locations_file> <output_prefix> <grib_file>...

Options:
  -h --help     Show this screen.
  --version     Show version.
"""
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import xarray as xr
import numpy as np
from glob import glob
import logging
from docopt import docopt
import json
from collections import defaultdict
from retrievals.data import ncutils


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)





def find_nearest(x, value):
    x = np.asarray(x)
    idx = (np.abs(x - value)).argmin()
    return x[idx]


def extract(files, locations, out_prefix):
    """
    Extract one grid point from many files and concat to a single dataset.

    :param files: List of files.
    :param locations: Locations of interest.
    :param out_prefix: Prefix under which to save the files.
    :return: List of written files.
    """
    grid_coords = dict()
    files_written = defaultdict(list)
    for i, f in enumerate(files):
        logging.info('Opening file ' + f)
        ds = xr.open_dataset(f, decode_times=False)
        for loc_name, loc in locations.items():
            if loc_name not in grid_coords:
                grid_lat = find_nearest(ds['lat'].values, loc['lat'])
                grid_lon = find_nearest(ds['lon'].values, loc['lon'])
                grid_coords[loc_name] = (grid_lat, grid_lon)
            grid_lat, grid_lon = grid_coords[loc_name]

            loc_ds = ds.sel(lat=grid_lat, lon=grid_lon, drop=False).compute()
            loc_ds['time'].attrs = {
                'standard_name': 'time',
                'units': 'day of year (0 based)',
            }
            loc_ds = loc_ds.drop(['w_stag', 'slat', 'slon'])
            loc_ds.attrs['loc_name'] = loc_name
            loc_ds.attrs['loc_lat'] = grid_lat
            loc_ds.attrs['loc_lon'] = grid_lon

            doy = int(loc_ds['time'].min()) + 1

            out_file = out_prefix + loc_name + '_{:03d}.nc'.format(doy)
            loc_ds.to_netcdf(out_file, unlimited_dims=['time', ])
            logging.info('Written to ' + out_file)
            files_written[loc_name].append(out_file)

    return files_written


def get_pressure(ds):
    """
    Calculate pressure from hybrid levels.
    :param ds: The waccm dataset.
    :return: Pressure as xarray.DataArray
    """
    p = ds['hyam'] * ds['P0'] + ds['hybm'] * ds['PS']
    p.attrs = {
        'standard_name': 'air_pressure',
        'units': 'Pa',
        'axis': 'Z',
        'direction': 'down',
    }
    return p


def find_files(grib_files):
    if len(grib_files) > 1:
        return sorted(grib_files)
    return sorted(glob(grib_files[0]))


def test():
    files = sorted(list(glob('/data/miradara1/archive/waccm2/f2000-t900-lat1.9lon2.5/*.nc')))
    locations = {'MAIDO': {'lat': -21.079816, 'lon': 55.383091}}
    nf = extract(files, locations, '/tmp/tmp_waccm2_f2000_')
    print(nf, 'files written')


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Extract WACCM 1.0')

    if arguments['extract']:
        out_prefix = arguments['<output_prefix>']
        with open(arguments['<locations_file>']) as f:
            locations = json.load(f)

        all_files = find_files(arguments['<grib_file>'])
        if not all_files:
            logging.error('No files found.')
            exit(1)

        logging.info('Extracting from {} files.'.format(len(all_files)))
        loc_files = extract(all_files, locations, out_prefix)

        logging.info('Concatting files')
        for loc_name, files in loc_files.items():
            out_file = out_prefix + loc_name + '.nc'
            ncutils.concat(files, out_file)
            logging.info('Written to ' + out_file)

        logging.info('Done. Have fun with your files ;)')
    else:
        logging.error('Unknown command.')
        exit(1)
