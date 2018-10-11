import pygrib
import numpy as np
from collections import defaultdict
from collections import namedtuple
from collections import Iterable
import xarray as xr


def grb_msg_to_xr(message, has_levels=True):
    """
    Convert a single grib message to xarray.

    :param message:
    :type message: pygrib message
    :param has_levels: If True, add a level coordinate.
    :rtype: xarray.DataArray
    """
    lons = np.linspace(-float(message['longitudeOfFirstGridPointInDegrees']),
                       float(message['longitudeOfLastGridPointInDegrees']),
                       int(message['Ni']))
    lats = np.linspace(float(message['latitudeOfFirstGridPointInDegrees']),
                       float(message['latitudeOfLastGridPointInDegrees']),
                       int(message['Nj']))

    coords = {
        'time': message.analDate,
        'lat': lats,
        'lon': lons,
    }

    if has_levels:
        coords['level'] = message.level

    # set up data variables
    values = message.values  # values in lat, lon
    attrs = dict()
    attrs['units'] = message.units
    attrs['standard_name'] = message.cfName
    attrs['long_name'] = message.name
    attrs['parameter_id'] = message.paramId

    da = xr.DataArray(data=values,
                      dims=('lat', 'lon'),
                      coords=coords,
                      name=message.name.lower().replace(' ', '_'),
                      attrs=attrs)

    # Expand dimensions
    if 'level' in coords:
        da = da.expand_dims('level', 2)
    da = da.expand_dims('time', len(coords) - 1)

    # Attributes
    da['lat'].attrs['standard_name'] = 'latitude'
    da['lat'].attrs['long_name'] = 'Latitude'
    da['lat'].attrs['units'] = 'degrees_north'
    da['lat'].attrs['axis'] = 'Y'

    da['lon'].attrs['standard_name'] = 'longitude'
    da['lon'].attrs['long_name'] = 'Longitude'
    da['lon'].attrs['units'] = 'degrees_east'
    da['lon'].attrs['axis'] = 'X'

    da['time'].attrs['standard_name'] = 'time'
    da['time'].attrs['long_name'] = 'Time'

    if 'level' in coords:
        da['level'].attrs['long_name'] = 'ECMWF model level'

    return da


class GribECMWF:
    def __init__(self, filename):
        self.filename = filename
        self.grbs = pygrib.open(filename)
        self.index = GribIndex(self.grbs, ['parameterName', 'level', 'analDate'])

    def get_dataset(self, parameter, level, time):
        message_numbers = self.index.sel(parameterName=parameter, level=level, analDate=time)

        if len(message_numbers) > 1:
            raise ValueError('Got multiple grib messages, but expected one.')
        elif len(message_numbers) == 0:
            raise KeyError('No messages found.')
        message_number = message_numbers[0]

        # read message
        message = self.grbs[message_number]
        has_levels = len(self.index.sel(parameterName=parameter, analDate=time)) > 1
        da = grb_msg_to_xr(message, has_levels)

        return da

    @staticmethod
    def _slugify(name):
        return name.lower().replace(' ', '_')


class GribIndex:
    def __init__(self, grbs, keys):
        self.grbs = grbs
        self.keys = set(keys)
        self.Index = namedtuple('Index', keys)

        # Create index
        self.index = None
        self.values = None
        self.create_index()

    def sel(self, **kwargs):
        """
        Get the message numbers by index.
        """
        for key in kwargs.keys():
            if key not in self.keys:
                raise KeyError(key + ' is not a valid indexer.')

        # format the selectors
        valid_values = dict()
        for key, selector in kwargs.items():
            valid_values[key] = self.sel_values(key, selector)

        message_numbers = list()
        for index, message_number in self.index.items():
            matched = True
            for key, selector in valid_values.items():
                matched = matched and getattr(index, key) in selector
            if matched:
                message_numbers.append(message_number)

        return message_numbers

    def sel_values(self, key, selector=None):
        if selector is None:
            # Retrun all values
            return self.values[key]

        if isinstance(selector, str):
            selector = [selector]
        elif not isinstance(selector, Iterable):
            selector = [selector]

        values = self.values[key]
        return [v for v in values if v in selector]

    def __getitem__(self, item):
        return self.values[item]

    def create_index(self):
        """
        Create an index of all messages by keys.
        """
        table = dict()
        values = defaultdict(set)
        self.grbs.seek(0)
        for msg in self.grbs:
            # Check if all keys are present
            present_keys = [msg.has_key(key) for key in self.keys]
            if False in present_keys:
                continue  # next message

            # Extract index values
            index_values = {key: getattr(msg, key) for key in self.keys}
            index = self.Index(**index_values)
            for key, value in index_values.items():
                values[key].add(value)
            table[index] = msg.messagenumber

        self.index = table
        self.values = values

