import pytest
import xarray as xr
import numpy as np
from retrievals.arts.atmosphere import Atmosphere
import os


TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data')


def _default_atmosphere():
    # Generate some stupid test data
    p_grid = np.logspace(1e5, 1-2, 100)
    t_field = np.linspace(10, -50, 100)
    z_field = np.linspace(0, 80e3, 100)
    o3_field = np.sin(z_field)

    ds = xr.Dataset(
        coords={'p': p_grid},
        data_vars={
            'z': ('p', z_field),
            't': ('p', t_field),
            'o3': ('p', o3_field),
        }
    )
    ds = ds.expand_dims('lat').expand_dims('lon')
    return Atmosphere.from_dataset(ds)


def test_from_dataset():
    _default_atmosphere()


def test_from_xml_data():
    basename = os.path.join(TEST_DATA_PATH, 'fascod-tropical/tropical.')
    atm = Atmosphere.from_arts_xml(basename)

    assert atm.vmr_field('C2H2').gridnames == ['Pressure', 'Latitude', 'Longitude']
    assert atm.t_field.gridnames == ['Pressure', 'Latitude', 'Longitude']
    assert atm.z_field.gridnames == ['Pressure', 'Latitude', 'Longitude']
    assert atm.t_field.data.shape[1:] == (1, 1)
    assert atm.z_field.data.shape[1:] == (1, 1)


def test_no_files_found():
    basename = os.path.join(TEST_DATA_PATH, 'fascod-tropical/tropical_xxxxxxxx')
    with pytest.raises(FileNotFoundError):
        Atmosphere.from_arts_xml(basename)


def test_fields():
    atm = _default_atmosphere()

    assert atm.t_field.gridnames == ['Pressure', 'Latitude', 'Longitude']
    assert atm.z_field.gridnames == ['Pressure', 'Latitude', 'Longitude']
    assert atm.t_field.data.shape[1:] == (1, 1)
    assert atm.z_field.data.shape[1:] == (1, 1)
    assert atm.vmr_field('O3').gridnames == ['Pressure', 'Latitude', 'Longitude']


def test_setters():
    atm = Atmosphere()
    p_grid = np.logspace(1e5, 1 - 2, 100)
    t_field = np.linspace(10, -50, 100)
    atm.set_t_field(p_grid, t_field)
    assert atm.t_field.data.shape == (len(p_grid), 1, 1)
