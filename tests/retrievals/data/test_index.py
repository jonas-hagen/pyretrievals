import pytest
from retrievals.data import index
import os
from glob import glob
import numpy as np


TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data')


def test_interval_dict():
    ind = index.IntervalDict()
    for i in range(0, 8, 2):
        ind[i, i + 1] = i
    assert list(ind.envelope(2.5, 6.5)) == [2, 4, 6]
    assert list(ind.containers(4.5)) == [4, ]
    # assert list(ind.closest(3.5)) == [2, 4]


def test_nc_index():
    pattern = 'ECMWF_OPER_v1_MAIDO_????????.nc'
    files = glob(os.path.join(TEST_DATA_PATH, pattern))
    ind = index.nc_index(TEST_DATA_PATH, pattern)

    assert len(set(ind.values)) == len(files)
    assert ind['time'].min() == np.datetime64('2018-01-01 00:00:00')
    assert ind['time'].max() == np.datetime64('2018-01-03 18:00:00')


def test_nc_index_update():
    pattern = 'ECMWF_OPER_v1_MAIDO_20180101.nc'
    ind = index.nc_index(TEST_DATA_PATH, pattern)

    assert len(set(ind.values)) == 1

    pattern = 'ECMWF_OPER_v1_MAIDO_????????.nc'
    ind = index.nc_index(TEST_DATA_PATH, pattern, index=ind)

    assert len(set(ind.values)) == 3
    assert ind['time'].min() == np.datetime64('2018-01-01 00:00:00')
    assert ind['time'].max() == np.datetime64('2018-01-03 18:00:00')
