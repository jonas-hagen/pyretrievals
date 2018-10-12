from retrievals.data.waccm import store
import pandas as pd
import os


TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data')
WACCM_FILE = os.path.join(TEST_DATA_PATH, 'waccm2_f2000_BERN_first5days.nc')


def test_waccm_location_single_file_store():
    ws = store.WaccmLocationSingleFileStore(WACCM_FILE)

    name, lat, lon = ws.location
    assert name == 'BERN'

    ds = ws.select_time('2018-01-01 12:00', '2018-01-03 12:00')
    assert pd.to_datetime(ds['time'].values[0]) == pd.to_datetime('2018-01-01 12:00')
    assert len(ds['time'].values) == 24 * 2 + 1
    assert 'pressure' in ds


def test_waccm_location_single_file_store_new_year():
    ws = store.WaccmLocationSingleFileStore(WACCM_FILE)

    # around happy new year!
    ds = ws.select_time('2017-12-30 12:00', '2018-01-02 12:00')
    assert pd.to_datetime(ds['time'].values[0]) == pd.to_datetime('2017-12-30 12:00')
    assert pd.to_datetime(ds['time'].values[-1]) == pd.to_datetime('2018-01-02 12:00')
    assert len(ds['time'].values) == 24 * 3 + 1
    assert 'pressure' in ds