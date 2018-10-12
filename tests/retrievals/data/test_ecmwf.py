from retrievals.data.ecmwf import store
import os


TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data')


def test_store():
    es = store.ECMWFLocationFileStore(TEST_DATA_PATH, 'ECMWF_OPER_v1_MAIDO_%Y%m%d.nc')
    assert len(es.file_names) == 3

    ds = es.select_time('2018-01-01 12', '2018-01-02 12')
    assert len(ds['time']) == 5
    assert 'pressure' in ds
