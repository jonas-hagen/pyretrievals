from retrievals.data import dtutils
from datetime import datetime
import pandas as pd
import os


TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data')


def test_date_glob():
    files = dict(dtutils.date_glob(TEST_DATA_PATH, 'ECMWF_OPER_v1_MAIDO_%Y%m%d.nc'))
    assert len(files) == 3
    assert files[datetime(2018, 1, 3)] == os.path.join(TEST_DATA_PATH, 'ECMWF_OPER_v1_MAIDO_20180103.nc')
    assert files[pd.Timestamp(datetime(2018, 1, 3))] == os.path.join(TEST_DATA_PATH, 'ECMWF_OPER_v1_MAIDO_20180103.nc')
