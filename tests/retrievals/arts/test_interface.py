from retrievals.arts.interface import ArtsController, Observation
from retrievals.arts.atmosphere import Atmosphere
from retrievals.arts.sensors import SensorGaussian

import os
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data')


def test_arts_controller():
    # Huge integration test, how could we do better?
    f0 = 142.17504e9

    ac = ArtsController()
    ac.setup(atmosphere_dim=3)
    ac.set_spectroscopy_from_file(
        os.path.join(TEST_DATA_PATH, 'Perrin_O3_142.xml'),
        ['O3', 'H2O-PWR98'],
    )
    ac.set_grids(
        f_grid=np.linspace(-50e6, 50e6, 300) + f0,
        p_grid=np.logspace(5, -1, 100),
        lat_grid=np.linspace(-4, 4, 5),
        lon_grid=np.linspace(-4, 4, 5)
    )
    ac.set_surface(1e3)

    basename = os.path.join(TEST_DATA_PATH, 'fascod-tropical/tropical.')
    atm = Atmosphere.from_arts_xml(basename)
    ac.set_atmosphere(atm)

    f_backend = np.linspace(-40e6, 40e6, 600) + f0
    ac.set_sensor(SensorGaussian(f_backend, np.array([10e3, ])))

    ac.set_observations([Observation(time=0, lat=0, lon=0, alt=12e3, za=90 - 22, aa=azimuth)
                         for azimuth in [90, -90]])

    ac.checked_calc()
    y = ac.y_calc()

    y_east, y_west = ac.y_calc()
    assert len(y_east) == 600
    assert np.abs(y_east[0] - 15) < 1
    assert np.abs(y_east[300] - 38) < 1

