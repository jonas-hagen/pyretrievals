from retrievals.arts.interface import ArtsController, Observation
from retrievals.arts.atmosphere import Atmosphere
from retrievals.arts.sensors import SensorGaussian
from retrievals.arts.retrieval import AbsSpecies
from retrievals import covmat

import os
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data')


def _setup_default_controller():
    """Provide an Arts controller with a complete Workspace and two retrieval quantities."""
    f0 = 142.17504e9

    ac = ArtsController()
    ac.setup(atmosphere_dim=3)
    ac.set_spectroscopy_from_file(
        os.path.join(TEST_DATA_PATH, 'Perrin_O3_142.xml'),
        ['O3', ],
    )
    ac.set_grids(
        f_grid=np.linspace(-500e6, 500e6, 300) + f0,
        p_grid=np.logspace(5, -1, 100),
        lat_grid=np.linspace(-4, 4, 5),
        lon_grid=np.linspace(-4, 4, 5)
    )
    ac.set_surface(1e3)

    basename = os.path.join(TEST_DATA_PATH, 'fascod-tropical/tropical.')
    atm = Atmosphere.from_arts_xml(basename)
    ac.set_atmosphere(atm)

    f_backend = np.linspace(-400e6, 400e6, 600) + f0
    ac.set_sensor(SensorGaussian(f_backend, np.array([800e6/600, ])))

    ac.set_observations([Observation(time=0, lat=0, lon=0, alt=12e3, za=90 - 22, aa=azimuth)
                         for azimuth in [90, -90]])
    ac.checked_calc()

    return ac


def test_y_calc():
    # Integration test
    ac = _setup_default_controller()
    y_east, y_west = ac.y_calc()

    assert len(y_east) == 600
    assert np.abs(y_east[0] - 1.5) < 1
    assert np.abs(y_east[300] - 35) < 1


def test_retrieval():
    """Perform a simple O3 retrieval."""
    ac = _setup_default_controller()
    ac.checked_calc()
    ac.y_calc()

    # Variance of y
    y_vars = 0.01 * np.ones(ac.n_y)

    # O3 covmat and retrieval setup
    sx = covmat.covmat_diagonal_sparse(1e-6 * np.ones_like(ac.p_grid))
    rq = AbsSpecies('O3', ac.p_grid, np.array([0]), np.array([0]), covmat=sx, unit='vmr')
    ac.define_retrieval([rq], y_vars)

    # Offset VMR fields a bit
    x_true = np.copy(ac.ws.vmr_field.value[0, :, 2, 0])
    ac.ws.Tensor4AddScalar(ac.ws.vmr_field, ac.ws.vmr_field, 0.5e-6)
    x_a = np.copy(ac.ws.vmr_field.value[0, :, 2, 0])

    ac.oem(method='gn')
    assert ac.oem_converged
    ac.ws.x2artsAtmAndSurf()
    x_hat = np.copy(ac.ws.vmr_field.value[0, :, 2, 0])

    # Compare values in altitudes between approx. 19 to 47 km
    poi = slice(22, 45)
    assert np.allclose(x_true[poi], x_hat[poi], atol=0.1e-6)
    assert np.allclose(x_a[poi]-0.5e-6, x_hat[poi], atol=0.1e-6)

    assert len(ac.oem_diagnostics) == 5

    # Check to_netcdf
    ds = ac.get_level2_xarray()
    assert 'uuid' in ds.attrs
    assert 'arts_version' in ds.attrs


def test_version():
    ac = ArtsController()
    assert ac.arts_version.startswith('arts-')
