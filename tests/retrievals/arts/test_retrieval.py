from retrievals.arts import retrieval as ret
from retrievals.arts.interface import ArtsController, Observation
from retrievals.arts.atmosphere import Atmosphere
from retrievals.arts.sensors import SensorGaussian
from retrievals.arts.retrieval import AbsSpecies, Polyfit
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


def _setup_retrieved_controller():
    ac = _setup_default_controller()

    y0, y1 = ac.y_calc()
    offset = 3
    ac.set_y([y0+offset, y1+offset])

    # Variance of y
    y_vars = 0.01 * np.ones(ac.n_y)

    # O3 covmat and retrieval setup
    sx1 = covmat.covmat_diagonal_sparse(1e-6 * np.ones_like(ac.p_grid))
    rq1 = AbsSpecies('O3', ac.p_grid, np.array([0]), np.array([0]), covmat=sx1, unit='vmr')

    sx2_0 = covmat.covmat_diagonal(np.array([5, 5]))
    sx2_1 = covmat.covmat_diagonal(np.array([2, 2]))
    rq2 = Polyfit(1, [sx2_0, sx2_1])

    ac.define_retrieval([rq1, rq2], y_vars)

    ac.oem(method='gn')
    return ac


def test_sizes_to_slices():
    sizes = [1, 3, 2]
    slices = ret._sizes_to_slices(sizes)
    assert slices == [slice(a, b) for a, b in [(0, 1), (1, 4), (4, 6)]]


def test_sliced_nd_array():
    def m(x, y):
        a = (10 * x + y) * np.ones((x, y))
        return a

    blocks = [[m(1, 1), m(1, 3), m(1, 2)],
              [m(3, 1), m(3, 3), m(3, 2)],
              [m(2, 1), m(2, 3), m(2, 2)]]
    M = np.block(blocks)

    slugs = {'A': (0, 1), 'B': (1, 4), 'C': (4, 6)}
    sna = ret.SlicedArray(M, slugs)

    assert np.all(sna['A'] == 11)
    assert np.all(sna['B'] == 33)
    assert np.all(sna['C'] == 22)
    assert np.all(sna['B', 'A'] == 31)


def test_retrieval_quantity_indices():
    ac = _setup_default_controller()

    y0, y1 = ac.y_calc()
    offset = 3
    ac.set_y([y0+offset, y1+offset])

    # Variance of y
    y_vars = 0.01 * np.ones(ac.n_y)

    # O3 covmat and retrieval setup
    sx1 = covmat.covmat_diagonal_sparse(1e-6 * np.ones_like(ac.p_grid))
    rq1 = AbsSpecies('O3', ac.p_grid, np.array([0]), np.array([0]), covmat=sx1, unit='vmr')

    sx2_0 = covmat.covmat_diagonal(np.array([5, 5]))
    sx2_1 = covmat.covmat_diagonal(np.array([2, 2]))
    rq2 = Polyfit(1, [sx2_0, sx2_1])

    ac.define_retrieval([rq1, rq2], y_vars)

    assert rq1._is_applied
    assert rq1.ws == ac.ws
    assert rq2._is_applied
    assert rq2.ws == ac.ws

    n_p = len(ac.p_grid)
    assert rq1._slice == slice(0, n_p)
    assert rq2._slice == [slice(a, b) for a, b in [(n_p, n_p + 2), (n_p + 2, n_p + 4)]]

    ac.oem(method='gn')
    assert ac.oem_converged

    # Check ozone retrieval
    assert rq1.avkm.shape == (n_p, n_p)
    assert np.all(np.abs(rq1.mr[22:45] - 1) <= 0.2)

    # Check if the baseline has been retrieved
    assert np.all(np.abs(rq2.x[0] - offset) < 0.01)


def test_retrieval_quantity_xarray():
    ac = _setup_retrieved_controller()
    assert ac.oem_converged

    rq1, rq2 = ac.retrieval_quantities
    level2 = ac.get_level2_xarray()
    assert 'y_baseline' in level2
    assert 'o3_offset' in level2
    assert 'observation' in level2

    mr = level2['o3_avkm'].sum(dim='o3_p_avk').values
    assert np.allclose(mr, rq1.mr)
