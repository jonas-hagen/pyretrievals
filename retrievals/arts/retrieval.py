"""
Representation of the quantites, that can be added by the WSMs:

retrievalAddAbsSpecies
retrievalAddFreqShift
retrievalAddFreqStretch
retrievalAddConstantVMRAbsSpecies
retrievalAddCatalogParameter
retrievalAddCatalogParameters
retrievalAddMagField
retrievalAddPointingZa
retrievalAddPolyfit
retrievalAddScatSpecies
retrievalAddSinefit
retrievalAddSpecialSpecies
retrievalAddWind
retrievalAddSurfaceQuantity
retrievalAddTemperature
"""

import numpy as np
from scipy import sparse
import xarray as xr
import pandas as pd
from contextlib import contextmanager

from .boilerplate import set_variable_by_xml

from retrievals.arts import boilerplate
from retrievals.arts.atmosphere import p2z_simple, z2p_simple
from retrievals import level2

import warnings


@contextmanager
def retrieval_def(ws):
    """Context manager for the RetrievalDef[Init/Close] context."""
    ws.covmat_block = []
    ws.covmat_inv_block = []

    ws.retrievalDefInit()
    yield ws
    ws.retrievalDefClose()


def _covmat_shape(g1, g2, g3):
    n = len(g1) * len(g2) * len(g3)
    return (n, n)


def _jqs_sizes(jqs):
    """Get the total size of the jacobian quantities."""
    jq_size = list()
    for jq in jqs:
        sz = 1
        for g in jq.grids:
            sz *= len(g)
        jq_size.append(sz)
    return jq_size


def _sizes_to_slices(jq_size):
    """Convert the sizes to start and end indices."""
    jq_start = [0] + list(np.cumsum(jq_size)[:-1])
    jq_slices = [slice(start, start + size) for start, size in zip(jq_start, jq_size)]
    return jq_slices


def _jqs_slices(jqs):
    """Get the start and end indices of jacobian quantities."""
    return _sizes_to_slices(_jqs_sizes(jqs))


class RetrievalQuantity:
    """A generic retrieval quantity."""

    def __init__(self, kind, covmat, **kwargs):
        self._kind = kind
        self._covmat = covmat
        self._args = kwargs

        self._is_applied = False
        self._sequence_id = None  # index inside jacobian quantities of workspace
        self._jacobian_quantity = None
        self._slice = None  # Holds the start and end indices for this quantity
        self._ws = None  # Associated workspace

        self._x = None
        self._avkm = None
        self._eo = None
        self._es = None

    def apply(self, ws):
        """
        Add this retrieval quantity to a Workspace and extract the indices from the jacobian quantities in the
        workspace. Calls `self.apply_retrieval()` and `self.apply_covmat(). Subclasses should override these methods
        instead of just `self.apply()`.
        """
        if self._is_applied:
            raise Exception('Same retrieval quantity cannot be added twice to the same workspace.')

        self._ws = ws
        self.apply_retrieval()
        self.apply_covmat()

        jqs = ws.jacobian_quantities.value
        self._sequence_id = len(jqs) - 1
        self._jacobian_quantity = jqs[self._sequence_id]
        self._slice = _jqs_slices(jqs)[self._sequence_id]
        self._is_applied = True

    def apply_retrieval(self):
        """Calls the corresponding `retrievalAdd...()` WSM of the workspace."""
        ws = self.ws
        boilerplate.clear_sparse(ws, ws.covmat_block)
        boilerplate.clear_sparse(ws, ws.covmat_inv_block)
        retrieval_add_wsm = getattr(ws, 'retrievalAdd' + self.kind)
        retrieval_add_wsm(**self.args)

    def apply_covmat(self):
        """Add the corresponding covmat block to the Workspace."""
        ws = self.ws
        set_variable_by_xml(ws, ws.covmat_block, self.covmat)
        ws.covmat_sxAddBlock(block=ws.covmat_block)

    def extract_result(self, x=None, avk=None, eo=None, es=None):
        """Extract the result and the corresponding values form the Workspace."""
        ws = self._ws
        if x is None:
            x = ws.x.value
        if avk is None:
            avk = ws.avk.value
        if eo is None:
            eo = ws.retrieval_eo.value
        if es is None:
            es = ws.retrieval_ss.value

        self._x = x[self._slice]
        self._avkm = avk[self._slice, self._slice]
        self._eo = eo[self._slice]
        self._es = es[self._slice]

    def to_xarray(self):
        prefix = self.slug + '_'
        shape = self.shape
        grid_names = [prefix + 'grid' + str(i+1) for i in range(len(self.shape))]
        grids = self.grids
        flat_grid_name = prefix + 'grid'

        coords = {n: c for n, c in zip(grid_names, grids)}

        ds = xr.Dataset(
            data_vars={
                prefix + 'x': (grid_names, np.reshape(self.x, shape)),
                prefix + 'mr': (grid_names, np.reshape(self.mr, shape)),
                prefix + 'eo': (grid_names, np.reshape(self.eo, shape)),
                prefix + 'es': (grid_names, np.reshape(self.es, shape)),
                prefix + 'avkm': ((flat_grid_name, flat_grid_name + '_avk'), self.avkm),
            },
            coords=coords,
            attrs={
                'maintag': self._jacobian_quantity.maintag,
                'subtag': self._jacobian_quantity.subtag,
                'subsubtag': self._jacobian_quantity.subsubtag,
                'analytical': self._jacobian_quantity.analytical,
                'mode': self._jacobian_quantity.mode,
                'perturbation': self._jacobian_quantity.perturbation,
            }
        )
        return ds

    @property
    def kind(self):
        return self._kind

    @property
    def covmat(self):
        return self._covmat

    @property
    def args(self):
        return self._args

    @property
    def num_elem(self):
        return self.covmat.shape[0]

    @property
    def grids(self):
        return self._jacobian_quantity.grids

    @property
    def shape(self):
        return tuple(map(len, self.grids))

    @property
    def ws(self):
        return self._ws

    def __str__(self):
        return self.kind

    @property
    def slug(self):
        """Create a slug. Useful for serialisation."""
        jq = self._jacobian_quantity
        if jq.maintag == 'Absorption species':
            return str.lower(jq.subtag)  # this is the species e.g. O3
        elif jq.maintag == 'Wind':
            return 'wind_' + str.lower(jq.subtag)  # component u, v, w
        elif jq.maintag == 'Polynomial baseline fit':
            return 'polyfit'  # Same for all coefficients
        else:
            return str.lower('_'.join((jq.maintag, jq.subtag, jq.subsubtag))).replace(' ', '-')

    @property
    def x(self):
        return self._x

    @property
    def avkm(self):
        return self._avkm

    @property
    def mr(self):
        return level2.avkm_mr(self._avkm)

    @property
    def eo(self):
        return self._eo

    @property
    def es(self):
        return self._es


class GriddedRetrievalQuantity(RetrievalQuantity):
    def __init__(self, kind, covmat, **kwargs):
        if 'g1' not in kwargs or 'g2' not in kwargs or 'g3' not in kwargs:
            raise ValueError('Grids g1, g2 and g3 are required for gridded retrieval quantities.')
        p_grid = kwargs['g1']
        lat_grid = kwargs['g2']
        lon_grid = kwargs['g3']
        if not covmat.shape == _covmat_shape(p_grid, lat_grid, lon_grid):
            expected = _covmat_shape(p_grid, lat_grid, lon_grid)
            raise ValueError(
                f'Covariance matrix must have shape according to retrieval grid elements: {expected[0]} x {expected[1]}')
        super().__init__(kind, covmat, **kwargs)

    def to_xarray(self):
        prefix = self.slug + '_'
        shape = self.shape
        grid_names = [prefix + n for n in ('p', 'lat', 'lon')]
        grids = self.grids
        flat_grid_name = prefix+'p' if self.dimensions == 1 else prefix + 'grid'

        coords = {n: c for n, c in zip(grid_names, grids)}

        ds = xr.Dataset(
            data_vars={
                prefix + 'x': (grid_names, np.reshape(self.x, shape)),
                prefix + 'mr': (grid_names, np.reshape(self.mr, shape)),
                prefix + 'eo': (grid_names, np.reshape(self.eo, shape)),
                prefix + 'es': (grid_names, np.reshape(self.es, shape)),
                prefix + 'fwhm': (grid_names, np.reshape(self.fwhm, shape)),
                prefix + 'offset': (grid_names, np.reshape(self.offset, shape)),
                prefix + 'avkm': ((flat_grid_name, flat_grid_name + '_avk'), self.avkm),
                prefix + 'z': (grid_names[0], self.z_grid),
            },
            coords=coords,
            attrs={
                'maintag': self._jacobian_quantity.maintag,
                'subtag': self._jacobian_quantity.subtag,
                'subsubtag': self._jacobian_quantity.subsubtag,
                'analytical': self._jacobian_quantity.analytical,
                'mode': self._jacobian_quantity.mode,
                'perturbation': self._jacobian_quantity.perturbation,
            }
        )
        return ds

    @property
    def shape(self):
        return tuple(map(len, (self.args['g1'], self.args['g2'], self.args['g3'])))

    @property
    def num_elem(self):
        shape = self.shape
        return shape[0] * shape[1] * shape[2]

    @property
    def p_grid(self):
        return self._args['g1']

    @property
    def z_grid(self):
        if self.dimensions > 1:
            warnings.warn('Z grid extraction not supported for multi dim retrieval grids. Using simple conversion.')
            return p2z_simple(self.p_grid)
        if self.lat_grid[0] not in self._ws.lat_grid.value or self.lon_grid[0] not in self._ws.lon_grid.value:
            warnings.warn('Z grid extraction not supported for arbitrary retrieval grids. Using simple conversion.')
            return p2z_simple(self.p_grid)

        z_field = self._ws.z_field.value
        lat_grid = self._ws.lat_grid.value
        lon_grid = self._ws.lon_grid.value
        i_lat = np.searchsorted(lat_grid, self.lat_grid[0])
        i_lon = np.searchsorted(lon_grid, self.lon_grid[0])
        return z_field[:, i_lat, i_lon]

    @property
    def lat_grid(self):
        return self._args['g2']

    @property
    def lon_grid(self):
        return self._args['g3']

    @property
    def dimensions(self):
        return len(list(filter(lambda x: x > 1, self.shape)))

    @property
    def fwhm(self):
        if self.dimensions > 1:
            raise NotImplementedError('FWHM not implemented for multi dim retrieval grids.')
        return level2.avkm_fwhm(self.z_grid, self.avkm)

    @property
    def offset(self):
        if self.dimensions > 1:
            raise NotImplementedError('Offset not implemented for multi dim retrieval grids.')
        return level2.avkm_offset(self.z_grid, self.avkm)


class AbsSpecies(GriddedRetrievalQuantity):
    def __init__(self, species, p_grid, lat_grid, lon_grid, covmat,
                 method='analytical', unit='rel', for_species_tag=1, dx=0.001):
        super().__init__('AbsSpecies', covmat,
                         species=species,
                         g1=p_grid,
                         g2=lat_grid,
                         g3=lon_grid,
                         method=method,
                         unit=unit,
                         for_species_tag=for_species_tag,
                         dx=dx)

    @property
    def species(self):
        return self.args['species']

    @property
    def slug(self):
        return self.species.lower()


class Wind(GriddedRetrievalQuantity):
    def __init__(self, component, p_grid, lat_grid, lon_grid, covmat, dfrequency=0.1):
        super().__init__('Wind', covmat,
                         component=component,
                         g1=p_grid,
                         g2=lat_grid,
                         g3=lon_grid,
                         dfrequency=dfrequency)

    @property
    def component(self):
        return self.args['component']

    @property
    def slug(self):
        return 'wind_' + self.component.lower()


class FreqShift(RetrievalQuantity):
    def __init__(self, sx, df=100e3):
        covmat = np.array([sx])
        super().__init__('FreqShift', covmat, df=df)

    @property
    def num_elem(self):
        return 1

    @property
    def slug(self):
        return 'freq_shift'


class Polyfit(RetrievalQuantity):
    def __init__(self, poly_order, covmats, pol_variation=True, los_variation=True, mblock_variation=True):
        if len(covmats) != poly_order + 1:
            raise ValueError('Must provide (poly_order + 1) covariance matrices.')

        self._covmats = covmats
        super().__init__('Polyfit', None,
                         poly_order=poly_order,
                         no_pol_variation=0 if pol_variation else 1,
                         no_los_variation=0 if los_variation else 1,
                         no_mblock_variation=0 if mblock_variation else 1)

    def apply(self, ws):
        if self._is_applied:
            raise Exception('Same retrieval quantity cannot be added twice to the same workspace.')

        self._ws = ws
        self.apply_retrieval()
        self.apply_covmat()

        n_jqs = self.poly_order + 1
        jqs = ws.jacobian_quantities.value
        self._sequence_id = slice(len(jqs) - n_jqs, len(jqs))
        self._jacobian_quantity = jqs[self._sequence_id]
        self._slice = _jqs_slices(jqs)[self._sequence_id]
        self._is_applied = True

    def apply_retrieval(self):
        ws = self.ws
        boilerplate.clear_sparse(ws, ws.covmat_block)
        boilerplate.clear_sparse(ws, ws.covmat_inv_block)
        ws.retrievalAddPolyfit(**self.args)

    def apply_covmat(self):
        for covmat in self._covmats:
            self.ws.covmat_sxAddBlock(block=covmat)

    def extract_result(self, x=None, avk=None, eo=None, es=None):
        """Extract the result and the corresponding values form the Workspace."""
        ws = self._ws
        if x is None:
            x = ws.x.value
        if avk is None:
            avk = ws.avk.value
        if eo is None:
            eo = ws.retrieval_eo.value
        if es is None:
            es = ws.retrieval_ss.value

        self._x = [x[s] for s in self._slice]
        self._avkm = [avk[s, s] for s in self._slice]
        self._eo = [eo[s] for s in self._slice]
        self._es = [es[s] for s in self._slice]

    def to_xarray(self):
        prefix = self.slug + '_'
        grid_names = ('poly_order', 'observation')
        grids = self.grids
        flat_grid_name = prefix + 'grid'

        coords = {n: c for n, c in zip(grid_names, grids)}

        ds = xr.Dataset(
            data_vars={
                prefix + 'x': (grid_names, np.stack(self.x)),
                prefix + 'mr': (grid_names, np.stack(self.mr)),
                prefix + 'eo': (grid_names, np.stack(self.eo)),
                prefix + 'es': (grid_names, np.stack(self.es)),
                prefix + 'avkm': ((grid_names[0], flat_grid_name, flat_grid_name + '_avk'), np.stack(self.avkm)),
            },
            coords=coords,
            attrs={
                'maintag': self._jacobian_quantity[0].maintag,
                'subtag': '',
                'subsubtag': '',
                'analytical': self._jacobian_quantity[0].analytical,
                'mode': self._jacobian_quantity[0].mode,
                'perturbation': self._jacobian_quantity[0].perturbation,
            }
        )
        return ds

    @property
    def slug(self):
        return 'poly_fit'

    @property
    def grids(self):
        jq_grids = self._jacobian_quantity[0].grids
        if len(jq_grids[0]) > 1 or len(jq_grids[1]) > 1 or len(jq_grids[2]) > 1:
            raise NotImplementedError()
        return [np.arange(self.poly_order+1), np.array(jq_grids[-1], dtype=np.int)]

    @property
    def covmat(self):
        return sparse.diags(self._covmats)

    @property
    def poly_order(self):
        return self.args['poly_order']

    @property
    def mr(self):
        return [level2.avkm_mr(a) for a in self._avkm]


class SlicedArray:
    def __init__(self, values, slugs_to_slices):
        self._values = values
        self.slugs = slugs_to_slices

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) == len(self._values.shape):
                idx = tuple(slice(*self.slugs[it]) for it in item)
                return self.values[idx]
            else:
                raise ValueError('Number of indexers must be one or be equal to number of dimensions.')
        else:
            a, b = self.slugs[item]
            n_dims = len(self._values.shape)
            return self._values[tuple(n_dims * [slice(a, b)])]

    @property
    def values(self):
        return self._values
