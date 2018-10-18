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
        """
        :param kind: The kind is used to identify the corresponding WSMs, for example for a kind of `Sinefit` the WSM
                     `retrievalAddSinefit` would be called upon apply with the keyword arguments.
        :param covmat: The covariance matrix for the retrieval.
        :param kwargs: Arguments to the `retrievalAdd...` WSM.
        """
        self._kind = kind
        self._covmat = covmat
        self._args = kwargs

        self._is_applied = False
        self._sequence_id = None  # index inside jacobian quantities of workspace
        self._jacobian_quantity = None
        self._slice = None  # Holds the start and end indices for this quantity
        self._ws = None  # Associated workspace

        self._xa = None
        self._x = None
        self._avkm = None
        self._eo = None
        self._es = None

    def apply(self, ws):
        """
        Add this retrieval quantity to a Workspace and extract the indices from the jacobian quantities.
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

    def extract_apriori(self, xa=None):
        """
        Extract the a priori values. Call WSM `xaStandard()` before calling this function!
        :param xa: The values in the `xa` vector just before the inversion. If None, copy from Workspace.
        """
        ws = self._ws
        if xa is None:
            xa = ws.xa.value
        self._xa = xa[self._slice]

    def extract_result(self, x=None, avk=None, eo=None, es=None):
        """
        Extract the result and the corresponding values form the Workspace.
        If the arguments are `None`, the variables are copied from the associated Workspace.
        """
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
        """
        Export this retrieval quantity.

        :rtype: xarray.Dataset
        """
        prefix = self.slug + '_'
        shape = self.shape
        grid_names = [prefix + gn for gn in self.grid_names]
        grids = self.grids
        flat_grid_name = prefix + 'grid' if self.dimensions > 1 else grid_names[0]

        coords = {n: c for n, c in zip(grid_names, grids)}

        ds = xr.Dataset(
            data_vars={
                prefix + 'x': (grid_names, np.reshape(self.x, shape, order='F')),
                prefix + 'xa': (grid_names, np.reshape(self.xa, shape, order='F')),
                prefix + 'mr': (grid_names, np.reshape(self.mr, shape, order='F')),
                prefix + 'eo': (grid_names, np.reshape(self.eo, shape, order='F')),
                prefix + 'es': (grid_names, np.reshape(self.es, shape, order='F')),
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
        """The arguments for the `retrievalAdd...` WSM."""
        return self._args

    @property
    def num_elem(self):
        """Number of elements, product of all grid lengths."""
        return self.covmat.shape[0]

    @property
    def grids(self):
        """Associated grids."""
        return self._jacobian_quantity.grids

    @property
    def grid_names(self):
        """Names of the grids."""
        return ['grid' + str(i + 1) for i in range(len(self.shape))]

    @property
    def shape(self):
        """Shape of grids."""
        return tuple(map(len, self.grids))

    @property
    def dimensions(self):
        return len(list(filter(lambda x: x > 1, self.shape)))

    @property
    def ws(self):
        """Associated workspace. Set by :py:meth:`apply`."""
        return self._ws

    def __str__(self):
        return self.kind

    @property
    def slug(self):
        """A slug according to the kind of this retrieval quantity, useful for serialisation."""
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
        """The retrieved values. Available after :py:meth:`extract_results`."""
        return self._x

    @property
    def xa(self):
        """The a priori values. Available after :py:meth:`extract_apriori`."""
        return self._xa

    @property
    def avkm(self):
        """The averaging kernel matrix. Available after :py:meth:`extract_results`."""
        return self._avkm

    @property
    def mr(self):
        """The measurement response. Available after :py:meth:`extract_results`."""
        return level2.avkm_mr(self._avkm)

    @property
    def eo(self):
        """The retrieval error associated with the observational system. Available after :py:meth:`extract_results`."""
        return self._eo

    @property
    def es(self):
        """The smoothing error. Available after :py:meth:`extract_results`."""
        return self._es


class GriddedRetrievalQuantity(RetrievalQuantity):
    """Retrieval quantity that is retrieved on a spatial grid."""

    def __init__(self, kind, p_grid, lat_grid, lon_grid, covmat, **kwargs):
        """
        :param kind: The kind is used to identify the corresponding WSMs, for example for a kind of `Sinefit` the WSM
                     `retrievalAddSinefit` would be called upon apply with the keyword arguments.
        :param p_grid: Pressure grid
        :param lat_grid: Latitude grid
        :param lon_grid: Longitude grid
        :param covmat: The covariance matrix for the retrieval.
        :param kwargs: Arguments to the `retrievalAdd...` WSM.
        """
        kwargs['g1'] = p_grid
        kwargs['g2'] = lat_grid
        kwargs['g3'] = lon_grid
        if not covmat.shape == _covmat_shape(p_grid, lat_grid, lon_grid):
            expected = _covmat_shape(p_grid, lat_grid, lon_grid)
            raise ValueError(
                f'Covariance matrix must have shape according to retrieval grid elements: {expected[0]} x {expected[1]}')
        super().__init__(kind, covmat, **kwargs)

    def to_xarray(self):
        prefix = self.slug + '_'
        shape = self.shape
        grid_names = [prefix + gn for gn in self.grid_names]

        ds = super().to_xarray()

        if self.dimensions == 1:
            ds[prefix + 'z'] = (grid_names[0], self.z_grid)
            ds[prefix + 'fwhm'] = (grid_names, np.reshape(self.fwhm, shape, order='F'))
            ds[prefix + 'offset'] = (grid_names, np.reshape(self.offset, shape, order='F'))

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

        atmosphere_dim = self._ws.atmosphere_dim.value
        lat_grid = self._ws.lat_grid.value
        lon_grid = self._ws.lon_grid.value
        if atmosphere_dim > 1:
            if self.lat_grid[0] not in lat_grid or self.lon_grid[0] not in lon_grid:
                warnings.warn('Z grid extraction not supported for arbitrary retrieval grids. Using simple conversion.')
                return p2z_simple(self.p_grid)

        z_field = self._ws.z_field.value

        if atmosphere_dim == 1:
            i_lat = i_lon = 0
        elif atmosphere_dim == 2:
            i_lon = 0
            i_lat = np.searchsorted(lat_grid, self.lat_grid[0])
        else:
            i_lat = np.searchsorted(lat_grid, self.lat_grid[0])
            i_lon = np.searchsorted(lon_grid, self.lon_grid[0])
        z_grid1 = z_field[:, i_lat, i_lon]
        p_grid1 = self._ws.p_grid.value
        idx = np.argsort(p_grid1)
        z_grid = np.interp(np.log(self.p_grid), np.log(p_grid1[idx]), z_grid1[idx])
        return z_grid

    @property
    def lat_grid(self):
        return self._args['g2']

    @property
    def lon_grid(self):
        return self._args['g3']

    @property
    def grid_names(self):
        return 'p', 'lat', 'lon'

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
    """Absorption species."""

    def __init__(self, species, p_grid, lat_grid, lon_grid, covmat,
                 method='analytical', unit='rel', for_species_tag=1, dx=0.001):
        super().__init__('AbsSpecies',
                         p_grid=p_grid,
                         lat_grid=lat_grid,
                         lon_grid=lon_grid,
                         covmat=covmat,
                         species=species,
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
    """Wind."""

    def __init__(self, component, p_grid, lat_grid, lon_grid, covmat, dfrequency=0.1):
        super().__init__('Wind',
                         p_grid=p_grid,
                         lat_grid=lat_grid,
                         lon_grid=lon_grid,
                         covmat=covmat,
                         component=component,
                         dfrequency=dfrequency)

    @property
    def component(self):
        return self.args['component']

    @property
    def slug(self):
        return 'wind_' + self.component.lower()


class FreqShift(RetrievalQuantity):
    """Backend frequency shift."""

    def __init__(self, var, df=100e3):
        covmat = np.array([var ** 2])
        super().__init__('FreqShift', covmat, df=df)

    def apply_covmat(self):
        ws = self.ws
        ws.covmat_block = sparse.bsr_matrix(self.covmat)
        ws.covmat_sxAddBlock(block=ws.covmat_block)

    @property
    def num_elem(self):
        return 1

    @property
    def slug(self):
        return 'freq_shift'


class Polyfit(RetrievalQuantity):
    """Polynomial baseline fit."""

    def __init__(self, poly_order, covmats, pol_variation=True, los_variation=True, mblock_variation=True):
        if len(covmats) != poly_order + 1:
            raise ValueError('Must provide (poly_order + 1) covariance matrices.')
        for covmat in covmats:
            if covmat.ndim != 2:
                raise ValueError('Covariance matrices must have ndim=2, but got {}.'.format(covmat.ndim))

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

    def extract_apriori(self, xa=None):
        """
        Extract the a priori values. Call WSM `xaStandard()` before calling this function!
        :param xa: The values in the `xa` vector just before the inversion. If None, copy from Workspace.
        """
        ws = self._ws
        if xa is None:
            xa = ws.xa.value
        self._xa = [xa[s] for s in self._slice]  # Always 0, but we implement it anyways.

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
                prefix + 'xa': (grid_names, np.stack(self.xa)),
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
        return [np.arange(self.poly_order + 1), np.array(jq_grids[-1], dtype=np.int)]

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
