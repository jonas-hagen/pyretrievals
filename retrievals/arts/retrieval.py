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
from contextlib import contextmanager

from .boilerplate import set_variable_by_xml


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


class RetrievalQuantity:
    """A generic retrieval quantity."""

    def __init__(self, kind, covmat, **kwargs):
        self._kind = kind
        self._covmat = covmat
        self._args = kwargs

    def apply(self, ws):
        retrieval_add_wsm = getattr(ws, 'retrievalAdd' + self.kind)
        retrieval_add_wsm(**self.args)
        set_variable_by_xml(ws, ws.covmat_block, self.covmat)
        ws.covmat_sxAddBlock(block=ws.covmat_block)
        print('--- 1 ---')
        pass

    @property
    def kind(self):
        return self._kind

    @property
    def covmat(self):
        return self._covmat

    @property
    def args(self):
        return self._args


class AbsSpecies(RetrievalQuantity):
    def __init__(self, species, p_grid, lat_grid, lon_grid, covmat,
                 method='analytical', unit='rel', for_species_tag=1, dx=0.001):
        if not covmat.shape == _covmat_shape(p_grid, lat_grid, lon_grid):
            expected = _covmat_shape(p_grid, lat_grid, lon_grid)[0]
            raise ValueError(
                f'Covariance matrix must have shape according to retrieval grid elements: {expected} x {expected}')
        super().__init__('AbsSpecies', covmat,
                         species=species,
                         g1=p_grid,
                         g2=lat_grid,
                         g3=lon_grid,
                         method=method,
                         unit=unit,
                         for_species_tag=for_species_tag,
                         dx=dx)


class Wind(RetrievalQuantity):
    def __init__(self, component, p_grid, lat_grid, lon_grid, covmat, dfrequency=0.1):
        if not covmat.shape == _covmat_shape(p_grid, lat_grid, lon_grid):
            expected = _covmat_shape(p_grid, lat_grid, lon_grid)[0]
            raise ValueError(
                f'Covariance matrix must have shape according to retrieval grid elements: {expected} x {expected}')
        super().__init__('Wind', covmat,
                         component=component,
                         g1=p_grid,
                         g2=lat_grid,
                         g3=lon_grid,
                         dfrequency=dfrequency)


class FreqShift(RetrievalQuantity):
    def __init__(self, sx, df=100e3):
        covmat = np.array([sx])
        super().__init__('FreqShift', covmat, df=df)


class Polyfit(RetrievalQuantity):
    def __init__(self, poly_order, covmats, pol_variation=True, los_variation=True, mblock_variation=True):
        if len(covmats) != poly_order + 1:
            raise ValueError('Must provide (poly_order + 1) covariance matrices.')

        self._covmats = covmats
        super().__init__('Polyfit', None,
                         no_pol_variation=0 if pol_variation else 1,
                         no_los_variation=0 if los_variation else 1,
                         no_mblock_variation=0 if mblock_variation else 1)

    def apply(self, ws):
        ws.retrievalAddPolyfit(**self.args)
        for covmat in self._covmats:
            ws.covmat_sxAddBlock(block=covmat)

    @property
    def covmat(self):
        return sparse.diags(self._covmats)
