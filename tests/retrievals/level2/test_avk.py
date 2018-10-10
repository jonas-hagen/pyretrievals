from retrievals import level2
import numpy as np


def _make_avk(n, fwhm=15, x0=50):
    """Make an avk with default center at 50 and default fwhm of 15."""
    x = np.linspace(0, 100, n)
    c = fwhm / (2*np.sqrt(2*np.log(2)))
    avk = np.exp(-((x-x0) / c) ** 2 / 2)
    return x, avk


def test_fwhm():
    x, avk = _make_avk(100)
    fwhm = level2.avk.fwhm(x, avk)
    assert abs(fwhm - 15) < 0.1


def test_avkm_fwhm():
    n = 200
    fwhms = np.linspace(5, 30, n)
    x, _ = _make_avk(n)
    avks = [_make_avk(n, fwhm=f)[1] for f in fwhms]
    avkm = np.vstack(avks)
    fwhm = level2.avkm_fwhm(x, avkm)
    assert np.allclose(fwhm, fwhms, atol=0.1)


def test_avkm_peak():
    n = 100
    x, avk = _make_avk(n)
    avkm = np.vstack(n * [avk, ])
    x_peak = level2.avkm_peak(x, avkm)
    assert np.allclose(x_peak, 50, atol=0.001)


def test_avkm_offset():
    n = 100
    x, avk = _make_avk(n)
    avkm = np.vstack(n * [avk, ])
    offset = level2.avkm_offset(x, avkm)
    assert np.allclose(offset, np.linspace(-50, 50, 100), atol=0.001)
