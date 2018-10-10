"""This module deals with the retrieval results."""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from numba import jit


def avkm_mr(avkm):
    """Calculate measurement response, which is the sum of the rows of the AVKM."""
    return np.sum(avkm, axis=1)


def fwhm(x, y):
    """Calculate the FWHM and peak height of a vector y with corresponding coordinates x."""
    i_max_y = np.argmax(y)
    y_norm = y - y[i_max_y] / 2
    max_x = x[i_max_y]
    spline = InterpolatedUnivariateSpline(x, y_norm, k=3, ext='raise')

    # FWHM
    r = spline.roots()
    if len(r) < 2:
        return np.nan
    center = r[np.argsort(np.abs(r - max_x))[0:2]]
    fwhm = np.abs(center[1] - center[0])
    return fwhm


def peak(x, y):
    """Calculate the FWHM and peak height of a vector y with corresponding coordinates x."""
    spline = InterpolatedUnivariateSpline(x, y, k=4, ext='raise')
    extrema = spline.derivative().roots()
    if len(extrema) < 1:
        return np.nan
    max_x = extrema[np.argmax(spline(extrema))]
    return max_x


def avkm_fwhm(x, avkm):
    """Calculate the FWHM of a AVKM."""
    n_rows, n_cols = avkm.shape
    if x.shape != (n_cols,):
        raise ValueError('Size of coordinate must match number of columns of AVKM.')
    fwhms = np.zeros((n_rows,))
    for i in range(n_rows):
        y = avkm[i, :]
        fwhms[i] = fwhm(x, y)
    return fwhms


def avkm_peak(x, avkm):
    """Calculate the peak height of AVKs in avkm in units of coordinate vector x."""
    n_rows, n_cols = avkm.shape
    if x.shape != (n_cols,):
        raise ValueError('Size of coordinate must match number of columns of AVKM.')
    peaks = np.zeros((n_rows,))
    for i in range(n_rows):
        y = avkm[i, :]
        peaks[i] = peak(x, y)  # x[np.argmax(y)]
    return peaks


def avkm_offset(x, avkm):
    """Calculate the offset of nominal height to peak height of AVKs in avkm in units of coordinate vector x."""
    n_rows, n_cols = avkm.shape
    if n_rows != n_cols:
        ValueError('AVKM must be square matrix.')
    if x.shape != (n_cols,):
        raise ValueError('Size of coordinate must match number of columns of AVKM.')
    return x - avkm_peak(x, avkm)





import time


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print(f'Elapsed time: {self.interval:.4f} seconds.')


def benchmark():
    for n in [10, 100, 500, 1000, 2000]:
        x = np.linspace(-10, 10, n)
        avk = np.exp(-(x / 3) ** 2 / 2)
        avkm = np.vstack(n * [avk, ])

        print(f'N = {n:>4}:', end='')
        with Timer():
            fwhm = avkm_fwhm(x, avkm)
            peak = avkm_peak(x, avkm)


if __name__ == '__main__':
    benchmark()
