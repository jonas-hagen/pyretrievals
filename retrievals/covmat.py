import numpy as np
from scipy import sparse
from numba import jit, njit


def corr_fun(fname):
    """
    Define the functional form of correlations.

    The following types are available:
        "exp": f(x) = exp(-x)
        "lin": f(x) = 1 - x * (1 - np.exp(-1))
        "gauss": f(x) = exp(-x^2)

    For all functional forms 'f(1) = exp(-1)', and they are usually used
    with a distance 'd', normalized to a correlation length 'c' as 'f(d/c)'.

    :param str fname: Name of functional form.
    :return: A function taking one numeric argument.
    :raises: NotImplementedError: If function name is not known.
    """

    def f_lin(r):
        return 1.0 - (1.0 - np.exp(-1.0)) * r

    def f_exp(r):
        return np.exp(-r)

    def f_gauss(r):
        return np.exp(-r ** 2)

    switcher = {
        'lin': f_lin,
        'exp': f_exp,
        'gauss': f_gauss,
    }
    try:
        return switcher[fname]
    except KeyError:
        raise NotImplementedError(
            f'Correlation type {fname} is not implemented, use one of ' + ', '.join(switcher.keys()))


def covmat_diagonal(sigma):
    """
    Create diagonal covariance matrix.

    :param sigma: n-vector containing the standard deviations.
    :return: A diagonal n x n matrix.
    """
    return np.diag(np.square(sigma))


def covmat_diagonal_sparse(sigma):
    """
    Create sparse diagonal covariance matrix.

    :param sigma: n-vector containing the standard deviations.
    :return: A sparse diagonal n x n matrix.
    """
    n = len(sigma)
    ind = np.arange(n)
    S = sparse.coo_matrix((np.square(sigma), (ind, ind)), shape=(n, n))
    return S


def covmat_1d(grid1, sigma1, cl1, grid2=None, sigma2=None, cl2=None, fname='lin', cutoff=0):
    """
    Creates a 1D covariance matrix for two retrieval quantities on given
    grids from a given functional form. Elements of the covariance matrix
    are computed as :math:`S_{i,j} = \sigma_i  \sigma_j  f(d_{i,j} / l_{i,j})`
    where :math:`d_{i,j}` is the distance between the two grid points and :math:`c_{i,j}`
    the mean of the correlation lengths of the grid points.

    If a cutoff value is given, elements with a correlation less than this
    are set to zero.

    :param grid1: The retrieval grid for the first retrieval quantity.
    :param sigma1: The variances of the first retrieval quantity.
    :param cl1: The correlations lengths of the first retrieval quantity.
    :param grid2: The retrieval grid for the second retrieval quantity. Default: grid1
    :param sigma2: The variances of the second retrieval quantity. Default: sigma1
    :param cl2: The correlations lengths of the second retrieval quantity. Default: cl1
    :param fname: Name of functional form of correlation. Default: 'lin'.
    :param cutoff: The cutoff value for covariance.
    :return: The covariance matrix.
    """
    if grid2 is None:
        grid2 = grid1
        sigma2 = sigma1
        cl2 = cl1

    f = corr_fun(fname)

    x2, x1 = np.meshgrid(grid2, grid1)
    dist = np.abs(x2 - x1)

    x2, x1 = np.meshgrid(cl2, cl1)
    cl = (x2 + x1) / 2

    var = sigma1[:, np.newaxis] @ sigma2[np.newaxis, :]
    Sc = f(dist / cl)

    if cutoff > 0:
        Sc[Sc < cutoff] = 0
    S = var * Sc

    return S


def covmat_1d_sparse(grid1, sigma1, cl1, grid2=None, sigma2=None, cl2=None, fname='lin', cutoff=0):
    """
    Same as `covmat_1d` but creates and returns a sparse matrix.

    :param grid1:
    :param sigma1:
    :param cl1:
    :param grid2:
    :param sigma2:
    :param cl2:
    :param fname:
    :param cutoff:
    :return:
    """

    @njit
    def impl(grid1, sigma1, cl1, grid2, sigma2, cl2, cutoff, f):
        n1 = len(grid1)
        n2 = len(grid2)
        row_ind = []
        col_ind = []
        values = []

        for i in range(n1):
            for j in range(n2):
                d = np.abs(grid1[i] - grid2[j])
                c = (cl1[i] + cl2[j]) / 2
                s = sigma1[i] * sigma2[j]
                value = f(d / c)
                if value >= cutoff:
                    row_ind.append(i)
                    col_ind.append(j)
                    values.append(s * value)
                elif j > i:
                    # we are right of diagonal
                    break

        return values, row_ind, col_ind, n1, n2

    if grid2 is None:
        grid2 = grid1
        sigma2 = sigma1
        cl2 = cl1

    f = njit(corr_fun(fname))
    values, row_ind, col_ind, n1, n2 = impl(grid1, sigma1, cl1, grid2, sigma2, cl2, cutoff, f)
    S = sparse.coo_matrix((values, (row_ind, col_ind)), shape=(n1, n2))
    return S


def covmat_3d(grid1, cl1, fname1,
              grid2, cl2, fname2,
              grid3, cl3, fname3,
              sigma, cutoff=0,
              use_separable=True, use_sparse=False):
    """
    Builds a correlation matrix for one retrieval species for a 3D retrieval grid.

    The correlation matrix is a 2 dimensional matrix with the 3 (spatial) dimensions stacked,
    where grid1 belongs to the fastest running dimension and grid3 to the slowest running dimension.

    :param grid1: Grid for first dimension.
    :param cl1: The correlations lengths for the first dimension, same shape as grid1.
    :param fname1: Name of functional form of correlation, one of 'exp', 'lin', 'gauss'.
    :param grid2: Grid for second dimension.
    :param cl2:
    :param fname2:
    :param grid3: Grid for third dimension.
    :param cl3:
    :param fname3:
    :param sigma: Variances for the retrieval quantity on 3d grid.
    :param cutoff: Correlation cut-off
    :param use_separable: Use separable or non-separable statistics.
    :param use_sparse: Build a sparse matrix if set to True.
    :return: The correlation matrix with index of grid1 running fastest.
    """
    assert grid1.shape == cl1.shape, 'Dimension mismatch of grid1 and cl1.'
    assert grid2.shape == cl2.shape, 'Dimension mismatch of grid2 and cl2.'
    assert grid3.shape == cl3.shape, 'Dimension mismatch of grid3 and cl3.'
    assert sigma.shape == (len(grid1), len(grid2), len(grid3)), 'Dimension mismatch of sigma and grids.'

    if use_separable:
        # Use separable statistics
        # Total correlation is just the product of separate correlations
        f1 = njit(corr_fun(fname1))
        f2 = njit(corr_fun(fname2))
        f3 = njit(corr_fun(fname3))

        @njit
        def f(r1, r2, r3):
            return f1(r1) * f2(r2) * f3(r3)
    else:
        # Use non separable statistics
        # All dimensions must have same correlation function
        if not fname1 == fname2 == fname3:
            raise ValueError(f'For non separable statistics, correlation function must be the same '
                             f'for all dimensions, got {fname1}, {fname2}, {fname3}.')
        f1 = njit(corr_fun(fname1))

        @njit
        def f(r1, r2, r3):
            return f1(np.sqrt(r1 ** 2 + r2 ** 2 + r3 ** 2))

    n1 = len(grid1)
    n2 = len(grid2)
    n3 = len(grid3)
    n = n1 * n2 * n3

    @njit
    def unravel(i):
        """Fast unravel index for 3 dims"""
        i1 = i // (n2 * n1)
        i -= i1 * (n2 * n1)
        i2 = i // n1
        i -= i2 * n1
        i3 = i
        return i1, i2, i3

    @njit
    def impl(n, grid1, cl1, grid2, cl2, grid3, cl3, f, sigma, cutoff):
        for idx1 in range(n):
            # index i belongs to grid1 and is the fastest running
            k1, j1, i1 = unravel(idx1)

            for idx2 in range(n):
                k2, j2, i2 = unravel(idx2)

                # determine distance and correlation for each grid separately
                d1 = np.abs(grid1[i1] - grid1[i2])
                c1 = (cl1[i1] + cl1[i2]) / 2

                d2 = np.abs(grid2[j1] - grid2[j2])
                c2 = (cl2[j1] + cl2[j2]) / 2

                d3 = np.abs(grid3[k1] - grid3[k2])
                c3 = (cl3[k1] + cl3[k2]) / 2

                # normalize distance with correlation length and compute total correlation
                value = f(d1 / c1, d2 / c2, d3 / c3)

                if value >= cutoff:
                    var = sigma[i1, j1, k1] * sigma[i2, j2, k2]
                    yield idx1, idx2, var * value

    generator = impl(n, grid1, cl1, grid2, cl2, grid3, cl3, f, sigma, cutoff)
    n = n1 * n2 * n3

    if use_sparse:
        row_ind = []
        col_ind = []
        values = []
        for idx1, idx2, x in generator:
            row_ind.append(idx1)
            col_ind.append(idx2)
            values.append(x)
        S = sparse.coo_matrix((values, (row_ind, col_ind)), shape=(n, n))
    else:
        S = np.zeros((n, n))
        for idx1, idx2, x in generator:
            S[idx1, idx2] = x

    return S


if __name__ == '__main__':
    # Do a benchmark
    # Set NUMBA_DISABLE_JIT=1 to disable JIT

    import os
    import timeit

    for with_jit in [True, False]:
        s = 'With JIT:' if with_jit else 'Python only:'
        print(s)
        os.environ['NUMBA_DISABLE_JIT'] = '0' if with_jit else '1'
        setup_stmt = 'import numpy as np; from retrievals import covmat; covmat.covmat_1d_sparse(np.arange(2), np.ones(2), 2*np.ones(2), cutoff=0.3);'

        for n in [100, 300, 500]:
            print('N = {0:5d} -> '.format(n), end='')
            stmt = 'n={0}; covmat.covmat_1d_sparse(np.arange(n), np.ones(n), 2*np.ones(n), cutoff=0.3)'.format(n)
            t = timeit.timeit(stmt=stmt, setup=setup_stmt, number=1)
            print('{0:6.5f} '.format(t))
        print()
