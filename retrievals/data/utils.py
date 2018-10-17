import numpy as np


def interpolate(x2, x1, y1, left=None, right=None, fill=None):
    """
    Convenient and safe wrapper around :py:func:`numpy.interp`. The coordinates and vaules `x1` and `y1` are sorted
    and shapes are checked before calling :py:func:`numpy.interp` that would otherwise return rubbish.

    :param x2: The x-coordinates of the interpolated values.
    :param x1: The x-coordinates of the data points.
    :param y1: The y-coordinates of the data points, same length as `x1`.
    :param left: Value to return for `x = min(x1)`, default is `y1[min(x1)]`.
    :param right: Value to return for `x = max(x1)`, default is `y1[max(x1)]`.
    :param fill: Shorthand for `left = right = fill`.
    :return: The interpolated values, same shape as `x2`.
    """
    if fill is not None:
        if left is not None or right is not None:
            raise ValueError('Either provide `fill` or {`left` and `right`}, but not both.')
        left = fill
        right = fill
    if x2.ndim > 1 or x1.ndim != 1 or y1.ndim != 1:
        raise ValueError('Only 1d arrays can be interpolated.')
    if x1.shape != y1.shape:
        raise ValueError('`x1` and `y1` must have same length.')

    sorted_idx = np.argsort(x1, kind='heapsort')
    x1_sorted = x1[sorted_idx]
    y1_sorted = y1[sorted_idx]
    y2 = np.interp(x2, x1_sorted, y1_sorted, left=left, right=right)
    return y2


def p_interpolate(p2, p1, y1, bottom=None, top=None, fill=None):
    """
    Same as :py:func:`interpolate` but for pressure coordinates. `bottom` and `top` are the fill values for bottom and
    top of atmosphere respectively. By default, the closest value is used.
    """
    return interpolate(np.log(p2), np.log(p1), y1, left=top, right=bottom, fill=fill)
