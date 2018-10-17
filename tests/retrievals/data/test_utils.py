from retrievals.data import utils
import numpy as np


def test_interpolate():
    n = 4
    x1 = np.linspace(-1.5, 1.5, n)
    x2 = np.linspace(-1, 1, n)

    def f(x):
        # a linear function
        return 3 * x + 5

    # Shuffle
    np.random.shuffle(x1)
    y1 = f(x1)

    y2 = utils.interpolate(x2, x1, y1)
    assert np.allclose(y2, f(x2))
