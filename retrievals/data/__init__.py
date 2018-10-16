"""
This package helps to exploit atmospheric data from different sources.
"""

from .dtutils import (
    date_glob,
    year_start,
    fz_dayofyear,
)

from .index import (
    nc_index,
)

from .utils import (
    interpolate,
    p_interpolate,
)