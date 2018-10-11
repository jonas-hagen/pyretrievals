"""
The problem with ECMWF data is, that it comes in many different formats Grib1, Grib2, NetCDF.
Often they are organized in different ways: By location, by time, by variable etc.
These tools aim to give an abstraction and make it comfortable and fast to extract the desired data.
"""

from .store import (
    ECMWFLocationFileStore,
)

from .levels import (
    hybrid_level,
)
