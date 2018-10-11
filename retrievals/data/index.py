import xarray as xr
from glob import glob
import os
import numpy as np


class IntervalDict(dict):
    """A dict to store and lookup values that are indexed by intervals."""

    def envelope(self, lower, upper):
        """Return a generator over all intervals that intersect with `(lower, upper)`."""
        for (a, b), v in self.items():
            if lower <= a <= upper or lower <= b <= upper or a <= lower <= b or a <= upper <= b:
                yield v

    def containers(self, x):
        """Get the intervals that contain the value."""
        return (v for (a, b), v in self.items() if a <= x <= b)

    def min(self):
        """Minimum of all lower bounds."""
        return min(a for a, _ in self.keys())

    def max(self):
        """Maximum of all upper bounds."""
        return max(b for _, b in self.keys())


def nc_index(path, pattern, index_var='time', index=None, **kwargs):
    """
    Create an index of netCDF files.

    :param path: Base path of the files.
    :param pattern: Glob pattern for the files.
    :param index_var: Variable to index. Default: 'time'
    :param index: Index to update. Files already present in the index are skipped.
    :type index: xarray.DataArray
    :param kwargs: Additional arguments to :py:func:`xarray.open_dataset()`
    :return: The new index.
    :rtype: xarray.DataArray
    """
    f_names = []
    f_values = []
    files = glob(os.path.join(path, pattern))
    if index is not None:
        f_names = list(index.values)
        f_values = list(index[index_var].values)
        known_files = set(os.path.join(path, fn) for fn in f_names)
    else:
        known_files = set()

    new_files = list(filter(lambda fn: fn not in known_files, files))
    for f in new_files:
        ds = xr.open_mfdataset(f, **kwargs)
        for value in ds[index_var].values:
            f_names.append(os.path.relpath(f, path))
            f_values.append(value)
    index = xr.DataArray(data=f_names, coords={index_var: np.array(f_values)}, dims=(index_var,), name='file_names')
    index.attrs['base_path'] = path
    index.attrs['pattern'] = pattern
    return index


def main():
    from retrievals.utils import Timer
    from glob import glob
    files = glob('/media/network/gantrisch/scratch/wirac/ecmwf_locations/ECMWF_OPER_v1_MAIDO_????????.nc')

    with Timer():
        ind = nc_index(files)
    for (a, b), f in ind.items():
        print(a, b, f)


def main2():
    from retrievals.utils import Timer

    path = '/media/network/gantrisch/scratch/wirac/ecmwf_locations'
    pattern1 = 'ECMWF_OPER_v1_MAIDO_????????.nc'
    pattern2 = 'ECMWF_OPER_v1_MAIDO_201801??.nc'
    pattern3 = 'ECMWF_OPER_v1_MAIDO_201802??.nc'

    with Timer():
        index = nc_index(path, pattern2, index_var='time')
    with Timer():
        index = nc_index(path, pattern3, index_var='time', index=index)
    print(index)


if __name__ == '__main__':
    main2()
