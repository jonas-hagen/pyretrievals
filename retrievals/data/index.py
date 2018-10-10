import xarray as xr


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


def nc_index(files, index_var='time', upper_index_var=None, index=None, **kwargs):
    """
    Index NetCDF files.

    :param files: List of file names.
    :param index_var: Variable name to use for indexing. Default: `time`
    :param upper_index_var: Variable name to use for the upper index if different from index_var.
    :param index: Index to update. Files already present in the index are skipped.
    :type index: IntervalDict
    :param kwargs: Additional arguments to :py:func:`xarray.open_dataset`.
    :rtype: IntervalDict
    """
    # TODO: Use netcdf-python for more speed?
    upper_index_var = upper_index_var or index_var
    if index is None:
        index = IntervalDict()
    new_files = filter(lambda fn: fn not in index.values(), files)
    for f in new_files:
        df = xr.open_dataset(f, **kwargs)
        lower = df[index_var].min().values
        upper = df[upper_index_var].max().values
        index[(lower, upper)] = f
        df.close()
    return index


def main():
    from retrievals.utils import Timer
    from glob import glob
    files = glob('/media/network/gantrisch/scratch/wirac/ecmwf_locations/ECMWF_OPER_v1_MAIDO_????????.nc')

    with Timer():
        ind = nc_index(files)
    for (a, b), f in ind.items():
        print(a, b, f)


if __name__ == '__main__':
    main()
