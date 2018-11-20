import os
from datetime import datetime
import pandas as pd


def date_glob_1(path, fmt):
    """
    Return all elements (files and dirs) in `path`, that match the format `fmt`.

    :param path: Base path.
    :param fmt: The format string for the file names as used
                by :py:meth:`datetime.datetime.strftime`
    :return: A dict `datetime -> file name`
    """
    if os.path.split(fmt)[0]:
        raise ValueError("fmt must not be a path.")
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        try:
            date = datetime.strptime(name, fmt)
        except ValueError:
            pass
        else:
            yield date, os.path.join(path, name)


def date_glob(path, fmt):
    """
    If the files are organized as `/path/to/folder/2018/result_2018-01-01.nc`, one can
    get all files with:

    >>> date_glob('/path/to/folder', ''%Y/result_%Y-%m-%d.nc')
    {datetime.datetime(2019, 1, 1): '2018/result_2018-01-01.nc', ... )

    If a field is present multiple times in a pattern (like `%Y` above), this function
    will try to use as
    much of the file path as possible and successively ignore parts of the path until
    eventually ending
    up with parts that can be parsed (the filename only in the example above).

    :param path: Base path.
    :param fmt: The format string for the file names as used
                by :py:meth:`datetime.datetime.strftime`
    :return: A dict `datetime -> file name`
    """
    fmt_parts = os.path.normpath(fmt).split(os.sep)
    dirs = [os.path.normpath(path)]  # the dirs to search
    for dir_fmt in fmt_parts:
        new_dirs = []
        for d in dirs:
            names = list(date_glob_1(d, dir_fmt))
            new_dirs += (name for _, name in names)
        dirs = new_dirs

    # parse dates
    for name in dirs:
        for i in range(len(fmt_parts), 0, -1):
            n = os.path.join(*name.split(os.sep)[-i:])
            f = os.path.join(*fmt_parts[-i:])
            try:
                date = datetime.strptime(n, f)
            except:
                # Part of filename does not match pattern, skip it
                pass
            else:
                yield date, os.path.join(path, name)
                break  # next file
        else:  # nobreak
            raise ValueError("Unable to parse {}".format(name))


def year_start(ts):
    """Return the first day of the year of the timestamp."""
    ts = pd.to_datetime(ts)
    return (ts - pd.Timedelta(ts.dayofyear - 1, "D")).floor("D")


def fz_dayofyear(ts, squash_leap=False):
    """
    Convert timestamp to fractional zero-based day-of-year.

    :param squash_leap: If true, Feb. 29 and Feb. 28 are the same day-of-year.
    """
    ts = pd.to_datetime(ts)
    doy = (ts - year_start(ts)).total_seconds() / (24 * 60 * 60)

    if ts.is_leap_year and squash_leap and doy >= 60:
        doy -= 1

    return doy
