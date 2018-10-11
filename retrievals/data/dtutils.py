import os
from datetime import datetime


def date_glob(path, fmt):
    """
    If the files are organized as `/path/to/folder/2018/ecmwf_2018-01-01.nc`, one can get all files with:

    >>> date_glob('/path/to/folder', ''%Y/ecmwf_%Y-%m-%d.nc')
    {datetime.datetime(2018, 1, 1): '2018/ecmwf_2018-01-01.nc', ... )

    :param path: Base path.
    :param fmt: The format string for the file names as used by :py:meth:`datetime.datetime.strftime`
    :return: A dict `datetime -> file name`
    """
    tree = dict()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            try:
                date = datetime.strptime(name, fmt)
            except ValueError:
                pass
            else:
                tree[date] = os.path.join(root, name)

    return tree

