"""
This module contains tools to access the NASA earth data database.

A login is required. The login data will be taken from one of the following sources (in that order):

1. Environment variables `NASA_EARTHDATA_LOGIN` and `NASA_EARTHDATA_PASSWORD`
2. The .env file in the current working directory (same variable names)
3. The users .netrc file from a line like this::

    machine urs.earthdata.nasa.gov login some_user password ABCdef123!


If you need to download many files and global data, `using wget`_ is faster.

.. _using wget: https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget
"""

import pydap.cas.urs
import xarray as xr
import os
import netrc
from dotenv import dotenv_values


class CredentialsNotFoundError(Exception):
    pass


def open_dataset(url, session=None):
    """
    Retrieve a dataset from a opendap url.

    :param url: Example: `https://acdisc.gesdisc.eosdis.nasa.gov:443/opendap/HDF-EOS5/Aura_MLS_Level2/ML2T.004/2017/MLS-Aura_L2GP-Temperature_v04-23-c01_2017d177.he5`
    :param session: Session to use for requests.
    :rtype: xarray.Dataset
    """
    if session is None:
        session = setup_session()
    store = xr.backends.PydapDataStore.open(url, session=session)
    ds = xr.open_dataset(store)
    return ds


def download(url, target_path, session=None):
    """
    Download a dataset.

    :param url: Example: `https://acdisc.gesdisc.eosdis.nasa.gov:443/opendap/HDF-EOS5/Aura_MLS_Level2/ML2T.004/2017/MLS-Aura_L2GP-Temperature_v04-23-c01_2017d177.he5.nc4`
    :param target_path: Path to store downloaded data.
    :param session: Session to use for requests.
    :return: Filename of target.
    """
    target_fn = os.path.join(target_path, os.path.basename(url))
    if session is None:
        session = setup_session()
    r = session.get(url)
    with open(target_fn, 'wb') as f:
        f.write(r.content)
    return target_fn


def setup_session():
    """Setup a default session."""
    login, password = load_credentials()
    session = pydap.cas.urs.setup_session(login, password)
    return session


def load_credentials():
    """
    Load the login data. The following sources (in that order) are considered:

    1. Environment variables `NASA_EARTHDATA_LOGIN` and `NASA_EARTHDATA_PASSWORD`
    2. The .env file in the current working directory (same variable names)
    3. The users .netrc file from a line like this::

        machine urs.earthdata.nasa.gov login some_user password ABCdef123!

    :return: username, password
    """
    # Check environment
    try:
        login = os.environ['NASA_EARTHDATA_LOGIN']
        password = os.environ['NASA_EARTHDATA_PASSWORD']
    except KeyError:
        pass
    else:
        return login, password

    # Check .env
    try:
        env = dotenv_values(dotenv_path='./.env')
        login = env['NASA_EARTHDATA_LOGIN']
        password = env['NASA_EARTHDATA_PASSWORD']
    except (FileNotFoundError, KeyError):
        pass
    else:
        return login, password

    # Check netrc (see https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget)
    try:
        ns = netrc.netrc()
        login, _, password = ns.authenticators('urs.earthdata.nasa.gov')
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    else:
        return login, password

    raise CredentialsNotFoundError('No credentials found. Looked in environment, ./.env, ~/.netrc')
