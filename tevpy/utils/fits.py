# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""FITS utility functions"""
from astropy.io import fits

__all__ = ['get_hdu', 'get_image_hdu', 'get_table_hdu']


def get_hdu(location):
    """Get one HDU for a given location

    location should be either a `file_name` or a file
    and HDU name in the format `file_name[hdu_name]`.
    """
    # TODO: Test all cases and give good exceptions / error messages
    if '[' in location:
        tokens = location.split('[')
        file_name = tokens[0]
        hdu_name = tokens[1][:-1]  # split off ']' at the end
        return fits.open(file_name)[hdu_name]
    else:
        file_name = location
        return fits.open(file_name)[0]


def get_image_hdu():
    """Get the first image HDU"""
    raise NotImplementedError


def get_table_hdu():
    """Get the first table HDU"""
    raise NotImplementedError
