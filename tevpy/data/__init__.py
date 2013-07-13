# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Example and test datasets.

Example how to load a dataset from file:

    >>> from tevpy import data
    >>> image = data.poisson_stats_image()

To get a summary table of available datasets::

    >>> from tevpy import data
    >>> data.list_datasets()

To download all datasets into a local cache::

    >>> from tevpy import data
    >>> data.download_datasets()
"""
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

included_datasets = ['poisson_stats_image',
                     ]

remote_datasets = ['fermi_galactic_center'
                   ]

datasets = included_datasets + remote_datasets

__all__ = ['list_datasets',
           'download_datasets',
           ] + datasets

def list_datasets():
    """List available datasets"""
    for name in datasets:
        docstring = eval('{0}.__doc__'.format(name))
        summary = docstring.split('\n')[0]
        print('{0:>25s} : {1}'.format(name, summary))


def download_datasets(names='all'):
    """Download all datasets in to a local cache.
    
    TODO: set this up and test
    """
    for name in remote_datasets:
        raise NotImplementedError
        # Check if available in cache
        # if not download to cache


def poisson_stats_image(extra_info=False, return_filenames=False):
    """Poisson statistics counts image of a Gaussian source on flat background.

    Parameters
    ----------
    extra_info : bool
        If true, a dict of images is returned.
    
    return_filenames : bool
        If true, return filenames instead of images

    Returns
    -------
    numpy array or dict of arrays or filenames, depending on the options

    See poissson_stats_image/README.md for further info.
    TODO: add better description (extract from README?)
    """
    if extra_info:
        out = dict()
        for name in ['counts', 'model', 'source', 'background']:
            filename = 'poisson_stats_image/{0}.fits.gz'.format(name)
            filename = get_pkg_data_filename(filename)
            if return_filenames:
                out[name] = filename
            else:
                data = fits.getdata(filename)
                out[name] = data
    else:
        filename = 'poisson_stats_image/counts.fits.gz'
        filename = get_pkg_data_filename(filename)
        if return_filenames:
            out = filename
        else:
            out = fits.getdata(filename)

    return out

def fermi_galactic_center():
    """Fermi high-energy counts image of the Galactic center region.
    
    TODO: document energy band, region, ... add script to produce the image 
    TODO: download from Dropbox
    """
    raise NotImplementedError
