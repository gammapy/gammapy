# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A selection of source catalogs of interest for gamma-ray astronomers.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.utils.data import download_file
from astropy.table import Table
from ..datasets import get_path

__all__ = ['load_catalog_atnf',
           'load_catalog_green',
           'load_catalog_snrcat',
           'load_catalog_tevcat',
           ]


def load_catalog_atnf(small_sample=False):
    """Load ATNF pulsar catalog.

    The `ATNF pulsar catalog <http://www.atnf.csiro.au/people/pulsar/psrcat/>`__
    is **the** collection of information on all pulsars.


    http://www.atnf.csiro.au/research/pulsar/psrcat/catalogueHistory.html

    For details, see
    `README file for FermiVelaRegion
    <https://github.com/gammapy/gammapy-extra/blob/master/datasets/vela_region/README.rst>`_.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog
    """
    if small_sample:
        filename = get_path('atnf/atnf_sample.txt')
        return Table.read(filename, format='ascii.csv', delimiter=' ')
    else:
        filename = get_path('catalogs/ATNF_v51.fits', location='remote')
        return Table.read(filename)


def load_catalog_green():
    """Load Green's supernova remnant catalog.

    TODO: document

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog
    """
    filename = get_path('catalogs/Green_2014-05.fits', location='remote')
    return Table.read(filename)


def load_catalog_tevcat():
    """Load TeVCat source catalog.

    This is a dump of TeVCat (http://tevcat.uchicago.edu/)
    as of 2014-009-24 created by Christoph Deil using this code:
    https://github.com/astropy/astroquery/pull/41

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog
    """
    filename = get_path('tev_catalogs/tevcat.fits.gz')
    return Table.read(filename)


def load_catalog_snrcat():
    """Load SNRcat supernova remnant catalog.

    `SNRcat <http://www.physics.umanitoba.ca/snr/SNRcat/>`__
    is a census of high-energy observations of Galactic supernova remnants.

    Unfortunately the full information is not available for
    programmatic access (only as html pages via their webpage).

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog
    """
    url = 'http://www.physics.umanitoba.ca/snr/SNRcat/SNRlist.php?textonly'
    filename = download_file(url, cache=True)
    # filename = get_path('catalogs/SNRcat.fits', location='remote')
    return Table.read(filename, format='ascii')

