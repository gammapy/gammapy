# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A selection of source catalogs of interest for gamma-ray astronomers.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.table import Table
from ..datasets import get_path

__all__ = ['load_catalog_atnf',
           'load_catalog_hess_galactic',
           # 'load_catalog_hgps',
           'load_catalog_green',
           'load_catalog_snrcat',
           'load_catalog_tevcat',
           ]


def load_catalog_atnf():
    """Load ATNF pulsar catalog.

    The `ATNF pulsar catalog <http://www.atnf.csiro.au/people/pulsar/psrcat/>`__
    is **the** collection of information on all pulsars.

    Unfortunately it's only available in a database format that can only
    be read with their software.

    This function loads a FITS copy of version 1.51 of the ATNF catalog:
    http://www.atnf.csiro.au/research/pulsar/psrcat/catalogueHistory.html

    The ``ATNF_v1.51.fits.gz`` file and ``make_atnf.py`` script are available
    `here <https://github.com/gammapy/gammapy-extra/blob/master/datasets/catalogs/>`__.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog
    """
    filename = get_path('catalogs/ATNF_v1.51.fits.gz', location='remote')
    return Table.read(filename)


def load_catalog_hess_galactic():
    """Load catalog with info on HESS Galactic sources from individual publications.

    Note that this is different from the
    `official "H.E.S.S. source catalog" <http://www.mpi-hd.mpg.de/hfm/HESS/pages/home/sources/>`__.

    The main difference is that we only include Galactic sources (and the ones
    from the large Magellanic cloud) and that we collect more information
    (morphological and spectral parameters, associations, paper links to ADS),
    using the latest publication for each source.

    Please drop us an email if you have any questions or find something incorrect or outdated.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog
    """
    filename = get_path('catalogs/hess_galactic_catalog.fits.gz', location='remote')
    return Table.read(filename)


def load_catalog_hgps():
    """Load the HESS Galactic plane survey (HGPS) catalog.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog
    """
    raise NotImplementedError


def load_catalog_green():
    """Load Green's supernova remnant catalog.

    This is the May 2014 version of the catalog, which contains 294 sources.

    References:

    - http://www.mrao.cam.ac.uk/surveys/snrs/
    - http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=VII/272
    - http://adsabs.harvard.edu/abs/2014BASI...42...47G

    The ``Green_2014-05.fits.gz`` file and ``make_green.py`` script are available
    `here <https://github.com/gammapy/gammapy-extra/blob/master/datasets/catalogs/>`__.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog
    """
    filename = get_path('catalogs/Green_2014-05.fits.gz', location='remote')
    return Table.read(filename)


def load_catalog_tevcat():
    """Load TeVCat source catalog.

    `TeVCat <http://tevcat.uchicago.edu/>`__ is an online catalog
    for TeV astronomy.

    Unfortunately the TeVCat is not available in electronic format.

    This is a dump of TeVCat as of 2014-09-24 created by scraping their
    web page using the script available
    `here <https://github.com/astropy/astroquery/pull/41>`__.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog
    """
    filename = get_path('catalogs/tevcat.fits.gz', location='remote')
    return Table.read(filename)


def load_catalog_snrcat():
    """Load SNRcat supernova remnant catalog.

    `SNRcat <http://www.physics.umanitoba.ca/snr/SNRcat/>`__
    is a census of high-energy observations of Galactic supernova remnants.

    Unfortunately the SNRCat information is not available in electronic format.

    This is a dump of http://www.physics.umanitoba.ca/snr/SNRcat/SNRlist.php?textonly from 2015-02-08.
    It doesn't contain position and extension columns and is really a HTML page with lots of hidden stuff that
    would have to be scraped, i.e. we can't be simply ``requests.get`` the latest version from there ... sigh.
    I've contacted them and they might provide a useful version of their catalog in the future ...

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog
    """
    filename = get_path('catalogs/SNRCat.csv', location='remote')
    return Table.read(filename, format='ascii.csv')

