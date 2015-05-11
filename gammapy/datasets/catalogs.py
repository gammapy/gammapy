# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A selection of source catalogs of interest for gamma-ray astronomers.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.utils.data import download_file
from astropy.coordinates import Angle
from astropy.table import Table, Column
from ..datasets import get_path

__all__ = ['load_catalog_atnf',
           'load_catalog_hess_galactic',
           # 'load_catalog_hgps',
           'load_catalog_green',
           'fetch_catalog_snrcat',
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


def fetch_catalog_snrcat(cache=True):
    """Fetch SNRcat supernova remnant catalog.

    `SNRcat <http://www.physics.umanitoba.ca/snr/SNRcat/>`__
    is a census of high-energy observations of Galactic supernova remnants.

    This function obtains the CSV table from
    http://www.physics.umanitoba.ca/snr/SNRcat/SNRdownload.php?table=SNR
    and adds some useful columns and unit information.

    Note that this only represents a subset of the information available in SNRCat,
    to get at the rest (e.g. observations and papers) we would have to scrape their
    web pages.

    Parameters
    ----------
    cache : bool, optional
        Whether to use the cache

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog
    """
    url = 'http://www.physics.umanitoba.ca/snr/SNRcat/SNRdownload.php?table=SNR'
    filename = download_file(url, cache=cache)

    # Note: currently the first line contains this comment, which we skip via `header_start=1`
    # This file was downloaded on 2015-05-11T03:00:55 CDT from http://www.physics.umanitoba.ca/snr/SNRcat
    table = Table.read(filename, format='ascii.csv', header_start=1, delimiter=';')

    # Fix the columns to have common column names and add unit info

    table.rename_column('G', 'Source_Name')
    table.rename_column('J', 'Source_JName')

    table.rename_column('G_lon', 'GLON')
    table['GLON'].unit = 'deg'
    table.rename_column('G_lat', 'GLAT')
    table['GLAT'].unit = 'deg'

    table.rename_column('J_ra', 'RAJ2000_str')
    table.rename_column('J_dec', 'DEJ2000_str')

    data = Angle(table['RAJ2000_str'], unit='hour').deg
    index = table.index_column('RAJ2000_str') + 1
    table.add_column(Column(data=data, name='RAJ2000', unit='deg'), index=index)

    data = Angle(table['DEJ2000_str'], unit='deg').deg
    index = table.index_column('DEJ2000_str') + 1
    table.add_column(Column(data=data, name='DEJ2000', unit='deg'), index=index)

    table.rename_column('age_min (yr)', 'age_min')
    table['age_min'].unit = 'year'
    table.rename_column('age_max (yr)', 'age_max')
    table['age_max'].unit = 'year'
    distance = np.mean([table['age_min'], table['age_max']], axis=0)
    index = table.index_column('age_max') + 1
    table.add_column(Column(distance, name='age', unit='year'), index=index)

    table.rename_column('distance_min (kpc)', 'distance_min')
    table['distance_min'].unit = 'kpc'
    table.rename_column('distance_max (kpc)', 'distance_max')
    table['distance_max'].unit = 'kpc'
    distance = np.mean([table['distance_min'], table['distance_max']], axis=0)
    index = table.index_column('distance_max') + 1
    table.add_column(Column(distance, name='distance', unit='kpc'), index=index)

    table.rename_column('size_radio', 'size_radio_str')
    table.rename_column('size_X', 'size_xray_str')

    table.rename_column('size_coarse (arcmin)', 'size_radio_mean')
    table['size_radio_mean'].unit = 'arcmin'

    # TODO: maybe we should parse the size_xray_str to get a float column?
    # https://github.com/gammapy/gammapy-extra/blob/master/datasets/catalogs/green_snrcat_check.ipynb

    return table
