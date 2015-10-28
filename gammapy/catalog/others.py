# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Other catalogs of interest for gamma-ray astronomy
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.table import Table
from ..datasets.core import gammapy_extra
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    'load_catalog_green',
    'load_catalog_tevcat',
    'SourceCatalogATNF',
    'SourceCatalogObjectATNF',
]


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
    filename = gammapy_extra.filename('datasets/catalogs/Green_2014-05.fits.gz')
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
    filename = gammapy_extra.filename('datasets/catalogs/tevcat.fits.gz')
    return Table.read(filename)


class SourceCatalogObjectATNF(SourceCatalogObject):
    """One source from the ATNF pulsar catalog.
    """
    pass


class SourceCatalogATNF(SourceCatalog):
    """ATNF pulsar catalog.

    The `ATNF pulsar catalog <http://www.atnf.csiro.au/people/pulsar/psrcat/>`__
    is **the** collection of information on all pulsars.

    Unfortunately it's only available in a database format that can only
    be read with their software.

    This function loads a FITS copy of version 1.51 of the ATNF catalog:
    http://www.atnf.csiro.au/research/pulsar/psrcat/catalogueHistory.html

    The ``ATNF_v1.51.fits.gz`` file and ``make_atnf.py`` script are available
    `here <https://github.com/gammapy/gammapy-extra/blob/master/datasets/catalogs/>`__.
    """
    name = 'ATNF'
    description = 'ATNF pulsar catalog'
    source_object_class = SourceCatalogObjectATNF

    def __init__(self, filename=None):
        if not filename:
            filename = gammapy_extra.filename('datasets/catalogs/ATNF_v1.51.fits.gz')
            self.table = Table.read(filename)
