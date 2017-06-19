# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Other catalogs of interest for gamma-ray astronomy
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.table import Table
from ..utils.scripts import make_path
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    'load_catalog_green',
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
    filename = make_path('$GAMMAPY_EXTRA/datasets/catalogs/Green_2014-05.fits.gz')
    return Table.read(filename)


# TODO: remove, or integrate with gammapy.astro.source.Pulsar !
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

    This function loads a FITS copy of version 1.54 of the ATNF catalog:
    http://www.atnf.csiro.au/research/pulsar/psrcat/catalogueHistory.html

    The ``ATNF_v1.54.fits.gz`` file and ``make_atnf.py`` script are available
    `here <https://github.com/gammapy/gammapy-extra/blob/master/datasets/catalogs/>`__.
    """
    name = 'ATNF'
    description = 'ATNF pulsar catalog'
    source_object_class = SourceCatalogObjectATNF

    def __init__(self, filename=None):
        filename = filename or make_path('$GAMMAPY_EXTRA/datasets/catalogs/ATNF_v1.54.fits.gz')
        self.table = Table.read(filename)
