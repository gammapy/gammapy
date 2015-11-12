# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .core import SourceCatalog, SourceCatalogObject
import os
from ..extern.pathlib import Path
from astropy.io import fits
from astropy.table import Table

__all__ = [
    'SourceCatalogHGPS',
    'SourceCatalogObjectHGPS',
]


class SourceCatalogObjectHGPS(SourceCatalogObject):
    """One object from the HGPS catalog.
    """
    pass


class SourceCatalogHGPS(SourceCatalog):
    """HESS Galactic plane survey (HGPS) source catalog.

    Note: this catalog isn't publicly available yet.
    For now you need to be a H.E.S.S. member with an account
    at MPIK to fetch it.
    """
    name = 'hgps'
    description = 'H.E.S.S. Galactic plane survey (HGPS) source catalog'
    source_object_class = SourceCatalogObjectHGPS

    def __init__(self, filename=None):
        if not filename:
            filename = Path(os.environ['HGPS_ANALYSIS']) / 'data/catalogs/HGPS3/HGPS_v0.3.1.fits'

        self.hdu_list = fits.open(str(filename))
        table = Table(self.hdu_list['HGPS_SOURCES'].data)

        self.components = Table(self.hdu_list['HGPS_COMPONENTS'].data)
        self.associations = Table(self.hdu_list['HGPS_ASSOCIATIONS'].data)
        super(SourceCatalogHGPS, self).__init__(table=table)
