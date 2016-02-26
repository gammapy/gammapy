# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging

from ..extern.bunch import Bunch
from astropy.io import fits

__all__ = ['FitsMapBunch']

log = logging.getLogger(__name__)


class FitsMapBunch(Bunch):
    """
    Minimal version of future Fits map container class.
    """
    @classmethod
    def read(cls, filename):
        """
        Create Bunch of maps from Fits file.

        Parameters
        ----------
        filename : str
            Fits file name.
        """
        kwargs = {}
        
        # list of map names to save maps order in fits file
        _map_names = []
        for hdu in fits.open(filename):
            try:
                name = hdu.header['HDUNAME']
            except KeyError:
                name = hdu.name.lower()
            kwargs[name] = hdu.data
            _map_names.append(name)
        kwargs['_map_names'] = _map_names
        return cls(**kwargs)


    def write(self, filename, header=None, **kwargs):
        """
        Write Bunch of maps to Fits file.

        Parameters
        ----------
        filename : str
            Fits file name.
 
        """
        hdulist = fits.HDUList()
        try:
            names = self._map_names
        except AttributeError:
            names = sorted(self)
        for name in names:
            hdu = fits.ImageHDU(data=self[name], header=header, name=name)
            hdulist.append(hdu)
        hdulist.writeto(filename, **kwargs)