# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging

from ..extern.bunch import Bunch
from astropy.io import fits

__all__ = ['MapsBunch']

log = logging.getLogger(__name__)


class MapsBunch(Bunch):
    """
    Minimal version of a future map container class.
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
        hdulist = fits.open(filename)
        kwargs = {}
        _map_names = []  # list of map names to save order in fits file
        kwargs['_ref_header'] = hdulist[0].header

        for hdu in hdulist:
            name =  hdu.header.get('HDUNAME', hdu.name.lower())
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
        header : `~astropy.io.fits.Header`
            Reference header to be used for all maps. 
        """
        hdulist = fits.HDUList()
        header = header or self.get('_ref_header')
        for name in self.get('_map_names', sorted(self)):
            hdu = fits.ImageHDU(data=self[name], header=header, name=name)
            hdulist.append(hdu)
        hdulist.writeto(filename, **kwargs)