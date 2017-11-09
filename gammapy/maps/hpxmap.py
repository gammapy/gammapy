# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import re
import numpy as np
from astropy.io import fits
from .base import MapBase
from .hpx import HpxGeom, HpxConv
from .geom import MapAxis, find_and_read_bands
from .utils import find_bintable_hdu, find_bands_hdu

__all__ = [
    'HpxMap',
]


class HpxMap(MapBase):
    """Base class for HEALPIX map classes.

    Parameters
    ----------
    geom : `~gammapy.maps.HpxGeom`
        HEALPix geometry object.
    data : `~numpy.ndarray`
        Data array.
    """

    def __init__(self, geom, data):
        super(HpxMap, self).__init__(geom, data)
        self._wcs2d = None
        self._hpx2wcs = None

    @classmethod
    def create(cls, nside=None, binsz=None, nest=True, map_type=None, coordsys='CEL',
               data=None, skydir=None, width=None, dtype='float32',
               region=None, axes=None, conv='gadf'):
        """Factory method to create an empty HEALPix map.

        Parameters
        ----------
        nside : int or `~numpy.ndarray`
            HEALPix NSIDE parameter.  This parameter sets the size of
            the spatial pixels in the map.
        binsz : float or `~numpy.ndarray`
            Approximate pixel size in degrees.  An NSIDE will be
            chosen that correponds to a pixel size closest to this
            value.  This option is superseded by nside.
        nest : bool
            True for HEALPix "NESTED" indexing scheme, False for "RING" scheme.
        coordsys : {'CEL', 'GAL'}, optional
            Coordinate system, either Galactic ('GAL') or Equatorial ('CEL').
        skydir : tuple or `~astropy.coordinates.SkyCoord`
            Sky position of map center.  Can be either a SkyCoord
            object or a tuple of longitude and latitude in deg in the
            coordinate system of the map.
        map_type : str
            Internal map representation.  Valid types are `HpxMapND`/`hpx` and
            `HpxMapSparse`/`hpx-sparse`.
        width : float
            Diameter of the map in degrees.  If None then an all-sky
            geometry will be created.
        axes : list
            List of `~MapAxis` objects for each non-spatial dimension.
        conv : str, optional
            FITS format convention ('fgst-ccube', 'fgst-template',
            'gadf').  Default is 'gadf'.
        """
        from .hpxnd import HpxMapND
        from .hpxsparse import HpxMapSparse

        hpx = HpxGeom.create(nside=nside, binsz=binsz,
                             nest=nest, coordsys=coordsys, region=region,
                             conv=conv, axes=axes, skydir=skydir, width=width)
        if cls.__name__ == 'HpxMapND':
            return HpxMapND(hpx, dtype=dtype)
        elif cls.__name__ == 'HpxMapSparse':
            return HpxMapSparse(hpx, dtype=dtype)
        elif map_type in [None, 'hpx', 'HpxMapND']:
            return HpxMapND(hpx, dtype=dtype)
        elif map_type in ['hpx-sparse', 'HpxMapSparse']:
            return HpxMapSparse(hpx, dtype=dtype)
        else:
            raise ValueError('Unregnized Map type: {}'.format(map_type))

    @classmethod
    def from_hdulist(cls, hdulist, hdu=None, hdu_bands=None):
        """Make a HpxMap object from a FITS HDUList.

        Parameters
        ----------
        hdulist :  `~astropy.io.fits.HDUList`
            HDU list containing HDUs for map data and bands.
        hdu : str        
            Name or index of the HDU with the map data.  If None then
            the method will try to load map data from the first
            BinTableHDU in the file.            
        hdu_bands : str
            Name or index of the HDU with the BANDS table.

        Returns
        -------
        hpx_map : `HpxMap`
            Map object
        """
        if hdu is None:
            hdu = find_bintable_hdu(hdulist)
        else:
            hdu = hdulist[hdu]

        if hdu_bands is None:
            hdu_bands = find_bands_hdu(hdulist, hdu)

        if hdu_bands is not None:
            hdu_bands = hdulist[hdu_bands]

        return cls.from_hdu(hdu, hdu_bands)

    def to_hdulist(self, **kwargs):

        extname = kwargs.get('extname', 'SKYMAP')
        # extname_bands = kwargs.get('extname_bands', self.geom.conv.bands_hdu)
        extname_bands = kwargs.get('extname_bands', 'BANDS')
        hdulist = [fits.PrimaryHDU(), self.make_hdu(**kwargs)]

        if self.geom.axes:
            hdulist += [self.geom.make_bands_hdu(extname=extname_bands)]
        return fits.HDUList(hdulist)

    @abc.abstractmethod
    def to_wcs(self, sum_bands=False, normalize=True):
        """Make a WCS object and convert HEALPIX data into WCS projection.

        Parameters
        ----------
        sum_bands : bool
            Sum over non-spatial axes before reprojecting.  If False
            then the WCS map will have the same dimensionality as the
            HEALPix one.
        normalize : bool
            True -> preserve integral by splitting HEALPIX values between bins

        Returns
        -------
        wcs : `~astropy.wcs.WCS`
            WCS object
        data : `~numpy.ndarray`
            Reprojected data
        """
        pass

    @abc.abstractmethod
    def to_swapped_scheme(self):
        """Return a new map with the opposite scheme (ring or nested).
        """
        pass

    @abc.abstractmethod
    def to_ud_graded(self, nside, preserve_counts=False):
        """Upgrade or downgrade the resolution of the map to the chosen nside.

        Parameters
        ----------
        nside : int
            NSIDE parameter of the new map.

        preserve_counts : bool
            Choose whether to preserve counts (total amplitude) or
            intensity (amplitude per unit solid angle).
        """
        pass

    def make_hdu(self, **kwargs):
        """Make a FITS HDU with input data.

        Parameters
        ----------
        extname : str
            The HDU extension name.
        extname_bands : str
            The HDU extension name for BANDS table.
        colbase : str
            The prefix for column names
        sparse : bool
            Set INDXSCHM to SPARSE and sparsify the map by only
            writing pixels with non-zero amplitude.
        """
        # FIXME: Should this be a method of HpxMapND?
        # FIXME: Should we assign extname in this method?

        from .hpxsparse import HpxMapSparse

        conv = kwargs.get('conv', HpxConv.create('gadf'))

        data = self.data
        shape = data.shape
        extname = kwargs.get('extname', conv.extname)
        extname_bands = kwargs.get('extname_bands', conv.bands_hdu)

        sparse = kwargs.get('sparse', True if isinstance(self, HpxMapSparse)
                            else False)
        header = self.geom.make_header()

        if self.geom.axes:
            header['BANDSHDU'] = extname_bands

        if sparse:
            header['INDXSCHM'] = 'SPARSE'

        cols = []
        if header['INDXSCHM'] == 'EXPLICIT':
            cols.append(fits.Column('PIX', 'J', array=self.geom._ipix))
        elif header['INDXSCHM'] == 'LOCAL':
            cols.append(fits.Column('PIX', 'J',
                                    array=np.arange(data.shape[-1])))

        cols += self._make_cols(header, conv)
        hdu = fits.BinTableHDU.from_columns(cols, header=header, name=extname)
        return hdu
