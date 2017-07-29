# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.io import fits
from .geom import find_and_read_bands
from .base import MapBase
from .wcs import WcsGeom

__all__ = [
    'WcsMap',
]


def find_hdu(hdulist):

    for hdu in hdulist:
        if hdu.data is not None:
            return hdu

    raise AttributeError('No Image HDU found.')


def find_image_hdu(hdulist):

    for hdu in hdulist:
        if hdu.data is not None and isinstance(hdu, fits.ImageHDU):
            return hdu

    raise AttributeError('No Image HDU found.')


def find_bintable_hdu(hdulist):

    for hdu in hdulist:
        if hdu.data is not None and isinstance(hdu, fits.BinTableHDU):
            return hdu

    raise AttributeError('No BinTable HDU found.')


class WcsMap(MapBase):
    """Base class for WCS map classes.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        A WCS geometry object.

    data : `~numpy.ndarray`
        Data array.
    """

    def __init__(self, geom, data=None):
        MapBase.__init__(self, geom, data)

    @classmethod
    def create(cls, map_type=None, npix=None, binsz=0.1, width=None,
               proj='CAR', coordsys='CEL', refpix=None,
               axes=None, skydir=None, dtype='float32'):
        """Factory method to create an empty WCS map.

        Parameters
        ----------
        map_type : str
            Internal map representation.  Valid types are `WcsMapND`/`wcs` and
            `WcsMapSparse`/`wcs-sparse`.
        npix : int or tuple or list
            Width of the map in pixels. A tuple will be interpreted as
            parameters for longitude and latitude axes.  For maps with
            non-spatial dimensions, list input can be used to define a
            different map width in each image plane.  This option
            supersedes width.
        width : float or tuple or list
            Width of the map in degrees.  A tuple will be interpreted
            as parameters for longitude and latitude axes.  For maps
            with non-spatial dimensions, list input can be used to
            define a different map width in each image plane.
        binsz : float or tuple or list
            Map pixel size in degrees.  A tuple will be interpreted
            as parameters for longitude and latitude axes.  For maps
            with non-spatial dimensions, list input can be used to
            define a different bin size in each image plane.
        skydir : tuple or `~astropy.coordinates.SkyCoord`
            Sky position of map center.  Can be either a SkyCoord
            object or a tuple of longitude and latitude in deg in the
            coordinate system of the map.
        coordsys : {'CEL', 'GAL'}, optional
            Coordinate system, either Galactic ('GAL') or Equatorial ('CEL').
        axes : list
            List of non-spatial axes.
        proj : string, optional
            Any valid WCS projection type. Default is 'CAR' (cartesian).
        refpix : tuple
            Reference pixel of the projection.  If None then this will
            be chosen to be center of the map.
        dtype : str, optional
            Data type, default is float32

        Returns
        -------
        map : `~WcsMap`
            A WCS map object.
        """
        from .wcsnd import WcsMapND
        # from .wcssparse import WcsMapSparse

        geom = WcsGeom.create(npix=npix, binsz=binsz, width=width,
                              proj=proj, skydir=skydir,
                              coordsys=coordsys, refpix=refpix, axes=axes)

        if map_type in [None, 'wcs', 'WcsMapND']:
            return WcsMapND(geom, dtype=dtype)
        elif map_type in ['wcs-sparse', 'WcsMapSparse']:
            raise NotImplementedError
        else:
            raise ValueError('Unregnized Map type: {}'.format(map_type))

    @classmethod
    def from_hdulist(cls, hdulist, **kwargs):
        """Make a WcsMap object from a FITS HDUList.

        Parameters
        ----------
        hdulist :  `~astropy.io.fits.HDUList`
            HDU list containing HDUs for map data and bands.
        hdu : str
            Name or index of the HDU with the map data.
        hdu_bands : str
            Name or index of the HDU with the BANDS table.

        Returns
        -------
        wcs_map : `WcsMap`
            Map object
        """
        extname = kwargs.get('hdu', None)
        if extname is None:
            hdu = find_hdu(hdulist)
        else:
            hdu = hdulist[extname]
        extname_bands = kwargs.get('hdu_bands', None)
        if 'BANDSHDU' in hdu.header and extname_bands is None:
            extname_bands = hdu.header['BANDSHDU']

        hdu_bands = None
        if extname_bands is not None:
            hdu_bands = hdulist[extname_bands]

        return cls.from_hdu(hdu, hdu_bands)

    def to_hdulist(self, extname=None, extname_bands='BANDS', sparse=False):

        if sparse:
            extname = 'SKYMAP' if extname is None else extname.upper()
        else:
            extname = 'PRIMARY' if extname is None else extname.upper()

        if sparse and extname == 'PRIMARY':
            raise ValueError(
                'Sparse maps cannot be written to the PRIMARY HDU.')

        hdu = self.make_hdu(extname=extname, extname_bands=extname_bands,
                            sparse=sparse)

        if extname == 'PRIMARY':
            hdulist = [hdu]
        else:
            hdulist = [fits.PrimaryHDU(), hdu]

        if self.geom.axes:
            hdulist += [self.geom.make_bands_hdu(extname=extname_bands)]
        return fits.HDUList(hdulist)

    def make_hdu(self, extname='SKYMAP', extname_bands='BANDS', sparse=False):
        """Make a FITS HDU from this map.

        Parameters
        ----------
        extname : str
            The HDU extension name.
        extname_bands : str
            The HDU extension name for BANDS table.
        sparse : bool
            Set INDXSCHM to SPARSE and sparsify the map by only
            writing pixels with non-zero amplitude.

        Returns
        -------
        hdu : `~astropy.io.fits.BinTableHDU` or `~astropy.io.fits.ImageHDU`
            HDU containing the map data.
        """
        data = self.data
        shape = data.shape
        header = self.geom.make_header()

        if self.geom.axes:
            header['BANDSHDU'] = extname_bands

        cols = []
        if sparse:

            if len(shape) == 2:
                data_flat = np.ravel(data)
                data_flat[~np.isfinite(data_flat)] = 0
                nonzero = np.where(data_flat > 0)
                cols.append(fits.Column('PIX', 'J', array=nonzero[0]))
                cols.append(fits.Column('VALUE', 'E',
                                        array=data_flat[nonzero].astype(float)))
            elif self.geom.npix[0].size == 1:
                data_flat = np.ravel(data).reshape(
                    shape[:-2] + (shape[-1] * shape[-2],))
                data_flat[~np.isfinite(data_flat)] = 0
                nonzero = np.where(data_flat > 0)
                channel = np.ravel_multi_index(nonzero[:-1], shape[:-2])
                cols.append(fits.Column('PIX', 'J', array=nonzero[-1]))
                cols.append(fits.Column('CHANNEL', 'I', array=channel))
                cols.append(fits.Column('VALUE', 'E',
                                        array=data_flat[nonzero].astype(float)))
            else:

                data_flat = []
                channel = []
                pix = []
                for i, _ in np.ndenumerate(self.geom.npix[0]):
                    data_i = np.ravel(data[i[::-1]])
                    data_i[~np.isfinite(data_i)] = 0
                    pix_i = np.where(data_i > 0)
                    data_i = data_i[pix_i]
                    data_flat += [data_i]
                    pix += pix_i
                    channel += [np.ones(data_i.size, dtype=int) *
                                np.ravel_multi_index(i[::-1], shape[:-2])]
                data_flat = np.concatenate(data_flat)
                pix = np.concatenate(pix)
                channel = np.concatenate(channel)
                cols.append(fits.Column('PIX', 'J', array=pix))
                cols.append(fits.Column('CHANNEL', 'I', array=channel))
                cols.append(fits.Column('VALUE', 'E',
                                        array=data_flat.astype(float)))

            hdu = fits.BinTableHDU.from_columns(cols, header=header,
                                                name=extname)
        elif extname == 'PRIMARY':
            hdu = fits.PrimaryHDU(data, header=header)
        else:
            hdu = fits.ImageHDU(data, header=header, name=extname)
        return hdu
