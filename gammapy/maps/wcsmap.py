# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.io import fits
from .geom import find_and_read_bands
from .base import MapBase
from .wcs import WcsGeom
from .utils import find_hdu, find_bands_hdu

__all__ = [
    'WcsMap',
]


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
               axes=None, skydir=None, dtype='float32', conv=None):
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
        conv : str, optional
            FITS format convention ('fgst-ccube', 'fgst-template',
            'gadf').  Default is 'gadf'.

        Returns
        -------
        map : `~WcsMap`
            A WCS map object.
        """
        from .wcsnd import WcsMapND
        # from .wcssparse import WcsMapSparse

        geom = WcsGeom.create(npix=npix, binsz=binsz, width=width,
                              proj=proj, skydir=skydir,
                              coordsys=coordsys, refpix=refpix, axes=axes,
                              conv=conv)

        if map_type in [None, 'wcs', 'WcsMapND']:
            return WcsMapND(geom, dtype=dtype)
        elif map_type in ['wcs-sparse', 'WcsMapSparse']:
            raise NotImplementedError
        else:
            raise ValueError('Unregnized Map type: {}'.format(map_type))

    @classmethod
    def from_hdulist(cls, hdulist, hdu=None, hdu_bands=None):
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
        if hdu is None:
            hdu = find_hdu(hdulist)
        else:
            hdu = hdulist[hdu]

        if hdu_bands is None:
            hdu_bands = find_bands_hdu(hdulist, hdu)

        if hdu_bands is not None:
            hdu_bands = hdulist[hdu_bands]

        return cls.from_hdu(hdu, hdu_bands)

    def to_hdulist(self, extname=None, extname_bands=None, sparse=False,
                   conv=None):

        if sparse:
            extname = 'SKYMAP' if extname is None else extname.upper()
        else:
            extname = 'PRIMARY' if extname is None else extname.upper()

        if sparse and extname == 'PRIMARY':
            raise ValueError(
                'Sparse maps cannot be written to the PRIMARY HDU.')

        if self.geom.axes:
            bands_hdu = self.geom.make_bands_hdu(extname=extname_bands,
                                                 conv=conv)
            extname_bands = bands_hdu.name

        hdu = self.make_hdu(extname=extname, extname_bands=extname_bands,
                            sparse=sparse, conv=conv)

        if extname == 'PRIMARY':
            hdulist = [hdu]
        else:
            hdulist = [fits.PrimaryHDU(), hdu]

        if self.geom.axes:
            hdulist += [bands_hdu]
        return fits.HDUList(hdulist)

    def make_hdu(self, extname='SKYMAP', extname_bands=None, sparse=False,
                 conv=None):
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

        if extname_bands is not None:
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
