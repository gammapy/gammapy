# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import numpy as np
from astropy.io import fits
from .base import Map
from .wcs import WcsGeom
from .utils import find_hdu, find_bands_hdu

__all__ = ["WcsMap"]


class WcsMap(Map):
    """Base class for WCS map classes.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        A WCS geometry object.
    data : `~numpy.ndarray`
        Data array.
    """

    @classmethod
    def create(
        cls,
        map_type="wcs",
        npix=None,
        binsz=0.1,
        width=None,
        proj="CAR",
        coordsys="CEL",
        refpix=None,
        axes=None,
        skydir=None,
        dtype="float32",
        conv="gadf",
        meta=None,
        unit="",
    ):
        """Factory method to create an empty WCS map.

        Parameters
        ----------
        map_type : {'wcs', 'wcs-sparse'}
            Map type.  Selects the class that will be used to
            instantiate the map.
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
        conv : {'fgst-ccube','fgst-template','gadf'}, optional
            FITS format convention.  Default is 'gadf'.
        meta : `~collections.OrderedDict`
            Dictionary to store meta data.
        unit : str or `~astropy.units.Unit`
            The unit of the map

        Returns
        -------
        map : `~WcsMap`
            A WCS map object.
        """
        from .wcsnd import WcsNDMap

        # from .wcssparse import WcsMapSparse

        geom = WcsGeom.create(
            npix=npix,
            binsz=binsz,
            width=width,
            proj=proj,
            skydir=skydir,
            coordsys=coordsys,
            refpix=refpix,
            axes=axes,
            conv=conv,
        )

        if map_type == "wcs":
            return WcsNDMap(geom, dtype=dtype, meta=meta, unit=unit)
        elif map_type == "wcs-sparse":
            raise NotImplementedError
        else:
            raise ValueError("Invalid map type: {!r}".format(map_type))

    @classmethod
    def from_hdulist(cls, hdu_list, hdu=None, hdu_bands=None):
        """Make a WcsMap object from a FITS HDUList.

        Parameters
        ----------
        hdu_list :  `~astropy.io.fits.HDUList`
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
            hdu = find_hdu(hdu_list)
        else:
            hdu = hdu_list[hdu]

        if hdu_bands is None:
            hdu_bands = find_bands_hdu(hdu_list, hdu)

        if hdu_bands is not None:
            hdu_bands = hdu_list[hdu_bands]

        return cls.from_hdu(hdu, hdu_bands)

    def to_hdulist(self, hdu=None, hdu_bands=None, sparse=False, conv=None):
        """Convert to `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        hdu : str
            Name or index of the HDU with the map data.
        hdu_bands : str
            Name or index of the HDU with the BANDS table.
        sparse : bool
            Sparsify the map by only writing pixels with non-zero
            amplitude.
        conv : {'fgst-ccube','fgst-template','gadf',None}, optional
            FITS format convention.  If None this will be set to the
            default convention of the map.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`

        """
        if sparse:
            hdu = "SKYMAP" if hdu is None else hdu.upper()
        else:
            hdu = "PRIMARY" if hdu is None else hdu.upper()

        if sparse and hdu == "PRIMARY":
            raise ValueError("Sparse maps cannot be written to the PRIMARY HDU.")

        if self.geom.axes:
            hdu_bands_out = self.geom.make_bands_hdu(
                hdu=hdu_bands, hdu_skymap=hdu, conv=conv
            )
            hdu_bands = hdu_bands_out.name
        else:
            hdu_bands = None

        hdu_out = self.make_hdu(hdu=hdu, hdu_bands=hdu_bands, sparse=sparse, conv=conv)

        hdu_out.header["META"] = json.dumps(self.meta)

        hdu_out.header["BUNIT"] = self.unit.to_string("fits")

        if hdu == "PRIMARY":
            hdulist = [hdu_out]
        else:
            hdulist = [fits.PrimaryHDU(), hdu_out]

        if self.geom.axes:
            hdulist += [hdu_bands_out]

        return fits.HDUList(hdulist)

    def make_hdu(self, hdu="SKYMAP", hdu_bands=None, sparse=False, conv=None):
        """Make a FITS HDU from this map.

        Parameters
        ----------
        hdu : str
            The HDU extension name.
        hdu_bands : str
            The HDU extension name for BANDS table.
        sparse : bool
            Set INDXSCHM to SPARSE and sparsify the map by only
            writing pixels with non-zero amplitude.

        Returns
        -------
        hdu : `~astropy.io.fits.BinTableHDU` or `~astropy.io.fits.ImageHDU`
            HDU containing the map data.
        """
        header = self.geom.make_header()

        if hdu_bands is not None:
            header["BANDSHDU"] = hdu_bands

        if sparse:
            hdu_out = self._make_hdu_sparse(self.data, self.geom.npix, hdu, header)
        elif hdu == "PRIMARY":
            hdu_out = fits.PrimaryHDU(self.data, header=header)
        else:
            hdu_out = fits.ImageHDU(self.data, header=header, name=hdu)

        return hdu_out

    @staticmethod
    def _make_hdu_sparse(data, npix, hdu, header):
        shape = data.shape

        # We make a copy, because below we modify `data` to handle non-finite entries
        # TODO: The code below could probably be simplified to use expressions
        # that create new arrays instead of in-place modifications
        # But first: do we want / need the non-finite entry handling at all and always cast to 64-bit float?
        data = data.copy()

        if len(shape) == 2:
            data_flat = np.ravel(data)
            data_flat[~np.isfinite(data_flat)] = 0
            nonzero = np.where(data_flat > 0)
            value = data_flat[nonzero].astype(float)
            cols = [
                fits.Column("PIX", "J", array=nonzero[0]),
                fits.Column("VALUE", "E", array=value),
            ]
        elif npix[0].size == 1:
            shape_flat = shape[:-2] + (shape[-1] * shape[-2],)
            data_flat = np.ravel(data).reshape(shape_flat)
            data_flat[~np.isfinite(data_flat)] = 0
            nonzero = np.where(data_flat > 0)
            channel = np.ravel_multi_index(nonzero[:-1], shape[:-2])
            value = data_flat[nonzero].astype(float)
            cols = [
                fits.Column("PIX", "J", array=nonzero[-1]),
                fits.Column("CHANNEL", "I", array=channel),
                fits.Column("VALUE", "E", array=value),
            ]
        else:
            data_flat = []
            channel = []
            pix = []
            for i, _ in np.ndenumerate(npix[0]):
                data_i = np.ravel(data[i[::-1]])
                data_i[~np.isfinite(data_i)] = 0
                pix_i = np.where(data_i > 0)
                data_i = data_i[pix_i]
                data_flat += [data_i]
                pix += pix_i
                channel += [
                    np.ones(data_i.size, dtype=int)
                    * np.ravel_multi_index(i[::-1], shape[:-2])
                ]

            pix = np.concatenate(pix)
            channel = np.concatenate(channel)
            value = np.concatenate(data_flat).astype(float)

            cols = [
                fits.Column("PIX", "J", array=pix),
                fits.Column("CHANNEL", "I", array=channel),
                fits.Column("VALUE", "E", array=value),
            ]

        return fits.BinTableHDU.from_columns(cols, header=header, name=hdu)
