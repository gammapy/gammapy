# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import json
import numpy as np
from astropy.io import fits
from .base import Map
from .hpx import HpxGeom, HpxConv
from .utils import find_bintable_hdu, find_bands_hdu

__all__ = ["HpxMap"]


class HpxMap(Map):
    """Base class for HEALPIX map classes.

    Parameters
    ----------
    geom : `~gammapy.maps.HpxGeom`
        HEALPix geometry object.
    data : `~numpy.ndarray`
        Data array.
    meta : `~collections.OrderedDict`
        Dictionary to store meta data.
    unit : `~astropy.units.Unit`
        The map unit
    """

    def __init__(self, geom, data, meta=None, unit=""):
        super(HpxMap, self).__init__(geom, data, meta, unit)
        self._wcs2d = None
        self._hpx2wcs = None

    @classmethod
    def create(
        cls,
        nside=None,
        binsz=None,
        nest=True,
        map_type="hpx",
        coordsys="CEL",
        data=None,
        skydir=None,
        width=None,
        dtype="float32",
        region=None,
        axes=None,
        conv="gadf",
        meta=None,
        unit="",
    ):
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
        map_type : {'hpx', 'hpx-sparse'}
            Map type.  Selects the class that will be used to
            instantiate the map.
        width : float
            Diameter of the map in degrees.  If None then an all-sky
            geometry will be created.
        axes : list
            List of `~MapAxis` objects for each non-spatial dimension.
        conv : {'fgst-ccube','fgst-template','gadf'}, optional
            Default FITS format convention that will be used when
            writing this map to a file.  Default is 'gadf'.
        meta : `~collections.OrderedDict`
            Dictionary to store meta data.
        unit : str or `~astropy.units.Unit`
            The map unit

        Returns
        -------
        map : `~HpxMap`
            A HPX map object.
        """
        from .hpxnd import HpxNDMap
        from .hpxsparse import HpxSparseMap

        hpx = HpxGeom.create(
            nside=nside,
            binsz=binsz,
            nest=nest,
            coordsys=coordsys,
            region=region,
            conv=conv,
            axes=axes,
            skydir=skydir,
            width=width,
        )
        if cls.__name__ == "HpxNDMap":
            return HpxNDMap(hpx, dtype=dtype, meta=meta, unit=unit)
        elif cls.__name__ == "HpxSparseMap":
            return HpxSparseMap(hpx, dtype=dtype, meta=meta, unit=unit)
        elif map_type == "hpx":
            return HpxNDMap(hpx, dtype=dtype, meta=meta, unit=unit)
        elif map_type == "hpx-sparse":
            return HpxSparseMap(hpx, dtype=dtype, meta=meta, unit=unit)
        else:
            raise ValueError("Unrecognized map type: {!r}".format(map_type))

    @classmethod
    def from_hdulist(cls, hdu_list, hdu=None, hdu_bands=None):
        """Make a HpxMap object from a FITS HDUList.

        Parameters
        ----------
        hdu_list :  `~astropy.io.fits.HDUList`
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
            hdu_out = find_bintable_hdu(hdu_list)
        else:
            hdu_out = hdu_list[hdu]

        if hdu_bands is None:
            hdu_bands = find_bands_hdu(hdu_list, hdu_out)

        hdu_bands_out = None
        if hdu_bands is not None:
            hdu_bands_out = hdu_list[hdu_bands]

        return cls.from_hdu(hdu_out, hdu_bands_out)

    def to_hdulist(self, hdu="SKYMAP", hdu_bands=None, sparse=False, conv=None):
        """Convert to `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        hdu : str
            The HDU extension name.
        hdu_bands : str
            The HDU extension name for BANDS table.
        sparse : bool
            Set INDXSCHM to SPARSE and sparsify the map by only
            writing pixels with non-zero amplitude.
        conv : {'fgst-ccube','fgst-template','gadf',None}, optional
            FITS format convention.  If None this will be set to the
            default convention of the map.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
        """
        if self.geom.axes:
            hdu_bands_out = self.geom.make_bands_hdu(
                hdu=hdu_bands, hdu_skymap=hdu, conv=conv
            )
            hdu_bands = hdu_bands_out.name
        else:
            hdu_bands_out = None
            hdu_bands = None

        hdu_out = self.make_hdu(hdu=hdu, hdu_bands=hdu_bands, sparse=sparse, conv=conv)
        hdu_out.header["META"] = json.dumps(self.meta)
        hdu_out.header["BUNIT"] = self.unit.to_string("fits")

        hdu_list = fits.HDUList([fits.PrimaryHDU(), hdu_out])

        if self.geom.axes:
            hdu_list.append(hdu_bands_out)

        return hdu_list

    @abc.abstractmethod
    def to_wcs(
        self,
        sum_bands=False,
        normalize=True,
        proj="AIT",
        oversample=2,
        width_pix=None,
        hpx2wcs=None,
    ):
        """Make a WCS object and convert HEALPIX data into WCS projection.

        Parameters
        ----------
        sum_bands : bool
            Sum over non-spatial axes before reprojecting.  If False
            then the WCS map will have the same dimensionality as the
            HEALPix one.
        normalize : bool
            Preserve integral by splitting HEALPIX values between bins?
        proj : str
            WCS-projection
        oversample : float
            Oversampling factor for WCS map. This will be the
            approximate ratio of the width of a HPX pixel to a WCS
            pixel. If this parameter is None then the width will be
            set from ``width_pix``.
        width_pix : int
            Width of the WCS geometry in pixels.  The pixel size will
            be set to the number of pixels satisfying ``oversample``
            or ``width_pix`` whichever is smaller.  If this parameter
            is None then the width will be set from ``oversample``.
        hpx2wcs : `~HpxToWcsMapping`
            Set the HPX to WCS mapping object that will be used to
            generate the WCS map.  If none then a new mapping will be
            generated based on ``proj`` and ``oversample`` arguments.

        Returns
        -------
        map_out : `~gammapy.maps.WcsMap`
            WCS map object.
        """
        pass

    @abc.abstractmethod
    def to_swapped(self):
        """Return a new map with the opposite scheme (ring or nested).

        Returns
        -------
        map : `~HpxMap`
            Map object.
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

        Returns
        -------
        map : `~HpxMap`
            Map object.
        """
        pass

    def make_hdu(self, hdu=None, hdu_bands=None, sparse=False, conv=None):
        """Make a FITS HDU with input data.

        Parameters
        ----------
        hdu : str
            The HDU extension name.
        hdu_bands : str
            The HDU extension name for BANDS table.
        sparse : bool
            Set INDXSCHM to SPARSE and sparsify the map by only
            writing pixels with non-zero amplitude.
        conv : {'fgst-ccube', 'fgst-template', 'gadf', None}, optional
            FITS format convention.  If None this will be set to the
            default convention of the map.

        Returns
        -------
        hdu_out : `~astropy.io.fits.BinTableHDU` or `~astropy.io.fits.ImageHDU`
            Output HDU containing map data.
        """
        convname = self.geom.conv if conv is None else conv
        conv = HpxConv.create(convname)
        hduname = conv.hduname if hdu is None else hdu
        hduname_bands = conv.bands_hdu if hdu_bands is None else hdu_bands

        header = self.geom.make_header(conv=conv)

        if self.geom.axes:
            header["BANDSHDU"] = hduname_bands

        if sparse:
            header["INDXSCHM"] = "SPARSE"

        cols = []
        if header["INDXSCHM"] == "EXPLICIT":
            array = self.geom._ipix
            cols.append(fits.Column("PIX", "J", array=array))
        elif header["INDXSCHM"] == "LOCAL":
            array = np.arange(self.data.shape[-1])
            cols.append(fits.Column("PIX", "J", array=array))

        cols += self._make_cols(header, conv)
        return fits.BinTableHDU.from_columns(cols, header=header, name=hduname)
