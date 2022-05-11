# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import json
import numpy as np
import astropy.units as u
from astropy.io import fits
from ..core import Map
from ..io import find_bands_hdu, find_bintable_hdu
from .geom import HpxGeom
from .io import HpxConv

__all__ = ["HpxMap"]


class HpxMap(Map):
    """Base class for HEALPIX map classes.

    Parameters
    ----------
    geom : `~gammapy.maps.HpxGeom`
        HEALPix geometry object.
    data : `~numpy.ndarray`
        Data array.
    meta : `dict`
        Dictionary to store meta data.
    unit : `~astropy.units.Unit`
        The map unit
    """

    @classmethod
    def create(
        cls,
        nside=None,
        binsz=None,
        nest=True,
        map_type="hpx",
        frame="icrs",
        data=None,
        skydir=None,
        width=None,
        dtype="float32",
        region=None,
        axes=None,
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
            chosen that corresponds to a pixel size closest to this
            value.  This option is superseded by nside.
        nest : bool
            True for HEALPix "NESTED" indexing scheme, False for "RING" scheme.
        frame : {"icrs", "galactic"}, optional
            Coordinate system, either Galactic ("galactic") or Equatorial ("icrs").
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
        meta : `dict`
            Dictionary to store meta data.
        unit : str or `~astropy.units.Unit`
            The map unit

        Returns
        -------
        map : `~HpxMap`
            A HPX map object.
        """
        from .ndmap import HpxNDMap

        hpx = HpxGeom.create(
            nside=nside,
            binsz=binsz,
            nest=nest,
            frame=frame,
            region=region,
            axes=axes,
            skydir=skydir,
            width=width,
        )
        if cls.__name__ == "HpxNDMap":
            return HpxNDMap(hpx, dtype=dtype, meta=meta, unit=unit)
        elif map_type == "hpx":
            return HpxNDMap(hpx, dtype=dtype, meta=meta, unit=unit)
        else:
            raise ValueError(f"Unrecognized map type: {map_type!r}")

    @classmethod
    def from_hdulist(
        cls, hdu_list, hdu=None, hdu_bands=None, format=None, colname=None
    ):
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
        format : str, optional
            FITS format convention.  By default files will be written
            to the gamma-astro-data-formats (GADF) format.  This
            option can be used to write files that are compliant with
            format conventions required by specific software (e.g. the
            Fermi Science Tools). The following formats are supported:

                - "gadf" (default)
                - "fgst-ccube"
                - "fgst-ltcube"
                - "fgst-bexpcube"
                - "fgst-srcmap"
                - "fgst-template"
                - "fgst-srcmap-sparse"
                - "galprop"
                - "galprop2"

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

        if format is None:
            format = HpxConv.identify_hpx_format(hdu_out.header)

        hpx_map = cls.from_hdu(hdu_out, hdu_bands_out, format=format, colname=colname)

        # exposure maps have an additional GTI hdu
        if format == "fgst-bexpcube" and "GTI" in hdu_list:
            hpx_map._unit = u.Unit("cm2 s")

        return hpx_map

    def to_hdulist(self, hdu="SKYMAP", hdu_bands=None, sparse=False, format="gadf"):
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
        format : str, optional
            FITS format convention.  By default files will be written
            to the gamma-astro-data-formats (GADF) format.  This
            option can be used to write files that are compliant with
            format conventions required by specific software (e.g. the
            Fermi Science Tools). The following formats are supported:

                - "gadf" (default)
                - "fgst-ccube"
                - "fgst-ltcube"
                - "fgst-bexpcube"
                - "fgst-srcmap"
                - "fgst-template"
                - "fgst-srcmap-sparse"
                - "galprop"
                - "galprop2"

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
        """
        if hdu_bands is None:
            hdu_bands = f"{hdu.upper()}_BANDS"

        if self.geom.axes:
            hdu_bands_out = self.geom.to_bands_hdu(hdu_bands=hdu_bands, format=format)
            hdu_bands = hdu_bands_out.name
        else:
            hdu_bands_out = None
            hdu_bands = None

        hdu_out = self.to_hdu(
            hdu=hdu, hdu_bands=hdu_bands, sparse=sparse, format=format
        )
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

    def to_hdu(self, hdu=None, hdu_bands=None, sparse=False, format=None):
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
        format : {'fgst-ccube', 'fgst-template', 'gadf', None}, optional
            FITS format convention.  If None this will be set to the
            default convention of the map.

        Returns
        -------
        hdu_out : `~astropy.io.fits.BinTableHDU` or `~astropy.io.fits.ImageHDU`
            Output HDU containing map data.
        """
        hpxconv = HpxConv.create(format)
        hduname = hpxconv.hduname if hdu is None else hdu
        hduname_bands = hpxconv.bands_hdu if hdu_bands is None else hdu_bands

        header = self.geom.to_header(format=format)

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

        cols += self._make_cols(header, hpxconv)
        return fits.BinTableHDU.from_columns(cols, header=header, name=hduname)
