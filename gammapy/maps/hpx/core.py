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
    meta : dict
        Dictionary to store metadata.
    unit : `~astropy.units.Unit`
        The map unit.
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
        nside : int or `~numpy.ndarray`, optional
            HEALPix NSIDE parameter. This parameter sets the size of
            the spatial pixels in the map. Default is None.
        binsz : float or `~numpy.ndarray`, optional
            Approximate pixel size in degrees. An NSIDE will be
            chosen that corresponds to a pixel size closest to this
            value. This option is superseded by ``nside``.
            Default is None.
        nest : bool, optional
            Indexing scheme. If True, "NESTED" scheme. If False, "RING" scheme.
            Default is True.
        map_type : {'hpx', 'hpx-sparse'}, optional
            Map type. Selects the class that will be used to
            instantiate the map. Default is "hpx".
        frame : {"icrs", "galactic"}
            Coordinate system, either Galactic ("galactic") or Equatorial ("icrs").
            Default is "icrs".
        data : `~numpy.ndarray`, optional
            Data array. Default is None.
        skydir : tuple or `~astropy.coordinates.SkyCoord`, optional
            Sky position of map center. Can be either a SkyCoord
            object or a tuple of longitude and latitude in deg in the
            coordinate system of the map. Default is None.
        width : float, optional
            Diameter of the map in degrees. If None then an all-sky
            geometry will be created. Default is None.
        dtype : str, optional
            Data type. Default is "float32".
        region : str, optional
            HEALPix region string. Default is None.
        axes : list, optional
            List of `~MapAxis` objects for each non-spatial dimension.
            Default is None.
        meta : `dict`, optional
            Dictionary to store the metadata. Default is None.
        unit : str or `~astropy.units.Unit`, optional
            The map unit. Default is "".

        Returns
        -------
        map : `~HpxMap`
            A HEALPix map object.
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
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list containing HDUs for map data and bands.
        hdu : str, optional
            Name or index of the HDU with the map data. If None then
            the method will try to load map data from the first
            BinTableHDU in the file.
            Default is None.
        hdu_bands : str, optional
            Name or index of the HDU with the BANDS table.
            Default is None.
        format : str, optional
            FITS format convention. By default, files will be written
            to the gamma-astro-data-formats (GADF) format. This
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
            Map object.
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
        hdu : str, optional
            The HDU extension name. Default is "SKYMAP".
        hdu_bands : str, optional
            The HDU extension name for BANDS table.
            Default is None.
        sparse : bool, optional
            Set INDXSCHM to SPARSE and sparsify the map by only
            writing pixels with non-zero amplitude.
            Default is False.
        format : str, optional
            FITS format convention. By default, files will be written
            to the gamma-astro-data-formats (GADF) format. This
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
            The FITS HDUList.
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
        """Make a WCS object and convert HEALPix data into WCS projection.

        Parameters
        ----------
        sum_bands : bool, optional
            Sum over non-spatial axes before reprojecting. If False
            then the WCS map will have the same dimensionality as the
            HEALPix one. Default is False.
        normalize : bool, optional
            Preserve integral by splitting HEALPix values between bins.
            Default is True.
        proj : str, optional
            WCS-projection. Default is "AIT".
        oversample : float, optional
            Oversampling factor for WCS map. This will be the
            approximate ratio of the width of a HEALPix pixel to a WCS
            pixel. If this parameter is None then the width will be
            set from ``width_pix``. Default is 2.
        width_pix : int, optional
            Width of the WCS geometry in pixels. The pixel size will
            be set to the number of pixels satisfying ``oversample``
            or ``width_pix`` whichever is smaller. If this parameter
            is None then the width will be set from ``oversample``.
            Default is None.
        hpx2wcs : `~HpxToWcsMapping`, optional
            Set the HEALPix to WCS mapping object that will be used to
            generate the WCS map. If None then a new mapping will be
            generated based on ``proj`` and ``oversample`` arguments.
            Default is None.

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
        hdu : str, optional
            The HDU extension name. Default is None.
        hdu_bands : str, optional
            The HDU extension name for BANDS table. Default is None.
        sparse : bool, optional
            Set INDXSCHM to SPARSE and sparsify the map by only
            writing pixels with non-zero amplitude.
            Default is False.
        format : {None, 'fgst-ccube', 'fgst-template', 'gadf'}
            FITS format convention. If None this will be set to the
            default convention of the map. Default is None.

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
