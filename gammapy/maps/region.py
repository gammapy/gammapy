import copy
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area, wcs_to_celestial_frame
from regions import FITSRegionParser, fits_region_objects_to_table
from gammapy.utils.regions import (
    compound_region_to_list,
    list_to_compound_region,
    make_region,
)
from .core import MapCoord
from .geom import Geom, MapAxes, MapAxis, pix_tuple_to_idx
from .wcs import WcsGeom

__all__ = ["RegionGeom"]


class RegionGeom(Geom):
    """Map geometry representing a region on the sky.

    Parameters
    ----------
    region : `~regions.SkyRegion`
        Region object.
    axes : list of `MapAxis`
        Non-spatial data axes.
    wcs : `~astropy.wcs.WCS`
        Optional wcs object to project the region if needed.
    """

    is_image = False
    is_regular = True
    is_allsky = False
    is_hpx = False
    _slice_spatial_axes = slice(0, 2)
    _slice_non_spatial_axes = slice(2, None)
    projection = "TAN"
    binsz = 0.01

    def __init__(self, region, axes=None, wcs=None):
        self._region = region
        self._axes = MapAxes.from_default(axes)

        if wcs is None and region is not None:
            wcs = WcsGeom.create(
                skydir=region.center,
                binsz=self.binsz,
                proj=self.projection,
                frame=self.frame,
            ).wcs

        self._wcs = wcs
        self.ndim = len(self.data_shape)

    @property
    def frame(self):
        if self.region is None:
            return "icrs"
        try:
            return self.region.center.frame.name
        except AttributeError:
            return wcs_to_celestial_frame(self.wcs).name

    @property
    def width(self):
        """Width of bounding box of the region"""
        if self.region is None:
            raise ValueError("Region definition required.")

        regions = compound_region_to_list(self.region)
        regions_pix = [_.to_pixel(self.wcs) for _ in regions]

        bbox = regions_pix[0].bounding_box

        for region_pix in regions_pix[1:]:
            bbox = bbox.union(region_pix.bounding_box)

        rectangle_pix = bbox.to_region()
        rectangle = rectangle_pix.to_sky(self.wcs)
        return u.Quantity([rectangle.width, rectangle.height])

    @property
    def region(self):
        return self._region

    @property
    def axes(self):
        return self._axes

    @property
    def wcs(self):
        return self._wcs

    @property
    def center_coord(self):
        """(`astropy.coordinates.SkyCoord`)"""
        return self.pix_to_coord(self.center_pix)

    @property
    def center_pix(self):
        return tuple((np.array(self.data_shape) - 1.0) / 2)[::-1]

    @property
    def center_skydir(self):
        """Center skydir"""
        if self.region is None:
            return SkyCoord(np.nan * u.deg, np.nan * u.deg)
        try:
            return self.region.center
        except AttributeError:
            xp, yp = self.wcs.wcs.crpix
            return SkyCoord.from_pixel(xp=xp, yp=yp, wcs=self.wcs)

    def contains(self, coords):
        if self.region is None:
            raise ValueError("Region definition required.")

        coords = MapCoord.create(coords, frame=self.frame, axis_names=self.axes.names)
        return self.region.contains(coords.skycoord, self.wcs)

    def separation(self, position):
        return self.center_skydir.separation(position)

    @property
    def data_shape(self):
        """Shape of the Numpy data array matching this geometry."""
        return self._shape[::-1]

    @property
    def _shape(self):
        return tuple((1, 1) + self.axes.shape)

    def get_coord(self, frame=None):
        """Get map coordinates from the geometry.

        Returns
        -------
        coord : `~MapCoord`
            Map coordinate object.
        """
        # TODO: support mode=edges?
        cdict = {}
        cdict["skycoord"] = self.center_skydir.reshape((1, 1))

        if self.axes is not None:
            coords = []
            for ax in self.axes:
                coords.append(ax.center)  # .reshape((-1, 1, 1)))

            coords = np.meshgrid(*coords)
            for idx, ax in enumerate(self.axes):
                cdict[ax.name] = coords[idx].reshape(self.data_shape)

        if frame is None:
            frame = self.frame

        return MapCoord.create(cdict, frame=self.frame).to_frame(frame)

    def pad(self):
        raise NotImplementedError("Padding of `RegionGeom` not supported")

    def crop(self):
        raise NotImplementedError("Cropping of `RegionGeom` not supported")

    def solid_angle(self):
        if self.region is None:
            raise ValueError("Region definition required.")

        area = self.region.to_pixel(self.wcs).area
        solid_angle = area * proj_plane_pixel_area(self.wcs) * u.deg ** 2
        return solid_angle.to("sr")

    def bin_volume(self):
        bin_volume = self.solid_angle() * np.ones(self.data_shape)

        for idx, ax in enumerate(self.axes):
            shape = self.ndim * [1]
            shape[-(idx + 3)] = -1
            bin_volume = bin_volume * ax.bin_width.reshape(tuple(shape))

        return bin_volume

    def to_cube(self, axes):
        axes = copy.deepcopy(self.axes) + axes
        return self._init_copy(axes=axes)

    def to_image(self):
        return self._init_copy(axes=None)

    def upsample(self, factor, axis_name):
        axes = self.axes.upsample(factor=factor, axis_name=axis_name)
        return self._init_copy(axes=axes)

    def downsample(self, factor, axis_name):
        axes = self.axes.downsample(factor=factor, axis_name=axis_name)
        return self._init_copy(axes=axes)

    def pix_to_coord(self, pix):
        lon = np.where(
            (-0.5 < pix[0]) & (pix[0] < 0.5),
            self.center_skydir.data.lon,
            np.nan * u.deg,
        )
        lat = np.where(
            (-0.5 < pix[1]) & (pix[1] < 0.5),
            self.center_skydir.data.lat,
            np.nan * u.deg,
        )
        coords = (lon, lat)

        for p, ax in zip(pix[self._slice_non_spatial_axes], self.axes):
            coords += (ax.pix_to_coord(p),)

        return coords

    def pix_to_idx(self, pix, clip=False):
        idxs = list(pix_tuple_to_idx(pix))

        for i, idx in enumerate(idxs[self._slice_non_spatial_axes]):
            if clip:
                np.clip(idx, 0, self.axes[i].nbin - 1, out=idx)
            else:
                np.putmask(idx, (idx < 0) | (idx >= self.axes[i].nbin), -1)

        return tuple(idxs)

    def coord_to_pix(self, coords):
        coords = MapCoord.create(coords, frame=self.frame, axis_names=self.axes.names)

        if self.region is None:
            pix = (0, 0)
        else:
            in_region = self.region.contains(coords.skycoord, wcs=self.wcs)

            x = np.zeros(coords.shape)
            x[~in_region] = np.nan

            y = np.zeros(coords.shape)
            y[~in_region] = np.nan

            pix = (x, y)

        pix += self.axes.coord_to_pix(coords)
        return pix

    def get_idx(self):
        idxs = [np.arange(n, dtype=float) for n in self.data_shape[::-1]]
        return np.meshgrid(*idxs[::-1], indexing="ij")[::-1]

    def _make_bands_cols(self):
        pass

    @classmethod
    def create(cls, region, **kwargs):
        """Create region.

        Parameters
        ----------
        region : str or `~regions.SkyRegion`
            Region
        axes : list of `MapAxis`
            Non spatial axes.

        Returns
        -------
        geom : `RegionGeom`
            Region geometry
        """
        if isinstance(region, str):
            region = make_region(region)

        return cls(region, **kwargs)

    def __repr__(self):
        axes = ["lon", "lat"] + [_.name for _ in self.axes]
        try:
            frame = self.center_skydir.frame.name
            lon = self.center_skydir.data.lon.deg
            lat = self.center_skydir.data.lat.deg
        except AttributeError:
            frame, lon, lat = "", np.nan, np.nan

        return (
            f"{self.__class__.__name__}\n\n"
            f"\tregion     : {self.region.__class__.__name__}\n"
            f"\taxes       : {axes}\n"
            f"\tshape      : {self.data_shape[::-1]}\n"
            f"\tndim       : {self.ndim}\n"
            f"\tframe      : {frame}\n"
            f"\tcenter     : {lon:.1f} deg, {lat:.1f} deg\n"
        )

    def __eq__(self, other):
        # check overall shape and axes compatibility
        if self.data_shape != other.data_shape:
            return False

        for axis, otheraxis in zip(self.axes, other.axes):
            if axis != otheraxis:
                return False

        # TODO: compare regions
        return True

    def _to_region_table(self):
        """Export region to a FITS region table."""
        if self.region is None:
            raise ValueError("Region definition required.")

        # TODO: make this a to_hdulist() method
        region_list = compound_region_to_list(self.region)
        pixel_region_list = []
        for reg in region_list:
            pixel_region_list.append(reg.to_pixel(self.wcs))
        table = fits_region_objects_to_table(pixel_region_list)
        table.meta.update(self.wcs.to_header())
        return table

    @classmethod
    def from_hdulist(cls, hdulist, format="ogip"):
        """Read region table and convert it to region list.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            HDU list
        format : {"ogip", "ogip-arf"}
            HDU format

        Returns
        -------
        geom : `RegionGeom`
            Region map geometry

        """
        if "REGION" in hdulist:
            region_table = Table.read(hdulist["REGION"])
            parser = FITSRegionParser(region_table)
            pix_region = parser.shapes.to_regions()
            wcs = WCS(region_table.meta)

            regions = []
            for reg in pix_region:
                regions.append(reg.to_sky(wcs))
            region = list_to_compound_region(regions)
        else:
            region, wcs = None, None

        if format == "ogip":
            hdu = "EBOUNDS"
        elif format == "ogip-arf":
            hdu = "SPECRESP"
        else:
            raise ValueError(f"Unknown format {format}")

        axis = MapAxis.from_table_hdu(hdulist[hdu], format=format)
        return cls(region=region, wcs=wcs, axes=[axis])

    def union(self, other):
        """Stack a RegionGeom by making the union"""
        if not self == other:
            raise ValueError("Can only make union if extra axes are equivalent.")
        if other.region:
            if self.region:
                self._region = self.region.union(other.region)
            else:
                self._region = other.region
