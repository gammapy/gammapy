import copy
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.wcs.utils import proj_plane_pixel_area, wcs_to_celestial_frame
from astropy.wcs import WCS
from regions import CircleSkyRegion, fits_region_objects_to_table, FITSRegionParser
from gammapy.utils.regions import make_region, compound_region_to_list, list_to_compound_region
from .base import MapCoord
from .geom import Geom, make_axes, pix_tuple_to_idx, MapAxis
from .utils import INVALID_INDEX, edges_from_lo_hi
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
    is_allsky = False
    is_hpx = False
    _slice_spatial_axes = slice(0, 2)
    _slice_non_spatial_axes = slice(2, None)
    projection = "TAN"
    binsz = 0.01

    def __init__(self, region, axes=None, wcs=None):
        self._region = region
        self._axes = make_axes(axes)

        if axes is not None:
            if len(axes) > 1 or axes[0].name not in ["energy", "energy_true"]:
                raise ValueError("RegionGeom currently only supports an energy axes.")

        if wcs is None and region is not None:
            wcs = WcsGeom.create(
                skydir=region.center,
                binsz=self.binsz,
                width=self.width,
                proj=self.projection,
                frame=self.frame,
            ).wcs

        self._wcs = wcs
        self.ndim = len(self.data_shape)

    @property
    def frame(self):
        try:
            return self.region.center.frame.name
        except AttributeError:
            return wcs_to_celestial_frame(self.wcs).name

    @property
    def width(self):
        if isinstance(self.region, CircleSkyRegion):
            return 2 * self.region.radius
        else:
            raise ValueError("Currently only circular regions supported")

    @property
    def region(self):
        if self._region is None:
            raise ValueError("Region definition required.")
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
        return self.region.center

    def contains(self, coords):
        idx = self.coord_to_idx(coords)
        return np.all(np.stack([t != INVALID_INDEX.int for t in idx]), axis=0)

    def separation(self, position):
        return self.center_skydir.separation(position)

    @property
    def data_shape(self):
        """Shape of the Numpy data array matching this geometry."""
        return self._shape[::-1]

    @property
    def _shape(self):
        npix_shape = [1, 1]
        ax_shape = [ax.nbin for ax in self.axes]
        return tuple(npix_shape + ax_shape)

    def get_coord(self, frame=None):
        """Get map coordinates from the geometry.

        Returns
        -------
        coord : `~MapCoord`
            Map coordinate object.
        """
        cdict = {}
        cdict["skycoord"] = self.center_skydir.reshape((1, 1))

        if self.axes is not None:
            for ax in self.axes:
                cdict[ax.name] = ax.center.reshape((-1, 1, 1))

        if frame is None:
            frame = self.frame

        return MapCoord.create(cdict, frame=self.frame).to_frame(frame)

    def pad(self):
        raise NotImplementedError("Padding of `RegionGeom` not supported")

    def crop(self):
        raise NotImplementedError("Cropping of `RegionGeom` not supported")

    def solid_angle(self):
        area = self.region.to_pixel(self.wcs).area
        solid_angle = area * proj_plane_pixel_area(self.wcs) * u.deg ** 2
        return solid_angle.to("sr")

    def bin_volume(self):
        return self.solid_angle() * self.axes[0].bin_width.reshape((-1, 1, 1))

    def to_cube(self, axes):
        axes = copy.deepcopy(self.axes) + axes
        return self._init_copy(axes=axes)

    def to_image(self):
        return self._init_copy(axes=None)

    def upsample(self, factor, axis):
        axes = copy.deepcopy(self.axes)
        idx = self.get_axis_index_by_name(axis)
        axes[idx] = axes[idx].upsample(factor)
        return self._init_copy(axes=axes)

    def downsample(self, factor, axis):
        axes = copy.deepcopy(self.axes)
        idx = self.get_axis_index_by_name(axis)
        axes[idx] = axes[idx].downsample(factor)
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
        coords = MapCoord.create(coords, frame=self.frame)
        in_region = self.region.contains(coords.skycoord, wcs=self.wcs)

        x = np.zeros(coords.shape)
        x[~in_region] = np.nan

        y = np.zeros(coords.shape)
        y[~in_region] = np.nan

        pix = (x, y)
        for coord, ax in zip(coords[self._slice_non_spatial_axes], self.axes):
            pix += (ax.coord_to_pix(coord),)

        return pix

    def get_idx(self):
        idxs = (0, 0)
        if self.axes is not None:
            for ax in self.axes:
                idxs += (np.arange(ax.nbin).reshape((-1, 1, 1)),)
        return np.broadcast_arrays(*idxs)

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
        lon = self.center_skydir.data.lon.deg
        lat = self.center_skydir.data.lat.deg

        return (
            f"{self.__class__.__name__}\n\n"
            f"\tregion     : {self.region.__class__.__name__}\n"
            f"\taxes       : {axes}\n"
            f"\tshape      : {self.data_shape[::-1]}\n"
            f"\tndim       : {self.ndim}\n"
            f"\tframe      : {self.center_skydir.frame.name}\n"
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
        region_list = compound_region_to_list(self.region)
        pixel_region_list = []
        for reg in region_list:
            pixel_region_list.append(reg.to_pixel(self.wcs))
        table = fits_region_objects_to_table(pixel_region_list)
        table.meta.update(self.wcs.to_header())
        return table

    @classmethod
    def from_hdulist(cls, hdulist, format="ogip"):
        """Read region table and convert it to region list."""

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

        ebounds = Table.read(hdulist["EBOUNDS"])
        emin = ebounds["E_MIN"].quantity
        emax = ebounds["E_MAX"].quantity

        edges = edges_from_lo_hi(emin, emax)
        axis = MapAxis.from_edges(edges, interp="log", name="energy")
        return cls(region=region, wcs=wcs, axes=[axis])
