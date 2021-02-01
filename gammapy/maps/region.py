import copy
from functools import lru_cache
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs.utils import proj_plane_pixel_area, wcs_to_celestial_frame
from regions import FITSRegionParser, fits_region_objects_to_table
from gammapy.utils.regions import (
    compound_region_to_list,
    list_to_compound_region,
    make_region,
)
from gammapy.maps.wcs import _check_width
from .core import MapCoord, Map
from .geom import Geom, MapAxes, MapAxis, pix_tuple_to_idx
from .wcs import WcsGeom

__all__ = ["RegionGeom"]


class RegionGeom(Geom):
    """Map geometry representing a region on the sky.
    The spatial component of the geometry is made up of a
    single pixel with an arbitrary shape and size. It can
    also have any number of non-spatial dimensions. This class
    represents the geometry for the `RegionNDMap` maps.

    Parameters
    ----------
    region : `~regions.SkyRegion`
        Region object.
    axes : list of `MapAxis`
        Non-spatial data axes.
    wcs : `~astropy.wcs.WCS`
        Optional wcs object to project the region if needed.
    binsz_wcs : `float`
        Angular bin size of the underlying `~WcsGeom` used to evaluate
        quantities in the region. Default size is 0.01 deg. This default
        value is adequate for the majority of use cases. If a wcs object
        is provided, the input of binsz_wcs is overridden.
    """

    is_image = False
    is_regular = True
    is_allsky = False
    is_hpx = False
    is_region = True

    _slice_spatial_axes = slice(0, 2)
    _slice_non_spatial_axes = slice(2, None)
    projection = "TAN"

    def __init__(self, region, axes=None, wcs=None, binsz_wcs=0.01):
        self._region = region
        self._axes = MapAxes.from_default(axes)
        self._binsz_wcs = binsz_wcs

        if wcs is None and region is not None:
            wcs = WcsGeom.create(
                skydir=region.center,
                binsz=binsz_wcs,
                proj=self.projection,
                frame=self.frame,
            ).wcs

        self._wcs = wcs
        self.ndim = len(self.data_shape)

        # define cached methods
        self.get_wcs_coord_and_weights = lru_cache()(self.get_wcs_coord_and_weights)

    @property
    def frame(self):
        """Coordinate system, either Galactic ("galactic") or Equatorial
            ("icrs")."""
        if self.region is None:
            return "icrs"
        try:
            return self.region.center.frame.name
        except AttributeError:
            return wcs_to_celestial_frame(self.wcs).name

    @property
    def width(self):
        """Width of bounding box of the region.

        Returns
        -------
        width : `~astropy.units.Quantity`
            Dimensions of the region in both spatial dimensions.
            Units: ``deg``
        """
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
        """`~regions.SkyRegion` object that defines the spatial component
        of the region geometry"""
        return self._region

    @property
    def axes(self):
        """List of non-spatial axes."""
        return self._axes

    @property
    def wcs(self):
        """WCS projection object."""
        return self._wcs

    @property
    def center_coord(self):
        """(`astropy.coordinates.SkyCoord`)"""
        return self.pix_to_coord(self.center_pix)

    @property
    def center_pix(self):
        """Pixel values corresponding to the center of the region"""
        return tuple((np.array(self.data_shape) - 1.0) / 2)[::-1]

    @property
    def center_skydir(self):
        """Sky coordinate of the center of the region"""
        if self.region is None:
            return SkyCoord(np.nan * u.deg, np.nan * u.deg)
        try:
            return self.region.center
        except AttributeError:
            xp, yp = self.wcs.wcs.crpix
            return SkyCoord.from_pixel(xp=xp, yp=yp, wcs=self.wcs)

    def contains(self, coords):
        """Check if a given map coordinate is contained in the region.
        Requires the `.region` attribute to be set.

        Parameters
        ----------
        coords : tuple, dict, `MapCoord` or `~astropy.coordinates.SkyCoord`
            Object containing coordinate arrays we wish to check for inclusion
            in the region.

        Returns
        -------
        mask : `~numpy.ndarray`
            Boolean Numpy array with the same shape as the input that indicates
            which coordinates are inside the region.
        """
        if self.region is None:
            raise ValueError("Region definition required.")

        coords = MapCoord.create(coords, frame=self.frame, axis_names=self.axes.names)
        return self.region.contains(coords.skycoord, self.wcs)

    def separation(self, position):
        """Angular distance between the center of the region and the given position.

        Parameters
        ----------
        position : `astropy.coordinates.SkyCoord`
            Sky coordinate we want the angular distance to.

        Returns
        -------
        sep : `~astropy.coordinates.Angle`
            The on-sky separation between the given coordinate and the region center.
        """
        return self.center_skydir.separation(position)

    @property
    def data_shape(self):
        """Shape of the Numpy data array matching this geometry."""
        return self._shape[::-1]

    @property
    def _shape(self):
        """Number of bins in each dimension.
        The spatial dimension is always (1, 1), as a
        `RegionGeom` is not pixelized further
        """
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
        """Get solid angle of the region.

        Returns
        -------
        angle : `~astropy.units.Quantity`
            Solid angle of the region. In sr.
            Units: ``sr``
        """
        if self.region is None:
            raise ValueError("Region definition required.")

        area = self.region.to_pixel(self.wcs).area
        solid_angle = area * proj_plane_pixel_area(self.wcs) * u.deg ** 2
        return solid_angle.to("sr")

    def bin_volume(self):
        """If the RegionGeom has a non-spatial axis, it
        returns the volume of the region. If not, it
        just retuns the solid angle size.

        Returns
        -------
        volume : `~astropy.units.Quantity`
            Volume of the region.
        """
        bin_volume = self.solid_angle() * np.ones(self.data_shape)

        for idx, ax in enumerate(self.axes):
            shape = self.ndim * [1]
            shape[-(idx + 3)] = -1
            bin_volume = bin_volume * ax.bin_width.reshape(tuple(shape))

        return bin_volume

    def to_wcs_geom(self, width_min=None):
        """Get the minimal equivalent geometry
        which contains the region.

        Parameters
         ----------
        width_min : `~astropy.quantity.Quantity`
        Minimal width for the resulting geometry.
        Can be a single number or two, for
        different minimum widths in each spatial dimension.

        Returns
        -------
        wcs_geom : `~WcsGeom`
            A WCS geometry object.
        """
        if width_min is not None:
            width = np.min([self.width.to_value("deg"), _check_width(width_min)], axis=0)
        else:
            width = self.width
        wcs_geom_region = WcsGeom(wcs=self.wcs, npix=self.wcs.array_shape)
        wcs_geom = wcs_geom_region.cutout(position=self.center_skydir, width=width)
        wcs_geom = wcs_geom.to_cube(self.axes)
        return wcs_geom

    def get_wcs_coord_and_weights(self, factor=10):
        """Get the array of spatial coordinates and corresponding weights

        The coordinates are the center of a pixel that intersects the region and
        the weights that represent which fraction of the pixel is contained
        in the region.

        Parameters
        ----------
        factor : int
            Oversampling factor to compute the weights

        Returns
        -------
        region_coord : `~MapCoord`
            MapCoord object with the coordinates inside
            the region.
        weights : `~np.array`
            Weights representing the fraction of each pixel
            contained in the region.
        """
        wcs_geom = self.to_wcs_geom().to_image()

        weights = wcs_geom.region_weights(
            regions=[self.region], oversampling_factor=factor
        )

        mask = (weights.data > 0)
        weights = weights.data[mask]

        # Get coordinates
        region_coord = wcs_geom.get_coord().apply_mask(mask)
        
        return region_coord, weights

    def to_binsz(self, binsz):
        """Returns self"""
        return self

    def to_cube(self, axes):
        """Append non-spatial axes to create a higher-dimensional geometry.

        Returns
        -------
        region : `~RegionGeom`
            RegionGeom with the added axes.
        """
        axes = copy.deepcopy(self.axes) + axes
        return self._init_copy(axes=axes)

    def to_image(self):
        """Remove non-spatial axes to create a 2D region.

        Returns
        -------
        region : `~RegionGeom`
            RegionGeom without any non-spatial axes.
        """
        return self._init_copy(axes=None)

    def upsample(self, factor, axis_name):
        """Upsample a non-spatial dimension of the region by a given factor.

        Returns
        -------
        region : `~RegionGeom`
            RegionGeom with the upsampled axis.
        """
        axes = self.axes.upsample(factor=factor, axis_name=axis_name)
        return self._init_copy(axes=axes)

    def downsample(self, factor, axis_name):
        """Downsample a non-spatial dimension of the region by a given factor.

        Returns
        -------
        region : `~RegionGeom`
            RegionGeom with the downsampled axis.
        """
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

        header = WcsGeom(wcs=self.wcs, npix=self.wcs.array_shape).to_header()
        table.meta.update(header)
        return table

    def to_hdulist(self, format="ogip"):
        """Convert geom to hdulist

        Parameters
        ----------
        format : {"ogip", "ogip-sherpa"}
            HDU format

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
            HDU list

        """
        hdulist = fits.HDUList()

        # energy bounds HDU
        energy_axis = self.axes["energy"]
        hdulist.append(energy_axis.to_table_hdu(format=format))

        # region HDU
        if self.region:
            region_table = self._to_region_table()
            region_hdu = fits.BinTableHDU(region_table, name="REGION")
            hdulist.append(region_hdu)

        return hdulist

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
            wcs = WcsGeom.from_header(region_table.meta).wcs

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

    def plot_region(self, ax=None, **kwargs):
        """Plot region in the sky.

        Parameters
        ----------
        ax : `~astropy.vizualisation.WCSAxes`
            Axes to plot on. If no axes are given,
            the region is shown using the minimal
            equivalent WCS geometry.
        **kwargs : dict
            Keyword arguments forwarded to `~regions.PixelRegion.as_artist`
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from astropy.visualization.wcsaxes import WCSAxes

        if ax is None:
            ax = plt.gca()

            if not isinstance(ax, WCSAxes):
                ax.remove()
                wcs_geom = self.to_wcs_geom()
                m = Map.from_geom(wcs_geom.to_image())
                fig, ax, cbar = m.plot(add_cbar=False)

        regions = compound_region_to_list(self.region)
        artists = [region.to_pixel(wcs=ax.wcs).as_artist() for region in regions]

        kwargs.setdefault("fc", "None")

        patches = PatchCollection(artists, **kwargs)
        ax.add_collection(patches)
        return ax
