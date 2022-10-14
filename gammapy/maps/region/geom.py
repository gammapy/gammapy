import copy
from functools import lru_cache
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.table import QTable, Table
from astropy.utils import lazyproperty
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs.utils import (
    proj_plane_pixel_area,
    proj_plane_pixel_scales,
    wcs_to_celestial_frame,
)
from regions import (
    CompoundSkyRegion,
    PixCoord,
    PointSkyRegion,
    RectanglePixelRegion,
    Regions,
    SkyRegion,
)
import matplotlib.pyplot as plt
from gammapy.utils.regions import (
    compound_region_center,
    compound_region_to_regions,
    regions_to_compound_region,
)
from gammapy.visualization.utils import ARTIST_TO_LINE_PROPERTIES
from ..axes import MapAxes
from ..coord import MapCoord
from ..core import Map
from ..geom import Geom, pix_tuple_to_idx
from ..utils import _check_width
from ..wcs import WcsGeom

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

    is_regular = True
    is_allsky = False
    is_hpx = False
    is_region = True

    _slice_spatial_axes = slice(0, 2)
    _slice_non_spatial_axes = slice(2, None)
    projection = "TAN"

    def __init__(self, region, axes=None, wcs=None, binsz_wcs="0.1 deg"):
        self._region = region
        self._axes = MapAxes.from_default(axes, n_spatial_axes=2)
        self._binsz_wcs = u.Quantity(binsz_wcs)

        if wcs is None and region is not None:
            if isinstance(region, CompoundSkyRegion):
                self._center = compound_region_center(region)
            else:
                self._center = region.center

            wcs = WcsGeom.create(
                binsz=binsz_wcs,
                skydir=self._center,
                proj=self.projection,
                frame=self._center.frame.name,
            ).wcs

        self._wcs = wcs
        self.ndim = len(self.data_shape)

        # define cached methods
        self.get_wcs_coord_and_weights = lru_cache()(self.get_wcs_coord_and_weights)

    def __setstate__(self, state):
        for key, value in state.items():
            if key in ["get_wcs_coord_and_weights"]:
                state[key] = lru_cache()(value)
        self.__dict__ = state

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
    def binsz_wcs(self):
        """Angular bin size of the underlying `~WcsGeom`

        Returns
        -------
        binsz_wcs: `~astropy.coordinates.Angle`
        """
        return Angle(proj_plane_pixel_scales(self.wcs), unit="deg")

    @lazyproperty
    def _rectangle_bbox(self):
        if self.region is None:
            raise ValueError("Region definition required.")

        regions = compound_region_to_regions(self.region)
        regions_pix = [_.to_pixel(self.wcs) for _ in regions]

        bbox = regions_pix[0].bounding_box

        for region_pix in regions_pix[1:]:
            bbox = bbox.union(region_pix.bounding_box)

        try:
            rectangle_pix = bbox.to_region()
        except ValueError:
            rectangle_pix = RectanglePixelRegion(
                center=PixCoord(*bbox.center[::-1]), width=1, height=1
            )
        return rectangle_pix.to_sky(self.wcs)

    @property
    def width(self):
        """Width of bounding box of the region.

        Returns
        -------
        width : `~astropy.units.Quantity`
            Dimensions of the region in both spatial dimensions.
            Units: ``deg``
        """
        rectangle = self._rectangle_bbox
        return u.Quantity([rectangle.width.to("deg"), rectangle.height.to("deg")])

    @property
    def region(self):
        """`~regions.SkyRegion` object that defines the spatial component
        of the region geometry"""
        return self._region

    @property
    def is_all_point_sky_regions(self):
        """Whether regions are all point regions"""
        regions = compound_region_to_regions(self.region)
        return np.all([isinstance(_, PointSkyRegion) for _ in regions])

    @property
    def axes(self):
        """List of non-spatial axes."""
        return self._axes

    @property
    def axes_names(self):
        """All axes names"""
        return ["lon", "lat"] + self.axes.names

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

    @lazyproperty
    def center_skydir(self):
        """Sky coordinate of the center of the region"""
        if self.region is None:
            return SkyCoord(np.nan * u.deg, np.nan * u.deg)

        return self._rectangle_bbox.center

    @property
    def npix(self):
        """Number of spatial pixels"""
        return (1, 1)

    def contains(self, coords):
        """Check if a given map coordinate is contained in the region.
        Requires the `.region` attribute to be set.

        For `PointSkyRegion` the method always returns true.

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

        if self.is_all_point_sky_regions:
            return np.ones(coords.skycoord.shape, dtype=bool)

        return self.region.contains(coords.skycoord, self.wcs)

    def contains_wcs_pix(self, pix):
        """Check if a given wcs pixel coordinate is contained in the region.

        For `PointSkyRegion` the method always returns true.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinates.

        Returns
        -------
        containment : `~numpy.ndarray`
            Bool array.
        """
        if self.is_all_point_sky_regions:
            return np.ones(pix[0].shape, dtype=bool)

        region_pix = self.region.to_pixel(self.wcs)
        return region_pix.contains(PixCoord(pix[0], pix[1]))

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
    def data_shape_axes(self):
        """Shape of data of the non-spatial axes and unit spatial axes."""
        return self.axes.shape[::-1] + (1, 1)

    @property
    def _shape(self):
        """Number of bins in each dimension.
        The spatial dimension is always (1, 1), as a
        `RegionGeom` is not pixelized further
        """
        return tuple((1, 1) + self.axes.shape)

    def get_coord(self, mode="center", frame=None, sparse=False, axis_name=None):
        """Get map coordinates from the geometry.

        Parameters
        ----------
        mode : {'center', 'edges'}
            Get center or edge coordinates for the non-spatial axes.
        frame : str or `~astropy.coordinates.Frame`
            Coordinate frame
        sparse : bool
            Compute sparse coordinates
        axis_name : str
            If mode = "edges", the edges will be returned for this axis only.


        Returns
        -------
        coord : `~MapCoord`
            Map coordinate object.
        """
        if mode == "edges" and axis_name is None:
            raise ValueError("Mode 'edges' requires axis name")

        coords = self.axes.get_coord(mode=mode, axis_name=axis_name)
        coords["skycoord"] = self.center_skydir.reshape((1, 1))

        if frame is None:
            frame = self.frame

        return MapCoord.create(coords, frame=self.frame).to_frame(frame)

    def _pad_spatial(self, pad_width):
        raise NotImplementedError("Spatial padding of `RegionGeom` not supported")

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

        # compound regions do not implement area()
        # so we use the mask representation and estimate the area
        # from the pixels in the mask using oversampling
        if isinstance(self.region, CompoundSkyRegion):
            # oversample by a factor of ten
            oversampling = 10.0
            wcs = self.to_binsz_wcs(self.binsz_wcs / oversampling).wcs
            pixel_region = self.region.to_pixel(wcs)
            mask = pixel_region.to_mask()
            area = np.count_nonzero(mask) / oversampling**2
        else:
            # all other types of regions should implement area
            area = self.region.to_pixel(self.wcs).area

        solid_angle = area * proj_plane_pixel_area(self.wcs) * u.deg**2
        return solid_angle.to("sr")

    def bin_volume(self):
        """If the RegionGeom has a non-spatial axis, it
        returns the volume of the region. If not, it
        just returns the solid angle size.

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
            width = np.max(
                [self.width.to_value("deg"), _check_width(width_min)], axis=0
            )
        else:
            width = self.width
        wcs_geom_region = WcsGeom(wcs=self.wcs, npix=self.wcs.array_shape)
        wcs_geom = wcs_geom_region.cutout(position=self.center_skydir, width=width)
        wcs_geom = wcs_geom.to_cube(self.axes)
        return wcs_geom

    def to_binsz_wcs(self, binsz):

        """Change the bin size of the underlying WCS geometry.

        Parameters
        ----------
        binzs : float, string or `~astropy.quantity.Quantity`

        Returns
        -------
        region : `~RegionGeom`
            A RegionGeom with the same axes and region as the input,
            but different wcs pixelization.
        """
        new_geom = RegionGeom(self.region, axes=self.axes, binsz_wcs=binsz)
        return new_geom

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
        wcs_geom = self.to_wcs_geom()

        weights = wcs_geom.to_image().region_weights(
            regions=[self.region], oversampling_factor=factor
        )

        mask = weights.data > 0
        weights = weights.data[mask]

        # Get coordinates
        coords = wcs_geom.get_coord(sparse=True).apply_mask(mask)
        return coords, weights

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

    def upsample(self, factor, axis_name=None):
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
        # inherited docstring
        if isinstance(coords, tuple) and len(coords) == len(self.axes):
            skydir = self.center_skydir.transform_to(self.frame)
            coords = (skydir.data.lon, skydir.data.lat) + coords
        elif isinstance(coords, dict):
            valid_keys = ["lon", "lat", "skycoord"]
            if not any([_ in coords for _ in valid_keys]):
                coords.setdefault("skycoord", self.center_skydir)

        coords = MapCoord.create(coords, frame=self.frame, axis_names=self.axes.names)

        if self.region is None:
            pix = (0, 0)
        else:
            in_region = self.contains(coords.skycoord)

            x = np.zeros(coords.skycoord.shape)
            x[~in_region] = np.nan

            y = np.zeros(coords.skycoord.shape)
            y[~in_region] = np.nan

            pix = (x, y)

        pix += self.axes.coord_to_pix(coords)
        return pix

    def get_idx(self):
        idxs = [np.arange(n, dtype=float) for n in self.data_shape[::-1]]
        return np.meshgrid(*idxs[::-1], indexing="ij")[::-1]

    def _make_bands_cols(self):
        return []

    @classmethod
    def create(cls, region, **kwargs):
        """Create region geometry.

        The input region can be passed in the form of a ds9 string and will be parsed
        internally by `~regions.Regions.parse`. See:

        * https://astropy-regions.readthedocs.io/en/stable/region_io.html
        * http://ds9.si.edu/doc/ref/region.html

        Parameters
        ----------
        region : str or `~regions.SkyRegion`
            Region definition
        **kwargs : dict
            Keyword arguments passed to `RegionGeom.__init__`

        Returns
        -------
        geom : `RegionGeom`
            Region geometry
        """
        return cls.from_regions(regions=region, **kwargs)

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

    def is_allclose(self, other, rtol_axes=1e-6, atol_axes=1e-6):
        """Compare two data IRFs for equivalency

        Parameters
        ----------
        other : `RegionGeom`
            Geom to compare against.
        rtol_axes : float
            Relative tolerance for the axes comparison.
        atol_axes : float
            Relative tolerance for the axes comparison.

        Returns
        -------
        is_allclose : bool
            Whether the geometry is all close.
        """
        if not isinstance(other, self.__class__):
            return TypeError(f"Cannot compare {type(self)} and {type(other)}")

        if self.data_shape != other.data_shape:
            return False

        axes_eq = self.axes.is_allclose(other.axes, rtol=rtol_axes, atol=atol_axes)
        # TODO: compare regions based on masks...
        regions_eq = True
        return axes_eq and regions_eq

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.is_allclose(other=other)

    def _to_region_table(self):
        """Export region to a FITS region table."""
        if self.region is None:
            raise ValueError("Region definition required.")

        region_list = compound_region_to_regions(self.region)

        pixel_region_list = []

        for reg in region_list:
            pixel_region_list.append(reg.to_pixel(self.wcs))

        table = Regions(pixel_region_list).serialize(format="fits")

        header = WcsGeom(wcs=self.wcs, npix=self.wcs.array_shape).to_header()
        table.meta.update(header)
        return table

    def to_hdulist(self, format="ogip", hdu_bands=None, hdu_region=None):
        """Convert geom to hdulist

        Parameters
        ----------
        format : {"gadf", "ogip", "ogip-sherpa"}
            HDU format
        hdu : str
            Name of the HDU with the map data.

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
            HDU list

        """
        if hdu_bands is None:
            hdu_bands = "HDU_BANDS"
        if hdu_region is None:
            hdu_region = "HDU_REGION"
        if format != "gadf":
            hdu_region = "REGION"

        hdulist = fits.HDUList()

        hdulist.append(self.axes.to_table_hdu(hdu_bands=hdu_bands, format=format))

        # region HDU
        if self.region:
            region_table = self._to_region_table()

            region_hdu = fits.BinTableHDU(region_table, name=hdu_region)
            hdulist.append(region_hdu)

        return hdulist

    @classmethod
    def from_regions(cls, regions, **kwargs):
        """Create region geom from list of regions

        The regions are combined with union to a compound region.

        Parameters
        ----------
        regions : list of `~regions.SkyRegion` or str
            Regions
        **kwargs: dict
            Keyword arguments forwarded to `RegionGeom`

        Returns
        -------
        geom : `RegionGeom`
            Region map geometry
        """
        if isinstance(regions, str):
            regions = Regions.parse(data=regions, format="ds9")
        elif isinstance(regions, SkyRegion):
            regions = [regions]
        elif isinstance(regions, SkyCoord):
            regions = [PointSkyRegion(center=regions)]

        if regions:
            regions = regions_to_compound_region(regions)

        return cls(region=regions, **kwargs)

    @classmethod
    def from_hdulist(cls, hdulist, format="ogip", hdu=None):
        """Read region table and convert it to region list.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            HDU list
        format : {"ogip", "ogip-arf", "gadf"}
            HDU format

        Returns
        -------
        geom : `RegionGeom`
            Region map geometry

        """
        region_hdu = "REGION"

        if format == "gadf" and hdu:
            region_hdu = hdu + "_" + region_hdu

        if region_hdu in hdulist:
            try:
                region_table = QTable.read(hdulist[region_hdu])
                regions_pix = Regions.parse(data=region_table, format="fits")
            except TypeError:
                # TODO: this is needed to support regions=0.5
                region_table = Table.read(hdulist[region_hdu])
                regions_pix = Regions.parse(data=region_table, format="fits")

            wcs = WcsGeom.from_header(region_table.meta).wcs
            regions = []

            for region_pix in regions_pix:
                # TODO: remove workaround once regions issue with fits serialization is sorted out
                # see https://github.com/astropy/regions/issues/400
                region_pix.meta["include"] = True
                regions.append(region_pix.to_sky(wcs))

            region = regions_to_compound_region(regions)
        else:
            region, wcs = None, None

        if format == "ogip":
            hdu_bands = "EBOUNDS"
        elif format == "ogip-arf":
            hdu_bands = "SPECRESP"
        elif format == "gadf":
            hdu_bands = hdu + "_BANDS"
        else:
            raise ValueError(f"Unknown format {format}")

        axes = MapAxes.from_table_hdu(hdulist[hdu_bands], format=format)
        return cls(region=region, wcs=wcs, axes=axes)

    def union(self, other):
        """Stack a RegionGeom by making the union"""
        if not self == other:
            raise ValueError("Can only make union if extra axes are equivalent.")
        if other.region:
            if self.region:
                self._region = self.region.union(other.region)
            else:
                self._region = other.region

    def plot_region(self, ax=None, kwargs_point=None, path_effect=None, **kwargs):
        """Plot region in the sky.

        Parameters
        ----------
        ax : `~astropy.visualization.WCSAxes`
            Axes to plot on. If no axes are given,
            the region is shown using the minimal
            equivalent WCS geometry.
        kwargs_point : dict
            Keyword arguments passed to `~matplotlib.lines.Line2D` for plotting
            of point sources
        path_effect : `~matplotlib.patheffects.PathEffect`
            Path effect applied to artists and lines.
        **kwargs : dict
            Keyword arguments forwarded to `~regions.PixelRegion.as_artist`

        Returns
        -------
        ax : `~astropy.visualization.WCSAxes`
            Axes to plot on.
        """
        kwargs_point = kwargs_point or {}

        if ax is None:
            ax = plt.gca()

            if not isinstance(ax, WCSAxes):
                ax.remove()
                wcs_geom = self.to_wcs_geom()
                m = Map.from_geom(geom=wcs_geom.to_image())
                ax = m.plot(add_cbar=False, vmin=-1, vmax=0)

        kwargs.setdefault("facecolor", "None")
        kwargs.setdefault("edgecolor", "tab:blue")
        kwargs_point.setdefault("marker", "*")

        for key, value in kwargs.items():
            key_point = ARTIST_TO_LINE_PROPERTIES.get(key, None)
            if key_point:
                kwargs_point[key_point] = value

        for region in compound_region_to_regions(self.region):
            region_pix = region.to_pixel(wcs=ax.wcs)

            if isinstance(region, PointSkyRegion):
                artist = region_pix.as_artist(**kwargs_point)
            else:
                artist = region_pix.as_artist(**kwargs)

            if path_effect:
                artist.add_path_effect(path_effect)

            ax.add_artist(artist)

        return ax
