# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
from functools import lru_cache
import numpy as np
import astropy.units as u
from astropy.convolution import Tophat2DKernel
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import overlap_slices
from astropy.utils import lazyproperty
from astropy.wcs import WCS
from astropy.wcs.utils import (
    celestial_frame_to_wcs,
    proj_plane_pixel_scales,
    wcs_to_celestial_frame,
)
from gammapy.utils.array import round_up_to_even, round_up_to_odd
from ..axes import MapAxes
from ..coord import MapCoord, skycoord_to_lonlat
from ..geom import Geom, get_shape, pix_tuple_to_idx
from ..utils import INVALID_INDEX, _check_binsz, _check_width

__all__ = ["WcsGeom"]


def cast_to_shape(param, shape, dtype):
    """Cast a tuple of parameter arrays to a given shape."""
    if not isinstance(param, tuple):
        param = [param]

    param = [np.array(p, ndmin=1, dtype=dtype) for p in param]

    if len(param) == 1:
        param = [param[0].copy(), param[0].copy()]

    for i, p in enumerate(param):

        if p.size > 1 and p.shape != shape:
            raise ValueError

        if p.shape == shape:
            continue

        param[i] = p * np.ones(shape, dtype=dtype)

    return tuple(param)


def get_resampled_wcs(wcs, factor, downsampled):
    """
    Get resampled WCS object.
    """
    wcs = wcs.deepcopy()

    if not downsampled:
        factor = 1.0 / factor

    wcs.wcs.cdelt *= factor
    wcs.wcs.crpix = (wcs.wcs.crpix - 0.5) / factor + 0.5
    return wcs


class WcsGeom(Geom):
    """Geometry class for WCS maps.

    This class encapsulates both the WCS transformation object and the
    the image extent (number of pixels in each dimension).  Provides
    methods for accessing the properties of the WCS object and
    performing transformations between pixel and world coordinates.

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        WCS projection object
    npix : tuple
        Number of pixels in each spatial dimension
    cdelt : tuple
        Pixel size in each image plane.  If none then a constant pixel size will be used.
    crpix : tuple
        Reference pixel coordinate in each image plane.
    axes : list
        Axes for non-spatial dimensions
    """

    _slice_spatial_axes = slice(0, 2)
    _slice_non_spatial_axes = slice(2, None)
    is_hpx = False
    is_region = False

    def __init__(self, wcs, npix, cdelt=None, crpix=None, axes=None):
        self._wcs = wcs
        self._frame = wcs_to_celestial_frame(wcs).name
        self._projection = wcs.wcs.ctype[0][5:]
        self._axes = MapAxes.from_default(axes, n_spatial_axes=2)

        if cdelt is None:
            cdelt = tuple(np.abs(self.wcs.wcs.cdelt))

        # Shape to use for WCS transformations
        wcs_shape = max([get_shape(t) for t in [npix, cdelt]])
        self._npix = cast_to_shape(npix, wcs_shape, int)
        self._cdelt = cast_to_shape(cdelt, wcs_shape, float)

        # By convention CRPIX is indexed from 1
        if crpix is None:
            crpix = tuple(1.0 + (np.array(self._npix) - 1.0) / 2.0)

        self._crpix = crpix

        # define cached methods
        self.get_coord = lru_cache()(self.get_coord)
        self.get_pix = lru_cache()(self.get_pix)

    def __setstate__(self, state):
        for key, value in state.items():
            if key in ["get_coord", "get_pix"]:
                state[key] = lru_cache()(value)

        self.__dict__ = state

    @property
    def data_shape(self):
        """Shape of the Numpy data array matching this geometry."""
        return self._shape[::-1]

    @property
    def axes_names(self):
        """All axes names"""
        return ["lon", "lat"] + self.axes.names

    @property
    def data_shape_axes(self):
        """Shape of data of the non-spatial axes and unit spatial axes."""
        return self.axes.shape[::-1] + (1, 1)

    @property
    def _shape(self):
        npix_shape = tuple([np.max(self.npix[0]), np.max(self.npix[1])])
        return npix_shape + self.axes.shape

    @property
    def _shape_edges(self):
        npix_shape = tuple([np.max(self.npix[0]) + 1, np.max(self.npix[1]) + 1])
        return npix_shape + self.axes.shape

    @property
    def shape_axes(self):
        """Shape of non-spatial axes."""
        return self._shape[self._slice_non_spatial_axes]

    @property
    def wcs(self):
        """WCS projection object."""
        return self._wcs

    @property
    def frame(self):
        """Coordinate system of the projection.

        Galactic ("galactic") or Equatorial ("icrs").
        """
        return self._frame

    def cutout_slices(self, geom, mode="partial"):
        """Compute cutout slices.

        Parameters
        ----------
        geom : `WcsGeom`
            Parent geometry
        mode : {"trim", "partial", "strict"}
            Cutout slices mode.

        Returns
        -------
        slices : dict
            Dictionary containing "parent-slices" and "cutout-slices".
        """
        position = geom.to_image().coord_to_pix(self.center_skydir)
        slices = overlap_slices(
            large_array_shape=geom.data_shape[-2:],
            small_array_shape=self.data_shape[-2:],
            position=position[::-1],
            mode=mode,
        )
        return {
            "parent-slices": slices[0],
            "cutout-slices": slices[1],
        }

    @property
    def projection(self):
        """Map projection."""
        return self._projection

    @property
    def is_allsky(self):
        """Flag for all-sky maps."""
        if np.all(np.isclose(self._npix[0] * self._cdelt[0], 360.0)) and np.all(
            np.isclose(self._npix[1] * self._cdelt[1], 180.0)
        ):
            return True
        else:
            return False

    @property
    def is_regular(self):
        """Is this geometry is regular in non-spatial dimensions (bool)?

        - False for multi-resolution or irregular geometries.
        - True if all image planes have the same pixel geometry.
        """
        if self.npix[0].size > 1:
            return False
        else:
            return True

    @property
    def width(self):
        """Tuple with image dimension in deg in longitude and latitude."""
        dlon = self._cdelt[0] * self._npix[0]
        dlat = self._cdelt[1] * self._npix[1]
        return (dlon, dlat) * u.deg

    @property
    def pixel_area(self):
        """Pixel area in deg^2."""
        # FIXME: Correctly compute solid angle for projection
        return self._cdelt[0] * self._cdelt[1]

    @property
    def npix(self):
        """Tuple with image dimension in pixels in longitude and latitude."""
        return self._npix

    @property
    def axes(self):
        """List of non-spatial axes."""
        return self._axes

    @property
    def ndim(self):
        return len(self.data_shape)

    @property
    def center_coord(self):
        """Map coordinate of the center of the geometry.

        Returns
        -------
        coord : tuple
        """
        return self.pix_to_coord(self.center_pix)

    @property
    def center_pix(self):
        """Pixel coordinate of the center of the geometry.

        Returns
        -------
        pix : tuple
        """
        return tuple((np.array(self.data_shape) - 1.0) / 2)[::-1]

    @property
    def center_skydir(self):
        """Sky coordinate of the center of the geometry.

        Returns
        -------
        pix : `~astropy.coordinates.SkyCoord`
        """
        return SkyCoord.from_pixel(self.center_pix[0], self.center_pix[1], self.wcs)

    @property
    def pixel_scales(self):
        """
        Pixel scale.

        Returns angles along each axis of the image at the CRPIX location once
        it is projected onto the plane of intermediate world coordinates.

        Returns
        -------
        angle: `~astropy.coordinates.Angle`
        """
        return Angle(proj_plane_pixel_scales(self.wcs), "deg")

    @classmethod
    def create(
        cls,
        npix=None,
        binsz=0.5,
        proj="CAR",
        frame="icrs",
        refpix=None,
        axes=None,
        skydir=None,
        width=None,
    ):
        """Create a WCS geometry object.

        Pixelization of the map is set with
        ``binsz`` and one of either ``npix`` or ``width`` arguments.
        For maps with non-spatial dimensions a different pixelization
        can be used for each image plane by passing a list or array
        argument for any of the pixelization parameters.  If both npix
        and width are None then an all-sky geometry will be created.

        Parameters
        ----------
        npix : int or tuple or list
            Width of the map in pixels. A tuple will be interpreted as
            parameters for longitude and latitude axes.  For maps with
            non-spatial dimensions, list input can be used to define a
            different map width in each image plane.  This option
            supersedes width.
        width : float or tuple or list or string
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
        frame : {"icrs", "galactic"}, optional
            Coordinate system, either Galactic ("galactic") or Equatorial ("icrs").
        axes : list
            List of non-spatial axes.
        proj : string, optional
            Any valid WCS projection type. Default is 'CAR' (Plate-Carrée projection).
            See `WCS supported projections <https://docs.astropy.org/en/stable/wcs/supported_projections.html>`__  # noqa: E501
        refpix : tuple
            Reference pixel of the projection.  If None this will be
            set to the center of the map.

        Returns
        -------
        geom : `~WcsGeom`
            A WCS geometry object.

        Examples
        --------
        >>> from gammapy.maps import WcsGeom
        >>> from gammapy.maps import MapAxis
        >>> axis = MapAxis.from_bounds(0,1,2)
        >>> geom = WcsGeom.create(npix=(100,100), binsz=0.1)
        >>> geom = WcsGeom.create(npix=(100,100), binsz="0.1deg")
        >>> geom = WcsGeom.create(npix=[100,200], binsz=[0.1,0.05], axes=[axis])
        >>> geom = WcsGeom.create(npix=[100,200], binsz=["0.1deg","0.05deg"], axes=[axis])
        >>> geom = WcsGeom.create(width=[5.0,8.0], binsz=[0.1,0.05], axes=[axis])
        >>> geom = WcsGeom.create(npix=([100,200],[100,200]), binsz=0.1, axes=[axis])
        """
        if skydir is None:
            crval = (0.0, 0.0)
        elif isinstance(skydir, tuple):
            crval = skydir
        elif isinstance(skydir, SkyCoord):
            xref, yref, frame = skycoord_to_lonlat(skydir, frame=frame)
            crval = (xref, yref)
        else:
            raise ValueError(f"Invalid type for skydir: {type(skydir)!r}")

        if width is not None:
            width = _check_width(width)

        binsz = _check_binsz(binsz)

        shape = max([get_shape(t) for t in [npix, binsz, width]])
        binsz = cast_to_shape(binsz, shape, float)

        # If both npix and width are None then create an all-sky geometry
        if npix is None and width is None:
            width = (360.0, 180.0)

        if npix is None:
            width = cast_to_shape(width, shape, float)
            npix = (
                np.rint(width[0] / binsz[0]).astype(int),
                np.rint(width[1] / binsz[1]).astype(int),
            )
        else:
            npix = cast_to_shape(npix, shape, int)

        if refpix is None:
            nxpix = int(npix[0].flat[0])
            nypix = int(npix[1].flat[0])
            refpix = ((nxpix + 1) / 2.0, (nypix + 1) / 2.0)

        # get frame class
        frame = SkyCoord(np.nan, np.nan, frame=frame, unit="deg").frame
        wcs = celestial_frame_to_wcs(frame, projection=proj)
        wcs.wcs.crpix = refpix
        wcs.wcs.crval = crval

        cdelt = (-binsz[0].flat[0], binsz[1].flat[0])
        wcs.wcs.cdelt = cdelt

        wcs.array_shape = npix[0].flat[0], npix[1].flat[0]
        wcs.wcs.datfix()
        return cls(wcs, npix, cdelt=binsz, axes=axes)

    @property
    def footprint(self):
        """Footprint of the geometry"""
        coords = self.wcs.calc_footprint()
        return SkyCoord(coords, frame=self.frame, unit="deg")

    @classmethod
    def from_aligned(cls, geom, skydir, width):
        """Create an aligned geometry from an existing one

        Parameters
        ----------
        geom : `~WcsGeom`
            A reference WCS geometry object.
        skydir : tuple or `~astropy.coordinates.SkyCoord`
            Sky position of map center.  Can be either a SkyCoord
            object or a tuple of longitude and latitude in deg in the
            coordinate system of the map.
        width : float or tuple or list or string
            Width of the map in degrees.  A tuple will be interpreted
            as parameters for longitude and latitude axes.  For maps
            with non-spatial dimensions, list input can be used to
            define a different map width in each image plane.

        Returns
        -------
        geom : `~WcsGeom`
            An aligned WCS geometry object with specified size and center.

        """
        width = _check_width(width) * u.deg
        npix = tuple(np.round(width / geom.pixel_scales).astype(int))
        xref, yref = geom.to_image().coord_to_pix(skydir)
        xref = int(np.floor(-xref + npix[0] / 2.0)) + geom.wcs.wcs.crpix[0]
        yref = int(np.floor(-yref + npix[1] / 2.0)) + geom.wcs.wcs.crpix[1]
        return cls.create(
            skydir=tuple(geom.wcs.wcs.crval),
            npix=npix,
            refpix=(xref, yref),
            frame=geom.frame,
            binsz=tuple(geom.pixel_scales.deg),
            axes=geom.axes,
            proj=geom.projection,
        )

    @classmethod
    def from_header(cls, header, hdu_bands=None, format="gadf"):
        """Create a WCS geometry object from a FITS header.

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            The FITS header
        hdu_bands : `~astropy.io.fits.BinTableHDU`
            The BANDS table HDU.
        format : {'gadf', 'fgst-ccube','fgst-template'}
            FITS format convention.

        Returns
        -------
        wcs : `~WcsGeom`
            WCS geometry object.
        """
        wcs = WCS(header, naxis=2)
        # TODO: see https://github.com/astropy/astropy/issues/9259
        wcs._naxis = wcs._naxis[:2]

        axes = MapAxes.from_table_hdu(hdu_bands, format=format)
        shape = axes.shape

        if hdu_bands is not None and "NPIX" in hdu_bands.columns.names:
            npix = hdu_bands.data.field("NPIX").reshape(shape + (2,))
            npix = (npix[..., 0], npix[..., 1])
            cdelt = hdu_bands.data.field("CDELT").reshape(shape + (2,))
            cdelt = (cdelt[..., 0], cdelt[..., 1])
        elif "WCSSHAPE" in header:
            wcs_shape = eval(header["WCSSHAPE"])
            npix = (wcs_shape[0], wcs_shape[1])
            cdelt = None
            wcs.array_shape = npix
        else:
            npix = (header["NAXIS1"], header["NAXIS2"])
            cdelt = None

        return cls(wcs, npix, cdelt=cdelt, axes=axes)

    def _make_bands_cols(self):

        cols = []
        if not self.is_regular:
            cols += [
                fits.Column(
                    "NPIX",
                    "2I",
                    dim="(2)",
                    array=np.vstack((np.ravel(self.npix[0]), np.ravel(self.npix[1]))).T,
                )
            ]
            cols += [
                fits.Column(
                    "CDELT",
                    "2E",
                    dim="(2)",
                    array=np.vstack(
                        (np.ravel(self._cdelt[0]), np.ravel(self._cdelt[1]))
                    ).T,
                )
            ]
            cols += [
                fits.Column(
                    "CRPIX",
                    "2E",
                    dim="(2)",
                    array=np.vstack(
                        (np.ravel(self._crpix[0]), np.ravel(self._crpix[1]))
                    ).T,
                )
            ]
        return cols

    def to_header(self):
        header = self.wcs.to_header()
        header.update(self.axes.to_header())
        shape = "{},{}".format(np.max(self.npix[0]), np.max(self.npix[1]))
        for ax in self.axes:
            shape += f",{ax.nbin}"

        header["WCSSHAPE"] = f"({shape})"
        return header

    def get_idx(self, idx=None, flat=False):
        pix = self.get_pix(idx=idx, mode="center")
        if flat:
            pix = tuple([p[np.isfinite(p)] for p in pix])
        return pix_tuple_to_idx(pix)

    def _get_pix_all(
        self, idx=None, mode="center", sparse=False, axis_name=("lon", "lat")
    ):
        """Get idx coordinate array without footprint of the projection applied"""
        pix_all = []

        for name, nbin in zip(self.axes_names, self._shape):
            if mode == "edges" and name in axis_name:
                pix = np.arange(-0.5, nbin, dtype=float)
            else:
                pix = np.arange(nbin, dtype=float)

            pix_all.append(pix)

        # TODO: improve varying bin size coordinate handling
        if idx is not None:
            pix_all = pix_all[self._slice_spatial_axes] + [float(t) for t in idx]

        return np.meshgrid(*pix_all[::-1], indexing="ij", sparse=sparse)[::-1]

    def get_pix(self, idx=None, mode="center"):
        """Get map pix coordinates from the geometry.

        Parameters
        ----------
        mode : {'center', 'edges'}
            Get center or edge pix coordinates for the spatial axes.

        Returns
        -------
        coord : tuple
            Map pix coordinate tuple.
        """
        pix = self._get_pix_all(idx=idx, mode=mode)
        coords = self.pix_to_coord(pix)
        m = np.isfinite(coords[0])
        for _ in pix:
            _[~m] = INVALID_INDEX.float
        return pix

    def get_coord(
        self, idx=None, mode="center", frame=None, sparse=False, axis_name=None
    ):
        """Get map coordinates from the geometry.

        Parameters
        ----------
        mode : {'center', 'edges'}
            Get center or edge coordinates for the spatial axes.
        frame : str or `~astropy.coordinates.Frame`
            Coordinate frame
        sparse : bool
            Compute sparse coordinates
        axis_name : str
            If mode = "edges", the edges will be returned for this axis.

        Returns
        -------
        coord : `~MapCoord`
            Map coordinate object.
        """
        if axis_name is None:
            axis_name = ("lon", "lat")

        if frame is None:
            frame = self.frame

        pix = self._get_pix_all(idx=idx, mode=mode, sparse=sparse, axis_name=axis_name)

        data = self.pix_to_coord(pix)

        coords = MapCoord.create(
            data=data, frame=self.frame, axis_names=self.axes.names
        )
        return coords.to_frame(frame)

    def coord_to_pix(self, coords):
        coords = MapCoord.create(coords, frame=self.frame, axis_names=self.axes.names)

        if coords.size == 0:
            return tuple([np.array([]) for i in range(coords.ndim)])

        # Variable Bin Size
        if not self.is_regular:
            idxs = self.axes.coord_to_idx(coords, clip=True)
            crpix = [t[idxs] for t in self._crpix]
            cdelt = [t[idxs] for t in self._cdelt]
            pix = world2pix(self.wcs, cdelt, crpix, (coords.lon, coords.lat))
            pix = list(pix)
        else:
            pix = self._wcs.wcs_world2pix(coords.lon, coords.lat, 0)

        pix += self.axes.coord_to_pix(coords)
        return tuple(pix)

    def pix_to_coord(self, pix):
        # Variable Bin Size
        if not self.is_regular:
            idxs = pix_tuple_to_idx(pix[self._slice_non_spatial_axes])
            crpix = [t[idxs] for t in self._crpix]
            cdelt = [t[idxs] for t in self._cdelt]
            coords = pix2world(self.wcs, cdelt, crpix, pix[self._slice_spatial_axes])
        else:
            coords = self._wcs.wcs_pix2world(pix[0], pix[1], 0)

        coords = (
            u.Quantity(coords[0], unit="deg", copy=False),
            u.Quantity(coords[1], unit="deg", copy=False),
        )

        coords += self.axes.pix_to_coord(pix[self._slice_non_spatial_axes])
        return coords

    def pix_to_idx(self, pix, clip=False):
        pix = pix_tuple_to_idx(pix)

        idx_non_spatial = self.axes.pix_to_idx(
            pix[self._slice_non_spatial_axes], clip=clip
        )

        if not self.is_regular:
            npix = (self.npix[0][idx_non_spatial], self.npix[1][idx_non_spatial])
        else:
            npix = self.npix

        idx_spatial = []

        for idx, npix_ in zip(pix[self._slice_spatial_axes], npix):
            if clip:
                idx = np.clip(idx, 0, npix_)
            else:
                idx = np.where((idx < 0) | (idx >= npix_), -1, idx)

            idx_spatial.append(idx)

        return tuple(idx_spatial) + idx_non_spatial

    def contains(self, coords):
        idx = self.coord_to_idx(coords)
        return np.all(np.stack([t != INVALID_INDEX.int for t in idx]), axis=0)

    def to_image(self):
        return self._image_geom

    @lazyproperty
    def _image_geom(self):
        npix = (np.max(self._npix[0]), np.max(self._npix[1]))
        cdelt = (np.max(self._cdelt[0]), np.max(self._cdelt[1]))
        return self.__class__(self._wcs, npix, cdelt=cdelt)

    def to_cube(self, axes):
        npix = (np.max(self._npix[0]), np.max(self._npix[1]))
        cdelt = (np.max(self._cdelt[0]), np.max(self._cdelt[1]))
        axes = copy.deepcopy(self.axes) + axes
        return self.__class__(
            self._wcs.deepcopy(),
            npix,
            cdelt=cdelt,
            axes=axes,
        )

    def _pad_spatial(self, pad_width):
        if np.isscalar(pad_width):
            pad_width = (pad_width, pad_width)

        npix = (self.npix[0] + 2 * pad_width[0], self.npix[1] + 2 * pad_width[1])
        wcs = self._wcs.deepcopy()
        wcs.wcs.crpix += np.array(pad_width)
        cdelt = copy.deepcopy(self._cdelt)
        return self.__class__(wcs, npix, cdelt=cdelt, axes=copy.deepcopy(self.axes))

    def crop(self, crop_width):
        if np.isscalar(crop_width):
            crop_width = (crop_width, crop_width)

        npix = (self.npix[0] - 2 * crop_width[0], self.npix[1] - 2 * crop_width[1])
        wcs = self._wcs.deepcopy()
        wcs.wcs.crpix -= np.array(crop_width)
        cdelt = copy.deepcopy(self._cdelt)
        return self.__class__(wcs, npix, cdelt=cdelt, axes=copy.deepcopy(self.axes))

    def downsample(self, factor, axis_name=None):
        if axis_name is None:
            if np.any(np.mod(self.npix, factor) > 0):
                raise ValueError(
                    f"Spatial shape not divisible by factor {factor!r} in all axes."
                    f" You need to pad prior to calling downsample."
                )

            npix = (self.npix[0] / factor, self.npix[1] / factor)
            cdelt = (self._cdelt[0] * factor, self._cdelt[1] * factor)
            wcs = get_resampled_wcs(self.wcs, factor, True)
            return self._init_copy(wcs=wcs, npix=npix, cdelt=cdelt)
        else:
            if not self.is_regular:
                raise NotImplementedError(
                    "Upsampling in non-spatial axes not supported for irregular geometries"
                )
            axes = self.axes.downsample(factor=factor, axis_name=axis_name)
            return self._init_copy(axes=axes)

    def upsample(self, factor, axis_name=None):
        if axis_name is None:
            npix = (self.npix[0] * factor, self.npix[1] * factor)
            cdelt = (self._cdelt[0] / factor, self._cdelt[1] / factor)
            wcs = get_resampled_wcs(self.wcs, factor, False)
            return self._init_copy(wcs=wcs, npix=npix, cdelt=cdelt)
        else:
            if not self.is_regular:
                raise NotImplementedError(
                    "Upsampling in non-spatial axes not supported for irregular geometries"
                )
            axes = self.axes.upsample(factor=factor, axis_name=axis_name)
            return self._init_copy(axes=axes)

    def to_binsz(self, binsz):
        """Change pixel size of the geometry.

        Parameters
        ----------
        binsz : float or tuple or list
            New pixel size in degree.

        Returns
        -------
        geom : `WcsGeom`
            Geometry with new pixel size.
        """
        return self.create(
            skydir=self.center_skydir,
            binsz=binsz,
            width=self.width,
            proj=self.projection,
            frame=self.frame,
            axes=copy.deepcopy(self.axes),
        )

    def solid_angle(self):
        """Solid angle array (`~astropy.units.Quantity` in ``sr``).

        The array has the same dimension as the WcsGeom object.

        To return solid angles for the spatial dimensions only use::

             WcsGeom.to_image().solid_angle()
        """
        return self._solid_angle

    @lazyproperty
    def _solid_angle(self):
        coord = self.get_coord(mode="edges").skycoord

        # define pixel corners
        low_left = coord[..., :-1, :-1]
        low_right = coord[..., 1:, :-1]
        up_left = coord[..., :-1, 1:]
        up_right = coord[..., 1:, 1:]

        # compute side lengths
        low = low_left.separation(low_right)
        left = low_left.separation(up_left)
        up = up_left.separation(up_right)
        right = low_right.separation(up_right)

        # compute enclosed angles
        angle_low_right = low_right.position_angle(up_right) - low_right.position_angle(
            low_left
        )
        angle_up_left = up_left.position_angle(up_right) - low_left.position_angle(
            up_left
        )

        # compute area assuming a planar triangle
        area_low_right = 0.5 * low * right * np.sin(angle_low_right)
        area_up_left = 0.5 * up * left * np.sin(angle_up_left)
        # TODO: for non-negative cdelt a negative solid angle is returned
        #  find out why and fix properly
        return np.abs(u.Quantity(area_low_right + area_up_left, "sr", copy=False))

    def bin_volume(self):
        """Bin volume (`~astropy.units.Quantity`)"""
        return self._bin_volume

    @lazyproperty
    def _bin_volume(self):
        """Cached property of bin volume"""
        value = self.to_image().solid_angle()

        if not self.is_image:
            value = value * self.axes.bin_volume()

        return value

    def separation(self, center):
        """Compute sky separation wrt a given center.

        Parameters
        ----------
        center : `~astropy.coordinates.SkyCoord`
            Center position

        Returns
        -------
        separation : `~astropy.coordinates.Angle`
            Separation angle array (2D)
        """
        coord = self.to_image().get_coord()
        return center.separation(coord.skycoord)

    def cutout(self, position, width, mode="trim", odd_npix=False):
        """
        Create a cutout around a given position.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Center position of the cutout region.
        width : tuple of `~astropy.coordinates.Angle`
            Angular sizes of the region in (lon, lat) in that specific order.
            If only one value is passed, a square region is extracted.
        mode : {'trim', 'partial', 'strict'}
            Mode option for Cutout2D, for details see `~astropy.nddata.utils.Cutout2D`.
        odd_npix : bool
            Force width to odd number of pixels.

        Returns
        -------
        cutout : `~gammapy.maps.WcsNDMap`
            Cutout map
        """
        width = _check_width(width) * u.deg

        binsz = self.pixel_scales
        width_npix = np.clip((width / binsz).to_value(""), 1, None)
        width = width_npix * binsz

        if odd_npix:
            width = round_up_to_odd(width_npix)

        dummy_data = np.empty(self.to_image().data_shape, dtype=bool)
        c2d = Cutout2D(
            data=dummy_data,
            wcs=self.wcs,
            position=position,
            # Cutout2D takes size with order (lat, lon)
            size=width[::-1],
            mode=mode,
        )
        return self._init_copy(wcs=c2d.wcs, npix=c2d.shape[::-1])

    def boundary_mask(self, width):
        """Create a mask applying binary erosion with a given width from geom edges

        Parameters
        ----------
        width : tuple of `~astropy.units.Quantity`
            Angular sizes of the margin in (lon, lat) in that specific order.
            If only one value is passed, the same margin is applied in (lon, lat).

        Returns
        -------
        mask_map : `~gammapy.maps.WcsNDMap` of boolean type
            Boundary mask

        """
        from .ndmap import WcsNDMap

        data = np.ones(self.data_shape, dtype=bool)
        return WcsNDMap.from_geom(self, data=data).binary_erode(
            width=2 * u.Quantity(width), kernel="box"
        )

    def region_mask(self, regions, inside=True):
        """Create a mask from a given list of regions

        The mask is filled such that a pixel inside the region is filled with
        "True". To invert the mask, e.g. to create a mask with exclusion regions
        the tilde (~) operator can be used (see example below).

        Parameters
        ----------
        regions : str, `~regions.Region` or list of `~regions.Region`
            Region or list of regions (pixel or sky regions accepted).
            A region can be defined as a string ind DS9 format as well.
            See http://ds9.si.edu/doc/ref/region.html for details.
        inside : bool
            For ``inside=True``, pixels in the region to True (the default).
            For ``inside=False``, pixels in the region are False.

        Returns
        -------
        mask_map : `~gammapy.maps.WcsNDMap` of boolean type
            Boolean region mask


        Examples
        --------
        Make an exclusion mask for a circular region::

            from regions import CircleSkyRegion
            from astropy.coordinates import SkyCoord, Angle
            from gammapy.maps import WcsNDMap, WcsGeom

            pos = SkyCoord(0, 0, unit='deg')
            geom = WcsGeom.create(skydir=pos, npix=100, binsz=0.1)

            region = CircleSkyRegion(
                SkyCoord(3, 2, unit='deg'),
                Angle(1, 'deg'),
            )

            # the Gammapy convention for exclusion regions is to take the inverse
            mask = ~geom.region_mask([region])

        Note how we made a list with a single region,
        since this method expects a list of regions.
        """
        from gammapy.maps import Map, RegionGeom

        if not self.is_regular:
            raise ValueError("Multi-resolution maps not supported yet")

        geom = RegionGeom.from_regions(regions, wcs=self.wcs)
        idx = self.get_idx()
        mask = geom.contains_wcs_pix(idx)

        if not inside:
            np.logical_not(mask, out=mask)

        return Map.from_geom(self, data=mask)

    def region_weights(self, regions, oversampling_factor=10):
        """Compute regions weights

        Parameters
        ----------
        regions : str, `~regions.Region` or list of `~regions.Region`
            Region or list of regions (pixel or sky regions accepted).
            A region can be defined as a string ind DS9 format as well.
            See http://ds9.si.edu/doc/ref/region.html for details.
        oversampling_factor : int
            Over-sampling factor to compute the region weights

        Returns
        -------
        map : `~gammapy.maps.WcsNDMap` of boolean type
            Weights region mask
        """
        geom = self.upsample(factor=oversampling_factor)
        m = geom.region_mask(regions=regions)
        m.data = m.data.astype(float)
        return m.downsample(factor=oversampling_factor, preserve_counts=False)

    def binary_structure(self, width, kernel="disk"):
        """Get binary structure

        Parameters
        ----------
        width : `~astropy.units.Quantity`, str or float
            If a float is given it interpreted as width in pixels. If an (angular)
            quantity is given it converted to pixels using ``geom.wcs.wcs.cdelt``.
            The width corresponds to radius in case of a disk kernel, and
            the side length in case of a box kernel.
        kernel : {'disk', 'box'}
            Kernel shape

        Returns
        -------
        structure : `~numoy.ndarray`
            Binary structure
        """
        width = u.Quantity(width)

        if width.unit.is_equivalent("deg"):
            width = width / self.pixel_scales

        width = round_up_to_odd(width.to_value(""))

        if kernel == "disk":
            disk = Tophat2DKernel(width[0])
            disk.normalize("peak")
            structure = disk.array
        elif kernel == "box":
            structure = np.ones(width)
        else:
            raise ValueError(f"Invalid kernel: {kernel!r}")

        shape = (1,) * len(self.axes) + structure.shape
        return structure.reshape(shape)

    def __repr__(self):
        lon = self.center_skydir.data.lon.deg
        lat = self.center_skydir.data.lat.deg
        lon_ref, lat_ref = self.wcs.wcs.crval

        return (
            f"{self.__class__.__name__}\n\n"
            f"\taxes       : {self.axes_names}\n"
            f"\tshape      : {self.data_shape[::-1]}\n"
            f"\tndim       : {self.ndim}\n"
            f"\tframe      : {self.frame}\n"
            f"\tprojection : {self.projection}\n"
            f"\tcenter     : {lon:.1f} deg, {lat:.1f} deg\n"
            f"\twidth      : {self.width[0][0]:.1f} x {self.width[1][0]:.1f}\n"
            f"\twcs ref    : {lon_ref:.1f} deg, {lat_ref:.1f} deg\n"
        )

    def to_odd_npix(self, max_radius=None):
        """Create a new geom object with an odd number of pixel and a maximum size.

        This is useful for PSF kernel creation.

        Parameters
        ----------
        max_radius : `~astropy.units.Quantity`
            Max. radius of the geometry (half the width)

        Returns
        -------
        geom : `WcsGeom`
            Geom with odd number of pixels
        """
        if max_radius is None:
            width = self.width.max()
        else:
            width = 2 * u.Quantity(max_radius)

        binsz = self.pixel_scales.max()

        width_npix = (width / binsz).to_value("")
        npix = round_up_to_odd(width_npix)
        return WcsGeom.create(
            skydir=self.center_skydir,
            binsz=binsz,
            npix=npix,
            proj=self.projection,
            frame=self.frame,
            axes=self.axes,
        )

    def to_even_npix(self):
        """Create a new geom object with an even number of pixel and a maximum size.

        Returns
        -------
        geom : `WcsGeom`
            Geom with odd number of pixels
        """
        width = self.width.max()
        binsz = self.pixel_scales.max()

        width_npix = (width / binsz).to_value("")
        npix = round_up_to_even(width_npix)
        return WcsGeom.create(
            skydir=self.center_skydir,
            binsz=binsz,
            npix=npix,
            proj=self.projection,
            frame=self.frame,
            axes=self.axes,
        )

    def is_aligned(self, other, tolerance=1e-6):
        """Check if WCS and extra axes are aligned.

        Parameters
        ----------
        other : `WcsGeom`
            Other geom.
        tolerance : float
            Tolerance for the comparison.

        Returns
        -------
        aligned : bool
            Whether geometries are aligned
        """
        for axis, otheraxis in zip(self.axes, other.axes):
            if axis != otheraxis:
                return False

        # check WCS consistency with a priori tolerance of 1e-6
        return self.wcs.wcs.compare(other.wcs.wcs, cmp=2, tolerance=tolerance)

    def is_allclose(self, other, rtol_axes=1e-6, atol_axes=1e-6, rtol_wcs=1e-6):
        """Compare two data IRFs for equivalency

        Parameters
        ----------
        other :  `WcsGeom`
            Geom to compare against
        rtol_axes : float
            Relative tolerance for the axes comparison.
        atol_axes : float
            Relative tolerance for the axes comparison.
        rtol_wcs : float
            Relative tolerance for the wcs comparison.

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

        # check WCS consistency with a priori tolerance of 1e-6
        # cmp=1 parameter ensures no comparison with ancillary information
        # see https://github.com/astropy/astropy/pull/4522/files
        wcs_eq = self.wcs.wcs.compare(other.wcs.wcs, cmp=1, tolerance=rtol_wcs)

        return axes_eq and wcs_eq

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if not (self.is_regular and other.is_regular):
            raise NotImplementedError(
                "Geom comparison is not possible for irregular geometries."
            )

        return self.is_allclose(other=other, rtol_wcs=1e-6, rtol_axes=1e-6)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return id(self)


def pix2world(wcs, cdelt, crpix, pix):
    """Perform pixel to world coordinate transformation.

    For a WCS projection with a given pixel size (CDELT) and reference pixel
    (CRPIX). This method can be used to perform WCS transformations
    for projections with different pixelizations but the same
    reference coordinate (CRVAL), projection type, and coordinate system.

    Parameters
    ----------
    wcs : `astropy.wcs.WCS`
        WCS transform object.
    cdelt : tuple
        Tuple of X/Y pixel size in deg.  Each element should have the
        same length as ``pix``.
    crpix : tuple
        Tuple of reference pixel parameters in X and Y dimensions.  Each
        element should have the same length as ``pix``.
    pix : tuple
        Tuple of pixel coordinates.
    """
    pix_ratio = [
        np.abs(wcs.wcs.cdelt[0] / cdelt[0]),
        np.abs(wcs.wcs.cdelt[1] / cdelt[1]),
    ]
    pix = (
        (pix[0] - (crpix[0] - 1.0)) / pix_ratio[0] + wcs.wcs.crpix[0] - 1.0,
        (pix[1] - (crpix[1] - 1.0)) / pix_ratio[1] + wcs.wcs.crpix[1] - 1.0,
    )
    return wcs.wcs_pix2world(pix[0], pix[1], 0)


def world2pix(wcs, cdelt, crpix, coord):
    pix_ratio = [
        np.abs(wcs.wcs.cdelt[0] / cdelt[0]),
        np.abs(wcs.wcs.cdelt[1] / cdelt[1]),
    ]
    pix = wcs.wcs_world2pix(coord[0], coord[1], 0)
    return (
        (pix[0] - (wcs.wcs.crpix[0] - 1.0)) * pix_ratio[0] + crpix[0] - 1.0,
        (pix[1] - (wcs.wcs.crpix[1] - 1.0)) * pix_ratio[1] + crpix[1] - 1.0,
    )
