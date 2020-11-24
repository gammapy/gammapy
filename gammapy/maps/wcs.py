# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
from functools import lru_cache
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import (
    celestial_frame_to_wcs,
    proj_plane_pixel_scales,
    wcs_to_celestial_frame,
)
from regions import SkyRegion
from .geom import (
    Geom,
    MapAxes,
    MapCoord,
    get_shape,
    pix_tuple_to_idx,
    skycoord_to_lonlat,
)
from .utils import INVALID_INDEX, slice_to_str, str_to_slice

__all__ = ["WcsGeom"]


def _check_width(width):
    """Check and normalise width argument.

    Always returns tuple (lon, lat) as float in degrees.
    """
    if isinstance(width, tuple):
        lon = Angle(width[0], "deg").deg
        lat = Angle(width[1], "deg").deg
        return lon, lat
    else:
        angle = Angle(width, "deg").deg
        if np.isscalar(angle):
            return angle, angle
        else:
            return tuple(angle)


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
    cutout_info : dict
        Dict with cutout info, if the `WcsGeom` was created by `WcsGeom.cutout()`
    """

    _slice_spatial_axes = slice(0, 2)
    _slice_non_spatial_axes = slice(2, None)
    is_hpx = False

    def __init__(self, wcs, npix, cdelt=None, crpix=None, axes=None, cutout_info=None):
        self._wcs = wcs
        self._frame = wcs_to_celestial_frame(wcs).name
        self._projection = wcs.wcs.ctype[0][5:]
        self._axes = MapAxes.from_default(axes)

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
        self._cutout_info = cutout_info

        # define cached methods
        self.get_coord = lru_cache()(self.get_coord)
        self.get_pix = lru_cache()(self.get_pix)
        self.solid_angle = lru_cache()(self.solid_angle)
        self.bin_volume = lru_cache()(self.bin_volume)
        self.to_image = lru_cache()(self.to_image)

    # workaround for the lru_cache pickle issue
    # see e.g. https://github.com/cloudpipe/cloudpickle/issues/178
    def __getstate__(self):
        state = self.__dict__.copy()
        for key, value in state.items():
            func = getattr(value, "__wrapped__", None)
            if func is not None:
                state[key] = func

        return state

    def __setstate__(self, state):
        for key, value in state.items():
            if key in ["get_coord", "solid_angle", "bin_volume", "to_image", "get_pix"]:
                state[key] = lru_cache()(value)

        self.__dict__ = state

    @property
    def data_shape(self):
        """Shape of the Numpy data array matching this geometry."""
        return self._shape[::-1]

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

    @property
    def cutout_info(self):
        """Cutout info dict."""
        return self._cutout_info

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
        frame : {"icrs", "galactic"}, optional
            Coordinate system, either Galactic ("galactic") or Equatorial ("icrs").
        axes : list
            List of non-spatial axes.
        proj : string, optional
            Any valid WCS projection type. Default is 'CAR' (cartesian).
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
        >>> geom = WcsGeom.create(npix=[100,200], binsz=[0.1,0.05], axes=[axis])
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

    @classmethod
    def from_header(cls, header, hdu_bands=None, format=None):
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
        else:
            npix = (header["NAXIS1"], header["NAXIS2"])
            cdelt = None

        if "PSLICE1" in header:
            cutout_info = {}
            cutout_info["parent-slices"] = (
                str_to_slice(header["PSLICE2"]),
                str_to_slice(header["PSLICE1"]),
            )
            cutout_info["cutout-slices"] = (
                str_to_slice(header["CSLICE2"]),
                str_to_slice(header["CSLICE1"]),
            )
        else:
            cutout_info = None

        return cls(wcs, npix, cdelt=cdelt, axes=axes, cutout_info=cutout_info)

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
        header = self.axes.to_header(header)
        shape = "{},{}".format(np.max(self.npix[0]), np.max(self.npix[1]))
        for ax in self.axes:
            shape += f",{ax.nbin}"
        header["WCSSHAPE"] = f"({shape})"

        if self.cutout_info is not None:
            slices_parent = self.cutout_info["parent-slices"]
            slices_cutout = self.cutout_info["cutout-slices"]

            header["PSLICE1"] = (slice_to_str(slices_parent[1]), "Parent slice")
            header["PSLICE2"] = (slice_to_str(slices_parent[0]), "Parent slice")
            header["CSLICE1"] = (slice_to_str(slices_cutout[1]), "Cutout slice")
            header["CSLICE2"] = (slice_to_str(slices_cutout[0]), "Cutout slice")

        return header

    def get_image_shape(self, idx):
        """Get the shape of the image plane at index ``idx``."""
        if self.is_regular:
            return int(self.npix[0]), int(self.npix[1])
        else:
            return int(self.npix[0][idx]), int(self.npix[1][idx])

    def get_idx(self, idx=None, flat=False):
        pix = self.get_pix(idx=idx, mode="center")
        if flat:
            pix = tuple([p[np.isfinite(p)] for p in pix])
        return pix_tuple_to_idx(pix)

    def _get_pix_all(self, idx=None, mode="center"):
        """Get idx coordinate array without footprint of the projection applied"""
        if mode == "edges":
            shape = self._shape_edges
        else:
            shape = self._shape

        if idx is None:
            pix = [np.arange(n, dtype=float) for n in shape]
        else:
            pix = [np.arange(n, dtype=float) for n in shape[self._slice_spatial_axes]]
            pix += [float(t) for t in idx]

        if mode == "edges":
            for pix_array in pix[self._slice_spatial_axes]:
                pix_array -= 0.5

        pix = np.meshgrid(*pix[::-1], indexing="ij")[::-1]
        return pix

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

    def get_coord(self, idx=None, flat=False, mode="center", frame=None):
        """Get map coordinates from the geometry.

        Parameters
        ----------
        mode : {'center', 'edges'}
            Get center or edge coordinates for the spatial axes.

        Returns
        -------
        coord : `~MapCoord`
            Map coordinate object.
        """
        pix = self._get_pix_all(idx=idx, mode=mode)
        coords = self.pix_to_coord(pix)

        if flat:
            is_finite = np.isfinite(coords[0])
            coords = tuple([c[is_finite] for c in coords])

        axes_names = ["lon", "lat"] + self.axes.names
        cdict = dict(zip(axes_names, coords))

        if frame is None:
            frame = self.frame

        return MapCoord.create(cdict, frame=self.frame).to_frame(frame)

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
        # TODO: copy idx to avoid modifying input pix?
        # pix_tuple_to_idx seems to always make a copy!?
        idxs = pix_tuple_to_idx(pix)
        if not self.is_regular:
            ibin = pix[self._slice_non_spatial_axes]
            ibin = pix_tuple_to_idx(ibin)
            for i, ax in enumerate(self.axes):
                np.clip(ibin[i], 0, ax.nbin - 1, out=ibin[i])
            npix = (self.npix[0][ibin], self.npix[1][ibin])
        else:
            npix = self.npix

        for i, idx in enumerate(idxs):
            if clip:
                if i < 2:
                    np.clip(idxs[i], 0, npix[i], out=idxs[i])
                else:
                    np.clip(idxs[i], 0, self.axes[i - 2].nbin - 1, out=idxs[i])
            else:
                if i < 2:
                    np.putmask(idxs[i], (idx < 0) | (idx >= npix[i]), -1)
                else:
                    np.putmask(idxs[i], (idx < 0) | (idx >= self.axes[i - 2].nbin), -1)

        return idxs

    def contains(self, coords):
        idx = self.coord_to_idx(coords)
        return np.all(np.stack([t != INVALID_INDEX.int for t in idx]), axis=0)

    def to_image(self):
        npix = (np.max(self._npix[0]), np.max(self._npix[1]))
        cdelt = (np.max(self._cdelt[0]), np.max(self._cdelt[1]))
        return self.__class__(
            self._wcs, npix, cdelt=cdelt, cutout_info=self.cutout_info
        )

    def to_cube(self, axes):
        npix = (np.max(self._npix[0]), np.max(self._npix[1]))
        cdelt = (np.max(self._cdelt[0]), np.max(self._cdelt[1]))
        axes = copy.deepcopy(self.axes) + axes
        return self.__class__(
            self._wcs.deepcopy(),
            npix,
            cdelt=cdelt,
            axes=axes,
            cutout_info=self.cutout_info,
        )

    def pad(self, pad_width):
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
        bin_volume = self.to_image().solid_angle()

        for idx, ax in enumerate(self.axes):
            shape = self.ndim * [1]
            shape[-(idx + 3)] = -1
            bin_volume = bin_volume * ax.bin_width.reshape(tuple(shape))

        return bin_volume

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

    def cutout(self, position, width, mode="trim"):
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

        Returns
        -------
        cutout : `~gammapy.maps.WcsNDMap`
            Cutout map
        """
        width = _check_width(width)
        dummy_data = np.empty(self.to_image().data_shape)
        c2d = Cutout2D(
            data=dummy_data,
            wcs=self.wcs,
            position=position,
            # Cutout2D takes size with order (lat, lon)
            size=width[::-1] * u.deg,
            mode=mode,
        )

        cutout_info = {
            "parent-slices": c2d.slices_original,
            "cutout-slices": c2d.slices_cutout,
        }

        return self._init_copy(
            wcs=c2d.wcs, npix=c2d.shape[::-1], cutout_info=cutout_info
        )

    def region_mask(self, regions, inside=True):
        """Create a mask from a given list of regions

        Parameters
        ----------
        regions : list of  `~regions.Region`
            Python list of regions (pixel or sky regions accepted)
        inside : bool
            For ``inside=True``, pixels in the region to True (the default).
            For ``inside=False``, pixels in the region are False.

        Returns
        -------
        mask_map : `~numpy.ndarray` of boolean type
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
            mask = geom.region_mask([region], inside=False)

        Note how we made a list with a single region,
        since this method expects a list of regions.

        The return ``mask`` is a boolean Numpy array.
        If you want a map object (e.g. for storing in FITS or plotting),
        this is how you can make the map::

            mask_map = WcsNDMap(geom=geom, data=mask)
            mask_map.plot()
        """
        from regions import PixCoord

        if not self.is_regular:
            raise ValueError("Multi-resolution maps not supported yet")

        idx = self.get_idx()
        pixcoord = PixCoord(idx[0], idx[1])

        mask = np.zeros(self.data_shape, dtype=bool)

        for region in regions:
            if isinstance(region, SkyRegion):
                region = region.to_pixel(self.wcs)
            mask += region.contains(pixcoord)

        if inside is False:
            np.logical_not(mask, out=mask)

        return mask

    def __repr__(self):
        axes = ["lon", "lat"] + [_.name for _ in self.axes]
        lon = self.center_skydir.data.lon.deg
        lat = self.center_skydir.data.lat.deg

        return (
            f"{self.__class__.__name__}\n\n"
            f"\taxes       : {axes}\n"
            f"\tshape      : {self.data_shape[::-1]}\n"
            f"\tndim       : {self.ndim}\n"
            f"\tframe      : {self.frame}\n"
            f"\tprojection : {self.projection}\n"
            f"\tcenter     : {lon:.1f} deg, {lat:.1f} deg\n"
            f"\twidth      : {self.width[0][0]:.1f} x {self.width[1][0]:.1f}\n"
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

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        if not (self.is_regular and other.is_regular):
            raise NotImplementedError(
                "Geom comparison is not possible for irregular geometries."
            )

        # check overall shape and axes compatibility
        if self.data_shape != other.data_shape:
            return False

        for axis, otheraxis in zip(self.axes, other.axes):
            if axis != otheraxis:
                return False

        # check WCS consistency with a priori tolerance of 1e-6
        # cmp=1 parameter ensures no comparison with ancillary information
        # see https://github.com/astropy/astropy/pull/4522/files
        return self.wcs.wcs.compare(other.wcs.wcs, cmp=1, tolerance=1e-6)

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
