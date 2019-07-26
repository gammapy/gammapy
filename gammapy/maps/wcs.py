# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
from collections import OrderedDict
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
from regions import SkyRegion
from .geom import MapGeom, MapCoord, pix_tuple_to_idx, skycoord_to_lonlat
from .geom import get_shape, make_axes, find_and_read_bands, axes_pix_to_coord
from .utils import INVALID_INDEX

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


# TODO: remove this function, move code to the one caller below
def _make_image_header(
    nxpix=100,
    nypix=100,
    binsz=0.1,
    xref=0,
    yref=0,
    proj="CAR",
    coordsys="GAL",
    xrefpix=None,
    yrefpix=None,
):
    """Generate a FITS header from scratch.

    Uses the same parameter names as the Fermi tool gtbin.

    If no reference pixel position is given it is assumed ot be
    at the center of the image.

    Parameters
    ----------
    nxpix : int, optional
        Number of pixels in x axis. Default is 100.
    nypix : int, optional
        Number of pixels in y axis. Default is 100.
    binsz : float, optional
        Bin size for x and y axes in units of degrees. Default is 0.1.
    xref : float, optional
        Coordinate system value at reference pixel for x axis. Default is 0.
    yref : float, optional
        Coordinate system value at reference pixel for y axis. Default is 0.
    proj : string, optional
        Projection type. Default is 'CAR' (cartesian).
    coordsys : {'CEL', 'GAL'}, optional
        Coordinate system. Default is 'GAL' (Galactic).
    xrefpix : float, optional
        Coordinate system reference pixel for x axis. Default is None.
    yrefpix: float, optional
        Coordinate system reference pixel for y axis. Default is None.

    Returns
    -------
    header : `~astropy.io.fits.Header`
        Header
    """
    nxpix = int(nxpix)
    nypix = int(nypix)
    if not xrefpix:
        xrefpix = (nxpix + 1) / 2.0
    if not yrefpix:
        yrefpix = (nypix + 1) / 2.0

    if coordsys == "CEL":
        ctype1, ctype2 = "RA---", "DEC--"
    elif coordsys == "GAL":
        ctype1, ctype2 = "GLON-", "GLAT-"
    else:
        raise ValueError("Unsupported coordsys: {!r}".format(coordsys))

    pars = {
        "NAXIS": 2,
        "NAXIS1": nxpix,
        "NAXIS2": nypix,
        "CTYPE1": ctype1 + proj,
        "CRVAL1": xref,
        "CRPIX1": xrefpix,
        "CUNIT1": "deg",
        "CDELT1": -binsz,
        "CTYPE2": ctype2 + proj,
        "CRVAL2": yref,
        "CRPIX2": yrefpix,
        "CUNIT2": "deg",
        "CDELT2": binsz,
    }

    header = fits.Header()
    header.update(pars)

    return header


class WcsGeom(MapGeom):
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
    conv : {'gadf', 'fgst-ccube', 'fgst-template'}
        Serialization format convention.  This sets the default format
        that will be used when writing this geometry to a file.
    """

    _slice_spatial_axes = slice(0, 2)
    _slice_non_spatial_axes = slice(2, None)
    is_hpx = False

    def __init__(self, wcs, npix, cdelt=None, crpix=None, axes=None, conv="gadf"):
        self._wcs = wcs
        self._coordsys = get_coordys(wcs)
        self._projection = get_projection(wcs)
        self._conv = conv
        self._axes = make_axes(axes, conv)

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

    @property
    def data_shape(self):
        """Shape of the Numpy data array matching this geometry."""
        return self._shape[::-1]

    @property
    def _shape(self):
        npix_shape = [np.max(self.npix[0]), np.max(self.npix[1])]
        ax_shape = [ax.nbin for ax in self.axes]
        return tuple(npix_shape + ax_shape)

    @property
    def _shape_edges(self):
        npix_shape = [np.max(self.npix[0]) + 1, np.max(self.npix[1]) + 1]
        ax_shape = [ax.nbin for ax in self.axes]
        return tuple(npix_shape + ax_shape)

    @property
    def shape_axes(self):
        """Shape of non-spatial axes."""
        return self._shape[self._slice_non_spatial_axes]

    @property
    def wcs(self):
        """WCS projection object."""
        return self._wcs

    @property
    def coordsys(self):
        """Coordinate system of the projection.

        Galactic ('GAL') or Equatorial ('CEL').
        """
        return self._coordsys

    @property
    def projection(self):
        """Map projection."""
        return self._projection

    @property
    def is_allsky(self):
        """Flag for all-sky maps."""
        if np.all(np.isclose(self._npix[0] * self._cdelt[0], 360.0)):
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
    def conv(self):
        """Name of default FITS convention associated with this geometry."""
        return self._conv

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
        coordsys="CEL",
        refpix=None,
        axes=None,
        skydir=None,
        width=None,
        conv="gadf",
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
        coordsys : {'CEL', 'GAL'}, optional
            Coordinate system, either Galactic ('GAL') or Equatorial ('CEL').
        axes : list
            List of non-spatial axes.
        proj : string, optional
            Any valid WCS projection type. Default is 'CAR' (cartesian).
        refpix : tuple
            Reference pixel of the projection.  If None this will be
            set to the center of the map.
        conv : string, optional
            FITS format convention ('fgst-ccube', 'fgst-template',
            'gadf').  Default is 'gadf'.

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
            xref, yref = (0.0, 0.0)
        elif isinstance(skydir, tuple):
            xref, yref = skydir
        elif isinstance(skydir, SkyCoord):
            xref, yref, frame = skycoord_to_lonlat(skydir, coordsys=coordsys)
        else:
            raise ValueError("Invalid type for skydir: {!r}".format(type(skydir)))

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
            refpix = (None, None)

        header = _make_image_header(
            nxpix=npix[0].flat[0],
            nypix=npix[1].flat[0],
            binsz=binsz[0].flat[0],
            xref=float(xref),
            yref=float(yref),
            proj=proj,
            coordsys=coordsys,
            xrefpix=refpix[0],
            yrefpix=refpix[1],
        )
        wcs = WCS(header)
        return cls(wcs, npix, cdelt=binsz, axes=axes, conv=conv)

    @classmethod
    def from_header(cls, header, hdu_bands=None):
        """Create a WCS geometry object from a FITS header.

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            The FITS header
        hdu_bands : `~astropy.io.fits.BinTableHDU`
            The BANDS table HDU.

        Returns
        -------
        wcs : `~WcsGeom`
            WCS geometry object.
        """
        wcs = WCS(header)
        naxis = wcs.naxis
        for i in range(naxis - 2):
            wcs = wcs.dropaxis(2)

        axes = find_and_read_bands(hdu_bands, header)
        shape = tuple([ax.nbin for ax in axes])
        conv = "gadf"

        # Discover FITS convention
        if hdu_bands is not None:
            if hdu_bands.name == "EBOUNDS":
                conv = "fgst-ccube"
            elif hdu_bands.name == "ENERGIES":
                conv = "fgst-template"

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

        return cls(wcs, npix, cdelt=cdelt, axes=axes, conv=conv)

    def _make_bands_cols(self, hdu=None, conv=None):

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

    def make_header(self):
        header = self.wcs.to_header()
        self._fill_header_from_axes(header)
        shape = "{},{}".format(np.max(self.npix[0]), np.max(self.npix[1]))
        for ax in self.axes:
            shape += ",{}".format(ax.nbin)
        header["WCSSHAPE"] = "({})".format(shape)
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

    def get_coord(self, idx=None, flat=False, mode="center"):
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

        axes_names = ["lon", "lat"] + [ax.name for ax in self.axes]
        cdict = OrderedDict(zip(axes_names, coords))

        return MapCoord.create(cdict, coordsys=self.coordsys)

    def coord_to_pix(self, coords):
        coords = MapCoord.create(coords, coordsys=self.coordsys)

        if coords.size == 0:
            return tuple([np.array([]) for i in range(coords.ndim)])

        c = self.coord_to_tuple(coords)
        # Variable Bin Size
        if not self.is_regular:
            idxs = tuple(
                [
                    np.clip(ax.coord_to_idx(c[i + 2]), 0, ax.nbin - 1)
                    for i, ax in enumerate(self.axes)
                ]
            )
            crpix = [t[idxs] for t in self._crpix]
            cdelt = [t[idxs] for t in self._cdelt]
            pix = world2pix(self.wcs, cdelt, crpix, (coords.lon, coords.lat))
            pix = list(pix)
        else:
            pix = self._wcs.wcs_world2pix(coords.lon, coords.lat, 0)

        for coord, ax in zip(c[self._slice_non_spatial_axes], self.axes):
            pix += [ax.coord_to_pix(coord)]

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

        coords += axes_pix_to_coord(self.axes, pix[self._slice_non_spatial_axes])
        return tuple(coords)

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
        return self.__class__(self._wcs, npix, cdelt=cdelt)

    def to_cube(self, axes):
        npix = (np.max(self._npix[0]), np.max(self._npix[1]))
        cdelt = (np.max(self._cdelt[0]), np.max(self._cdelt[1]))
        axes = copy.deepcopy(self.axes) + axes
        return self.__class__(self._wcs.deepcopy(), npix, cdelt=cdelt, axes=axes)

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

    def downsample(self, factor, axis=None):
        if axis is None:
            if np.any(np.mod(self.npix, factor) > 0):
                raise ValueError(
                    "Spatial shape not divisible by factor {!r} in all axes."
                    " You need to pad prior to calling downsample.".format(factor)
                )

            npix = (self.npix[0] / factor, self.npix[1] / factor)
            cdelt = (self._cdelt[0] * factor, self._cdelt[1] * factor)
            wcs = get_resampled_wcs(self.wcs, factor, True)
            return self._init_copy(wcs=wcs, npix=npix, cdelt=cdelt)
        else:
            if not self.is_regular:
                raise NotImplementedError(
                    "Upsampling in non-spatial axes not"
                    " supported for irregular geometries"
                )

            axes = copy.deepcopy(self.axes)
            idx = self.get_axis_index_by_name(axis)
            axes[idx] = axes[idx].downsample(factor)
            return self._init_copy(axes=axes)

    def upsample(self, factor, axis=None):
        if axis is None:
            npix = (self.npix[0] * factor, self.npix[1] * factor)
            cdelt = (self._cdelt[0] / factor, self._cdelt[1] / factor)
            wcs = get_resampled_wcs(self.wcs, factor, False)
            return self._init_copy(wcs=wcs, npix=npix, cdelt=cdelt)
        else:
            if not self.is_regular:
                raise NotImplementedError(
                    "Upsampling in non-spatial axes not"
                    " supported for irregular geometries"
                )
            axes = copy.deepcopy(self.axes)
            idx = self.get_axis_index_by_name(axis)
            axes[idx] = axes[idx].upsample(factor)
            return self._init_copy(axes=axes)

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

        return u.Quantity(area_low_right + area_up_left, "sr", copy=False)

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

        return self._init_copy(wcs=c2d.wcs, npix=c2d.shape[::-1])

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
        str_ = self.__class__.__name__
        str_ += "\n\n"
        axes = ["lon", "lat"] + [_.name for _ in self.axes]
        str_ += "\taxes       : {}\n".format(", ".join(axes))
        str_ += "\tshape      : {}\n".format(self.data_shape[::-1])
        str_ += "\tndim       : {}\n".format(self.ndim)
        str_ += "\tcoordsys   : {}\n".format(self.coordsys)
        str_ += "\tprojection : {}\n".format(self.projection)
        lon = self.center_skydir.data.lon.deg
        lat = self.center_skydir.data.lat.deg
        str_ += "\tcenter     : {:.1f} deg, {:.1f} deg\n".format(lon, lat)
        str_ += "\twidth      : {width[0][0]:.1f} x {width[1][0]:.1f}\n".format(
            width=self.width
        )
        return str_

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        # check overall shape and axes compatibility
        if self.data_shape != other.data_shape:
            return False

        for axis, otheraxis in zip(self.axes, other.axes):
            if axis != otheraxis:
                return False

        # check WCS consistency with a priori tolerance of 1e-6
        return self.wcs.wcs.compare(other.wcs.wcs, tolerance=1e-6)

    def __ne__(self, other):
        return not self.__eq__(other)


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


def get_projection(wcs):
    return wcs.wcs.ctype[0][5:]


def get_coordys(wcs):
    if "RA" in wcs.wcs.ctype[0]:
        return "CEL"
    elif "GLON" in wcs.wcs.ctype[0]:
        return "GAL"
    else:
        raise ValueError("Unrecognized WCS coordinate system.")
