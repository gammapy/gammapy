# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import numpy as np
from collections import OrderedDict
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates.angle_utilities import angular_separation
from astropy.coordinates import Angle
import astropy.wcs.utils
import astropy.units as u
from regions import SkyRegion
from ..utils.scripts import make_path
from ..utils.wcs import get_resampled_wcs
from .geom import MapGeom, MapCoord, pix_tuple_to_idx, skycoord_to_lonlat
from .geom import get_shape, make_axes
from .geom import find_and_read_bands

__all__ = [
    'WcsGeom',
]


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


# TODO: remove this function, move code to the one caller below
def _make_image_header(nxpix=100, nypix=100, binsz=0.1, xref=0, yref=0,
                       proj='CAR', coordsys='GAL',
                       xrefpix=None, yrefpix=None):
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
        xrefpix = (nxpix + 1) / 2.
    if not yrefpix:
        yrefpix = (nypix + 1) / 2.

    if coordsys == 'CEL':
        ctype1, ctype2 = 'RA---', 'DEC--'
    elif coordsys == 'GAL':
        ctype1, ctype2 = 'GLON-', 'GLAT-'
    else:
        raise ValueError('Unsupported coordsys: {}'.format(proj))

    pars = {
        'NAXIS': 2, 'NAXIS1': nxpix, 'NAXIS2': nypix,
        'CTYPE1': ctype1 + proj,
        'CRVAL1': xref, 'CRPIX1': xrefpix, 'CUNIT1': 'deg', 'CDELT1': -binsz,
        'CTYPE2': ctype2 + proj,
        'CRVAL2': yref, 'CRPIX2': yrefpix, 'CUNIT2': 'deg', 'CDELT2': binsz,
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
    _slice_non_spatial_axes = slice(2, -1)

    def __init__(self, wcs, npix, cdelt=None, crpix=None, axes=None, conv='gadf'):
        self._wcs = wcs
        self._coordsys = get_coordys(wcs)
        self._projection = get_projection(wcs)
        self._conv = conv
        self._axes = make_axes(axes, conv)

        self._shape = tuple([ax.nbin for ax in self._axes])
        if cdelt is None:
            cdelt = (np.abs(self.wcs.wcs.cdelt[0]),
                     np.abs(self.wcs.wcs.cdelt[1]))

        # Shape to use for WCS transformations
        wcs_shape = max([get_shape(t) for t in [npix, cdelt]])
        if np.sum(wcs_shape) > 1 and wcs_shape != self._shape:
            raise ValueError

        self._npix = cast_to_shape(npix, wcs_shape, int)
        self._cdelt = cast_to_shape(cdelt, wcs_shape, float)
        # By convention CRPIX is indexed from 1
        if crpix is None:
            self._crpix = (1.0 + (self._npix[0] - 1.0) / 2.,
                           1.0 + (self._npix[1] - 1.0) / 2.)
        self._width = (self._cdelt[0] * self._npix[0],
                       self._cdelt[1] * self._npix[1])

        # FIXME: Determine center coord from CRVAL
        self._center_pix = tuple([(self._npix[0].flat[0] - 1.0) / 2.,
                                  (self._npix[1].flat[0] - 1.0) / 2.] +
                                 [(float(ax.nbin) - 1.0) / 2. for ax in self.axes])
        self._center_coord = self.pix_to_coord(self._center_pix)
        self._center_skydir = SkyCoord.from_pixel(self._center_pix[0],
                                                  self._center_pix[1],
                                                  self.wcs)

    @property
    def wcs(self):
        """WCS projection object."""
        return self._wcs

    @property
    def coordsys(self):
        """Coordinate system of the projection, either Galactic ('GAL') or
        Equatorial ('CEL')."""
        return self._coordsys

    @property
    def projection(self):
        """Map projection."""
        return self._projection

    @property
    def is_allsky(self):
        """Flag for all-sky maps."""
        if np.all(np.isclose(self._npix[0] * self._cdelt[0], 360.)):
            return True
        else:
            return False

    @property
    def is_regular(self):
        """Flag identifying whether this geometry is regular in non-spatial
        dimensions.  False for multi-resolution or irregular
        geometries.  If true all image planes have the same pixel
        geometry.
        """
        if self.npix[0].size > 1:
            return False
        else:
            return True

    @property
    def width(self):
        """Tuple with image dimension in deg in longitude and latitude."""
        return self._width

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
    def shape(self):
        """Shape of non-spatial axes."""
        return self._shape

    @property
    def ndim(self):
        return len(self._axes) + 2

    @property
    def center_coord(self):
        """Map coordinate of the center of the geometry.

        Returns
        -------
        coord : tuple
        """
        return self._center_coord

    @property
    def center_pix(self):
        """Pixel coordinate of the center of the geometry.

        Returns
        -------
        pix : tuple
        """
        return self._center_pix

    @property
    def center_skydir(self):
        """Sky coordinate of the center of the geometry.

        Returns
        -------
        pix : `~astropy.coordinates.SkyCoord`
        """
        return self._center_skydir

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
        return Angle(astropy.wcs.utils.proj_plane_pixel_scales(self.wcs), 'deg')

    @classmethod
    def create(cls, npix=None, binsz=0.5, proj='CAR', coordsys='CEL', refpix=None,
               axes=None, skydir=None, width=None, conv='gadf'):
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
            raise ValueError(
                'Invalid type for skydir: {}'.format(type(skydir)))

        shape = max([get_shape(t) for t in [npix, binsz, width]])
        binsz = cast_to_shape(binsz, shape, float)

        # If both npix and width are None then create an all-sky geometry
        if npix is None and width is None:
            width = (360., 180.)

        if npix is None:
            width = cast_to_shape(width, shape, float)
            npix = (np.rint(width[0] / binsz[0]).astype(int),
                    np.rint(width[1] / binsz[1]).astype(int),)
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
        conv = 'gadf'

        # Discover FITS convention
        if hdu_bands is not None:
            if hdu_bands.name == 'EBOUNDS':
                conv = 'fgst-ccube'
            elif hdu_bands.name == 'ENERGIES':
                conv = 'fgst-template'

        if hdu_bands is not None and 'NPIX' in hdu_bands.columns.names:
            npix = hdu_bands.data.field('NPIX').reshape(shape + (2,))
            npix = (npix[..., 0], npix[..., 1])
            cdelt = hdu_bands.data.field('CDELT').reshape(shape + (2,))
            cdelt = (cdelt[..., 0], cdelt[..., 1])
        elif 'WCSSHAPE' in header:
            wcs_shape = eval(header['WCSSHAPE'])
            npix = (wcs_shape[0], wcs_shape[1])
            cdelt = None
        else:
            npix = (header['NAXIS1'], header['NAXIS2'])
            cdelt = None

        return cls(wcs, npix, cdelt=cdelt, axes=axes, conv=conv)

    def _make_bands_cols(self, hdu=None, conv=None):

        cols = []
        if not self.is_regular:
            cols += [fits.Column('NPIX', '2I', dim='(2)',
                                 array=np.vstack((np.ravel(self.npix[0]),
                                                  np.ravel(self.npix[1]))).T), ]
            cols += [fits.Column('CDELT', '2E', dim='(2)',
                                 array=np.vstack((np.ravel(self._cdelt[0]),
                                                  np.ravel(self._cdelt[1]))).T), ]
            cols += [fits.Column('CRPIX', '2E', dim='(2)',
                                 array=np.vstack((np.ravel(self._crpix[0]),
                                                  np.ravel(self._crpix[1]))).T), ]
        return cols

    def make_header(self):
        header = self.wcs.to_header()
        self._fill_header_from_axes(header)
        shape = '{},{}'.format(np.max(self.npix[0]),
                               np.max(self.npix[1]))
        for ax in self.axes:
            shape += ',{}'.format(ax.nbin)
        header['WCSSHAPE'] = '({})'.format(shape)
        return header

    def distance_to_edge(self, skydir):
        """Angular distance from the given direction and
        the edge of the projection."""
        raise NotImplementedError

    def get_image_shape(self, idx):
        """Get the shape of the image plane at index ``idx``."""

        if self.is_regular:
            return int(self.npix[0]), int(self.npix[1])
        else:
            return int(self.npix[0][idx]), int(self.npix[1][idx])

    def get_image_wcs(self, idx):
        raise NotImplementedError

    def get_idx(self, idx=None, flat=False):
        pix = self.get_pix(idx=idx, mode='center')
        if flat:
            pix = tuple([p[np.isfinite(p)] for p in pix])
        return pix_tuple_to_idx(pix)

    def get_pix(self, idx=None, mode='center'):
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
        # FIXME: Figure out if there is some way to employ open/sparse
        # vectors

        # FIXME: It would be more efficient to split this into one
        # method that computes indices and a method that casts
        # those to floats and adds the appropriate offset

        npix = copy.deepcopy(self.npix)

        if mode == 'edges':
            for pix_num in npix[:2]:
                pix_num += 1

        if self.axes and not self.is_regular:

            shape = (np.max(self._npix[0]), np.max(self._npix[1]))

            if idx is None:
                shape = shape + self.shape
            else:
                shape = shape + (1,) * len(self.axes)

            pix2 = [np.full(shape, np.nan, dtype=float)
                    for i in range(2 + len(self.axes))]
            for idx_img in np.ndindex(self.shape):

                if idx is not None and idx_img != idx:
                    continue

                npix0, npix1 = npix[0][idx_img], npix[1][idx_img]
                pix_img = np.meshgrid(np.arange(npix0), np.arange(npix1),
                                      indexing='ij', sparse=False)

                if idx is None:
                    s_img = (slice(0, npix0), slice(0, npix1),) + idx_img
                else:
                    s_img = (slice(0, npix0), slice(0, npix1),) + \
                            (0,) * len(self.axes)

                pix2[0][s_img] = pix_img[0]
                pix2[1][s_img] = pix_img[1]
                for j in range(len(self.axes)):
                    pix2[j + 2][s_img] = idx_img[j]
            pix = [t.T for t in pix2]
        else:
            pix = [np.arange(npix[0], dtype=float),
                   np.arange(npix[1], dtype=float)]

            if idx is None:
                pix += [np.arange(ax.nbin, dtype=float) for ax in self.axes]
            else:
                pix += [float(t) for t in idx]

            pix = np.meshgrid(*pix[::-1], indexing='ij', sparse=False)[::-1]

        if mode == 'edges':
            for pix_array in pix[self._slice_spatial_axes]:
                pix_array -= 0.5

        coords = self.pix_to_coord(pix)
        m = np.isfinite(coords[0])
        for i in range(len(pix)):
            pix[i][~m] = np.nan
        return pix

    def get_coord(self, idx=None, flat=False, mode='center'):
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
        pix = self.get_pix(idx=idx, mode=mode)
        coords = self.pix_to_coord(pix)

        if flat:
            coords = tuple([c[np.isfinite(c)] for c in coords])

        axes_names = ['lon', 'lat'] + [ax.name for ax in self.axes]
        cdict = OrderedDict(zip(axes_names, coords))

        return MapCoord.create(cdict, coordsys=self.coordsys)

    def coord_to_pix(self, coords):
        coords = MapCoord.create(coords, coordsys=self.coordsys)
        if coords.size == 0:
            return tuple([np.array([]) for i in range(coords.ndim)])

        c = self.coord_to_tuple(coords)
        # Variable Bin Size
        if self.axes and self.npix[0].size > 1:
            bins = [ax.coord_to_pix(c[i + 2])
                    for i, ax in enumerate(self.axes)]
            idxs = [np.clip(ax.coord_to_idx(c[i + 2]), 0, ax.nbin - 1)
                    for i, ax in enumerate(self.axes)]
            crpix = [t[idxs] for t in self._crpix]
            cdelt = [t[idxs] for t in self._cdelt]
            pix = world2pix(self.wcs, cdelt, crpix, (coords.lon, coords.lat))
            pix = list(pix) + bins
        else:
            pix = self._wcs.wcs_world2pix(np.array(coords.lon, ndmin=1, copy=False),
                                          np.array(coords.lat, ndmin=1, copy=False), 0)
            for i, ax in enumerate(self.axes):
                pix += [ax.coord_to_pix(c[i + 2])]

        return tuple(pix)

    def pix_to_coord(self, pix):
        # Variable Bin Size
        if self.axes and self.npix[0].size > 1:
            idxs = pix_tuple_to_idx([pix[2 + i] for i, ax
                                     in enumerate(self.axes)])
            vals = [ax.pix_to_coord(pix[2 + i])
                    for i, ax in enumerate(self.axes)]
            crpix = [t[idxs] for t in self._crpix]
            cdelt = [t[idxs] for t in self._cdelt]
            coords = pix2world(self.wcs, cdelt, crpix, pix[:2])
            coords += vals
        else:
            coords = self._wcs.wcs_pix2world(pix[0], pix[1], 0)
            for i, ax in enumerate(self.axes):
                coords += [ax.pix_to_coord(pix[i + 2])]

        return tuple(coords)

    def pix_to_idx(self, pix, clip=False):
        idxs = pix_tuple_to_idx(pix, copy=True)
        if not self.is_regular:
            ibin = [pix[2 + i] for i, ax in enumerate(self.axes)]
            ibin = pix_tuple_to_idx(ibin, copy=True)
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
                    np.putmask(idxs[i], (idx < 0) | (
                            idx >= self.axes[i - 2].nbin), -1)

        return idxs

    def contains(self, coords):
        idx = self.coord_to_idx(coords)
        return np.all(np.stack([t != -1 for t in idx]), axis=0)

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
        npix = (self.npix[0] + 2 * pad_width[0],
                self.npix[1] + 2 * pad_width[1])
        wcs = self._wcs.deepcopy()
        wcs.wcs.crpix += np.array(pad_width)
        return self.__class__(wcs, npix, cdelt=copy.deepcopy(self._cdelt),
                              axes=copy.deepcopy(self.axes))

    def crop(self, crop_width):
        if np.isscalar(crop_width):
            crop_width = (crop_width, crop_width)
        npix = (self.npix[0] - 2 * crop_width[0],
                self.npix[1] - 2 * crop_width[1])
        wcs = self._wcs.deepcopy()
        wcs.wcs.crpix -= np.array(crop_width)
        return self.__class__(wcs, npix, cdelt=copy.deepcopy(self._cdelt),
                              axes=copy.deepcopy(self.axes))

    def downsample(self, factor):

        if (not np.all(np.mod(self.npix[0], factor) == 0) or
                not np.all(np.mod(self.npix[1], factor) == 0)):
            raise ValueError('Data shape is not divisible by {} in all axes.'
                             ' Pad image prior to downsampling to correct'
                             ' shape.'.format(factor))

        npix = (self.npix[0] / factor, self.npix[1] / factor)
        cdelt = (self._cdelt[0] * factor, self._cdelt[1] * factor)
        wcs = get_resampled_wcs(self.wcs, factor, True)
        return self.__class__(wcs, npix, cdelt=cdelt,
                              axes=copy.deepcopy(self.axes))

    def upsample(self, factor):
        npix = (self.npix[0] * factor, self.npix[1] * factor)
        cdelt = (self._cdelt[0] / factor, self._cdelt[1] / factor)
        wcs = get_resampled_wcs(self.wcs, factor, False)
        return self.__class__(wcs, npix, cdelt=cdelt,
                              axes=copy.deepcopy(self.axes))

    def solid_angle(self):
        """Solid angle array (`~astropy.units.Quantity` in ``sr``).

        The array has the same dimension as the WcsGeom object.
        To return solid angles for the spatial dimensions only use: WcsGeom.to_image().solid_angle()
        """
        coord = self.get_coord(mode='edges')
        lon = coord.lon * np.pi / 180.
        lat = coord.lat * np.pi / 180.

        # Compute solid angle using the approximation that it's
        # the product between angular separation of pixel corners.
        # First index is "y", second index is "x"
        ylo_xlo = lon[..., :-1, :-1], lat[..., :-1, :-1]
        ylo_xhi = lon[..., :-1, 1:], lat[..., :-1, 1:]
        yhi_xlo = lon[..., 1:, :-1], lat[..., 1:, :-1]

        dx = angular_separation(*(ylo_xlo + ylo_xhi))
        dy = angular_separation(*(ylo_xlo + yhi_xlo))

        return u.Quantity(dx * dy, 'sr')

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

    def region_mask(self, regions, inside=False):
        """Create a mask from a given list of regions

        TODO: implement list of region for each axis

        Parameters
        ----------
        regions : list of  `~regions.PixelRegion` or `~regions.SkyRegion` objects
            A list of regions on the sky (defined in pixel or sky coordinates).
        inside : bool
            Default ``inside=True`` sets pixels in the region to True.
            For ``inside=False``, pixels in the region are False.

        Returns
        -------
        mask_map : `~numpy.ndarray` of boolean type
            the mask array

        Example
        -------

        Example building a mask and storing it in a `~gammapy.maps.WcsNDMap' ::
            from regions import CircleSkyRegion
            from astropy.coordinates import SkyCoord, Angle
            from gammapy.maps import WcsNDMap, WcsGeom

            pos = SkyCoord(0.,0., unit='deg')
            geom = WcsGeom.create(skydir=pos, npix=100, binsz=0.1)

            pos_region = SkyCoord(3.,3., unit='deg')
            excluded_region = CircleSkyRegion(pos_region, Angle(0.5, 'deg'))

            # We want to exclude points inside excluded_region
            mask = geom.region_mask([excluded_region], inside=False)

            # Now we create a map
            mask_map = WcsNDMap(geom=geom, data=mask)
            mask_map.plot()
        """
        from regions import PixCoord

        if not self.is_regular:
            raise NotImplementedError("Region mask is not working yet with multi-resolution maps")

        idx = self.get_idx()
        pixcoord = PixCoord(idx[0], idx[1])

        mask = np.zeros(idx[0].shape, dtype=bool)

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
        str_ += "\tnpix      : {npix[0][0]} x {npix[1][0]} pix\n".format(npix=self.npix)
        str_ += "\tcoordsys  : {}\n".format(self.coordsys)
        str_ += "\tprojection: {}\n".format(self.projection)
        lon, lat = self.center_skydir.data.lon.deg, self.center_skydir.data.lat.deg
        str_ += "\tcenter    : {lon:.1f} deg, {lat:.1f} deg\n".format(lon=lon, lat=lat)
        str_ += ("\twidth     : {width[0][0]:.1f} x {width[1][0]:.1f} "
                 "deg\n".format(width=self.width))
        str_ += "\tndim      : {}\n".format(self.ndim)
        axes = [_.name for _ in self.axes]
        str_ += "\taxes      : {}\n".format(", ".join(axes))
        return str_


def create_wcs(skydir, coordsys='CEL', projection='AIT',
               cdelt=1.0, crpix=1., axes=None):
    """Create a WCS object.

    Parameters
    ----------
    skydir : `~astropy.coordinates.SkyCoord`
        Sky coordinate of the WCS reference point
    coordsys : str
        TODO
    projection : str
        TODO
    cdelt : float
        TODO
    crpix : float or (float,float)
        In the first case the same value is used for x and y axes
    axes : list
        List of non-spatial axes
    """
    naxis = 2
    if axes is not None:
        naxis += len(axes)

    w = WCS(naxis=naxis)

    if coordsys == 'CEL':
        w.wcs.ctype[0] = 'RA---{}'.format(projection)
        w.wcs.ctype[1] = 'DEC--{}'.format(projection)
        w.wcs.crval[0] = skydir.icrs.ra.deg
        w.wcs.crval[1] = skydir.icrs.dec.deg
    elif coordsys == 'GAL':
        w.wcs.ctype[0] = 'GLON-{}'.format(projection)
        w.wcs.ctype[1] = 'GLAT-{}'.format(projection)
        w.wcs.crval[0] = skydir.galactic.l.deg
        w.wcs.crval[1] = skydir.galactic.b.deg
    else:
        raise ValueError('Unrecognized coordinate system.')

    try:
        w.wcs.crpix[0] = crpix[0]
        w.wcs.crpix[1] = crpix[1]
    except:
        w.wcs.crpix[0] = crpix
        w.wcs.crpix[1] = crpix
    w.wcs.cdelt[0] = -cdelt
    w.wcs.cdelt[1] = cdelt

    w = WCS(w.to_header())
    # FIXME: Figure out what to do here
    # if naxis == 3 and energies is not None:
    #    w.wcs.crpix[2] = 1
    #    w.wcs.crval[2] = energies[0]
    #    w.wcs.cdelt[2] = energies[1] - energies[0]
    #    w.wcs.ctype[2] = 'Energy'
    #    w.wcs.cunit[2] = 'MeV'

    return w


def wcs_add_energy_axis(wcs, energies):
    """Copy a WCS object, and add on the energy axis.

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        WCS
    energies : array-like
        Array of energies
    """
    if wcs.naxis != 2:
        raise ValueError('WCS naxis must be 2. Got: {}'.format(wcs.naxis))

    w = WCS(naxis=3)
    w.wcs.crpix[0] = wcs.wcs.crpix[0]
    w.wcs.crpix[1] = wcs.wcs.crpix[1]
    w.wcs.ctype[0] = wcs.wcs.ctype[0]
    w.wcs.ctype[1] = wcs.wcs.ctype[1]
    w.wcs.crval[0] = wcs.wcs.crval[0]
    w.wcs.crval[1] = wcs.wcs.crval[1]
    w.wcs.cdelt[0] = wcs.wcs.cdelt[0]
    w.wcs.cdelt[1] = wcs.wcs.cdelt[1]

    w = WCS(w.to_header())
    w.wcs.crpix[2] = 1
    w.wcs.crval[2] = energies[0]
    w.wcs.cdelt[2] = energies[1] - energies[0]
    w.wcs.ctype[2] = 'Energy'

    return w


def offset_to_sky(skydir, offset_lon, offset_lat,
                  coordsys='CEL', projection='AIT'):
    """Convert a cartesian offset (X,Y) in the given projection into
    a pair of spherical coordinates."""
    offset_lon = np.array(offset_lon, ndmin=1)
    offset_lat = np.array(offset_lat, ndmin=1)

    w = create_wcs(skydir, coordsys, projection)
    pixcrd = np.vstack((offset_lon, offset_lat)).T

    return w.wcs_pix2world(pixcrd, 0)


def sky_to_offset(skydir, lon, lat, coordsys='CEL', projection='AIT'):
    """Convert sky coordinates to a projected offset.

    This function is the inverse of offset_to_sky.
    """
    w = create_wcs(skydir, coordsys, projection)
    skycrd = np.vstack((lon, lat)).T

    if len(skycrd) == 0:
        return skycrd

    return w.wcs_world2pix(skycrd, 0)


def offset_to_skydir(skydir, offset_lon, offset_lat,
                     coordsys='CEL', projection='AIT'):
    """Convert a cartesian offset (X,Y) in the given projection into
    a SkyCoord."""
    offset_lon = np.array(offset_lon, ndmin=1)
    offset_lat = np.array(offset_lat, ndmin=1)

    w = create_wcs(skydir, coordsys, projection)
    return SkyCoord.from_pixel(offset_lon, offset_lat, w, 0)


def pix2world(wcs, cdelt, crpix, pix):
    """Perform pixel to world coordinate transformation for a WCS
    projection with a given pixel size (CDELT) and reference pixel
    (CRPIX).  This method can be used to perform WCS transformations
    for projections with different pixelizations but the same
    reference coordinate (CRVAL), projection type, and coordinate
    system.

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
    pix_ratio = [np.abs(wcs.wcs.cdelt[0] / cdelt[0]),
                 np.abs(wcs.wcs.cdelt[1] / cdelt[1])]
    pix = ((pix[0] - (crpix[0] - 1.0)) / pix_ratio[0] + wcs.wcs.crpix[0] - 1.0,
           (pix[1] - (crpix[1] - 1.0)) / pix_ratio[1] + wcs.wcs.crpix[1] - 1.0)
    return wcs.wcs_pix2world(pix[0], pix[1], 0)


def world2pix(wcs, cdelt, crpix, coord):
    pix_ratio = [np.abs(wcs.wcs.cdelt[0] / cdelt[0]),
                 np.abs(wcs.wcs.cdelt[1] / cdelt[1])]
    pix = wcs.wcs_world2pix(coord[0], coord[1], 0)
    return ((pix[0] - (wcs.wcs.crpix[0] - 1.0)) * pix_ratio[0] + crpix[0] - 1.0,
            (pix[1] - (wcs.wcs.crpix[1] - 1.0)) * pix_ratio[1] + crpix[1] - 1.0)


def skydir_to_pix(skydir, wcs):
    """Convert skydir object to pixel coordinates.

    Gracefully handles 0-d coordinate arrays.

    Parameters
    ----------
    skydir : `~astropy.coordinates.SkyCoord`
        TODO
    wcs : `~astropy.wcs.WCS`
        TODO

    Returns
    -------
    xp, yp : `numpy.ndarray`
       The pixel coordinates
    """
    if len(skydir.shape) > 0 and len(skydir) == 0:
        return [np.empty(0), np.empty(0)]

    return skydir.to_pixel(wcs, origin=0)


def pix_to_skydir(xpix, ypix, wcs):
    """Convert pixel coordinates to a skydir object.

    Gracefully handles 0-d coordinate arrays.
    Always returns a celestial coordinate.

    Parameters
    ----------
    xpix, ypix : `numpy.ndarray`
        TODO
    wcs : `~astropy.wcs.WCS`
        TODO
    """
    xpix = np.array(xpix)
    ypix = np.array(ypix)

    if xpix.ndim > 0 and len(xpix) == 0:
        return SkyCoord(np.empty(0), np.empty(0), unit='deg', frame='icrs')

    return SkyCoord.from_pixel(xpix, ypix, wcs, origin=0).transform_to('icrs')


def get_projection(wcs):
    return wcs.wcs.ctype[0][5:]


def get_coordys(wcs):
    if 'RA' in wcs.wcs.ctype[0]:
        return 'CEL'
    elif 'GLON' in wcs.wcs.ctype[0]:
        return 'GAL'
    else:
        raise ValueError('Unrecognized WCS coordinate system.')


def wcs_to_axes(w, npix):
    """Generate a sequence of bin edge vectors corresponding to the
    axes of a WCS object."""
    npix = npix[::-1]

    x = np.linspace(-(npix[0]) / 2., (npix[0]) / 2.,
                    npix[0] + 1) * np.abs(w.wcs.cdelt[0])
    y = np.linspace(-(npix[1]) / 2., (npix[1]) / 2.,
                    npix[1] + 1) * np.abs(w.wcs.cdelt[1])

    cdelt2 = np.log10((w.wcs.cdelt[2] + w.wcs.crval[2]) / w.wcs.crval[2])

    z = (np.linspace(0, npix[2], npix[2] + 1)) * cdelt2
    z += np.log10(w.wcs.crval[2])

    return x, y, z


def wcs_to_coords(w, shape):
    """Generate an N x D list of pixel center coordinates where N is
    the number of pixels and D is the dimensionality of the map."""
    if w.naxis == 2:
        y, x = wcs_to_axes(w, shape)
    elif w.naxis == 3:
        z, y, x = wcs_to_axes(w, shape)
    else:
        raise ValueError('WCS naxis must be 2 or 3. Got: {}'.format(w.naxis))

    x = 0.5 * (x[1:] + x[:-1])
    y = 0.5 * (y[1:] + y[:-1])

    if w.naxis == 2:
        x = np.ravel(np.ones(shape) * x[:, np.newaxis])
        y = np.ravel(np.ones(shape) * y[np.newaxis, :])
        return np.vstack((x, y))

    z = 0.5 * (z[1:] + z[:-1])
    x = np.ravel(np.ones(shape) * x[:, np.newaxis, np.newaxis])
    y = np.ravel(np.ones(shape) * y[np.newaxis, :, np.newaxis])
    z = np.ravel(np.ones(shape) * z[np.newaxis, np.newaxis, :])

    return np.vstack((x, y, z))


def get_map_skydir(filename, maphdu=0):
    filename = str(make_path(filename))
    with fits.open(filename, memmap=False) as hdu_list:
        wcs = WCS(hdu_list[maphdu].header)
    return wcs_to_skydir(wcs)


def wcs_to_skydir(wcs):
    lon = wcs.wcs.crval[0]
    lat = wcs.wcs.crval[1]
    coordsys = get_coordys(wcs)
    if coordsys == 'GAL':
        return SkyCoord(lon, lat, unit='deg', frame='galactic').transform_to('icrs')
    else:
        return SkyCoord(lon, lat, unit='deg', frame='icrs')


def is_galactic(wcs):
    coordsys = get_coordys(wcs)
    if coordsys == 'GAL':
        return True
    elif coordsys == 'CEL':
        return False
    else:
        raise ValueError('Unsupported coordinate system: {}'.format(coordsys))
