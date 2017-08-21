# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from ..image.utils import make_header
from .geom import MapGeom, MapCoords, pix_tuple_to_idx, skydir_to_lonlat
from .geom import MapAxis, get_shape, make_axes_cols, make_axes
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
    axes : list
        Axes for non-spatial dimensions
    """

    def __init__(self, wcs, npix, cdelt=None, axes=None, conv='gadf'):
        self._wcs = wcs
        self._coordsys = get_coordsys(wcs)
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
    def allsky(self):
        """Flag for all-sky maps."""
        if np.all(np.isclose(self._npix[0] * self._cdelt[0], 360.)):
            return True
        else:
            return False

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

    @classmethod
    def create(cls, npix=None, binsz=0.5, proj='CAR', coordsys='CEL', refpix=None,
               axes=None, skydir=None, width=None, conv=None):
        """Create a WCS geometry object.  Pixelization of the map is set with
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
            xref, yref = skydir_to_lonlat(skydir, coordsys=coordsys)
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

        # FIXME: Need to propagate refpix

        header = make_header(npix[0].flat[0], npix[1].flat[0],
                             binsz[0].flat[0], xref, yref,
                             proj, coordsys, refpix, refpix)
        wcs = WCS(header)
        return cls(wcs, npix, cdelt=binsz, axes=axes, conv=conv)

    @classmethod
    def from_header(cls, header, hdu_bands=None, conv=None):
        """Create a WCS geometry object from a FITS header.

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            The FITS header
        hdu_bands : `~astropy.fits.BinTableHDU` 
            The BANDS table HDU.
        conv : str
            Override FITS format convention.

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

        # FIXME: Propagate CRPIX

        if hdu_bands is not None and 'NPIX' in hdu_bands.columns.names:
            npix = hdu_bands.data.field('NPIX').reshape(shape + (2,))
            npix = (npix[..., 0], npix[..., 1])
            cdelt = hdu_bands.data.field('CDELT').reshape(shape + (2,))
            cdelt = (cdelt[..., 0], cdelt[..., 1])
            crpix = hdu_bands.data.field('CRPIX').reshape(shape + (2,))
            crpix = (crpix[..., 0], crpix[..., 1])
        elif 'WCSSHAPE' in header:
            wcs_shape = eval(header['WCSSHAPE'])
            npix = (wcs_shape[0], wcs_shape[1])
            cdelt = None
            crpix = None
        else:
            npix = (header['NAXIS1'], header['NAXIS2'])
            cdelt = None
            crpix = None

        return cls(wcs, npix, cdelt=cdelt, axes=axes, conv=conv)

    def make_bands_hdu(self, extname=None, conv=None):

        conv = self._conv if conv is None else conv
        header = self.make_header(conv)
        axis_names = None

        # FIXME: Check whether convention is compatible with
        # dimensionality of geometry

        if conv == 'fgst-ccube':
            extname = 'EBOUNDS'
            axis_names = ['energy']
        elif conv == 'fgst-template':
            extname = 'ENERGIES'
            axis_names = ['energy']
        elif extname is None and conv == 'gadf':
            extname = 'BANDS'

        cols = make_axes_cols(self.axes, axis_names)
        if self.npix[0].size > 1:
            cols += [fits.Column('NPIX', '2I', dim='(2)',
                                 array=np.vstack((np.ravel(self.npix[0]),
                                                  np.ravel(self.npix[1]))).T), ]
            cols += [fits.Column('CDELT', '2E', dim='(2)',
                                 array=np.vstack((np.ravel(self._cdelt[0]),
                                                  np.ravel(self._cdelt[1]))).T), ]
            cols += [fits.Column('CRPIX', '2E', dim='(2)',
                                 array=np.vstack((np.ravel(self._crpix[0]),
                                                  np.ravel(self._crpix[1]))).T), ]

        hdu = fits.BinTableHDU.from_columns(cols, header, name=extname)
        return hdu

    def make_header(self, conv=None):
        header = self.wcs.to_header()
        self._fill_header_from_axes(header)
        header['WCSSHAPE'] = '({},{})'.format(np.max(self.npix[0]),
                                              np.max(self.npix[1]))
        return header

    def distance_to_edge(self, skydir):
        """Angular distance from the given direction and
        the edge of the projection."""
        raise NotImplementedError

    def get_image_shape(self, idx):
        """Get the shape of the image plane at index ``idx``."""

        if self.npix[0].size > 1:
            return (int(self.npix[0][idx]), int(self.npix[1][idx]))
        else:
            return (int(self.npix[0]), int(self.npix[1]))

    def get_image_wcs(self, idx):
        raise NotImplementedError

    def get_pixels(self, idx=None, local=False):
        return pix_tuple_to_idx(self._get_pix_coords(idx=idx,
                                                     mode='center'))

    def _get_pix_coords(self, idx=None, mode='center'):

        # FIXME: Figure out if there is some way to employ open/sparse
        # vectors

        npix = copy.deepcopy(self.npix)

        if mode == 'edge':
            npix[0] += 1
            npix[1] += 1

        if self.axes and self.npix[0].size > 1:

            pix = [np.array([], dtype=float)
                   for i in range(2 + len(self.axes))]
            for idx_img in np.ndindex(self.shape[::-1]):

                idx_img = idx_img[::-1]
                if idx is not None and idx_img != idx:
                    continue

                npix0, npix1 = npix[0][idx_img], npix[1][idx_img]
                ntot = npix0 * npix1
                pix_img = np.unravel_index(np.arange(ntot, dtype=int),
                                           (npix0, npix1), order='F')
                pix[0] = np.concatenate((pix[0], pix_img[0].astype(float)))
                pix[1] = np.concatenate((pix[1], pix_img[1].astype(float)))
                for j in range(len(self.axes)):
                    pix[2 + j] = np.concatenate((pix[2 + j],
                                                 idx_img[j] *
                                                 np.ones(ntot, dtype=float)))

        else:
            pix = [np.arange(npix[0], dtype=float),
                   np.arange(npix[1], dtype=float)]

            if idx is None:
                pix += [np.arange(ax.nbin, dtype=float) for ax in self.axes]
            else:
                pix += list(idx)

            pix = np.meshgrid(*pix[::-1], indexing='ij', sparse=False)[::-1]

        if mode == 'edges':
            for i in range(len(pix)):
                pix[i] -= 0.5

        coords = self.pix_to_coord(pix)
        m = np.isfinite(coords[0])
        return tuple([np.ravel(t[m]) for t in pix])
#        shape = np.broadcast(*coords).shape
#        m = [np.isfinite(c) for c in coords]
#        m = np.broadcast_to(np.prod(m),shape)
#        return tuple([np.ravel(np.broadcast_to(t,shape)[m]) for t in pix])

    def get_coords(self, idx=None):

        pix = self.get_pixels(idx=idx)
        return self.pix_to_coord(pix)

    def coord_to_pix(self, coords):

        c = MapCoords.create(coords)

        # Variable Bin Size
        if self.axes and self.npix[0].size > 1:
            bins = [ax.coord_to_pix(c[i + 2])
                    for i, ax in enumerate(self.axes)]
            idxs = [ax.coord_to_idx(c[i + 2])
                    for i, ax in enumerate(self.axes)]
            crpix = [t[idxs] for t in self._crpix]
            cdelt = [t[idxs] for t in self._cdelt]
            pix = world2pix(self.wcs, cdelt, crpix, (c.lon, c.lat))
            pix = tuple(list(pix) + bins)
        else:
            pix = self._wcs.wcs_world2pix(c.lon, c.lat, 0)
            for i, ax in enumerate(self.axes):
                pix += [ax.coord_to_pix(c[i + 2])]

        return pix

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

    def pix_to_idx(self, pix):
        idxs = pix_tuple_to_idx(pix)
        if self.npix[0].size > 1:
            ibin = [pix[2 + i] for i, ax in enumerate(self.axes)]
            ibin = pix_tuple_to_idx(ibin)
            npix = (self.npix[0][ibin], self.npix[1][ibin])
        else:
            npix = self.npix

        for i, idx in enumerate(idxs):
            if i < 2:
                idxs[i][(idx < 0) | (idx >= npix[i])] = -1
            else:
                idxs[i][(idx < 0) | (idx >= self.axes[i - 2].nbin)] = -1
        return idxs

    def contains(self, coords):
        raise NotImplementedError

    def to_image(self):
        npix = (np.max(self._npix[0]), np.max(self._npix[1]))
        cdelt = (np.max(self._cdelt[0]), np.max(self._cdelt[1]))
        return self.__class__(self._wcs, npix, cdelt=cdelt)

    def to_cube(self, axes):
        npix = (np.max(self._npix[0]), np.max(self._npix[1]))
        cdelt = (np.max(self._cdelt[0]), np.max(self._cdelt[1]))
        axes = copy.deepcopy(self.axes) + axes
        return self.__class__(self._wcs.deepcopy(), npix, cdelt=cdelt, axes=axes)

    def to_slice(self, slices):
        raise NotImplementedError


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
        raise Exception('WCS naxis must be 2. Got: {}'.format(wcs.naxis))

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


def get_coordsys(wcs):
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
        raise Exception('WCS naxis must be 2 or 3. Got: {}'.format(w.naxis))

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
    with fits.open(filename) as hdulist:
        wcs = WCS(hdulist[maphdu].header)
    return wcs_to_skydir(wcs)


def wcs_to_skydir(wcs):
    lon = wcs.wcs.crval[0]
    lat = wcs.wcs.crval[1]
    coordsys = get_coordsys(wcs)
    if coordsys == 'GAL':
        return SkyCoord(lon, lat, unit='deg', frame='galactic').transform_to('icrs')
    else:
        return SkyCoord(lon, lat, unit='deg', frame='icrs')


def is_galactic(wcs):
    coordsys = get_coordsys(wcs)
    if coordsys == 'GAL':
        return True
    elif coordsys == 'CEL':
        return False
    else:
        raise ValueError('Unsupported coordinate system: {}'.format(coordsys))
