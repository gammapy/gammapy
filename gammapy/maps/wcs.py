# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from ..image.utils import make_header
from .geom import MapGeom, MapCoords, pix_tuple_to_idx, skydir_to_lonlat
from .geom import MapAxis, get_shape

__all__ = [
    'WCSGeom',
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


class WCSGeom(MapGeom):
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

    def __init__(self, wcs, npix, cdelt=None, axes=None):
        self._wcs = wcs
        self._coordsys = get_coordsys(wcs)
        self._axes = axes if axes is not None else []
        for i, ax in enumerate(self.axes):
            if isinstance(ax, np.ndarray):
                self.axes[i] = MapAxis(ax)
            if self.axes[i].name == '':
                self.axes[i].set_name('axis%i' % i)

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
    def width(self):
        """Tuple with image dimension in deg in longitude and latitude."""
        return self._width

    @property
    def npix(self):
        """Tuple with image dimension in pixels in longitude and latitude."""
        return self._npix

    @property
    def axes(self):
        """List of non-spatial axes."""
        return self._axes

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
               axes=None, skydir=None, width=None):
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
            Reference pixel of the projection.  If None then this will
            be chosen to be center of the map.

        Returns
        -------
        geom : `~WCSGeom`
            A WCS geometry object.

        Examples
        --------
        >>> from gammapy.maps import WCSGeom
        >>> from gammapy.maps import MapAxis
        >>> axis = MapAxis.from_bounds(0,1,2)
        >>> geom = SkyImage.create(npix=(100,100), binsz=0.1)
        >>> geom = SkyImage.create(npix=[100,200], binsz=[0.1,0.05], axes=[axis])
        >>> geom = SkyImage.create(width=[5.0,8.0], binsz=[0.1,0.05], axes=[axis])
        >>> geom = SkyImage.create(npix=([100,200],[100,200]), binsz=0.1, axes=[axis])

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
        return cls(wcs, npix, cdelt=binsz, axes=axes)

    def distance_to_edge(self, skydir):
        """Angular distance from the given direction and
        the edge of the projection."""
        raise NotImplementedError

    def get_pixels(self):

        if self.axes and self.npix[0].size > 1:

            pix = [np.array([], dtype=int) for i in range(2 + len(self.axes))]
            for i, t in np.ndenumerate(self.npix[0]):

                npix = self.npix[0][i] * self.npix[1][i]
                o = np.unravel_index(np.arange(npix, dtype=int),
                                     (self.npix[0][i], self.npix[1][i]))
                pix[0] = np.concatenate((pix[0], o[0]))
                pix[1] = np.concatenate((pix[1], o[1]))
                for j in range(len(self.axes)):
                    pix[2 + j] = np.concatenate((pix[2 + j],
                                                 i[j] * np.ones(npix, dtype=int)))

        else:
            pix = [np.arange(self.npix[0], dtype=int),
                   np.arange(self.npix[1], dtype=int)]
            for i, ax in enumerate(self.axes):
                pix += [np.arange(ax.nbin, dtype=int)]
            pix = np.meshgrid(*pix, indexing='ij')

        coords = self.pix_to_coord(pix)
        m = np.isfinite(coords[0])
        return tuple([np.ravel(t[m]) for t in pix])

    def get_coords(self):

        pix = self.get_pixels()
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
        return pix_tuple_to_idx(pix)

    def contains(self, coords):
        raise NotImplementedError

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
