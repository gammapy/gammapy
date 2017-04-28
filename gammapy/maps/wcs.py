# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from ..image.utils import make_header
from .geom import MapGeom, MapCoords, val_to_pix

__all__ = [
    'WCSGeom',
]


class WCSGeom(MapGeom):
    """Container for WCS object and image extent.

    Class that encapsulates both a WCS object and the definition of
    the image extent (number of pixels).  Also provides a number of
    helper methods for accessing the properties of the WCS object.

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        WCS projection object
    npix : list
        Number of pixels in each spatial dimension
    axes : list
        Axes for non-spatial dimensions
    """

    def __init__(self, wcs, npix, axes=None):
        self._wcs = wcs
        self._npix = np.array(npix, ndmin=1)
        self._coordsys = get_coordsys(wcs)
        self._axes = axes if axes is not None else []

        cdelt0 = np.abs(self.wcs.wcs.cdelt[0])
        cdelt1 = np.abs(self.wcs.wcs.cdelt[1])

        self._width = np.array([cdelt0 * self._npix[0],
                                cdelt1 * self._npix[1]])
        self._pix_center = np.array([(self._npix[0] - 1.0) / 2.,
                                     (self._npix[1] - 1.0) / 2.])
        self._pix_size = np.array([cdelt0, cdelt1])
        self._skydir = SkyCoord.from_pixel(self._pix_center[0],
                                           self._pix_center[1],
                                           self.wcs)

    @property
    def wcs(self):
        """TODO."""
        return self._wcs

    @property
    def coordsys(self):
        """TODO."""
        return self._coordsys

    @property
    def skydir(self):
        """Sky coordinate of the image center (TODO: type?)."""
        return self._skydir

    @property
    def width(self):
        """Dimensions of the image (TODO: type?)."""
        return self._width

    @property
    def npix(self):
        """TODO."""
        return self._npix

    @classmethod
    def from_skydir(cls, skydir, cdelt, npix, coordsys='CEL', projection='AIT'):
        """TODO."""
        npix = np.array(npix, ndmin=1)
        crpix = npix / 2. + 0.5
        wcs = create_wcs(skydir, coordsys, projection,
                         cdelt, crpix)
        return cls(wcs, npix)

    @classmethod
    def create(cls, nxpix=100, nypix=100, binsz=0.1, xref=0, yref=0,
               proj='CAR', coordsys='CEL', xrefpix=None, yrefpix=None,
               axes=None):
        """TODO."""
        header = make_header(nxpix, nypix, binsz, xref, yref,
                             proj, coordsys, xrefpix, yrefpix)
        wcs = WCS(header)
        return cls(wcs, [nxpix, nypix], axes)

    def distance_to_edge(self, skydir):
        """Angular distance from the given direction and
        the edge of the projection."""
        xpix, ypix = skydir.to_pixel(self.wcs, origin=0)
        deltax = np.array((xpix - self._pix_center[0]) * self._pix_size[0],
                          ndmin=1)
        deltay = np.array((ypix - self._pix_center[1]) * self._pix_size[1],
                          ndmin=1)

        deltax = np.abs(deltax) - 0.5 * self._width[0]
        deltay = np.abs(deltay) - 0.5 * self._width[1]

        m0 = (deltax < 0) & (deltay < 0)
        m1 = (deltax > 0) & (deltay < 0)
        m2 = (deltax < 0) & (deltay > 0)
        m3 = (deltax > 0) & (deltay > 0)
        mx = np.abs(deltax) <= np.abs(deltay)
        my = np.abs(deltay) < np.abs(deltax)

        delta = np.zeros(len(deltax))
        delta[(m0 & mx) | (m3 & my) | m1] = deltax[(m0 & mx) | (m3 & my) | m1]
        delta[(m0 & my) | (m3 & mx) | m2] = deltay[(m0 & my) | (m3 & mx) | m2]
        return delta

    def coord_to_pix(self, coords):
        """TODO.
        """
        c = MapCoords.create(coords)
        pix = self._wcs.wcs_world2pix(c.lon, c.lat, 0)
        for i, ax in enumerate(self.axes):
            pix += val_to_pix(ax, c[i + 2])
        return pix

    def pix_to_coord(self, pix):
        """TODO.
        """
        pass


def create_wcs(skydir, coordsys='CEL', projection='AIT',
               cdelt=1.0, crpix=1., naxis=2, energies=None):
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
    naxis : {2, 3}
       Number of dimensions of the projection.
    energies : array-like
       Array of energies that defines the third dimension if naxis=3.
    """
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
    if naxis == 3 and energies is not None:
        w.wcs.crpix[2] = 1
        w.wcs.crval[2] = energies[0]
        w.wcs.cdelt[2] = energies[1] - energies[0]
        w.wcs.ctype[2] = 'Energy'
        w.wcs.cunit[2] = 'MeV'

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
    hdulist = fits.open(filename)
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
