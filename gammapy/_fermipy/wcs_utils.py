# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from astropy.extern import six
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord


class WCSProj(object):
    """Class that encapsulates both a WCS object and the definition of
    the image extent (number of pixels).  Also provides a number of
    helper methods for accessing the properties of the WCS object."""

    def __init__(self, wcs, npix):
        self._wcs = wcs
        self._npix = np.array(npix, ndmin=1)
        self._coordsys = get_coordsys(wcs)

        cdelt0 = np.abs(self.wcs.wcs.cdelt[0])
        cdelt1 = np.abs(self.wcs.wcs.cdelt[1])

        xindex = 0
        yindex = 1

        self._width = np.array([cdelt0 * self._npix[xindex],
                                cdelt1 * self._npix[yindex]])
        self._pix_center = np.array([(self._npix[xindex] - 1.0) / 2.,
                                     (self._npix[yindex] - 1.0) / 2.])
        self._pix_size = np.array([cdelt0, cdelt1])
        self._skydir = SkyCoord.from_pixel(self._pix_center[0],
                                           self._pix_center[1],
                                           self.wcs)

    @property
    def wcs(self):
        return self._wcs

    @property
    def coordsys(self):
        return self._coordsys

    @property
    def skydir(self):
        """Return the sky coordinate of the image center."""
        return self._skydir

    @property
    def width(self):
        """Return the dimensions of the image."""
        return self._width

    @property
    def npix(self):
        return self._npix

    @staticmethod
    def create(skydir, cdelt, npix, coordsys='CEL', projection='AIT'):
        npix = np.array(npix, ndmin=1)
        crpix = npix / 2. + 0.5
        wcs = create_wcs(skydir, coordsys, projection,
                         cdelt, crpix)
        return WCSProj(wcs, npix)

    def distance_to_edge(self, skydir):
        """Return the angular distance from the given direction and
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


def create_wcs(skydir, coordsys='CEL', projection='AIT',
               cdelt=1.0, crpix=1., naxis=2, energies=None):
    """Create a WCS object.

    Parameters
    ----------
    skydir : `~astropy.coordinates.SkyCoord`
        Sky coordinate of the WCS reference point.
    coordsys : str

    projection : str

    cdelt : float

    crpix : float or (float,float)
        In the first case the same value is used for x and y axes
    naxis : {2, 3}
       Number of dimensions of the projection.
    energies : array-like
       Array of energies that defines the third dimension if naxis=3.
    """

    w = WCS(naxis=naxis)

    if coordsys == 'CEL':
        w.wcs.ctype[0] = 'RA---%s' % (projection)
        w.wcs.ctype[1] = 'DEC--%s' % (projection)
        w.wcs.crval[0] = skydir.icrs.ra.deg
        w.wcs.crval[1] = skydir.icrs.dec.deg
    elif coordsys == 'GAL':
        w.wcs.ctype[0] = 'GLON-%s' % (projection)
        w.wcs.ctype[1] = 'GLAT-%s' % (projection)
        w.wcs.crval[0] = skydir.galactic.l.deg
        w.wcs.crval[1] = skydir.galactic.b.deg
    else:
        raise Exception('Unrecognized coordinate system.')

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
       Array of energies.
    """
    if wcs.naxis != 2:
        raise Exception(
            'wcs_add_energy_axis, input WCS naxis != 2 %i' % wcs.naxis)
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
    """Convert sky coordinates to a projected offset.  This function
    is the inverse of offset_to_sky."""

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

    wcs : `~astropy.wcs.WCS`

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
    xpix : `numpy.ndarray`

    ypix : `numpy.ndarray`

    wcs : `~astropy.wcs.WCS`

    """
    xpix = np.array(xpix)
    ypix = np.array(ypix)

    if xpix.ndim > 0 and len(xpix) == 0:
        return SkyCoord(np.empty(0), np.empty(0), unit='deg',
                        frame='icrs')

    return SkyCoord.from_pixel(xpix, ypix, wcs,
                               origin=0).transform_to('icrs')


def get_coordsys(wcs):
    if 'RA' in wcs.wcs.ctype[0]:
        return 'CEL'
    elif 'GLON' in wcs.wcs.ctype[0]:
        return 'GAL'
    else:
        raise Exception('Unrecognized WCS coordinate system.')


def get_target_skydir(config, ref_skydir=None):
    if ref_skydir is None:
        ref_skydir = SkyCoord(0.0, 0.0, unit=u.deg)

    radec = config.get('radec', None)

    if isinstance(radec, six.text_types):
        return SkyCoord(radec, unit=u.deg)
    elif isinstance(radec, list):
        return SkyCoord(radec[0], radec[1], unit=u.deg)

    ra = config.get('ra', None)
    dec = config.get('dec', None)

    if ra is not None and dec is not None:
        return SkyCoord(ra, dec, unit=u.deg)

    glon = config.get('glon', None)
    glat = config.get('glat', None)

    if glon is not None and glat is not None:
        return SkyCoord(glon, glat, unit=u.deg,
                        frame='galactic').transform_to('icrs')

    offset_ra = config.get('offset_ra', None)
    offset_dec = config.get('offset_dec', None)

    if offset_ra is not None and offset_dec is not None:
        return offset_to_skydir(ref_skydir, offset_ra, offset_dec,
                                coordsys='CEL')[0]

    offset_glon = config.get('offset_glon', None)
    offset_glat = config.get('offset_glat', None)

    if offset_glon is not None and offset_glat is not None:
        return offset_to_skydir(ref_skydir, offset_glon, offset_glat,
                                coordsys='GAL')[0].transform_to('icrs')

    return ref_skydir


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
        raise Exception("Wrong number of WCS axes %i" % w.naxis)

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


def wcs_to_skydir(wcs):
    lon = wcs.wcs.crval[0]
    lat = wcs.wcs.crval[1]

    coordsys = get_coordsys(wcs)

    if coordsys == 'GAL':
        return SkyCoord(lon, lat, unit='deg',
                        frame='galactic').transform_to('icrs')
    else:
        return SkyCoord(lon, lat, unit='deg', frame='icrs')


def is_galactic(wcs):
    coordsys = get_coordsys(wcs)
    if coordsys == 'GAL':
        return True
    elif coordsys == 'CEL':
        return False
    else:
        raise Exception('Unsupported coordinate system: %s' % coordsys)


def extract_mapcube_region(infile, skydir, outfile, maphdu=0):
    """Extract a region out of an all-sky mapcube file.

    Parameters
    ----------

    infile : str
        Path to mapcube file.

    skydir : `~astropy.coordinates.SkyCoord`

    """

    h = fits.open(os.path.expandvars(infile))

    npix = 200
    shape = list(h[maphdu].data.shape)
    shape[1] = 200
    shape[2] = 200

    wcs = WCS(h[maphdu].header)
    skywcs = WCS(h[maphdu].header, naxis=[1, 2])
    coordsys = get_coordsys(skywcs)

    region_wcs = wcs.deepcopy()

    if coordsys == 'CEL':
        region_wcs.wcs.crval[0] = skydir.ra.deg
        region_wcs.wcs.crval[1] = skydir.dec.deg
    elif coordsys == 'GAL':
        region_wcs.wcs.crval[0] = skydir.galactic.l.deg
        region_wcs.wcs.crval[1] = skydir.galactic.b.deg
    else:
        raise Exception('Unrecognized coordinate system.')

    region_wcs.wcs.crpix[0] = npix // 2 + 0.5
    region_wcs.wcs.crpix[1] = npix // 2 + 0.5

    from reproject import reproject_interp
    data, footprint = reproject_interp(h, region_wcs.to_header(),
                                       hdu_in=maphdu,
                                       shape_out=shape)

    hdu_image = fits.PrimaryHDU(data, header=region_wcs.to_header())
    hdulist = fits.HDUList([hdu_image, h['ENERGIES']])
    hdulist.writeto(outfile, clobber=True)
