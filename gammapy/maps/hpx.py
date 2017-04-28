# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for dealing with HEALPix projections and mappings
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import re
import numpy as np
from astropy.extern.six.moves import range
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from .wcs import WCSGeom
from .geom import MapGeom, MapCoords, val_to_bin, bin_to_val

# TODO: What should be part of the public API?
__all__ = [
    # 'HPX_Conv',
    # 'HPX_FITS_CONVENTIONS',
    # 'HPX_ORDER_TO_PIXSIZE',
    'HPXGeom',
    # 'HpxToWcsMapping',
]

# This is an approximation of the size of HEALPIX pixels (in degrees)
# for a particular order.   It is used to convert from HEALPIX to WCS-based
# projections
HPX_ORDER_TO_PIXSIZE = np.array([32.0, 16.0, 8.0, 4.0, 2.0, 1.0,
                                 0.50, 0.25, 0.1, 0.05, 0.025, 0.01,
                                 0.005, 0.002])


class HPX_Conv(object):
    """Data structure to define how a HEALPIX map is stored to FITS.
    """

    def __init__(self, convname, **kwargs):
        self.convname = convname
        self.colstring = kwargs.get('colstring', 'CHANNEL')
        self.firstcol = kwargs.get('firstcol', 1)
        self.extname = kwargs.get('extname', 'SKYMAP')
        self.energy_hdu = kwargs.get('energy_hdu', 'EBOUNDS')
        self.quantity_type = kwargs.get('quantity_type', 'integral')
        self.coordsys = kwargs.get('coordsys', 'COORDSYS')

    def colname(self, indx):
        return '{}{}'.format(self.colstring, indx)


# Various conventions for storing HEALPIX maps in FITS files
HPX_FITS_CONVENTIONS = OrderedDict()
HPX_FITS_CONVENTIONS['FGST_CCUBE'] = HPX_Conv('FGST_CCUBE')
HPX_FITS_CONVENTIONS['FGST_LTCUBE'] = HPX_Conv(
    'FGST_LTCUBE', colstring='COSBINS', extname='EXPOSURE', energy_hdu='CTHETABOUNDS')
HPX_FITS_CONVENTIONS['FGST_BEXPCUBE'] = HPX_Conv(
    'FGST_BEXPCUBE', colstring='ENERGY', extname='HPXEXPOSURES', energy_hdu='ENERGIES')
HPX_FITS_CONVENTIONS['FGST_SRCMAP'] = HPX_Conv('FGST_SRCMAP', extname=None, quantity_type='differential')
HPX_FITS_CONVENTIONS['FGST_TEMPLATE'] = HPX_Conv('FGST_TEMPLATE', colstring='ENERGY', energy_hdu='ENERGIES')
HPX_FITS_CONVENTIONS['FGST_SRCMAP_SPARSE'] = HPX_Conv(
    'FGST_SRCMAP_SPARSE', colstring=None, extname=None, quantity_type='differential')
HPX_FITS_CONVENTIONS['GALPROP'] = HPX_Conv(
    'GALPROP', colstring='Bin', extname='SKYMAP2',
    energy_hdu='ENERGIES', quantity_type='differential', coordsys='COORDTYPE')
HPX_FITS_CONVENTIONS['GALPROP2'] = HPX_Conv(
    'GALPROP', colstring='Bin', extname='SKYMAP2',
    energy_hdu='ENERGIES', quantity_type='differential')


def unravel_hpx_index(idx, npix):
    """Convert flattened global map index to an index tuple.

    Parameters
    ----------
    idx : `~numpy.ndarray`
        Flat index.
    npix : `~numpy.ndarray`
        Number of pixels in each band.

    Returns
    -------
    idx : tuple of `~numpy.ndarray`
        Index array for each dimension of the map.
    """
    dpix = np.zeros(npix.size, dtype='i')
    dpix[1:] = np.cumsum(npix.flat[:-1])
    bidx = np.searchsorted(np.cumsum(npix.flat), idx + 1)
    pix = idx - dpix[bidx]
    return tuple([pix] + list(np.unravel_index(bidx, npix.shape)))


def ravel_hpx_index(idx, npix):
    """Convert map index tuple to a flattened index.

    Parameters
    ----------
    idx : tuple of `~numpy.ndarray`
    """
    if len(idx) == 1:
        return idx

    idx0 = idx[0]
    idx1 = np.ravel_multi_index(idx[1:], npix.shape)
    npix = np.concatenate((np.array([0]), npix.flat[:-1]))

    return idx0 + np.cumsum(npix)[idx1]


def coords_to_vec(lon, lat):
    """Converts longitude and latitude coordinates to a unit 3-vector.

    Returns
    -------
    array(3,n) with v_x[i],v_y[i],v_z[i] = directional cosines
    """
    phi = np.radians(lon)
    theta = (np.pi / 2) - np.radians(lat)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    xVals = sin_t * np.cos(phi)
    yVals = sin_t * np.sin(phi)
    zVals = cos_t

    # Stack them into the output array
    out = np.vstack((xVals, yVals, zVals)).swapaxes(0, 1)
    return out


def get_pixel_size_from_nside(nside):
    """Estimate of the pixel size from the HEALPIX nside coordinate.

    This just uses a lookup table to provide a nice round number
    for each HEALPIX order.
    """
    order = nside_to_order(nside)
    if np.any(order < 0) or np.any(order > 13):
        raise ValueError('HEALPIX order must be between 0 to 13. Value is: {}'.format(order))

    return HPX_ORDER_TO_PIXSIZE[order]


def hpx_to_axes(h, npix):
    """Generate a sequence of bin edge vectors corresponding to the axes of a HPX object.
    """
    x = h.ebins
    z = np.arange(npix[-1] + 1)

    return x, z


def hpx_to_coords(h, shape):
    """Generate an N x D list of pixel center coordinates.

    ``N`` is the number of pixels and ``D`` is the dimensionality of the map.
    """
    x, z = hpx_to_axes(h, shape)

    x = np.sqrt(x[0:-1] * x[1:])
    z = z[:-1] + 0.5

    x = np.ravel(np.ones(shape) * x[:, np.newaxis])
    z = np.ravel(np.ones(shape) * z[np.newaxis, :])

    return np.vstack((x, z))


def make_hpx_to_wcs_mapping_centers(hpx, wcs):
    """Make the mapping data needed to from from HPX pixelization to a WCS-based array.

    Parameters
    ----------
    hpx : `~fermipy.hpx_utils.HPX`
        The HEALPIX mapping
    wcs : `~astropy.wcs.WCS`
        The WCS mapping

    Returns
    -------
    ipixs : array(nx,ny)
        HEALPIX pixel indices for each WCS pixel
        -1 indicates the wcs pixel does not contain the center of a HEALPIX pixel
    mult_val : array
        (nx,ny) of 1.
    npix : tuple
        (nx,ny) with the shape of the WCS grid
    """
    npix = (int(wcs.wcs.crpix[0] * 2), int(wcs.wcs.crpix[1] * 2))
    mult_val = np.ones(npix).T.flatten()
    sky_crds = hpx.get_sky_coords()
    pix_crds = wcs.wcs_world2pix(sky_crds, 0).astype(int)
    ipixs = -1 * np.ones(npix, int).T.flatten()
    pix_index = npix[1] * pix_crds[0:, 0] + pix_crds[0:, 1]

    if hpx._ipix is None:
        for ipix, pix_crd in enumerate(pix_index):
            ipixs[pix_crd] = ipix
    else:
        for pix_crd, ipix in zip(pix_index, hpx._ipix):
            ipixs[pix_crd] = ipix

    ipixs = ipixs.reshape(npix).T.flatten()

    return ipixs, mult_val, npix


def make_hpx_to_wcs_mapping(hpx, wcs):
    """Make the pixel mapping from HPX- to a WCS-based geometry.

    Parameters
    ----------
    hpx : `~gammapy.maps.hpx.HPXGeom`
       The HEALPIX geometry
    wcs : `~gammapy.maps.wcs.WCSGeom`
       The WCS geometry

    Returns
    -------
    ipix : `~numpy.ndarray`
        array(nx,ny) of HEALPIX pixel indices for each wcs pixel
    mult_val : `~numpy.ndarray`
        array(nx,ny) of 1./number of WCS pixels pointing at each HEALPIX pixel
    npix : tuple
        tuple(nx,ny) with the shape of the WCS grid
    """
    import healpy as hp
    npix = wcs.npix[:2]

    # FIXME: Calculation of WCS pixel centers should be moved into a
    # method of WCSGeom
    pix_crds = np.dstack(np.meshgrid(np.arange(npix[0]), np.arange(npix[1])))
    pix_crds = pix_crds.swapaxes(0, 1).reshape((-1, 2))
    sky_crds = wcs.wcs.wcs_pix2world(pix_crds, 0)
    sky_crds *= np.radians(1.)
    sky_crds[0:, 1] = (np.pi / 2) - sky_crds[0:, 1]

    mask = ~np.any(np.isnan(sky_crds), axis=1)
    ipix = -1 * np.ones((len(hpx.nside), npix[0] * npix[1]), int)
    m = mask[None, :] * np.ones_like(ipix, dtype=bool)

    ipix[m] = hp.pixelfunc.ang2pix(hpx.nside[..., None],
                                   sky_crds[:, 1][mask][None, ...],
                                   sky_crds[:, 0][mask][None, ...],
                                   hpx.nest).flatten()

    # Here we are counting the number of HEALPIX pixels each WCS pixel
    # points to and getting a multiplicative factor that tells use how
    # to split up the counts in each HEALPIX pixel (by dividing the
    # corresponding WCS pixels by the number of associated HEALPIX
    # pixels).
    mult_val = np.ones_like(ipix, dtype=float)
    for i, t in enumerate(ipix):
        count = np.unique(t, return_counts=True)
        idx = np.searchsorted(count[0], t)
        mult_val[i, ...] = 1. / count[1][idx]

    if hpx.nside.size == 1:
        ipix = np.squeeze(ipix, axis=0)
        mult_val = np.squeeze(mult_val, axis=0)

    return ipix, mult_val, npix


# TODO: it says "old" in the function name. Remove?
def make_hpx_to_wcs_mapping_old(hpx, wcs):
    """Make the mapping data needed to from from HPX pixelization to a
    WCS-based array

    Parameters
    ----------
    hpx     : `~gammapy.maps.HPX`
       The HEALPIX mapping (an HPX object)

    wcs     : `~astropy.wcs.WCS`
       The wcs mapping (a pywcs.wcs object)

    Returns
    -------
      ipixs    :  array(nx,ny) of HEALPIX pixel indices for each wcs pixel
      mult_val :  array(nx,ny) of 1./number of wcs pixels pointing at each HEALPIX pixel
      npix     :  tuple(nx,ny) with the shape of the wcs grid

    """
    import healpy as hp
    wcs = wcs.wcs

    npix = (int(wcs.wcs.crpix[0] * 2), int(wcs.wcs.crpix[1] * 2))
    pix_crds = np.dstack(np.meshgrid(np.arange(npix[0]),
                                     np.arange(npix[1]))).swapaxes(0, 1).reshape((npix[0] * npix[1], 2))

    sky_crds = wcs.wcs_pix2world(pix_crds, 0)

    sky_crds *= np.radians(1.)
    sky_crds[0:, 1] = (np.pi / 2) - sky_crds[0:, 1]

    fullmask = np.isnan(sky_crds)
    mask = (fullmask[0:, 0] + fullmask[0:, 1]) == 0
    ipixs = -1 * np.ones(npix, int).T.flatten()
    ipixs[mask] = hp.pixelfunc.ang2pix(hpx.nside, sky_crds[0:, 1][mask],
                                       sky_crds[0:, 0][mask], hpx.nest)

    # Here we are counting the number of HEALPIX pixels each WCS pixel points to;
    # this could probably be vectorized by filling a histogram.
    d_count = {}
    for ipix in ipixs:
        if ipix in d_count:
            d_count[ipix] += 1
        else:
            d_count[ipix] = 1

    # Here we are getting a multiplicative factor that tells use how to split up
    # the counts in each HEALPIX pixel (by dividing the corresponding WCS pixels
    # by the number of associated HEALPIX pixels).
    # This could also likely be vectorized.
    mult_val = np.ones(ipixs.shape)
    for i, ipix in enumerate(ipixs):
        mult_val[i] /= d_count[ipix]

    ipixs = ipixs.reshape(npix).flatten()
    mult_val = mult_val.reshape(npix).flatten()
    return ipixs, mult_val, npix


def match_hpx_pixel(nside, nest, nside_pix, ipix_ring):
    """TODO
    """
    import healpy as hp
    ipix_in = np.arange(12 * nside * nside)
    vecs = hp.pix2vec(nside, ipix_in, nest)
    pix_match = hp.vec2pix(nside_pix, vecs[0], vecs[1], vecs[2]) == ipix_ring
    return ipix_in[pix_match]


def parse_hpxregion(region):
    """Parse the ``HPX_REG`` header keyword into a list of tokens.
    """
    m = re.match(r'([A-Za-z\_]*?)\((.*?)\)', region)

    if m is None:
        raise Exception('Failed to parse hpx region string.')

    if not m.group(1):
        return re.split(',', m.group(2))
    else:
        return [m.group(1)] + re.split(',', m.group(2))


def is_power2(n):
    """Check if an integer is a power of 2."""
    return (n > 0) & ((n & (n - 1)) == 0)


def nside_to_order(nside):
    """Compute the ORDER given NSIDE.

    Returns -1 for NSIDE values that are not a power of 2.
    """
    nside = np.array(nside, ndmin=1)
    order = -1 * np.ones_like(nside)
    m = is_power2(nside)
    order[m] = np.log2(nside[m]).astype(int)
    return order


def upix_to_pix(upix):
    """Get the pixel index and nside from a unique pixel number."""
    nside = np.power(2, np.floor(np.log2(upix / 4)) / 2).astype(int)
    pix = upix - 4 * np.power(nside, 2)
    return pix, nside


def pix_to_upix(pix, nside):
    """Compute the unique pixel number from the pixel number and nside."""
    return pix + 4 * np.power(nside, 2)


class HPXGeom(MapGeom):
    """Geometry class for HEALPIX maps and cubes.

    This class performs
    mapping between partial-sky indices (pixel number within a HEALPIX
    region) and all-sky indices (pixel number within an all-sky
    HEALPIX map).  Multi-band HEALPIX geometries use a global indexing
    scheme that assigns a unique pixel number based on the all-sky
    index and band index.  In the single-band case the global index is
    the same as the HEALPIX index.

    Parameters
    ----------
    nside : `~numpy.ndarray`
        HEALPIX nside parameter, the total number of pixels is
        12*nside*nside.  For multi-dimensional maps one can pass
        either a single nside value or a vector of nside values
        defining the pixel size for each image plane.  If nside is not
        a scalar then its dimensionality should match that of the
        non-spatial axes.
    nest : bool
        True -> 'NESTED', False -> 'RING' indexing scheme
    coordsys : str
        Coordinate system, 'CEL' | 'GAL'
    region : str
        Region string defining the spatial geometry of the map.  If
        none the map will encompass the whole sky.
    axes : list
        Axes for non-spatial dimensions.
    sparse : bool
        If True defer allocation of partial- to all-sky index mapping arrays.
    """

    def __init__(self, nside, nest=True,
                 coordsys='CEL', region=None,
                 axes=None, conv=HPX_Conv('FGST_CCUBE'), sparse=False):

        self._nside = np.array(nside, ndmin=1)
        self._axes = axes if axes is not None else []
        self._shape = tuple([len(ax) - 1 for ax in self._axes])
        if self.nside.size > 1 and self.nside.shape != self._shape:
            raise Exception('Wrong dimensionality for nside.  nside must '
                            'be a scalar or have a dimensionality consistent '
                            'with the axes argument.')

        self._order = nside_to_order(self._nside)
        self._nest = nest
        self._coordsys = coordsys
        self._region = region
        self._maxpix = 12 * self._nside * self._nside
        self._maxpix = self._maxpix * np.ones(self._shape, dtype=int)

        self._ipix = None
        self._rmap = None
        self._npix = self._maxpix
        if self._region:
            self._create_lookup(self._region)

        self._npix = self._npix * np.ones(self._shape, dtype=int)
        self._conv = conv

    def _create_lookup(self, region):
        """Create local-to-global and global-to-local pixel lookup tables."""
        ipix = [self.get_index_list(nside, self._nest, region)
                for nside in self._nside.flat]
        self._ipix = [ravel_hpx_index((p, i * np.ones_like(p)),
                                      np.ravel(self._maxpix)) for i, p in
                      enumerate(ipix)]
        self._npix = np.array([len(t) for t in self._ipix])
        if self.nside.ndim > 1:
            self._npix = self._npix.reshape(self.nside.shape)
        self._ipix = np.concatenate(self._ipix)
        self._rmap = {}
        for i, ipixel in enumerate(self._ipix.flat):
            self._rmap[ipixel] = i

    def local_to_global(self, idx):
        """Compute a global index (partial-sky) from a global (all-sky)
        index."""
        pass

    def global_to_local(self, idx):
        """Compute a local (partial-sky) index from a global (all-sky)
        index."""
        pass

    def __getitem__(self, sliced):
        """This implements the global-to-local index lookup.

        For all-sky maps it just returns the input array.
        For partial-sky maps it returns the local indices corresponding to the indices in the
        input array, and -1 for those pixels that are outside the
        selected region.  For multi-dimensional maps with a different
        ``NSIDE`` in each band the global index is an unrolled index for
        both HEALPIX pixel number and image slice.

        Parameters
        ----------
        sliced: `~numpy.ndarray`
            An array of pixel indices.
            If this is a tuple, list, or
            array of integers it will be interpreted as a global
            (raveled) index.  If this argument is a tuple of lists or
            arrays it will be interpreted as a list of unraveled index
            vectors.

        Returns
        -------
        idx_local : `~numpy.ndarray`
            An array of local HEALPIX pixel indices.
        """
        # Convert to tuple representation
        if (isinstance(sliced, int) or
                (isinstance(sliced, tuple) and isinstance(sliced[0], int)) or
                isinstance(sliced, np.ndarray)):
            sliced = unravel_hpx_index(np.array(sliced, ndmin=1), self._maxpix)

        if self.nside.size == 1:
            idx = np.array(sliced[0], ndmin=1)
        else:
            idx = ravel_hpx_index(sliced, self._maxpix)

        if self._rmap is not None:
            retval = np.empty((idx.size), 'i')
            retval.fill(-1)
            m = np.in1d(idx.flat, self._ipix)
            retval[m] = np.searchsorted(self._ipix, idx.flat[m])
            retval = retval.reshape(idx.shape)
        else:
            retval = idx

        if self.nside.size == 1:
            retval = ravel_hpx_index([retval] + list(sliced[1:]),
                                     self.npix)

        return retval

    def coord_to_pix(self, coords):
        import healpy as hp
        c = MapCoords.create(coords)
        phi = np.radians(c.lon)
        theta = np.pi / 2. - np.radians(c.lat)

        if self.axes:

            bins = []
            for i, ax in enumerate(self.axes):
                bins += [val_to_bin(ax, c[i + 2])]

            # Ravel multi-dimensional indices
            ibin = np.ravel_multi_index(bins, self._shape)

            if self.nside.size > 1:
                nside = self.nside[ibin]
            else:
                nside = self.nside

            pix = hp.ang2pix(nside, theta, phi, nest=self.nest)
            pix = tuple([pix] + bins)
        else:
            pix = hp.ang2pix(self.nside, theta, phi, nest=self.nest),

        return pix

    def pix_to_coord(self, pix):
        import healpy as hp
        if self.axes:

            bins = []
            vals = []
            for i, ax in enumerate(self.axes):
                bins += [pix[1 + i]]
                vals += [bin_to_val(ax, pix[1 + i])]

            # Ravel multi-dimensional indices
            ibin = np.ravel_multi_index(bins, self._shape)

            if self.nside.size > 1:
                nside = self.nside[ibin]
            else:
                nside = self.nside

            ipix = np.round(pix[0]).astype(int)
            theta, phi = hp.pix2ang(nside, ipix, nest=self.nest)
            coords = [np.degrees(np.pi / 2. - theta), np.degrees(phi)]
            coords = tuple(coords + [vals])
        else:
            ipix = np.round(pix[0]).astype(int)
            theta, phi = hp.pix2ang(self.nside, ipix, nest=self.nest)
            coords = (np.degrees(np.pi / 2. - theta), np.degrees(phi))

        return coords

    @property
    def axes(self):
        """Non-spatial axes."""
        return self._axes

    @property
    def ndim(self):
        return len(self._axes) + 2

    @property
    def ordering(self):
        if self._nest:
            return 'NESTED'
        return 'RING'

    @property
    def nside(self):
        """NSIDE in each band."""
        return self._nside

    @property
    def order(self):
        return self._order

    @property
    def nest(self):
        return self._nest

    @property
    def npix(self):
        """Number of pixels in each band.

        For partial-sky geometries this can be less than the
        number of pixels for the band NSIDE.
        """
        return self._npix

    @property
    def conv(self):
        return self._conv

    @property
    def coordsys(self):
        return self._coordsys

    @property
    def region(self):
        return self._region

    @property
    def ipix(self):
        """HEALPIX pixel and band indices for every pixel in the map."""
        if self.nside.shape == self._maxpix.shape:
            return unravel_hpx_index(self._ipix, self._maxpix)
        else:
            maxpix = np.ravel(self._maxpix * np.arange(self.npix.size))[None, :]
            maxpix = maxpix * np.ones([self._npix[0]] + list(self._shape), dtype=int)
            maxpix = np.ravel(maxpix.T)
            ipix = np.ravel(self._ipix[None, :] * np.ones(self._shape, dtype=int)[..., None])
            return unravel_hpx_index(ipix + maxpix, self._maxpix)

    def ud_graded_hpx(self, order):
        """TODO.
        """
        if np.any(self.order < 0):
            raise ValueError('Upgrade and degrade only implemented for standard maps')

        return self.__class__(2 ** order, self.nest, self.coordsys,
                              self.region, self.axes, self.conv)

    def make_swapped_hpx(self):
        """TODO.
        """
        return self.__class__(self.nside, not self.nest, self.coordsys,
                              self.region, self.axes, self.conv)

    def copy_and_drop_axes(self):
        """TODO.
        """
        return self.__class__(self.nside[0], not self.nest, self.coordsys,
                              self.region, None, self.conv)

    @classmethod
    def create(cls, nside, nest, coordsys='CEL', region=None,
               axes=None, conv=HPX_Conv('FGST_CCUBE')):
        """Create a HPX object.

        Parameters
        ----------
        nside : `~numpy.ndarray`
            HEALPIX nside parameter
        nest : bool
            True for HEALPIX "NESTED" indexing scheme, False for "RING" scheme
        coordsys : str
            "CEL" or "GAL"
        region  : str
            Allows for partial-sky mappings
        axes : list
            List of axes for non-spatial dimensions
        """
        return cls(nside, nest, coordsys, region, axes, conv)

    @staticmethod
    def identify_HPX_convention(header):
        """Identify the convention used to write this file."""
        # Hopefully the file contains the HPX_CONV keyword specifying
        # the convention used
        try:
            return header['HPX_CONV']
        except KeyError:
            pass

        # Try based on the EXTNAME keyword
        extname = header.get('EXTNAME', None)
        if extname == 'HPXEXPOSURES':
            return 'FGST_BEXPCUBE'
        elif extname == 'SKYMAP2':
            if 'COORDTYPE' in header.keys():
                return 'GALPROP'
            else:
                return 'GALPROP2'

        # Check the name of the first column
        colname = header['TTYPE1']
        if colname == 'PIX':
            colname = header['TTYPE2']

        if colname == 'KEY':
            return 'FGST_SRCMAP_SPARSE'
        elif colname == 'ENERGY1':
            return 'FGST_TEMPLATE'
        elif colname == 'COSBINS':
            return 'FGST_LTCUBE'
        elif colname == 'Bin0':
            return 'GALPROP'
        elif colname == 'CHANNEL1':
            if extname == 'SKYMAP':
                return 'FGST_CCUBE'
            else:
                return 'FGST_SRCMAP'
        else:
            raise ValueError('Could not identify HEALPIX convention')

    @classmethod
    def from_header(cls, header, axes=None):
        """Create an HPX object from a FITS header.

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            The FITS header
        axes  : list
            List of non-spatial axes
        """
        convname = HPXGeom.identify_HPX_convention(header)
        conv = HPX_FITS_CONVENTIONS[convname]

        if header['PIXTYPE'] != 'HEALPIX':
            raise Exception('PIXTYPE != HEALPIX')
        if header['ORDERING'] == 'RING':
            nest = False
        elif header['ORDERING'] == 'NESTED':
            nest = True
        else:
            raise Exception('ORDERING != RING | NESTED')

        if 'NSIDE' in header:
            nside = header['NSIDE']
        elif 'ORDER' in header:
            nside = 2 ** header['ORDER']
        else:
            raise Exception('Failed to extract NSIDE or ORDER.')

        try:
            coordsys = header[conv.coordsys]
        except KeyError:
            coordsys = header['COORDSYS']

        try:
            region = header['HPX_REG']
        except KeyError:
            try:
                region = header['HPXREGION']
            except KeyError:
                region = None

        return cls(nside, nest, coordsys, region, axes=axes, conv=conv)

    def make_header(self):
        """"Build and return FITS header for this HEALPIX map."""
        cards = [
            fits.Card("TELESCOP", "GLAST"),
            fits.Card("INSTRUME", "LAT"),
            fits.Card(self._conv.coordsys, self.coordsys),
            fits.Card("PIXTYPE", "HEALPIX"),
            fits.Card("ORDERING", self.ordering),
            fits.Card("ORDER", self._order[0]),
            fits.Card("NSIDE", self._nside[0]),
            fits.Card("FIRSTPIX", 0),
            fits.Card("LASTPIX", self._maxpix[0] - 1),
            fits.Card("HPX_CONV", self._conv.convname),
        ]

        if self.coordsys == 'CEL':
            cards.append(fits.Card('EQUINOX', 2000.0,
                                   'Equinox of RA & DEC specifications'))

        if self.region:
            cards.append(fits.Card('HPX_REG', self._region))

        return fits.Header(cards)

    def make_ebounds_hdu(self, extname='EBOUNDS'):
        """Make a FITS HDU with the energy bin boundaries.

        Parameters
        ----------
        extname : str
            The HDU extension name
        """
        emin = self._axes[:-1]
        emax = self._axes[1:]

        cols = [fits.Column('CHANNEL', 'I', array=np.arange(1, len(self._zbins + 1))),
                fits.Column('E_MIN', '1E', unit='keV', array=1000 * emin),
                fits.Column('E_MAX', '1E', unit='keV', array=1000 * emax)]
        hdu = fits.BinTableHDU.from_columns(cols, self.make_header(), name=extname)
        return hdu

    def make_energies_hdu(self, extname='ENERGIES'):
        """Make a FITS HDU with the energy bin centers.

        Parameters
        ----------
        extname : str
            The HDU extension name
        """
        ectr = np.sqrt(self._axes[1:] * self._axes[:-1])
        cols = [fits.Column('ENERGY', '1E', unit='MeV', array=ectr)]
        hdu = fits.BinTableHDU.from_columns(
            cols, self.make_header(), name=extname)
        return hdu

    def write(self, data, outfile, extname='SKYMAP', clobber=True):
        """Write input data to a FITS file.

        Parameters
        ----------
        data :
            The data being stored
        outfile : str
            The name of the output file
        extname :
            The HDU extension name
        clobber : bool
            True -> overwrite existing files
        """
        hdu_prim = fits.PrimaryHDU()
        hdu_hpx = self.make_hdu(data, extname=extname)
        hl = [hdu_prim, hdu_hpx]

        if self.conv.energy_hdu == 'EBOUNDS':
            hdu_energy = self.make_energy_bounds_hdu()
        elif self.conv.energy_hdu == 'ENERGIES':
            hdu_energy = self.make_energies_hdu()

        if hdu_energy is not None:
            hl.append(hdu_energy)

        hdulist = fits.HDUList(hl)
        hdulist.writeto(outfile, clobber=clobber)

    @staticmethod
    def get_index_list(nside, nest, region):
        """Get list of pixels indices for all the pixels in a region.

        Parameters
        ----------
        nside : int
            HEALPIX nside parameter
        nest : bool
            True for 'NESTED', False = 'RING'
        region : str
            HEALPIX region string
        """
        import healpy as hp
        tokens = parse_hpxregion(region)

        if tokens[0] == 'DISK':
            vec = coords_to_vec(float(tokens[1]), float(tokens[2]))
            ilist = hp.query_disc(nside, vec[0], np.radians(float(tokens[3])),
                                  inclusive=False, nest=nest)
        elif tokens[0] == 'DISK_INC':
            vec = coords_to_vec(float(tokens[1]), float(tokens[2]))
            ilist = hp.query_disc(nside, vec[0], np.radians(float(tokens[3])),
                                  inclusive=True, fact=int(tokens[4]),
                                  nest=nest)
        elif tokens[0] == 'HPX_PIXEL':
            nside_pix = int(tokens[2])
            if tokens[1] == 'NESTED':
                ipix_ring = hp.nest2ring(nside_pix, int(tokens[3]))
            elif tokens[1] == 'RING':
                ipix_ring = int(tokens[3])
            else:
                raise ValueError('Did not recognize ordering scheme: {}'.format(tokens[1]))
            ilist = match_hpx_pixel(nside, nest, nside_pix, ipix_ring)
        else:
            raise ValueError('Did not recognize region type: {}'.format(tokens[0]))

        return ilist

    @staticmethod
    def get_ref_dir(region, coordsys):
        """Get the reference direction for a given  HEALPIX region string.

        Parameters
        ----------
        region : str
            A string describing a HEALPIX region
        coordsys : {'CEL', 'GAL'}
            Coordinate system
        """
        import healpy as hp
        frame = 'galactic' if coordsys == 'GAL' else 'icrs'

        if region is None:
            return SkyCoord(0., 0., frame=frame, unit="deg")

        tokens = parse_hpxregion(region)
        if tokens[0] in ['DISK', 'DISK_INC']:
            lon, lat = float(tokens[1]), float(tokens[2])
            return SkyCoord(lon, lat, frame=frame, unit='deg')
        elif tokens[0] == 'HPX_PIXEL':
            nside_pix = int(tokens[2])
            ipix_pix = int(tokens[3])
            if tokens[1] == 'NESTED':
                nest_pix = True
            elif tokens[1] == 'RING':
                nest_pix = False
            else:
                raise ValueError('Did not recognize ordering scheme: {}'.format(tokens[1]))
            theta, phi = hp.pix2ang(nside_pix, ipix_pix, nest_pix)
            lat = np.degrees((np.pi / 2) - theta)
            lon = np.degrees(phi)
            return SkyCoord(lon, lat, frame=frame, unit='deg')
        else:
            raise ValueError('HPX.get_ref_dir did not recognize region type: {}'.format(tokens[0]))

        return None

    @staticmethod
    def get_region_size(region):
        """Get the approximate size of region (in degrees) from a HEALPIX region string.
        """
        if region is None:
            return 180.
        tokens = parse_hpxregion(region)
        if tokens[0] in ['DISK', 'DISK_INC']:
            return float(tokens[3])
        elif tokens[0] == 'HPX_PIXEL':
            pixel_size = get_pixel_size_from_nside(int(tokens[2]))
            return 2. * pixel_size
        else:
            raise Exception('Did not recognize region type: {}'.format(tokens[0]))

    def make_wcs(self, naxis=2, proj='CAR', axes=None, oversample=2):
        """Make a WCS projection appropriate for this HPX pixelization.

        Parameters
        ----------
        naxis : int
            Set the number of axes that will be extracted from the
            HEALPIX geometry.  If None then all dimensions of the
            HEALPIX geometry will be copied to the WCS geometry.
        proj : str
            Projection type of WCS geometry.
        oversample : int
            Factor by which the WCS pixel size will be chosen to
            oversample the HEALPIX map.

        Returns
        -------
        wcs : `~gammapy.maps.wcs.WCSGeom`
            WCS geometry
        """
        if naxis < 2 or naxis > self.ndim:
            raise ValueError('naxis must be between 2 or the total number '
                             'of dimensions.')

        w = WCS(naxis=naxis)
        skydir = self.get_ref_dir(self._region, self.coordsys)

        if self.coordsys == 'CEL':
            w.wcs.ctype[0] = 'RA---{}'.format(proj)
            w.wcs.ctype[1] = 'DEC--{}'.format(proj)
            w.wcs.crval[0] = skydir.ra.deg
            w.wcs.crval[1] = skydir.dec.deg
        elif self.coordsys == 'GAL':
            w.wcs.ctype[0] = 'GLON-{}'.format(proj)
            w.wcs.ctype[1] = 'GLAT-{}'.format(proj)
            w.wcs.crval[0] = skydir.galactic.l.deg
            w.wcs.crval[1] = skydir.galactic.b.deg
        else:
            raise ValueError('Unrecognized coordinate system.')

        pixsize = np.min(get_pixel_size_from_nside(self.nside))
        roisize = self.get_region_size(self._region)
        allsky = False
        if roisize > 45:
            roisize = 90
            allsky = True

        npixels = int(2. * roisize / pixsize) * oversample
        crpix = npixels / 2.

        if allsky:
            w.wcs.crpix[0] = 2 * crpix
            npix = (2 * npixels, npixels)
        else:
            w.wcs.crpix[0] = crpix
            npix = (npixels, npixels)

        w.wcs.crpix[1] = crpix
        w.wcs.cdelt[0] = -pixsize / oversample
        w.wcs.cdelt[1] = pixsize / oversample

        if naxis == 3:
            w.wcs.crpix[2] = 1
            w.wcs.ctype[2] = 'Energy'
            if energies is not None:
                w.wcs.crval[2] = 10 ** energies[0]
                w.wcs.cdelt[2] = 10 ** energies[1] - 10 ** energies[0]

        w = WCS(w.to_header())
        wcs_proj = WCSGeom(w, npix)
        return wcs_proj

    def get_coords(self):
        """Get the coordinates of all the pixels in this pixelization."""
        import healpy as hp
        if self._ipix is None:
            theta, phi = hp.pix2ang(self.nside, range(self.npix), self.nest)
        else:
            theta, phi = hp.pix2ang(self.nside, self.ipix, self.nest)

        lat = np.degrees((np.pi / 2) - theta)
        lon = np.degrees(phi)
        return np.vstack([lon, lat]).T

    def get_skydirs(self):
        lonlat = self.get_coords()
        return SkyCoord(ra=lonlat.T[0], dec=lonlat.T[1], unit='deg')

    def skydir_to_pixel(self, skydir):
        """Return the pixel index of a SkyCoord object."""
        if self.coordsys in ['CEL', 'EQU']:
            skydir = skydir.transform_to('icrs')
            lon = skydir.ra.deg
            lat = skydir.dec.deg
        else:
            skydir = skydir.transform_to('galactic')
            lon = skydir.l.deg
            lat = skydir.b.deg

        return self.get_pixel_indices(lat, lon)


class HpxToWcsMapping(object):
    """Stores the indices need to convert from HEALPIX to WCS.

    Parameters
    ----------
    TODO
    """

    def __init__(self, hpx, wcs, mapping_data=None):
        self._hpx = hpx
        self._wcs = wcs
        if mapping_data is None:
            self._ipix, self._mult_val, self._npix = make_hpx_to_wcs_mapping(
                self.hpx, self.wcs.wcs)
        else:
            self._ipix = mapping_data['ipix']
            self._mult_val = mapping_data['mult_val']
            self._npix = mapping_data['npix']
        self._lmap = self._hpx[self._ipix]
        self._valid = self._lmap > 0

    @property
    def hpx(self):
        """The HEALPIX projection"""
        return self._hpx

    @property
    def wcs(self):
        """The WCS projection"""
        return self._wcs

    @property
    def ipix(self):
        """An array(nx,ny) of the global HEALPIX pixel indices for each WCS pixel"""
        return self._ipix

    @property
    def mult_val(self):
        """An array(nx,ny) of 1/number of WCS pixels pointing at each HEALPIX pixel"""
        return self._mult_val

    @property
    def npix(self):
        """A tuple(nx,ny) of the shape of the WCS grid"""
        return self._npix

    @property
    def lmap(self):
        """An array(nx,ny) giving the mapping of the local HEALPIX pixel
        indices for each WCS pixel"""
        return self._lmap

    @property
    def valid(self):
        """An array(nx,ny) of bools giving if each WCS pixel in inside the
        HEALPIX region"""
        return self._valid

    def write(self, fitsfile, clobber=True):
        """Write this mapping to a FITS file, to avoid having to recompute it
        """
        from fermipy.skymap import Map
        hpx_header = self._hpx.make_header()
        index_map = Map(self.ipix, self.wcs)
        mult_map = Map(self.mult_val, self.wcs)
        prim_hdu = index_map.create_primary_hdu()
        mult_hdu = index_map.create_image_hdu()
        for key in ['COORDSYS', 'ORDERING', 'PIXTYPE',
                    'ORDERING', 'ORDER', 'NSIDE',
                    'FIRSTPIX', 'LASTPIX']:
            prim_hdu.header[key] = hpx_header[key]
            mult_hdu.header[key] = hpx_header[key]

        hdulist = fits.HDUList([prim_hdu, mult_dhu])
        hdulist.writeto(fitsfile, clobber=clobber)

    @classmethod
    def read(cls, filename):
        """Read a FITS file and use it to make a mapping."""
        from fermipy.skymap import Map
        index_map = Map.read(filename)
        mult_map = Map.read(filename, hdu=1)
        ff = fits.open(filename)
        hpx = HPXGeom.from_header(ff[0])
        mapping_data = dict(ipix=index_map.counts,
                            mult_val=mult_map.counts,
                            npix=mult_map.counts.shape)
        return cls(hpx, index_map.wcs, mapping_data)

    def fill_wcs_map_from_hpx_data(self, hpx_data, wcs_data, normalize=True):
        """Fill the WCS map from the hpx data using the pre-calculated mappings.

        Parameters
        ----------
        hpx_data : TODO
            The input HEALPIX data
        wcs_data : TODO
            The data array being filled
        normalize : bool
            True -> preserve integral by splitting HEALPIX values between bins
        """
        # FIXME, there really ought to be a better way to do this
        hpx_data_flat = hpx_data.flatten()
        wcs_data_flat = np.zeros((wcs_data.size))
        lmap_valid = self._lmap[self._valid]
        wcs_data_flat[self._valid] = hpx_data_flat[lmap_valid]
        if normalize:
            wcs_data_flat *= self._mult_val
        wcs_data.flat = wcs_data_flat

    def make_wcs_data_from_hpx_data(self, hpx_data, wcs, normalize=True):
        """Create and fill a WCS map from the HEALPIX data using the pre-calculated mappings.

        Parameters
        ----------
        hpx_data : TODO
            The input HEALPIX data
        wcs : TODO
            The WCS object
        normalize : bool
            True -> preserve integral by splitting HEALPIX values between bins
        """
        wcs_data = np.zeros(wcs.npix)
        self.fill_wcs_map_from_hpx_data(hpx_data, wcs_data, normalize)
        return wcs_data
