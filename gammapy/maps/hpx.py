# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for dealing with HEALPix projections and mappings."""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import re
import copy
import numpy as np
from astropy.extern import six
from astropy.extern.six.moves import range
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from .wcs import WcsGeom
from .geom import MapGeom, MapCoords, MapAxis, bin_to_val, pix_tuple_to_idx
from .geom import coordsys_to_frame, skydir_to_lonlat, make_axes_cols
from .geom import find_and_read_bands, make_axes

# TODO: What should be part of the public API?
__all__ = [
    # 'HpxConv',
    # 'HPX_FITS_CONVENTIONS',
    # 'HPX_ORDER_TO_PIXSIZE',
    'HpxGeom',
    # 'HpxToWcsMapping',
]

# This is an approximation of the size of HEALPIX pixels (in degrees)
# for a particular order.   It is used to convert from HEALPIX to WCS-based
# projections
HPX_ORDER_TO_PIXSIZE = np.array([32.0, 16.0, 8.0, 4.0, 2.0, 1.0,
                                 0.50, 0.25, 0.1, 0.05, 0.025, 0.01,
                                 0.005, 0.002])


class HpxConv(object):
    """Data structure to define how a HEALPIX map is stored to FITS.
    """

    def __init__(self, convname, **kwargs):
        self.convname = convname
        self.colstring = kwargs.get('colstring', 'CHANNEL')
        self.firstcol = kwargs.get('firstcol', 1)
        self.extname = kwargs.get('extname', 'SKYMAP')
        self.bands_hdu = kwargs.get('bands_hdu', 'EBOUNDS')
        self.quantity_type = kwargs.get('quantity_type', 'integral')
        self.coordsys = kwargs.get('coordsys', 'COORDSYS')

    def colname(self, indx):
        return '{}{}'.format(self.colstring, indx)

    @classmethod
    def create(cls, convname='GADF'):
        return copy.deepcopy(HPX_FITS_CONVENTIONS[convname])


# Various conventions for storing HEALPIX maps in FITS files
HPX_FITS_CONVENTIONS = OrderedDict()
HPX_FITS_CONVENTIONS[None] = HpxConv('GADF', bands_hdu='BANDS')
HPX_FITS_CONVENTIONS['GADF'] = HpxConv('GADF', bands_hdu='BANDS')
HPX_FITS_CONVENTIONS['FGST_CCUBE'] = HpxConv('FGST_CCUBE')
HPX_FITS_CONVENTIONS['FGST_LTCUBE'] = HpxConv(
    'FGST_LTCUBE', colstring='COSBINS', extname='EXPOSURE', bands_hdu='CTHETABOUNDS')
HPX_FITS_CONVENTIONS['FGST_BEXPCUBE'] = HpxConv(
    'FGST_BEXPCUBE', colstring='ENERGY', extname='HPXEXPOSURES', bands_hdu='ENERGIES')
HPX_FITS_CONVENTIONS['FGST_SRCMAP'] = HpxConv(
    'FGST_SRCMAP', extname=None, quantity_type='differential')
HPX_FITS_CONVENTIONS['FGST_TEMPLATE'] = HpxConv(
    'FGST_TEMPLATE', colstring='ENERGY', bands_hdu='ENERGIES')
HPX_FITS_CONVENTIONS['FGST_SRCMAP_SPARSE'] = HpxConv(
    'FGST_SRCMAP_SPARSE', colstring=None, extname=None, quantity_type='differential')
HPX_FITS_CONVENTIONS['GALPROP'] = HpxConv(
    'GALPROP', colstring='Bin', extname='SKYMAP2',
    bands_hdu='ENERGIES', quantity_type='differential', coordsys='COORDTYPE')
HPX_FITS_CONVENTIONS['GALPROP2'] = HpxConv(
    'GALPROP', colstring='Bin', extname='SKYMAP2',
    bands_hdu='ENERGIES', quantity_type='differential')


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
    if npix.size == 1:
        return tuple([idx])

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

    Returns
    -------
    idx : `~numpy.ndarray`
    """
    if len(idx) == 1:
        return idx[0]

    # TODO: raise exception for indices that are out of bounds

    idx0 = idx[0]
    idx1 = np.ravel_multi_index(idx[1:], npix.shape, mode='clip')
    npix = np.concatenate((np.array([0]), npix.flat[:-1]))

    return idx0 + np.cumsum(npix)[idx1]


def lonlat_to_colat(lon, lat):
    phi = np.radians(lon)
    theta = (np.pi / 2) - np.radians(lat)
    return phi, theta


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


def get_nside_from_pixel_size(pixsz):
    """Get the NSIDE that is closest to the given pixel size.

    Parameters
    ----------
    pix : `~numpy.ndarray`
        Pixel size in degrees.

    Returns
    -------
    nside : `~numpy.ndarray`
        NSIDE parameter.
    """
    import healpy as hp
    pixsz = np.array(pixsz, ndmin=1)
    nside = 2 ** np.linspace(1, 14, 14, dtype=int)
    nside_pixsz = np.degrees(hp.nside2resol(nside))
    return nside[np.argmin(np.abs(nside_pixsz - pixsz[..., None]), axis=-1)]


def get_pixel_size_from_nside(nside):
    """Estimate of the pixel size from the HEALPIX nside coordinate.

    This just uses a lookup table to provide a nice round number
    for each HEALPIX order.
    """
    order = nside_to_order(nside)
    if np.any(order < 0) or np.any(order > 13):
        raise ValueError(
            'HEALPIX order must be between 0 to 13. Value is: {}'.format(order))

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
    hpx : `~gammapy.maps.hpx.HpxGeom`
       The HEALPIX geometry
    wcs : `~gammapy.maps.wcs.WcsGeom`
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
    npix = wcs.npix

    # FIXME: Calculation of WCS pixel centers should be moved into a
    # method of WcsGeom
    pix_crds = np.dstack(np.meshgrid(np.arange(npix[0]), np.arange(npix[1])))
    pix_crds = pix_crds.swapaxes(0, 1).reshape((-1, 2))
    sky_crds = wcs.wcs.wcs_pix2world(pix_crds, 0)
    sky_crds *= np.radians(1.)
    sky_crds[0:, 1] = (np.pi / 2) - sky_crds[0:, 1]

    mask = ~np.any(np.isnan(sky_crds), axis=1)
    ipix = -1 * np.ones((len(hpx.nside), int(npix[0] * npix[1])), int)
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


class HpxGeom(MapGeom):
    """Geometry class for HEALPIX maps.

    This class performs mapping between partial-sky indices (pixel
    number within a HEALPIX region) and all-sky indices (pixel number
    within an all-sky HEALPIX map).  Multi-band HEALPIX geometries use
    a global indexing scheme that assigns a unique pixel number based
    on the all-sky index and band index.  In the single-band case the
    global index is the same as the HEALPIX index.

    By default the constructor will return an all-sky map.
    Partial-sky maps can be defined with the ``region`` argument.

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
    region : str or tuple
        Spatial geometry for partial-sky maps.  If none the map will
        encompass the whole sky.  String input will be parsed
        according to HPX_REG header keyword conventions.  Tuple
        input can be used to define an explicit list of pixels
        encompassed by the geometry.
    axes : list
        Axes for non-spatial dimensions.
    conv : str
        Convention for FITS serialization format.
    sparse : bool
        If True defer allocation of partial- to all-sky index mapping
        arrays.  This option is only compatible with partial-sky maps
        with an analytic geometry (e.g. DISK).
    """

    def __init__(self, nside, nest=True, coordsys='CEL', region=None,
                 axes=None, conv='GADF', sparse=False):

        # FIXME: Figure out what to do when sparse=True
        # FIXME: Require NSIDE to be power of two when nest=True

        self._nside = np.array(nside, ndmin=1)
        self._axes = make_axes(axes, conv)
        self._shape = tuple([ax.nbin for ax in self._axes])
        if self.nside.size > 1 and self.nside.shape != self._shape:
            raise Exception('Wrong dimensionality for nside.  nside must '
                            'be a scalar or have a dimensionality consistent '
                            'with the axes argument.')

        self._order = nside_to_order(self._nside)
        self._nest = nest
        self._coordsys = coordsys
        self._maxpix = 12 * self._nside * self._nside
        self._maxpix = self._maxpix * np.ones(self._shape, dtype=int)

        self._ipix = None
        self._rmap = None
        self._create_lookup(region)

        if self._ipix is not None:
            self._rmap = {}
            for i, ipixel in enumerate(self._ipix.flat):
                self._rmap[ipixel] = i

        self._npix = self._npix * np.ones(self._shape, dtype=int)
        self._conv = conv
        self._center_skydir = self.get_ref_dir(region, self.coordsys)
        self._center_coord = tuple(list(skydir_to_lonlat(self._center_skydir)) +
                                   [ax.pix_to_coord((float(ax.nbin) - 1.0) / 2.) for ax in self.axes])
        self._center_pix = self.coord_to_pix(self._center_coord)

    def _create_lookup(self, region):
        """Create local-to-global pixel lookup table."""

        if isinstance(region, six.string_types):
            ipix = [self.get_index_list(nside, self._nest, region)
                    for nside in self._nside.flat]
            self._ibnd = np.concatenate([i * np.ones_like(p, dtype='int16') for
                                         i, p in enumerate(ipix)])
            self._ipix = [ravel_hpx_index((p, i * np.ones_like(p)),
                                          np.ravel(self._maxpix)) for i, p in
                          enumerate(ipix)]
            self._region = region
            self._indxschm = 'EXPLICIT'
            self._npix = np.array([len(t) for t in self._ipix])
            if self.nside.ndim > 1:
                self._npix = self._npix.reshape(self.nside.shape)
            self._ipix = np.concatenate(self._ipix)

        elif isinstance(region, tuple):

            # FIXME: How to determine reference direction for explicit
            # geom?

            self._ipix = ravel_hpx_index(region, self._maxpix)
            self._region = 'explicit'
            self._indxschm = 'EXPLICIT'
            if len(region) == 1:
                self._npix = np.array([len(region[0])])
            else:
                self._npix = np.zeros(self._shape, dtype=int)
                idx = np.ravel_multi_index(region[1:], self._shape)
                cnt = np.unique(idx, return_counts=True)
                self._npix.flat[cnt[0]] = cnt[1]

        elif region is None:
            self._region = None
            self._indxschm = 'IMPLICIT'
            self._npix = self._maxpix

        else:
            raise ValueError(
                'Invalid input for region string: {}'.format(region))

    def local_to_global(self, idx_local):
        """Compute a local index (partial-sky) from a global (all-sky)
        index.

        Returns
        -------
        idx_global : tuple
            A tuple of pixel index vectors with global HEALPIX pixel
            indices.
        """

        if self._ipix is None:
            return idx_local

        if self.nside.size > 1:
            idx = ravel_hpx_index(idx_local, self._npix)
        else:
            idx_tmp = tuple([idx_local[0]] +
                            [np.zeros(t.shape, dtype=int)
                             for t in idx_local[1:]])
            idx = ravel_hpx_index(idx_tmp, self._npix)

        idx_global = unravel_hpx_index(self._ipix[idx], self._maxpix)
        return idx_global[:1] + idx_local[1:]

    def global_to_local(self, idx_global):
        """Compute a global (all-sky) index from a local (partial-sky)
        index.

        Returns
        -------
        idx_local : tuple
            A tuple of pixel index vectors with local HEALPIX pixel
            indices.
        """

        if self.nside.size == 1:
            idx = np.array(idx_global[0], ndmin=1)
        else:
            idx = ravel_hpx_index(idx_global, self._maxpix)

        if self._rmap is not None:
            retval = np.empty((idx.size), 'i')
            retval.fill(-1)
            m = np.in1d(idx.flat, self._ipix)
            retval[m] = np.searchsorted(self._ipix, idx.flat[m])
            retval = retval.reshape(idx.shape)
        else:
            retval = idx

        if self.nside.size == 1:
            retval = tuple([retval] + list(idx_global[1:]))
        else:
            retval = unravel_hpx_index(retval, self._npix)

        return retval

    def __getitem__(self, idx_global):
        """This implements the global-to-local index lookup.

        For all-sky maps it just returns the input array.  For
        partial-sky maps it returns the local indices corresponding to
        the indices in the input array, and -1 for those pixels that
        are outside the selected region.  For multi-dimensional maps
        with a different ``NSIDE`` in each band the global index is an
        unrolled index for both HEALPIX pixel number and image slice.

        Parameters
        ----------
        idx_global: `~numpy.ndarray`
            An array of global (all-sky) pixel indices.  If this is a
            tuple, list, or array of integers it will be interpreted
            as a global (raveled) index.  If this argument is a tuple
            of lists or arrays it will be interpreted as a list of
            unraveled index vectors.

        Returns
        -------
        idx_local : `~numpy.ndarray`
            An array of local HEALPIX pixel indices.
        """
        # Convert to tuple representation
        if (isinstance(idx_global, int) or
                (isinstance(idx_global, tuple) and isinstance(idx_global[0], int)) or
                isinstance(idx_global, np.ndarray)):
            idx_global = unravel_hpx_index(np.array(idx_global, ndmin=1),
                                           self._maxpix)

        if self.nside.size == 1:
            idx = np.array(idx_global[0], ndmin=1)
        else:
            idx = ravel_hpx_index(idx_global, self._maxpix)

        if self._rmap is not None:
            retval = np.empty((idx.size), 'i')
            retval.fill(-1)
            m = np.in1d(idx.flat, self._ipix)
            retval[m] = np.searchsorted(self._ipix, idx.flat[m])
            retval = retval.reshape(idx.shape)
        else:
            retval = idx

        if self.nside.size == 1:
            retval = ravel_hpx_index([retval] + list(idx_global[1:]),
                                     self.npix)

        return retval

    def coord_to_pix(self, coords):
        import healpy as hp
        c = MapCoords.create(coords)
        phi = np.radians(c.lon)
        theta = np.pi / 2. - np.radians(c.lat)

        if self.axes:

            bins = []
            idxs = []
            for i, ax in enumerate(self.axes):
                bins += [ax.coord_to_pix(c[i + 2])]
                idxs += [ax.coord_to_idx(c[i + 2])]

            # FIXME: Figure out how to handle coordinates out of
            # bounds of non-spatial dimensions
            if self.nside.size > 1:
                nside = self.nside[idxs]
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
                vals += [ax.pix_to_coord(pix[1 + i])]

            idxs = pix_tuple_to_idx(bins)

            if self.nside.size > 1:
                nside = self.nside[idxs]
            else:
                nside = self.nside

            ipix = np.round(pix[0]).astype(int)
            theta, phi = hp.pix2ang(nside, ipix, nest=self.nest)
            coords = [np.degrees(phi), np.degrees(np.pi / 2. - theta)]
            coords = tuple(coords + vals)
        else:
            ipix = np.round(pix[0]).astype(int)
            theta, phi = hp.pix2ang(self.nside, ipix, nest=self.nest)
            coords = (np.degrees(phi), np.degrees(np.pi / 2. - theta))

        return coords

    def pix_to_idx(self, pix):

        # FIXME: Correctly apply bounds on non-spatial pixel
        # coordinates
        idx = list(pix_tuple_to_idx(pix))
        idx_local = self.global_to_local(idx)
        for i, _ in enumerate(idx):
            idx[i][idx_local[i] < 0] = -1
            if i > 0:
                idx[i][idx[i] > self.axes[i - 1].nbin - 1] = -1

        return tuple(idx)

    def to_slice(self, slices, drop_axes=True):

        if len(slices) == 0 and self.ndim == 2:
            return copy.deepcopy(self)

        if len(slices) != self.ndim - 2:
            raise ValueError

        nside = np.ones(self.shape, dtype=int) * self.nside
        nside = np.squeeze(nside[slices])

        axes = [ax.slice(s) for ax, s in zip(self.axes, slices)]
        if drop_axes:
            axes = [ax for ax in axes if ax.nbin > 1]
            slice_dims = [0] + [i + 1 for i,
                                ax in enumerate(axes) if ax.nbin > 1]
        else:
            slice_dims = np.arange(self.ndim)

        if self.region == 'explicit':
            idxs = [np.arange(ax.nbin)[s] for ax, s in zip(self.axes, slices)]
            pix = self.get_pixels()
            m = np.all([np.in1d(t, i) for i, t in zip(idxs, pix[1:])], axis=0)
            region = tuple([t[m]
                            for i, t in enumerate(pix) if i in slice_dims])
        else:
            region = self.region

        return HpxGeom(nside, self.nest, self.coordsys, region=region, axes=axes,
                       conv=self.conv)

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
        """ORDER in each band (NSIDE = 2**ORDER).  Set to -1 for bands with
        NSIDE that is not a power of 2.
        """
        return self._order

    @property
    def nest(self):
        return self._nest

    @property
    def npix(self):
        """Number of pixels in each band.

        For partial-sky geometries this can
        be less than the number of pixels for the band NSIDE.
        """
        return self._npix

    @property
    def conv(self):
        return self._conv

    @property
    def coordsys(self):
        return self._coordsys

    @property
    def projection(self):
        """Map projection."""
        return 'HPX'

    @property
    def region(self):
        """Region string."""
        return self._region

    @property
    def allsky(self):
        """Flag for all-sky maps."""
        if self._region is None:
            return True
        else:
            return False

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
    def ipix(self):
        """HEALPIX pixel and band indices for every pixel in the map."""
        return self.get_pixels()

    def ud_graded_hpx(self, order):
        """Upgrade or downgroad the resolution of this geometry to the given
        order.

        Returns
        -------
        geom : `~HpxGeom`
            A HEALPix geoemtry object.
        """
        if np.any(self.order < 0):
            raise ValueError(
                'Upgrade and degrade only implemented for standard maps')

        # FIXME: Pass ipix as argument

        return self.__class__(2 ** order, self.nest, coordsys=self.coordsys,
                              region=self.region, axes=self.axes, conv=self.conv)

    def to_swapped(self):
        """Make a copy of this geometry with a swapped ORDERING.  (NEST->RING
        or vice versa)

        Returns
        -------
        geom : `~HpxGeom`
            A HEALPix geoemtry object.
        """
        # FIXME: Pass ipix as argument

        return self.__class__(self.nside, not self.nest, coordsys=self.coordsys,
                              region=self.region, axes=self.axes, conv=self.conv)

    def to_image(self):
        return self.__class__(np.max(self.nside), not self.nest, coordsys=self.coordsys,
                              region=self.region, conv=self.conv)

    def to_cube(self, axes):
        axes = copy.deepcopy(self.axes) + axes
        return self.__class__(np.max(self.nside), not self.nest, coordsys=self.coordsys,
                              region=self.region, conv=self.conv, axes=axes)

    @classmethod
    def create(cls, nside=None, binsz=None, nest=True, coordsys='CEL', region=None,
               axes=None, conv='GADF', skydir=None, width=None):
        """Create an HpxGeom object.

        Parameters
        ----------
        nside : int or `~numpy.ndarray`
            HEALPix NSIDE parameter.  This parameter sets the size of
            the spatial pixels in the map.
        binsz : float or `~numpy.ndarray`
            Approximate pixel size in degrees.  An NSIDE will be
            chosen that correponds to a pixel size closest to this
            value.  This option is superseded by nside.
        nest : bool
            True for HEALPIX "NESTED" indexing scheme, False for "RING" scheme
        coordsys : {'CEL', 'GAL'}, optional
            Coordinate system, either Galactic ('GAL') or Equatorial ('CEL').
        skydir : tuple or `~astropy.coordinates.SkyCoord`
            Sky position of map center.  Can be either a SkyCoord
            object or a tuple of longitude and latitude in deg in the
            coordinate system of the map.
        region  : str
            HPX region string.  Allows for partial-sky maps.
        width : float
            Diameter of the map in degrees.  If set the map will
            encompass all pixels within a circular region centered on
            ``skydir``.
        axes : list
            List of axes for non-spatial dimensions.
        conv : str
            Convention for FITS file format.

        Returns
        -------
        geom : `~HpxGeom`
            A HEALPix geometry object.

        Examples
        --------
        >>> from gammapy.maps import HpxGeom
        >>> from gammapy.maps import MapAxis
        >>> axis = MapAxis.from_bounds(0,1,2)
        >>> geom = HpxGeom.create(nside=16)
        >>> geom = HpxGeom.create(binsz=0.1, width=10.0)
        >>> geom = HpxGeom.create(nside=64, width=10.0, axes=[axis])
        >>> geom = HpxGeom.create(nside=[32,64], width=10.0, axes=[axis])

        """

        if nside is None and binsz is None:
            raise ValueError('Either nside or binsz must be defined.')

        if nside is None and binsz is not None:
            nside = get_nside_from_pixel_size(binsz)

        if skydir is None:
            lonlat = (0.0, 0.0)
        elif isinstance(skydir, tuple):
            lonlat = skydir
        elif isinstance(skydir, SkyCoord):
            lonlat = skydir_to_lonlat(skydir, coordsys=coordsys)
        else:
            raise ValueError(
                'Invalid type for skydir: {}'.format(type(skydir)))

        if region is None and width is not None:
            region = 'DISK({:f},{:f},{:f})'.format(lonlat[0], lonlat[1],
                                                   width / 2.)

        return cls(nside, nest=nest, coordsys=coordsys, region=region,
                   axes=axes, conv=conv)

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
    def from_header(cls, header, hdu_bands=None, pix=None):
        """Create an HPX object from a FITS header.

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            The FITS header
        hdu_bands : `~astropy.fits.BinTableHDU` 
            The BANDS table HDU.
        pix : tuple
            List of pixel index vectors defining the pixels
            encompassed by the geometry.  For EXPLICIT geometries with
            HPX_REG undefined this tuple defines the geometry.

        Returns
        -------
        hpx : `~HpxGeom`
            HEALPix geometry.
        """
        convname = HpxGeom.identify_HPX_convention(header)
        conv = HPX_FITS_CONVENTIONS[convname]

        axes = find_and_read_bands(hdu_bands)
        shape = [ax.nbin for ax in axes]

        if header['PIXTYPE'] != 'HEALPIX':
            raise Exception('PIXTYPE != HEALPIX')
        if header['ORDERING'] == 'RING':
            nest = False
        elif header['ORDERING'] == 'NESTED':
            nest = True
        else:
            raise Exception('ORDERING != RING | NESTED')

        if hdu_bands is not None and 'NSIDE' in hdu_bands.columns.names:
            nside = hdu_bands.data.field('NSIDE').reshape(shape)
        elif 'NSIDE' in header:
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

        return cls(nside, nest, coordsys=coordsys, region=region,
                   axes=axes, conv=conv)

    @classmethod
    def from_hdu(cls, hdu, hdu_bands=None):
        """Create an HPX object from a BinTable HDU.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            The FITS HDU
        hdu_bands : `~astropy.io.fits.BinTableHDU`
            The BANDS table HDU

        Returns
        -------
        hpx : `~HpxGeom`
            HEALPix geometry.
        """
        # FIXME: Need correct handling of IMPLICIT and EXPLICIT maps

        # if HPX region is not defined then geometry is defined by
        # the set of all pixels in the table
        if not 'HPX_REG' in hdu.header:
            pix = (hdu.data.field('PIX'), hdu.data.field('CHANNEL'))
        else:
            pix = None

        return cls.from_header(hdu.header, hdu_bands=hdu_bands, pix=pix)

    def make_header(self, **kwargs):
        """"Build and return FITS header for this HEALPIX map."""

        header = fits.Header()

        conv = kwargs.get('conv', HPX_FITS_CONVENTIONS['GADF'])

        # FIXME: For some sparse maps we may want to allow EXPLICIT
        # with an empty region string
        indxschm = kwargs.get('indxschm',
                              'EXPLICIT' if self._region else 'IMPLICIT')

        if indxschm is None:
            if self._region is None:
                indxschm = 'IMPLICIT'
            elif self.nside.size == 1:
                indxschm = 'EXPLICIT'
            else:
                indxschm = 'LOCAL'

        # FIXME: Set TELESCOP and INSTRUME from convention type

        header["TELESCOP"] = "GLAST"
        header["INSTRUME"] = "LAT"
        header[conv.coordsys] = self.coordsys
        header["PIXTYPE"] = "HEALPIX"
        header["ORDERING"] = self.ordering
        header["INDXSCHM"] = indxschm
        header["ORDER"] = np.max(self._order)
        header["NSIDE"] = np.max(self._nside)
        header["FIRSTPIX"] = 0
        header["LASTPIX"] = np.max(self._maxpix) - 1
        header["HPX_CONV"] = conv.convname

        self._fill_header_from_axes(header)

        if self.coordsys == 'CEL':
            header['EQUINOX'] = (2000.0,
                                 'Equinox of RA & DEC specifications')

        if self.region:
            header['HPX_REG'] = self._region

        return header

    def make_bands_hdu(self, extname='BANDS'):

        header = self.make_header()
        cols = make_axes_cols(self.axes)
        if self.nside.size > 1:
            cols += [fits.Column('NSIDE', 'I', array=np.ravel(self.nside)), ]
        hdu = fits.BinTableHDU.from_columns(cols, header, name=extname)
        return hdu

    def make_ebounds_hdu(self, extname='EBOUNDS'):
        """Make a FITS HDU with the energy bin boundaries.

        Parameters
        ----------
        extname : str
            The HDU extension name
        """
        # TODO: Assert if the number of axes is wrong?

        emin = self._axes[0].edges[:-1]
        emax = self._axes[0].edges[1:]

        cols = [fits.Column('CHANNEL', 'I', array=np.arange(1, self._axes[0].nbin)),
                fits.Column('E_MIN', '1E', unit='keV', array=1000 * emin),
                fits.Column('E_MAX', '1E', unit='keV', array=1000 * emax)]
        hdu = fits.BinTableHDU.from_columns(
            cols, self.make_header(), name=extname)
        return hdu

    def make_energies_hdu(self, extname='ENERGIES'):
        """Make a FITS HDU with the energy bin centers.

        Parameters
        ----------
        extname : str
            The HDU extension name
        """
        ectr = np.sqrt(self._axes[0][1:] * self._axes[0][:-1])
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

        if self.conv.bands_hdu == 'EBOUNDS':
            hdu_energy = self.make_ebounds_hdu()
        elif self.conv.bands_hdu == 'ENERGIES':
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

        Returns
        -------
        ilist : `~numpy.ndarray`
            List of pixel indices.
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
                raise ValueError(
                    'Did not recognize ordering scheme: {}'.format(tokens[1]))
            ilist = match_hpx_pixel(nside, nest, nside_pix, ipix_ring)
        else:
            raise ValueError(
                'Did not recognize region type: {}'.format(tokens[0]))

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
        frame = coordsys_to_frame(coordsys)

        if region is None or isinstance(region, tuple):
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
                raise ValueError(
                    'Did not recognize ordering scheme: {}'.format(tokens[1]))
            theta, phi = hp.pix2ang(nside_pix, ipix_pix, nest_pix)
            lat = np.degrees((np.pi / 2) - theta)
            lon = np.degrees(phi)
            return SkyCoord(lon, lat, frame=frame, unit='deg')
        else:
            raise ValueError(
                'HPX.get_ref_dir did not recognize region type: {}'.format(tokens[0]))

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
            raise Exception(
                'Did not recognize region type: {}'.format(tokens[0]))

    def make_wcs(self, proj='AIT', oversample=2, drop_axes=True):
        """Make a WCS projection appropriate for this HPX pixelization.

        Parameters
        ----------
        drop_axes : bool
            Drop non-spatial axes from the
            HEALPIX geometry.  If False then all dimensions of the
            HEALPIX geometry will be copied to the WCS geometry.
        proj : str
            Projection type of WCS geometry.
        oversample : int
            Factor by which the WCS pixel size will be chosen to
            oversample the HEALPIX map.

        Returns
        -------
        wcs : `~gammapy.maps.wcs.WcsGeom`
            WCS geometry
        """

        skydir = self.get_ref_dir(self._region, self.coordsys)
        binsz = np.min(get_pixel_size_from_nside(self.nside)) / oversample
        width = (2.0 * self.get_region_size(self._region) +
                 np.max(get_pixel_size_from_nside(self.nside)))

        if width > 90.:
            width = (min(360., width), min(180.0, width))

        if drop_axes:
            axes = None
        else:
            axes = copy.deepcopy(self.axes)

        geom = WcsGeom.create(width=width, binsz=binsz, coordsys=self.coordsys,
                              axes=axes, skydir=skydir, proj=proj)

        return geom

    def get_pixels(self, idx=None, local=False):

        if idx is not None and np.any(np.array(idx) >= np.array(self._shape)):
            raise ValueError('Image index out of range: {}'.format(idx))

        if self._ipix is None and idx is None:
            ipix = np.concatenate([np.arange(t) for t in self._maxpix.flat])
            if not len(self._shape):
                return ipix,

            ibnd = np.concatenate([i * np.ones(t, dtype=int)
                                   for i, t in enumerate(self._maxpix.flat)])
            ibnd = list(np.unravel_index(ibnd, self._shape))
            pix = tuple([ipix] + ibnd)
        elif self._ipix is None:
            ipix = np.arange(self._maxpix[idx])
            ibnd = [t * np.ones(len(ipix), dtype=int) for t in idx]
            pix = tuple([ipix] + ibnd)
        elif self.nside.shape == self._maxpix.shape or self.region == 'explicit':

            if idx is not None:
                npix_sum = np.concatenate(([0], np.cumsum(self._npix)))
                idx_ravel = np.ravel_multi_index(idx, self._shape)
                s = slice(npix_sum[idx_ravel], npix_sum[idx_ravel + 1])
            else:
                s = slice(None)

            pix = unravel_hpx_index(self._ipix[s], self._maxpix)
        else:

            # For fixed nside we only store an ipix vector for the
            # first plane. Here we construct global pixel index vectors
            # for all planes
            if idx is None:

                nimg = np.prod(self._shape)
                npix = self._npix.flat[0]
                ibnd = np.ravel(
                    np.arange(nimg)[:, None] * np.ones(npix, dtype=int)[None, :])
                ibnd = np.unravel_index(ibnd, self._shape, order='F')
                ipix = np.ravel(self._ipix[None, :]
                                * np.ones(nimg, dtype=int)[:, None])
                pix = tuple([ipix] + list(ibnd))
            else:

                npix = self._npix.flat[0]
                ibnd = [t * np.ones(npix, dtype=int) for t in idx]
                ipix = self._ipix.copy()
                pix = tuple([ipix] + list(ibnd))

        if local:
            return self.global_to_local(pix)
        else:
            return pix

    def get_coords(self, idx=None):

        pix = self.get_pixels(idx=idx)
        return self.pix_to_coord(pix)

    def contains(self, coords):

        idx = self.coord_to_idx(coords)
        return np.all(np.stack([t != -1 for t in idx]), axis=0)

    def get_skydirs(self):
        """Get the sky coordinates of all the pixels in this geometry. """
        coords = self.get_coords()
        frame = 'galactic' if self.coordsys == 'GAL' else 'icrs'
        return SkyCoord(coords[0], coords[1], unit='deg', frame=frame)

    def skydir_to_pixel(self, skydir):
        """Return the pixel index of a SkyCoord object."""

        # FIXME: What should this method do for maps with non-spatial
        # dimensions

        if self.coordsys in ['CEL', 'EQU']:
            skydir = skydir.transform_to('icrs')
            lon = skydir.ra.deg
            lat = skydir.dec.deg
        else:
            skydir = skydir.transform_to('galactic')
            lon = skydir.l.deg
            lat = skydir.b.deg

        return self.pix_to_coord((lat, lon))


class HpxToWcsMapping(object):
    """Stores the indices need to convert from HEALPIX to WCS.

    Parameters
    ----------
    hpx : `~HpxGeom`
        HEALPix geometry object.
    wcs : `~gammapy.maps.wcs.WcsGeom`
        WCS geometry object.
    """

    def __init__(self, hpx, wcs, ipix, mult_val, npix):
        self._hpx = hpx
        self._wcs = wcs
        self._ipix = ipix
        self._mult_val = mult_val
        self._npix = npix
        self._lmap = self._hpx[self._ipix]
        self._valid = self._lmap >= 0

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

        hdulist = fits.HDUList([prim_hdu, mult_hdu])
        hdulist.writeto(fitsfile, clobber=clobber)

    @classmethod
    def create(cls, hpx, wcs):
        ipix, mult_val, npix = make_hpx_to_wcs_mapping(hpx, wcs)
        return cls(hpx, wcs, ipix, mult_val, npix)

    @classmethod
    def read(cls, filename):
        """Read a FITS file and use it to make a mapping."""
        from fermipy.skymap import Map
        index_map = Map.read(filename)
        mult_map = Map.read(filename, hdu=1)

        with fits.open(filename) as ff:
            hpx = HpxGeom.from_header(ff[0])
            ipix = index_map.counts
            mult_val = mult_map.counts
            npix = mult_map.counts.shape
        return cls(hpx, index_map.wcs, ipix, mult_val, npix)

    def fill_wcs_map_from_hpx_data(self, hpx_data, wcs_data, normalize=True):
        """Fill the WCS map from the hpx data using the pre-calculated mappings.

        Parameters
        ----------
        hpx_data : `~numpy.ndarray`
            The input HEALPIX data
        wcs_data : `~numpy.ndarray`
            The data array being filled
        normalize : bool
            True -> preserve integral by splitting HEALPIX values between bins
        """

        # FIXME: Do we want to flatten mapping arrays?

        # HPX images have (1,N) dimensionality by convention
        # hpx_data = np.squeeze(hpx_data)

        if self._valid.ndim == 1:
            shape = tuple([t.flat[0] for t in self._npix])
        else:
            shape = hpx_data.shape[:-1] + \
                tuple([t.flat[0] for t in self._npix])
        valid = np.where(self._valid.reshape(shape))
        lmap = self._lmap[self._valid]
        mult_val = self._mult_val[self._valid]

        wcs_slice = [slice(None) for i in range(wcs_data.ndim - 2)]
        wcs_slice = tuple(wcs_slice + list(valid)[::-1][:2])

        hpx_slice = [slice(None) for i in range(wcs_data.ndim - 2)]
        hpx_slice = tuple(hpx_slice + [lmap])

        if normalize:
            wcs_data[wcs_slice] = mult_val * hpx_data[hpx_slice]
        else:
            wcs_data[wcs_slice] = hpx_data[hpx_slice]

        return wcs_data

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
