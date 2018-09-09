# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for dealing with HEALPix projections and mappings."""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import re
import copy
import numpy as np
from ..extern import six
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from ..utils.scripts import make_path
from .wcs import WcsGeom
from .geom import MapGeom, MapCoord, pix_tuple_to_idx
from .geom import coordsys_to_frame, skycoord_to_lonlat
from .geom import find_and_read_bands, make_axes

# Not sure if we should expose this in the docs or not:
# HPX_FITS_CONVENTIONS, HpxConv
__all__ = ["HpxGeom"]

# Approximation of the size of HEALPIX pixels (in degrees) for a particular order.
# Used to convert from HEALPIX to WCS-based projections.
HPX_ORDER_TO_PIXSIZE = np.array(
    [32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.50, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.002]
)


class HpxConv(object):
    """Data structure to define how a HEALPIX map is stored to FITS."""

    def __init__(self, convname, **kwargs):
        self.convname = convname
        self.colstring = kwargs.get("colstring", "CHANNEL")
        self.firstcol = kwargs.get("firstcol", 1)
        self.hduname = kwargs.get("hduname", "SKYMAP")
        self.bands_hdu = kwargs.get("bands_hdu", "EBOUNDS")
        self.quantity_type = kwargs.get("quantity_type", "integral")
        self.coordsys = kwargs.get("coordsys", "COORDSYS")

    def colname(self, indx):
        return "{}{}".format(self.colstring, indx)

    @classmethod
    def create(cls, convname="gadf"):
        return copy.deepcopy(HPX_FITS_CONVENTIONS[convname])


HPX_FITS_CONVENTIONS = {}
"""Various conventions for storing HEALPIX maps in FITS files"""
HPX_FITS_CONVENTIONS[None] = HpxConv("gadf", bands_hdu="BANDS")
HPX_FITS_CONVENTIONS["gadf"] = HpxConv("gadf", bands_hdu="BANDS")
HPX_FITS_CONVENTIONS["fgst-ccube"] = HpxConv("fgst-ccube")
HPX_FITS_CONVENTIONS["fgst-ltcube"] = HpxConv(
    "fgst-ltcube", colstring="COSBINS", hduname="EXPOSURE", bands_hdu="CTHETABOUNDS"
)
HPX_FITS_CONVENTIONS["fgst-bexpcube"] = HpxConv(
    "fgst-bexpcube", colstring="ENERGY", hduname="HPXEXPOSURES", bands_hdu="ENERGIES"
)
HPX_FITS_CONVENTIONS["fgst-srcmap"] = HpxConv(
    "fgst-srcmap", hduname=None, quantity_type="differential"
)
HPX_FITS_CONVENTIONS["fgst-template"] = HpxConv(
    "fgst-template", colstring="ENERGY", bands_hdu="ENERGIES"
)
HPX_FITS_CONVENTIONS["fgst-srcmap-sparse"] = HpxConv(
    "fgst-srcmap-sparse", colstring=None, hduname=None, quantity_type="differential"
)
HPX_FITS_CONVENTIONS["galprop"] = HpxConv(
    "galprop",
    colstring="Bin",
    hduname="SKYMAP2",
    bands_hdu="ENERGIES",
    quantity_type="differential",
    coordsys="COORDTYPE",
)
HPX_FITS_CONVENTIONS["galprop2"] = HpxConv(
    "galprop",
    colstring="Bin",
    hduname="SKYMAP2",
    bands_hdu="ENERGIES",
    quantity_type="differential",
)


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

    dpix = np.zeros(npix.size, dtype="i")
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
    idx1 = np.ravel_multi_index(idx[1:], npix.shape, mode="clip")
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

    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = cos_t

    # Stack them into the output array
    out = np.vstack((x, y, z)).swapaxes(0, 1)
    return out


def get_nside_from_pix_size(pixsz):
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


def get_pix_size_from_nside(nside):
    """Estimate of the pixel size from the HEALPIX nside coordinate.

    This just uses a lookup table to provide a nice round number
    for each HEALPIX order.
    """
    order = nside_to_order(nside)
    if np.any(order < 0) or np.any(order > 13):
        raise ValueError("HEALPIX order must be 0 to 13. Got: {!r}".format(order))

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
    hpx : `~gammapy.maps.HpxGeom`
        The HEALPIX geometry.
    wcs : `~gammapy.maps.WcsGeom`
        The WCS geometry.

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
    hpx : `~gammapy.maps.HpxGeom`
       The HEALPIX geometry
    wcs : `~gammapy.maps.WcsGeom`
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

    ipix[m] = hp.ang2pix(
        hpx.nside[..., None],
        sky_crds[:, 1][mask][None, ...],
        sky_crds[:, 0][mask][None, ...],
        hpx.nest,
    ).flatten()

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


def match_hpx_pix(nside, nest, nside_pix, ipix_ring):
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
    m = re.match(r"([A-Za-z\_]*?)\((.*?)\)", region)

    if m is None:
        raise ValueError("Failed to parse hpx region string: {!r}".format(region))

    if not m.group(1):
        return re.split(",", m.group(2))
    else:
        return [m.group(1)] + re.split(",", m.group(2))


def get_hpxregion_dir(region, coordsys):
    """Get the reference direction for a HEALPIX region string.

    Parameters
    ----------
    region : str
        A string describing a HEALPIX region
    coordsys : {'CEL', 'GAL'}
        Coordinate system
    """
    import healpy as hp

    frame = coordsys_to_frame(coordsys)

    if region is None:
        return SkyCoord(0., 0., frame=frame, unit="deg")

    tokens = parse_hpxregion(region)
    if tokens[0] in ["DISK", "DISK_INC"]:
        lon, lat = float(tokens[1]), float(tokens[2])
        return SkyCoord(lon, lat, frame=frame, unit="deg")
    elif tokens[0] == "HPX_PIXEL":
        nside_pix = int(tokens[2])
        ipix_pix = int(tokens[3])
        if tokens[1] == "NESTED":
            nest_pix = True
        elif tokens[1] == "RING":
            nest_pix = False
        else:
            raise ValueError("Invalid ordering scheme: {!r}".format(tokens[1]))
        theta, phi = hp.pix2ang(nside_pix, ipix_pix, nest_pix)
        lat = np.degrees((np.pi / 2) - theta)
        lon = np.degrees(phi)
        return SkyCoord(lon, lat, frame=frame, unit="deg")
    else:
        raise ValueError("Invalid region type: {!r}".format(tokens[0]))


def get_hpxregion_size(region):
    """Get the approximate size of region (in degrees) from a HEALPIX region string.
    """
    tokens = parse_hpxregion(region)
    if tokens[0] in {"DISK", "DISK_INC"}:
        return float(tokens[3])
    elif tokens[0] == "HPX_PIXEL":
        pix_size = get_pix_size_from_nside(int(tokens[2]))
        return 2. * pix_size
    else:
        raise ValueError("Invalid region type: {!r}".format(tokens[0]))


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


def get_superpixels(idx, nside_subpix, nside_superpix, nest=True):
    """Compute the indices of superpixels that contain a subpixel.

    Parameters
    ----------
    idx : `~numpy.ndarray`
        Array of HEALPix pixel indices for subpixels of NSIDE
        ``nside_subpix``.
    nside_subpix  : int or `~numpy.ndarray`
        NSIDE of subpixel.
    nside_superpix : int or `~numpy.ndarray`
        NSIDE of superpixel.
    nest : bool
        If True, assume NESTED pixel ordering, otherwise, RING pixel
        ordering.

    Returns
    -------
    idx_super : `~numpy.ndarray`
        Indices of HEALpix pixels of nside ``nside_superpix`` that
        contain pixel indices ``idx`` of nside ``nside_subpix``.
    """

    import healpy as hp

    idx = np.array(idx)
    nside_superpix = np.asarray(nside_superpix)
    nside_subpix = np.asarray(nside_subpix)

    if not nest:
        idx = hp.ring2nest(nside_subpix, idx)

    if np.any(~is_power2(nside_superpix)) or np.any(~is_power2(nside_subpix)):
        raise ValueError("NSIDE must be a power of 2.")

    ratio = np.array((nside_subpix // nside_superpix) ** 2, ndmin=1)
    idx //= ratio

    if not nest:
        m = idx == -1
        idx[m] = 0
        idx = hp.nest2ring(nside_superpix, idx)
        idx[m] = -1

    return idx


def get_subpixels(idx, nside_superpix, nside_subpix, nest=True):
    """Compute the indices of subpixels contained within superpixels.
    This function returns an output array with one additional
    dimension of size N for subpixel indices where N is the maximum
    number of subpixels for any pair of ``nside_superpix`` and
    ``nside_subpix``.  If the number of subpixels is less than N the
    remaining subpixel indices will be set to -1.

    Parameters
    ----------
    idx : `~numpy.ndarray`
        Array of HEALPix pixel indices for superpixels of NSIDE
        ``nside_superpix``.
    nside_superpix : int or `~numpy.ndarray`
        NSIDE of superpixel.
    nside_subpix  : int or `~numpy.ndarray`
        NSIDE of subpixel.
    nest : bool
        If True, assume NESTED pixel ordering, otherwise, RING pixel
        ordering.

    Returns
    -------
    idx_sub : `~numpy.ndarray`
        Indices of HEALpix pixels of nside ``nside_subpix`` contained
        within pixel indices ``idx`` of nside ``nside_superpix``.
    """
    import healpy as hp

    if not nest:
        idx = hp.ring2nest(nside_superpix, idx)

    idx = np.asarray(idx)
    nside_superpix = np.asarray(nside_superpix)
    nside_subpix = np.asarray(nside_subpix)

    if np.any(~is_power2(nside_superpix)) or np.any(~is_power2(nside_subpix)):
        raise ValueError("NSIDE must be a power of 2.")

    # number of subpixels in each superpixel
    npix = np.array((nside_subpix // nside_superpix) ** 2, ndmin=1)
    x = np.arange(np.max(npix), dtype=int)
    idx = idx * npix

    if not np.all(npix[0] == npix):
        x = np.broadcast_to(x, idx.shape + x.shape)
        idx = idx[..., None] + x
        idx[x >= np.broadcast_to(npix[..., None], x.shape)] = -1
    else:
        idx = idx[..., None] + x

    if not nest:
        m = idx == -1
        idx[m] = 0
        idx = hp.nest2ring(nside_subpix[..., None], idx)
        idx[m] = -1

    return idx


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

    is_hpx = True

    def __init__(
        self,
        nside,
        nest=True,
        coordsys="CEL",
        region=None,
        axes=None,
        conv="gadf",
        sparse=False,
    ):

        # FIXME: Figure out what to do when sparse=True
        # FIXME: Require NSIDE to be power of two when nest=True

        self._nside = np.array(nside, ndmin=1)
        self._axes = make_axes(axes, conv)
        self._shape = tuple([ax.nbin for ax in self._axes])
        if self.nside.size > 1 and self.nside.shape != self._shape:
            raise ValueError(
                "Wrong dimensionality for nside. nside must "
                "be a scalar or have a dimensionality consistent "
                "with the axes argument."
            )

        self._order = nside_to_order(self._nside)
        self._nest = nest
        self._coordsys = coordsys
        self._maxpix = 12 * self._nside * self._nside
        self._maxpix = self._maxpix * np.ones(self._shape, dtype=int)
        self._sparse = sparse

        self._ipix = None
        self._rmap = None
        self._region = region
        self._create_lookup(region)

        if self._ipix is not None:
            self._rmap = {}
            for i, ipix in enumerate(self._ipix.flat):
                self._rmap[ipix] = i

        self._npix = self._npix * np.ones(self._shape, dtype=int)
        self._conv = conv
        self._center_skydir = self._get_ref_dir()
        lon, lat, frame = skycoord_to_lonlat(self._center_skydir)
        self._center_coord = tuple(
            [lon, lat]
            + [ax.pix_to_coord((float(ax.nbin) - 1.0) / 2.) for ax in self.axes]
        )
        self._center_pix = self.coord_to_pix(self._center_coord)

    @property
    def data_shape(self):
        """Shape of the Numpy data array matching this geometry."""
        npix_shape = [np.max(self.npix)]
        ax_shape = [ax.nbin for ax in self.axes]
        return tuple(npix_shape + ax_shape)[::-1]

    def _create_lookup(self, region):
        """Create local-to-global pixel lookup table."""

        if isinstance(region, six.string_types):
            ipix = [
                self.get_index_list(nside, self._nest, region)
                for nside in self._nside.flat
            ]
            self._ibnd = np.concatenate(
                [i * np.ones_like(p, dtype="int16") for i, p in enumerate(ipix)]
            )
            self._ipix = [
                ravel_hpx_index((p, i * np.ones_like(p)), np.ravel(self._maxpix))
                for i, p in enumerate(ipix)
            ]
            self._region = region
            self._indxschm = "EXPLICIT"
            self._npix = np.array([len(t) for t in self._ipix])
            if self.nside.ndim > 1:
                self._npix = self._npix.reshape(self.nside.shape)
            self._ipix = np.concatenate(self._ipix)

        elif isinstance(region, tuple):

            region = [np.asarray(t) for t in region]
            m = np.any(np.stack([t >= 0 for t in region]), axis=0)
            region = [t[m] for t in region]

            self._ipix = ravel_hpx_index(region, self._maxpix)
            self._ipix = np.unique(self._ipix)
            region = unravel_hpx_index(self._ipix, self._maxpix)
            self._region = "explicit"
            self._indxschm = "EXPLICIT"
            if len(region) == 1:
                self._npix = np.array([len(region[0])])
            else:
                self._npix = np.zeros(self._shape, dtype=int)
                idx = np.ravel_multi_index(region[1:], self._shape)
                cnt = np.unique(idx, return_counts=True)
                self._npix.flat[cnt[0]] = cnt[1]

        elif region is None:
            self._region = None
            self._indxschm = "IMPLICIT"
            self._npix = self._maxpix

        else:
            raise ValueError("Invalid region string: {!r}".format(region))

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
            idx_tmp = tuple(
                [idx_local[0]] + [np.zeros(t.shape, dtype=int) for t in idx_local[1:]]
            )
            idx = ravel_hpx_index(idx_tmp, self._npix)

        idx_global = unravel_hpx_index(self._ipix[idx], self._maxpix)
        return idx_global[:1] + tuple(idx_local[1:])

    def global_to_local(self, idx_global, ravel=False):
        """Compute global (all-sky) index from a local (partial-sky) index.

        Parameters
        ----------
        idx_global : tuple
            A tuple of pixel indices with global HEALPix pixel indices.
        ravel : bool
            Return a raveled index.

        Returns
        -------
        idx_local : tuple
            A tuple of pixel indices with local HEALPIX pixel indices.
        """
        if self.nside.size == 1:
            idx = np.array(idx_global[0], ndmin=1)
        else:
            idx = ravel_hpx_index(idx_global, self._maxpix)

        if self._rmap is not None:
            retval = np.full(idx.size, -1, "i")
            m = np.in1d(idx.flat, self._ipix)
            retval[m] = np.searchsorted(self._ipix, idx.flat[m])
            retval = retval.reshape(idx.shape)
        else:
            retval = idx

        if self.nside.size == 1:
            idx_local = tuple([retval] + list(idx_global[1:]))
        else:
            idx_local = unravel_hpx_index(retval, self._npix)

        m = np.any(np.stack([t == -1 for t in idx_local]), axis=0)
        for i, t in enumerate(idx_local):
            idx_local[i][m] = -1

        if not ravel:
            return idx_local
        else:
            return ravel_hpx_index(idx_local, self.npix)

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
        idx_global : `~numpy.ndarray`
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
        if (
            isinstance(idx_global, int)
            or (isinstance(idx_global, tuple) and isinstance(idx_global[0], int))
            or isinstance(idx_global, np.ndarray)
        ):
            idx_global = unravel_hpx_index(np.array(idx_global, ndmin=1), self._maxpix)

        return self.global_to_local(idx_global, ravel=True)

    def coord_to_pix(self, coords):
        import healpy as hp

        coords = MapCoord.create(coords, coordsys=self.coordsys)
        theta, phi = coords.theta, coords.phi

        c = self.coord_to_tuple(coords)

        if self.axes:
            bins = []
            idxs = []
            for i, ax in enumerate(self.axes):
                bins += [ax.coord_to_pix(c[i + 2])]
                idxs += [ax.coord_to_idx(c[i + 2])]

            # FIXME: Figure out how to handle coordinates out of
            # bounds of non-spatial dimensions
            if self.nside.size > 1:
                nside = self.nside[tuple(idxs)]
            else:
                nside = self.nside

            m = ~np.isfinite(theta)
            theta[m] = 0.0
            phi[m] = 0.0
            pix = hp.ang2pix(nside, theta, phi, nest=self.nest)
            pix = tuple([pix] + bins)
            if np.any(m):
                for p in pix:
                    p[m] = -1
        else:
            pix = (hp.ang2pix(self.nside, theta, phi, nest=self.nest),)

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
            m = ipix == -1
            ipix[m] = 0
            theta, phi = hp.pix2ang(nside, ipix, nest=self.nest)
            coords = [np.degrees(phi), np.degrees(np.pi / 2. - theta)]
            coords = tuple(coords + vals)
            if np.any(m):
                for c in coords:
                    c[m] = np.nan
        else:
            ipix = np.round(pix[0]).astype(int)
            theta, phi = hp.pix2ang(self.nside, ipix, nest=self.nest)
            coords = (np.degrees(phi), np.degrees(np.pi / 2. - theta))

        return coords

    def pix_to_idx(self, pix, clip=False):
        # FIXME: Look for better method to clip HPX indices
        # TODO: copy idx to avoid modifying input pix?
        # pix_tuple_to_idx seems to always make a copy!?
        idx = pix_tuple_to_idx(pix)
        idx_local = self.global_to_local(idx)
        for i, _ in enumerate(idx):

            if clip:
                if i > 0:
                    np.clip(idx[i], 0, self.axes[i - 1].nbin - 1, out=idx[i])
                else:
                    np.clip(idx[i], 0, None, out=idx[i])
            else:
                if i > 0:
                    mask = (idx[i] < 0) | (idx[i] >= self.axes[i - 1].nbin)
                    np.putmask(idx[i], mask, -1)
                else:
                    mask = (idx_local[i] < 0) | (idx[i] < 0)
                    np.putmask(idx[i], mask, -1)

        return tuple(idx)

    def to_slice(self, slices, drop_axes=True):
        if len(slices) == 0 and self.ndim == 2:
            return copy.deepcopy(self)

        if len(slices) != self.ndim - 2:
            raise ValueError()

        nside = np.ones(self.shape, dtype=int) * self.nside
        nside = np.squeeze(nside[slices])

        axes = [ax.slice(s) for ax, s in zip(self.axes, slices)]
        if drop_axes:
            axes = [ax for ax in axes if ax.nbin > 1]
            slice_dims = [0] + [i + 1 for i, ax in enumerate(axes) if ax.nbin > 1]
        else:
            slice_dims = np.arange(self.ndim)

        if self.region == "explicit":
            idx = self.get_idx()
            slices = (slice(None),) + slices
            idx = [p[slices[::-1]] for p in idx]
            idx = [p[p != -1] for p in idx]
            if drop_axes:
                idx = [idx[i] for i in range(len(idx)) if i in slice_dims]
            region = tuple(idx)
        else:
            region = self.region

        return self.__class__(nside, self.nest, self.coordsys, region, axes, self.conv)

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
        """Number of dimensions (int)."""
        return len(self._axes) + 2

    @property
    def ordering(self):
        """HEALPix ordering ('NESTED' or 'RING')."""
        return "NESTED" if self.nest else "RING"

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
        """Is HEALPix order nested? (bool)."""
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
        """Name of default FITS convention associated with this geometry."""
        return self._conv

    @property
    def hpx_conv(self):
        return HPX_FITS_CONVENTIONS[self.conv]

    @property
    def coordsys(self):
        return self._coordsys

    @property
    def projection(self):
        """Map projection."""
        return "HPX"

    @property
    def region(self):
        """Region string."""
        return self._region

    @property
    def is_allsky(self):
        """Flag for all-sky maps."""
        if self._region is None:
            return True
        else:
            return False

    @property
    def is_regular(self):
        """Flag identifying whether this geometry is regular in non-spatial dimensions.

        False for multi-resolution or irregular geometries.
        If true all image planes have the same pixel geometry.
        """
        if self.nside.size > 1 or self.region == "explicit":
            return False
        else:
            return True

    @property
    def center_coord(self):
        """Map coordinates of the center of the geometry (tuple)."""
        return self._center_coord

    @property
    def center_pix(self):
        """Pixel coordinates of the center of the geometry (tuple)."""
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
        return self.get_idx()

    def to_ud_graded(self, order):
        """Upgrade or downgrade the resolution of this geometry to the given
        order.  This method does not preserve the geometry footprint.

        Returns
        -------
        geom : `~HpxGeom`
            A HEALPix geometry object.
        """
        if np.any(self.order < 0):
            raise ValueError("Upgrade and degrade only implemented for standard maps")

        axes = copy.deepcopy(self.axes)
        return self.__class__(
            2 ** order,
            self.nest,
            coordsys=self.coordsys,
            region=self.region,
            axes=axes,
            conv=self.conv,
        )

    def to_swapped(self):
        """Make a copy of this geometry with a swapped ORDERING.  (NEST->RING
        or vice versa)

        Returns
        -------
        geom : `~HpxGeom`
            A HEALPix geometry object.
        """
        axes = copy.deepcopy(self.axes)
        return self.__class__(
            self.nside,
            not self.nest,
            coordsys=self.coordsys,
            region=self.region,
            axes=axes,
            conv=self.conv,
        )

    def to_image(self):
        return self.__class__(
            np.max(self.nside),
            self.nest,
            coordsys=self.coordsys,
            region=self.region,
            conv=self.conv,
        )

    def to_cube(self, axes):
        axes = copy.deepcopy(self.axes) + axes
        return self.__class__(
            np.max(self.nside),
            self.nest,
            coordsys=self.coordsys,
            region=self.region,
            conv=self.conv,
            axes=axes,
        )

    def _get_neighbors(self, idx):
        import healpy as hp

        nside = self._get_nside(idx)
        idx_nb = (hp.get_all_neighbours(nside, idx[0], nest=self.nest),)
        idx_nb += tuple([t[None, ...] * np.ones_like(idx_nb[0]) for t in idx[1:]])

        return idx_nb

    def pad(self, pad_width):
        if self.is_allsky:
            raise ValueError("Cannot pad an all-sky map.")

        idx = self.get_idx(flat=True)
        idx_r = ravel_hpx_index(idx, self._maxpix)

        # TODO: Pre-filter indices to find those close to the edge
        idx_nb = self._get_neighbors(idx)
        idx_nb = ravel_hpx_index(idx_nb, self._maxpix)

        for _ in range(pad_width):
            # Mask of neighbors that are not contained in the geometry
            # TODO: change this to numpy.isin when we require Numpy 1.13+
            # Here and everywhere in Gamampy -> search for "isin"
            # see https://github.com/gammapy/gammapy/pull/1371
            mask_edge = np.in1d(idx_nb, idx_r, invert=True).reshape(idx_nb.shape)
            idx_edge = idx_nb[mask_edge]
            idx_edge = np.unique(idx_edge)
            idx_r = np.sort(np.concatenate((idx_r, idx_edge)))
            idx_nb = unravel_hpx_index(idx_edge, self._maxpix)
            idx_nb = self._get_neighbors(idx_nb)
            idx_nb = ravel_hpx_index(idx_nb, self._maxpix)

        idx = unravel_hpx_index(idx_r, self._maxpix)
        return self.__class__(
            self.nside.copy(),
            self.nest,
            coordsys=self.coordsys,
            region=idx,
            conv=self.conv,
            axes=copy.deepcopy(self.axes),
        )

    def crop(self, crop_width):
        if self.is_allsky:
            raise ValueError("Cannot crop an all-sky map.")

        idx = self.get_idx(flat=True)
        idx_r = ravel_hpx_index(idx, self._maxpix)

        # TODO: Pre-filter indices to find those close to the edge
        idx_nb = self._get_neighbors(idx)
        idx_nb = ravel_hpx_index(idx_nb, self._maxpix)

        for _ in range(crop_width):
            # Mask of pixels that have at least one neighbor not
            # contained in the geometry
            mask_edge = np.in1d(idx_nb, idx_r, invert=True)
            mask_edge = np.any(mask_edge, axis=0)
            idx_r = idx_r[~mask_edge]
            idx_nb = idx_nb[:, ~mask_edge]

        idx = unravel_hpx_index(idx_r, self._maxpix)
        return self.__class__(
            self.nside.copy(),
            self.nest,
            coordsys=self.coordsys,
            region=idx,
            conv=self.conv,
            axes=copy.deepcopy(self.axes),
        )

    def upsample(self, factor):
        if not is_power2(factor):
            raise ValueError("Upsample factor must be a power of 2.")

        if self.is_allsky:
            return self.__class__(
                self.nside * factor,
                self.nest,
                coordsys=self.coordsys,
                region=self.region,
                conv=self.conv,
                axes=copy.deepcopy(self.axes),
            )

        idx = list(self.get_idx(flat=True))
        nside = self._get_nside(idx)

        idx_new = get_subpixels(idx[0], nside, nside * factor, nest=self.nest)
        for i in range(1, len(idx)):
            idx[i] = idx[i][..., None] * np.ones(idx_new.shape, dtype=int)

        idx[0] = idx_new
        return self.__class__(
            self.nside * factor,
            self.nest,
            coordsys=self.coordsys,
            region=tuple(idx),
            conv=self.conv,
            axes=copy.deepcopy(self.axes),
        )

    def downsample(self, factor):
        if not is_power2(factor):
            raise ValueError("Downsample factor must be a power of 2.")

        if self.is_allsky:
            return self.__class__(
                self.nside // factor,
                self.nest,
                coordsys=self.coordsys,
                region=self.region,
                conv=self.conv,
                axes=copy.deepcopy(self.axes),
            )

        idx = list(self.get_idx(flat=True))
        nside = self._get_nside(idx)
        idx_new = get_superpixels(idx[0], nside, nside // factor, nest=self.nest)
        idx[0] = idx_new
        return self.__class__(
            self.nside // factor,
            self.nest,
            coordsys=self.coordsys,
            region=tuple(idx),
            conv=self.conv,
            axes=copy.deepcopy(self.axes),
        )

    @classmethod
    def create(
        cls,
        nside=None,
        binsz=None,
        nest=True,
        coordsys="CEL",
        region=None,
        axes=None,
        conv="gadf",
        skydir=None,
        width=None,
    ):
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
        >>> from gammapy.maps import HpxGeom, MapAxis
        >>> axis = MapAxis.from_bounds(0,1,2)
        >>> geom = HpxGeom.create(nside=16)
        >>> geom = HpxGeom.create(binsz=0.1, width=10.0)
        >>> geom = HpxGeom.create(nside=64, width=10.0, axes=[axis])
        >>> geom = HpxGeom.create(nside=[32,64], width=10.0, axes=[axis])
        """
        if nside is None and binsz is None:
            raise ValueError("Either nside or binsz must be defined.")

        if nside is None and binsz is not None:
            nside = get_nside_from_pix_size(binsz)

        if skydir is None:
            lon, lat = (0.0, 0.0)
        elif isinstance(skydir, tuple):
            lon, lat = skydir
        elif isinstance(skydir, SkyCoord):
            lon, lat, frame = skycoord_to_lonlat(skydir, coordsys=coordsys)
        else:
            raise ValueError("Invalid type for skydir: {!r}".format(type(skydir)))

        if region is None and width is not None:
            region = "DISK({:f},{:f},{:f})".format(lon, lat, width / 2.)

        return cls(
            nside, nest=nest, coordsys=coordsys, region=region, axes=axes, conv=conv
        )

    @staticmethod
    def identify_hpx_convention(header):
        """Identify the convention used to write this file."""
        # Hopefully the file contains the HPX_CONV keyword specifying
        # the convention used
        if "HPX_CONV" in header:
            return header["HPX_CONV"].lower()

        # Try based on the EXTNAME keyword
        hduname = header.get("EXTNAME", None)
        if hduname == "HPXEXPOSURES":
            return "fgst-bexpcube"
        elif hduname == "SKYMAP2":
            if "COORDTYPE" in header.keys():
                return "galprop"
            else:
                return "galprop2"

        # Check the name of the first column
        colname = header["TTYPE1"]
        if colname == "PIX":
            colname = header["TTYPE2"]

        if colname == "KEY":
            return "fgst-srcmap-sparse"
        elif colname == "ENERGY1":
            return "fgst-template"
        elif colname == "COSBINS":
            return "fgst-ltcube"
        elif colname == "Bin0":
            return "galprop"
        elif colname == "CHANNEL1" or colname == "CHANNEL0":
            if hduname == "SKYMAP":
                return "fgst-ccube"
            else:
                return "fgst-srcmap"
        else:
            raise ValueError("Could not identify HEALPIX convention")

    @classmethod
    def from_header(cls, header, hdu_bands=None, pix=None):
        """Create an HPX object from a FITS header.

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            The FITS header
        hdu_bands : `~astropy.io.fits.BinTableHDU`
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
        convname = HpxGeom.identify_hpx_convention(header)
        conv = HPX_FITS_CONVENTIONS[convname]

        axes = find_and_read_bands(hdu_bands)
        shape = [ax.nbin for ax in axes]

        if header["PIXTYPE"] != "HEALPIX":
            raise ValueError(
                "Header PIXTYPE must be 'HEALPIX'. Got: {}".format(header["PIXTYPE"])
            )

        if header["ORDERING"] == "RING":
            nest = False
        elif header["ORDERING"] == "NESTED":
            nest = True
        else:
            raise ValueError(
                "Header ORDERING must be RING or NESTED. Got: {}".format(
                    header["ORDERING"]
                )
            )

        if hdu_bands is not None and "NSIDE" in hdu_bands.columns.names:
            nside = hdu_bands.data.field("NSIDE").reshape(shape).astype(int)
        elif "NSIDE" in header:
            nside = header["NSIDE"]
        elif "ORDER" in header:
            nside = 2 ** header["ORDER"]
        else:
            raise ValueError("Failed to extract NSIDE or ORDER.")

        try:
            coordsys = header[conv.coordsys]
        except KeyError:
            coordsys = header.get("COORDSYS", "CEL")

        try:
            region = header["HPX_REG"]
        except KeyError:
            try:
                region = header["HPXREGION"]
            except KeyError:
                region = None

        return cls(
            nside, nest, coordsys=coordsys, region=region, axes=axes, conv=convname
        )

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
        if "HPX_REG" not in hdu.header:
            pix = (hdu.data.field("PIX"), hdu.data.field("CHANNEL"))
        else:
            pix = None

        return cls.from_header(hdu.header, hdu_bands=hdu_bands, pix=pix)

    def make_header(self, **kwargs):
        """"Build and return FITS header for this HEALPIX map."""
        header = fits.Header()
        conv = kwargs.get("conv", HPX_FITS_CONVENTIONS[self.conv])

        # FIXME: For some sparse maps we may want to allow EXPLICIT
        # with an empty region string
        indxschm = kwargs.get("indxschm", None)

        if indxschm is None:
            if self._region is None:
                indxschm = "IMPLICIT"
            elif self.is_regular == 1:
                indxschm = "EXPLICIT"
            else:
                indxschm = "LOCAL"

        if "FGST" in conv.convname.upper():
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
        header["HPX_CONV"] = conv.convname.upper()

        if self.coordsys == "CEL":
            header["EQUINOX"] = (2000.0, "Equinox of RA & DEC specifications")

        if self.region:
            header["HPX_REG"] = self._region

        return header

    def _make_bands_cols(self):
        cols = []
        if self.nside.size > 1:
            cols += [fits.Column("NSIDE", "I", array=np.ravel(self.nside))]
        return cols

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

        # TODO: this should return something more friendly than a tuple
        # e.g. a namedtuple or a dict
        tokens = parse_hpxregion(region)

        reg_type = tokens[0]
        if reg_type == "DISK":
            lon, lat = float(tokens[1]), float(tokens[2])
            radius = np.radians(float(tokens[3]))
            vec = coords_to_vec(lon, lat)[0]
            ilist = hp.query_disc(nside, vec, radius, inclusive=False, nest=nest)
        elif reg_type == "DISK_INC":
            lon, lat = float(tokens[1]), float(tokens[2])
            radius = np.radians(float(tokens[3]))
            vec = coords_to_vec(lon, lat)[0]
            fact = int(tokens[4])
            ilist = hp.query_disc(
                nside, vec, radius, inclusive=True, nest=nest, fact=fact
            )
        elif reg_type == "HPX_PIXEL":
            nside_pix = int(tokens[2])
            if tokens[1] == "NESTED":
                ipix_ring = hp.nest2ring(nside_pix, int(tokens[3]))
            elif tokens[1] == "RING":
                ipix_ring = int(tokens[3])
            else:
                raise ValueError("Invalid ordering scheme: {!r}".format(tokens[1]))
            ilist = match_hpx_pix(nside, nest, nside_pix, ipix_ring)
        else:
            raise ValueError("Invalid region type: {!r}".format(reg_type))

        return ilist

    def _get_ref_dir(self):
        """Compute the reference direction for this geometry."""
        import healpy as hp

        frame = coordsys_to_frame(self.coordsys)

        if self.region == "explicit":
            idx = unravel_hpx_index(self._ipix, self._maxpix)
            nside = self._get_nside(idx)
            vec = hp.pix2vec(nside, idx[0], nest=self.nest)
            vec = np.array([np.mean(t) for t in vec])
            lonlat = hp.vec2ang(vec, lonlat=True)
            return SkyCoord(lonlat[0], lonlat[1], frame=frame, unit="deg")

        return get_hpxregion_dir(self.region, self.coordsys)

    def _get_region_size(self):
        import healpy as hp

        if self.region is None:
            return 180.
        if self.region == "explicit":
            idx = unravel_hpx_index(self._ipix, self._maxpix)
            nside = self._get_nside(idx)
            ang = hp.pix2ang(nside, idx[0], nest=self.nest, lonlat=True)
            frame = coordsys_to_frame(self.coordsys)
            dirs = SkyCoord(ang[0], ang[1], unit="deg", frame=frame)
            return np.max(dirs.separation(self.center_skydir).deg)

        return get_hpxregion_size(self.region)

    def _get_nside(self, idx):
        if self.nside.size > 1:
            return self.nside[tuple(idx[1:])]
        else:
            return self.nside

    def make_wcs(self, proj="AIT", oversample=2, drop_axes=True, width_pix=None):
        """Make a WCS projection appropriate for this HPX pixelization.

        Parameters
        ----------
        drop_axes : bool
            Drop non-spatial axes from the
            HEALPIX geometry.  If False then all dimensions of the
            HEALPIX geometry will be copied to the WCS geometry.
        proj : str
            Projection type of WCS geometry.
        oversample : float
            Oversampling factor for WCS map. This will be the
            approximate ratio of the width of a HPX pixel to a WCS
            pixel. If this parameter is None then the width will be
            set from ``width_pix``.
        width_pix : int
            Width of the WCS geometry in pixels.  The pixel size will
            be set to the number of pixels satisfying ``oversample``
            or ``width_pix`` whichever is smaller.  If this parameter
            is None then the width will be set from ``oversample``.

        Returns
        -------
        wcs : `~gammapy.maps.WcsGeom`
            WCS geometry
        """
        pix_size = get_pix_size_from_nside(self.nside)
        binsz = np.min(pix_size) / oversample
        width = 2.0 * self._get_region_size() + np.max(pix_size)

        if width_pix is not None and int(width / binsz) > width_pix:
            binsz = width / width_pix

        if width > 90.:
            width = min(360., width), min(180.0, width)

        if drop_axes:
            axes = None
        else:
            axes = copy.deepcopy(self.axes)

        return WcsGeom.create(
            width=width,
            binsz=binsz,
            coordsys=self.coordsys,
            axes=axes,
            skydir=self.center_skydir,
            proj=proj,
        )

    def get_idx(self, idx=None, local=False, flat=False):
        if idx is not None and np.any(np.array(idx) >= np.array(self._shape)):
            raise ValueError("Image index out of range: {!r}".format(idx))

        # Regular all- and partial-sky maps
        if self.is_regular:

            pix = [np.arange(np.max(self._npix))]
            if idx is None:
                pix += [np.arange(ax.nbin, dtype=int) for ax in self.axes]
            else:
                pix += [t for t in idx]
            pix = np.meshgrid(*pix[::-1], indexing="ij", sparse=False)[::-1]
            pix = self.local_to_global(pix)

        # Non-regular all-sky
        elif self.is_allsky and not self.is_regular:

            shape = (np.max(self.npix),)
            if idx is None:
                shape = shape + self.shape
            else:
                shape = shape + (1,) * len(self.axes)
            pix = [np.full(shape, -1, dtype=int) for i in range(1 + len(self.axes))]
            for idx_img in np.ndindex(self.shape):

                if idx is not None and idx_img != idx:
                    continue

                npix = self._npix[idx_img]
                if idx is None:
                    s_img = (slice(0, npix),) + idx_img
                else:
                    s_img = (slice(0, npix),) + (0,) * len(self.axes)

                pix[0][s_img] = np.arange(self._npix[idx_img])
                for j in range(len(self.axes)):
                    pix[j + 1][s_img] = idx_img[j]
            pix = [p.T for p in pix]

        # Explicit pixel indices
        else:

            if idx is not None:
                npix_sum = np.concatenate(([0], np.cumsum(self._npix)))
                idx_ravel = np.ravel_multi_index(idx, self._shape)
                s = slice(npix_sum[idx_ravel], npix_sum[idx_ravel + 1])
            else:
                s = slice(None)
            pix_flat = unravel_hpx_index(self._ipix[s], self._maxpix)

            shape = (np.max(self.npix),)
            if idx is None:
                shape = shape + self.shape
            else:
                shape = shape + (1,) * len(self.axes)
            pix = [np.full(shape, -1, dtype=int) for _ in range(1 + len(self.axes))]

            for idx_img in np.ndindex(self.shape):

                if idx is not None and idx_img != idx:
                    continue

                npix = int(self._npix[idx_img])
                if idx is None:
                    s_img = (slice(0, npix),) + idx_img
                else:
                    s_img = (slice(0, npix),) + (0,) * len(self.axes)

                if self.axes:
                    m = np.all(
                        np.stack([pix_flat[i + 1] == t for i, t in enumerate(idx_img)]),
                        axis=0,
                    )
                    pix[0][s_img] = pix_flat[0][m]
                else:
                    pix[0][s_img] = pix_flat[0]

                for j in range(len(self.axes)):
                    pix[j + 1][s_img] = idx_img[j]

            pix = [p.T for p in pix]

        if local:
            pix = self.global_to_local(pix)

        if flat:
            pix = tuple([p[p != -1] for p in pix])

        return pix

    def get_coord(self, idx=None, flat=False):
        pix = self.get_idx(idx=idx, flat=flat)
        coords = self.pix_to_coord(pix)
        cdict = OrderedDict([("lon", coords[0]), ("lat", coords[1])])

        for i, axis in enumerate(self.axes):
            cdict[axis.name] = coords[i + 2]

        return MapCoord.create(cdict, coordsys=self.coordsys)

    def contains(self, coords):
        idx = self.coord_to_idx(coords)
        return np.all(np.stack([t != -1 for t in idx]), axis=0)

    def get_skydirs(self):
        """Get the sky coordinates of all the pixels in this geometry. """
        coords = self.get_coord()
        frame = "galactic" if self.coordsys == "GAL" else "icrs"
        return SkyCoord(coords[0], coords[1], unit="deg", frame=frame)

    def solid_angle(self):
        """Solid angle array (`~astropy.units.Quantity` in ``sr``).

        The array has the same dimensionality as ``map.nside``
        since all pixels have the same solid angle.
        """
        import healpy as hp

        return Quantity(hp.nside2pixarea(self.nside), "sr")

    def __repr__(self):
        str_ = self.__class__.__name__
        str_ += "\n\n"
        axes = ["skycoord"] + [_.name for _ in self.axes]
        str_ += "\taxes       : {}\n".format(", ".join(axes))
        str_ += "\tshape      : {}\n".format(self.data_shape[::-1])
        str_ += "\tndim       : {}\n".format(self.ndim)
        str_ += "\tnside      : {nside[0]}\n".format(nside=self.nside)
        str_ += "\tnested     : {}\n".format(self.nest)
        str_ += "\tcoordsys   : {}\n".format(self.coordsys)
        str_ += "\tprojection : {}\n".format(self.projection)
        lon, lat = self.center_skydir.data.lon.deg, self.center_skydir.data.lat.deg
        str_ += "\tcenter     : {lon:.1f} deg, {lat:.1f} deg\n".format(lon=lon, lat=lat)
        return str_


class HpxToWcsMapping(object):
    """Stores the indices need to convert from HEALPIX to WCS.

    Parameters
    ----------
    hpx : `~HpxGeom`
        HEALPix geometry object.
    wcs : `~gammapy.maps.WcsGeom`
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

    @classmethod
    def create(cls, hpx, wcs):
        """Create an object that maps pixels from HEALPix geometry ``hpx`` to
        WCS geometry ``wcs``.

        Parameters
        ----------
        hpx : `~HpxGeom`
            HEALPix geometry object.
        wcs : `~gammapy.maps.WcsGeom`
            WCS geometry object.

        Returns
        -------
        hpx2wcs : `~HpxToWcsMapping`

        """
        ipix, mult_val, npix = make_hpx_to_wcs_mapping(hpx, wcs)
        return cls(hpx, wcs, ipix, mult_val, npix)

    def fill_wcs_map_from_hpx_data(
        self, hpx_data, wcs_data, normalize=True, fill_nan=True
    ):
        """Fill the WCS map from the hpx data using the pre-calculated mappings.

        Parameters
        ----------
        hpx_data : `~numpy.ndarray`
            The input HEALPIX data
        wcs_data : `~numpy.ndarray`
            The data array being filled
        normalize : bool
            True -> preserve integral by splitting HEALPIX values between bins
        fill_nan : bool
            Fill pixels outside the HPX geometry with NaN.
        """
        # FIXME: Do we want to flatten mapping arrays?

        shape = tuple([t.flat[0] for t in self._npix])
        if self._valid.ndim != 1:
            shape = hpx_data.shape[:-1] + shape

        valid = np.where(self._valid.reshape(shape))
        lmap = self._lmap[self._valid]
        mult_val = self._mult_val[self._valid]

        wcs_slice = [slice(None) for _ in range(wcs_data.ndim - 2)]
        wcs_slice = tuple(wcs_slice + list(valid)[::-1][:2])

        hpx_slice = [slice(None) for _ in range(wcs_data.ndim - 2)]
        hpx_slice = tuple(hpx_slice + [lmap])

        if normalize:
            wcs_data[wcs_slice] = mult_val * hpx_data[hpx_slice]
        else:
            wcs_data[wcs_slice] = hpx_data[hpx_slice]

        if fill_nan:
            valid = np.swapaxes(self._valid.reshape(shape), -1, -2)
            valid = valid * np.ones_like(wcs_data, dtype=bool)
            wcs_data[~valid] = np.nan

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
