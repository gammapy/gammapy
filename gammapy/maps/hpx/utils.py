# Licensed under a 3-clause BSD style license - see LICENSE.rst
import re
import numpy as np
from astropy.utils import lazyproperty
from gammapy.utils.array import is_power2
from ..utils import INVALID_INDEX

# Approximation of the size of HEALPIX pixels (in degrees) for a particular order.
# Used to convert from HEALPIX to WCS-based projections.
HPX_ORDER_TO_PIXSIZE = np.array(
    [32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.50, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.002]
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
        raise ValueError(f"HEALPIX order must be 0 to 13. Got: {order!r}")

    return HPX_ORDER_TO_PIXSIZE[order]


def match_hpx_pix(nside, nest, nside_pix, ipix_ring):
    """TODO: document."""
    import healpy as hp

    ipix_in = np.arange(12 * nside * nside)
    vecs = hp.pix2vec(nside, ipix_in, nest)
    pix_match = hp.vec2pix(nside_pix, vecs[0], vecs[1], vecs[2]) == ipix_ring
    return ipix_in[pix_match]


def parse_hpxregion(region):
    """Parse the ``HPX_REG`` header keyword into a list of tokens."""
    m = re.match(r"([A-Za-z\_]*?)\((.*?)\)", region)

    if m is None:
        raise ValueError(f"Failed to parse hpx region string: {region!r}")

    if not m.group(1):
        return re.split(",", m.group(2))
    else:
        return [m.group(1)] + re.split(",", m.group(2))


def nside_to_order(nside):
    """Compute the ORDER given NSIDE.

    Returns -1 for NSIDE values that are not a power of 2.
    """
    nside = np.array(nside, ndmin=1)
    order = -1 * np.ones_like(nside)
    m = is_power2(nside)
    order[m] = np.log2(nside[m]).astype(int)
    return order


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
        m = idx == INVALID_INDEX.int
        idx[m] = 0
        idx = hp.nest2ring(nside_superpix, idx)
        idx[m] = INVALID_INDEX.int

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
        idx[x >= np.broadcast_to(npix[..., None], x.shape)] = INVALID_INDEX.int
    else:
        idx = idx[..., None] + x

    if not nest:
        m = idx == INVALID_INDEX.int
        idx[m] = 0
        idx = hp.nest2ring(nside_subpix[..., None], idx)
        idx[m] = INVALID_INDEX.int

    return idx


class HpxToWcsMapping:
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

    @property
    def hpx(self):
        """HEALPIX projection."""
        return self._hpx

    @property
    def wcs(self):
        """WCS projection."""
        return self._wcs

    @property
    def ipix(self):
        """An array(nx,ny) of the global HEALPIX pixel indices for each WCS pixel."""
        return self._ipix

    @property
    def mult_val(self):
        """An array(nx,ny) of 1/number of WCS pixels pointing at each HEALPIX pixel."""
        return self._mult_val

    @property
    def npix(self):
        """A tuple(nx,ny) of the shape of the WCS grid."""
        return self._npix

    @lazyproperty
    def lmap(self):
        """Array ``(nx, ny)`` mapping local HEALPIX pixel indices for each WCS pixel."""

        return self.hpx.global_to_local(self.ipix, ravel=True)

    @property
    def valid(self):
        """Array ``(nx, ny)`` of bool: which WCS pixel in inside the HEALPIX region."""
        return self.lmap >= 0

    @classmethod
    def create(cls, hpx, wcs):
        """Create HEALPix to WCS geometry pixel mapping.

        Parameters
        ----------
        hpx : `~HpxGeom`
            HEALPix geometry object.
        wcs : `~gammapy.maps.WcsGeom`
            WCS geometry object.

        Returns
        -------
        hpx2wcs : `~HpxToWcsMapping`
            Mapping

        """
        import healpy as hp

        npix = wcs.npix

        # FIXME: Calculation of WCS pixel centers should be moved into a
        # method of WcsGeom
        pix_crds = np.dstack(np.meshgrid(np.arange(npix[0]), np.arange(npix[1])))
        pix_crds = pix_crds.swapaxes(0, 1).reshape((-1, 2))
        sky_crds = wcs.wcs.wcs_pix2world(pix_crds, 0)
        sky_crds *= np.radians(1.0)
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
            mult_val[i, ...] = 1.0 / count[1][idx]

        if hpx.nside.size == 1:
            ipix = np.squeeze(ipix, axis=0)
            mult_val = np.squeeze(mult_val, axis=0)

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
        if self.valid.ndim != 1:
            shape = hpx_data.shape[:-1] + shape

        valid = np.where(self.valid.reshape(shape))
        lmap = self.lmap[self.valid]
        mult_val = self._mult_val[self.valid]

        wcs_slice = [slice(None) for _ in range(wcs_data.ndim - 2)]
        wcs_slice = tuple(wcs_slice + list(valid)[::-1][:2])

        hpx_slice = [slice(None) for _ in range(wcs_data.ndim - 2)]
        hpx_slice = tuple(hpx_slice + [lmap])

        if normalize:
            wcs_data[wcs_slice] = mult_val * hpx_data[hpx_slice]
        else:
            wcs_data[wcs_slice] = hpx_data[hpx_slice]

        if fill_nan:
            valid = np.swapaxes(self.valid.reshape(shape), -1, -2)
            valid = valid * np.ones_like(wcs_data, dtype=bool)
            wcs_data[~valid] = np.nan

        return wcs_data

    def fill_hpx_map_from_wcs_data(self, wcs_data, hpx_data, normalize=True):
        """Fill the HPX map from the WCS data using the pre-calculated mappings.

        Parameters
        ----------
        wcs_data : `~numpy.ndarray`
            The input WCS data
        hpx_data : `~numpy.ndarray`
            The data array being filled
        normalize : bool
            True -> preserve integral by splitting HEALPIX values between bins
        """

        shape = tuple([t.flat[0] for t in self._npix])
        if self.valid.ndim != 1:
            shape = hpx_data.shape[:-1] + shape

        valid = np.where(self.valid.reshape(shape))
        lmap = self.lmap[self.valid]
        mult_val = self._mult_val[self.valid]

        wcs_slice = [slice(None) for _ in range(wcs_data.ndim - 2)]
        wcs_slice = tuple(wcs_slice + list(valid)[::-1][:2])

        hpx_slice = [slice(None) for _ in range(wcs_data.ndim - 2)]
        hpx_slice = tuple(hpx_slice + [lmap])

        if normalize:
            hpx_data[hpx_slice] = 1 / mult_val * wcs_data[wcs_slice]
        else:
            hpx_data[hpx_slice] = wcs_data[wcs_slice]

        return hpx_data
