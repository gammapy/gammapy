# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from .hpxmap import HpxMap
from .hpx import HPXGeom, HpxToWcsMapping

__all__ = [
    'HpxMapND',
]


class HpxMapND(HpxMap):
    """Representation of a N+2D map using HEALPIX with two spatial
    dimensions and N non-spatial dimensions.

    This class uses a numpy
    array to represent the sequence of HEALPix image planes.  As such
    it can only be used for maps with the same geometry (NSIDE and
    HPX_REG) in every plane.  Following the convention of WCS-based
    maps this class uses a column-wise ordering for the data array
    with the spatial dimension being tied to the last index of the
    array.

    Parameters
    ----------
    hpx : `~gammapy.maps.hpx.HPXGeom`
        HEALPIX geometry object.
    data : `~numpy.ndarray`
        HEALPIX data array.
        If none then an empty array will be allocated.
    """

    def __init__(self, hpx, data=None):

        npix = np.unique(hpx.npix)
        if len(npix) > 1:
            raise Exception('HpxMapND can only be instantiated from a '
                            'HPX geometry with the same nside in '
                            'every plane.')

        shape = tuple(list(hpx._shape[::-1]) + [npix[0]])
        if data is None:
            data = np.zeros(shape)
        elif data.shape != shape:
            raise ValueError('Wrong shape for input data array. Expected {} '
                             'but got {}'.format(shape, data.shape))

        HpxMap.__init__(self, hpx, data)
        self._wcs2d = None
        self._hpx2wcs = None

    @classmethod
    def from_hdu(cls, hdu, axes):
        """Make a HpxMapND object from a FITS HDU.

        Parameters
        ----------
        hdu : `~astropy.fits.BinTableHDU`
            The FITS HDU
        axes : list
            List of axes for non-spatial dimensions.
        """
        hpx = HPXGeom.from_header(hdu.header, axes)
        shape = tuple([ax.nbin for ax in axes[::-1]])
        shape_data = shape + tuple(np.unique(hpx.npix))

        # FIXME: We need to assert here if the file is incompatible
        # with an ND-array
        # TODO: Should we support extracting slices?

        colnames = hdu.columns.names
        cnames = []
        if hdu.header['INDXSCHM'] == 'SPARSE':
            pix = hdu.data.field('PIX')
            vals = hdu.data.field('VALUE')
            if 'CHANNEL' in hdu.data.columns.names:
                chan = hdu.data.field('CHANNEL')
                chan = np.unravel_index(chan, shape)
                idx = chan + (pix,)
            else:
                idx = (pix,)

            data = np.zeros(shape_data)
            data[idx] = vals
        else:
            for c in colnames:
                if c.find(hpx.conv.colstring) == 0:
                    cnames.append(c)
            nbin = len(cnames)
            data = np.ndarray(shape_data)
            if len(cnames) == 1:
                data[:] = hdu.data.field(cnames[0])
            else:
                for i, cname in enumerate(cnames):
                    idx = np.unravel_index(i, shape)
                    data[idx] = hdu.data.field(cname)
        return cls(hpx, data)

    def make_wcs_mapping(self, sum_bands=False, proj='AIT', oversample=2):
        """Make a HEALPix to WCS mapping object.

        Parameters
        ----------
        sum_bands : bool
           sum over non-spatial dimensions before reprojecting
        proj  : str
           WCS-projection
        oversample : int
           Oversampling factor for WCS map
        normalize : bool
           True -> preserve integral by splitting HEALPIX values between bins

        Returns
        -------
        wcs : `~astropy.wcs.WCS`
            WCS object
        data : `~numpy.ndarray`
            Reprojected data
        """
        self._wcs_proj = proj
        self._wcs_oversample = oversample
        self._wcs2d = self.hpx.make_wcs(proj=proj, oversample=oversample,
                                        drop_axes=True)
        self._hpx2wcs = HpxToWcsMapping.create(self.hpx, self._wcs2d)

    def to_wcs(self, sum_bands=False, normalize=True, proj='AIT', oversample=2):

        from .wcsnd import WcsMapND

        # FIXME: Check whether the old mapping is still valid and reuse it
        self.make_wcs_mapping(oversample=oversample)
        hpx_data = self.data

        if sum_bands:
            hpx_data = np.apply_over_axes(np.sum, hpx_data,
                                          axes=np.arange(hpx_data.ndim - 1))
            wcs_shape = tuple(self._hpx2wcs.npix)
            wcs_data = np.zeros(wcs_shape).T
            wcs = self.hpx.make_wcs(proj=proj,
                                    oversample=oversample,
                                    drop_axes=True)
        else:
            wcs_shape = tuple(self._hpx2wcs.npix) + self.hpx._shape
            wcs_data = np.zeros(wcs_shape).T
            wcs = self.hpx.make_wcs(proj=proj,
                                    oversample=oversample,
                                    drop_axes=False)

        self._hpx2wcs.fill_wcs_map_from_hpx_data(hpx_data, wcs_data, normalize)
        return WcsMapND(wcs, wcs_data)

    def get_pixel_skydirs(self):
        """Get a list of sky coordinates for the centers of every pixel. """
        return self._hpx.get_skydirs()

    def sum_over_axes(self):
        """Sum over all non-spatial dimensions."""
        # We sum over axis 0 in the array,
        # and drop the energy binning in the hpx object
        return self.__class__(self.hpx.copy_and_drop_axes(),
                              np.apply_over_axes(np.sum, self.data,
                                                 axes=np.arange(self.data.ndim - 1)))

    def get_by_coords(self, coords, interp=None):
        pix = self.hpx.coord_to_pix(coords)
        return self.get_by_pix(pix)

    def get_by_pix(self, pix):
        # FIXME: Support local indexing here?
        # FIXME: Support slicing?
        # FIXME: What to return for pixels outside the map

        # Reverse ordering and convert to local pixel indices
        pix = pix[::-1]
        pix_local = tuple(pix[:-1] + (self.hpx[pix[-1]],))

        return self.data[pix_local]

    def _interp_by_coord(self, coords):
        """Interpolate map values."""
        import healpy as hp
        raise NotImplementedError

    def _interpolate_cube(self, coords):
        """Perform interpolation on a HEALPIX cube.

        If egy is None, then interpolation will be performed
        on the existing energy planes.
        """
        import healpy as hp
        raise NotImplementedError

    def swap_scheme(self):
        """Return a new map with the opposite scheme (ring or nested).
        """
        import healpy as hp
        hpx_out = self.hpx.make_swapped_hpx()
        if self.hpx.nest:
            if self.data.ndim == 2:
                data_out = np.vstack([hp.pixelfunc.reorder(
                    self.data[i], n2r=True) for i in range(self.data.shape[0])])
            else:
                data_out = hp.pixelfunc.reorder(self.data, n2r=True)
        else:
            if self.data.ndim == 2:
                data_out = np.vstack([hp.pixelfunc.reorder(
                    self.data[i], r2n=True) for i in range(self.data.shape[0])])
            else:
                data_out = hp.pixelfunc.reorder(self.data, r2n=True)
        return self.__class__(hpx_out, data_out)

    def ud_grade(self, order, preserve_counts=False):
        """Upgrade or downgrade the resolution of the map to the chosen order.
        """
        import healpy as hp
        new_hpx = self.hpx.ud_graded_hpx(order)
        nebins = len(new_hpx.evals)
        shape = self.data.shape

        if preserve_counts:
            power = -2.
        else:
            power = 0

        if len(shape) == 1:
            new_data = hp.pixelfunc.ud_grade(self.data,
                                             nside_out=new_hpx.nside,
                                             order_in=new_hpx.ordering,
                                             order_out=ew_hpx.ordering,
                                             power=power)
        else:
            new_data = [hp.pixelfunc.ud_grade(self.data[i],
                                              nside_out=new_hpx.nside,
                                              order_in=new_hpx.ordering,
                                              order_out=new_hpx.ordering,
                                              power=power)
                        for i in range(shape[0])]
            new_data = np.vstack(new_data)

        return self.__class__(new_hpx, new_data)
