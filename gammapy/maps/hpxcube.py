# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from .hpxmap import HpxMap
from .hpx import HPXGeom

__all__ = [
    'HpxCube',
]


class HpxCube(HpxMap):
    """Representation of a N+2D map using HEALPIX.  This class uses a
    numpy array to represent the sequence of HEALPix image planes.  As
    such it can only be used for maps with the same geometry (NSIDE
    and HPX_REG) in every plane.

    Parameters
    ----------
    hpx : `~gammapy.maps.hpx.HPXGeom`
        HEALPIX geometry object.
    data : `~numpy.ndarray`
        HEALPIX data array.  If none then an empty array will be
        allocated.

    """

    def __init__(self, hpx, data=None):

        npix = np.unique(hpx.npix)
        if len(npix) > 1:
            raise Exception('HpxCube can only be instantiated from a '
                            'HPX geometry with the same nside in '
                            'every plane.')

        shape = tuple(list(hpx._shape) + [npix[0]])
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
        """Make a HpxCube object from a FITS HDU.

        Parameters
        ----------
        hdu : `~astropy.fits.BinTableHDU`
            The FITS HDU
        axes : list
            List of axes for non-spatial dimensions.
        """
        hpx = HPXGeom.from_header(hdu.header, axes)
        shape = tuple([ax.nbin for ax in axes])
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

    def make_wcs(self, sum_bands=False, proj='CAR', oversample=2,
                 normalize=True):
        """Make a WCS object and convert HEALPIX data into WCS projection.

        NOTE: this re-calculates the mapping, if you have already
        calculated the mapping it is much faster to use
        convert_to_cached_wcs() instead

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
        (WCS object, np.ndarray() with reprojected data)
        """
        self._wcs_proj = proj
        self._wcs_oversample = oversample
        self._wcs2d = self.hpx.make_wcs(2, proj=proj, oversample=oversample)
        self._hpx2wcs = HpxToWcsMapping(self.hpx, self._wcs2d)
        wcs, wcs_data = self.to_cached_wcs(self.counts, sum_ebins,
                                           normalize)
        return wcs, wcs_data

    def to_cached_wcs(self, hpx_in, sum_ebins=False, normalize=True):
        """Make a WCS object and convert HEALPIX data into WCS projection.

        Parameters
        ----------
        hpx_in  : `~numpy.ndarray`
           HEALPIX input data
        sum_ebins : bool
           sum energy bins over energy bins before reprojecting
        normalize  : bool
           True -> preserve integral by splitting HEALPIX values between bins

        Returns
        -------
        wcs : `~astropy.wcs.WCS`
            WCS object
        data : `~numpy.ndarray`
            Reprojected data
        """
        if self._hpx2wcs is None:
            raise Exception('HpxMap.convert_to_cached_wcs() called '
                            'before make_wcs_from_hpx()')

        if len(hpx_in.shape) == 1:
            wcs_data = np.ndarray(self._hpx2wcs.npix)
            loop_ebins = False
            hpx_data = hpx_in
        elif len(hpx_in.shape) == 2:
            if sum_ebins:
                wcs_data = np.ndarray(self._hpx2wcs.npix)
                hpx_data = hpx_in.sum(0)
                loop_ebins = False
            else:
                wcs_data = np.ndarray((self.counts.shape[0],
                                       self._hpx2wcs.npix[0],
                                       self._hpx2wcs.npix[1]))
                hpx_data = hpx_in
                loop_ebins = True
        else:
            dim = len(hpx_in.shape)
            raise Exception('Wrong dimension for HpxMap: {}'.format(dim))

        if loop_ebins:
            for i in range(hpx_data.shape[0]):
                self._hpx2wcs.fill_wcs_map_from_hpx_data(
                    hpx_data[i], wcs_data[i], normalize)
                pass

            wcs_data.reshape(
                (self.counts.shape[0], self._hpx2wcs.npix[0], self._hpx2wcs.npix[1]))
            # replace the WCS with a 3D one
            wcs = self.hpx.make_wcs(3, proj=self._wcs_proj,
                                    energies=self.hpx.ebins,
                                    oversample=self._wcs_oversample)
        else:
            self._hpx2wcs.fill_wcs_map_from_hpx_data(
                hpx_data, wcs_data, normalize)
            wcs_data.reshape(self._hpx2wcs.npix)
            wcs = self._wcs2d

        return wcs, wcs_data

    def get_pixel_skydirs(self):
        """Get a list of sky coordinates for the centers of every pixel. """
        sky_coords = self._hpx.get_sky_coords()
        frame = 'galactic' if self.hpx.coordsys == 'GAL' else 'icrs'
        return SkyCoord(sky_coords[0], sky_coords[1], frame=frame, unit='deg')

    def sum_over_axes(self):
        """Sum over all non-spatial dimensions."""
        # We sum over axis 0 in the array,
        # and drop the energy binning in the hpx object
        return HpxCube(np.sum(self.counts, axis=0),
                       self.hpx.copy_and_drop_axes())

    def get_by_coord(self, coords, interp=None):
        pix = self.hpx.coord_to_pix(coords)
        return self.get_by_pix(pix)

    def get_by_pix(self, pix):
        pix = self.hpx[pix]
        if self.data.ndim == 2:
            return self.data[:, pix]
        else:
            return self.data[pix]

    def _interp_by_coord(self, coords):
        """Interpolate map values.
        """
        import healpy as hp
        if self.data.ndim == 1:
            theta = np.pi / 2. - np.radians(lat)
            phi = np.radians(lon)
            return hp.pixelfunc.get_interp_val(self.data, theta,
                                               phi, nest=self.hpx.nest)
        else:
            return self._interpolate_cube(coords)

    def _interpolate_cube(self, coords):
        """Perform interpolation on a HEALPIX cube.

        If egy is None, then interpolation will be performed
        on the existing energy planes.
        """
        import healpy as hp
        shape = np.broadcast(lon, lat, egy).shape
        lon = lon * np.ones(shape)
        lat = lat * np.ones(shape)
        theta = np.pi / 2. - np.radians(lat)
        phi = np.radians(lon)
        vals = []
        for i, _ in enumerate(self.hpx.evals):
            v = hp.pixelfunc.get_interp_val(self.counts[i], theta,
                                            phi, nest=self.hpx.nest)
            vals += [np.expand_dims(np.array(v, ndmin=1), -1)]

        vals = np.concatenate(vals, axis=-1)

        if egy is None:
            return vals.T

        egy = egy * np.ones(shape)

        if interp_log:
            xvals = utils.val_to_pix(np.log(self.hpx.evals), np.log(egy))
        else:
            xvals = utils.val_to_pix(self.hpx.evals, egy)

        vals = vals.reshape((-1, vals.shape[-1]))
        xvals = np.ravel(xvals)
        v = map_coordinates(vals, [np.arange(vals.shape[0]), xvals],
                            order=1)
        return v.reshape(shape)

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
        shape = self.counts.shape

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
