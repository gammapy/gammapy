# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from .utils import unpack_seq
from .geom import MapCoords, pix_tuple_to_idx, coord_to_idx
from .hpxmap import HpxMap
from .hpx import HpxGeom, HpxToWcsMapping, nside_to_order

__all__ = [
    'HpxMapND',
]


class HpxMapND(HpxMap):
    """Representation of a N+2D map using HEALPix with two spatial
    dimensions and N non-spatial dimensions.

    This class uses a N+1D numpy array to represent the sequence of
    HEALPix image planes.  Following the convention of WCS-based maps
    this class uses a column-wise ordering for the data array with the
    spatial dimension being tied to the last index of the array.

    Parameters
    ----------
    hpx : `~gammapy.maps.hpx.HpxGeom`
        HEALPIX geometry object.
    data : `~numpy.ndarray`
        HEALPIX data array.
        If none then an empty array will be allocated.

    """

    def __init__(self, hpx, data=None, dtype='float32'):

        shape = tuple([np.max(hpx.npix)] + [ax.nbin for ax in hpx.axes])
        if data is None:

            if hpx.npix.size > 1:
                data = np.nan * np.ones(shape, dtype=dtype).T
                pix = hpx.get_pixels(local=True)
                data[pix[::-1]] = 0.0
            else:
                data = np.zeros(shape, dtype=dtype).T

        elif data.shape != shape[::-1]:
            raise ValueError('Wrong shape for input data array. Expected {} '
                             'but got {}'.format(shape, data.shape))

        HpxMap.__init__(self, hpx, data)
        self._wcs2d = None
        self._hpx2wcs = None

    @classmethod
    def from_hdu(cls, hdu, hdu_bands=None):
        """Make a HpxMapND object from a FITS HDU.

        Parameters
        ----------
        hdu : `~astropy.fits.BinTableHDU`
            The FITS HDU
        hdu_bands  : `~astropy.fits.BinTableHDU`
            The BANDS table HDU
        """
        hpx = HpxGeom.from_header(hdu.header, hdu_bands)
        shape = tuple([ax.nbin for ax in hpx.axes[::-1]])
        shape_data = shape + tuple([np.max(hpx.npix)])

        # TODO: Should we support extracting slices?

        map_out = cls(hpx)

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
            map_out.set_by_idx(idx[::-1], vals)
        else:
            for c in colnames:
                if c.find(hpx.conv.colstring) == 0:
                    cnames.append(c)
            nbin = len(cnames)
            data = np.ndarray(shape_data)
            if len(cnames) == 1:
                map_out.data = hdu.data.field(cnames[0])
            else:
                for i, cname in enumerate(cnames):
                    idx = np.unravel_index(i, shape)
                    map_out.data[idx + (slice(None),)] = hdu.data.field(cname)
        return map_out

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

        if sum_bands and self.hpx.nside.size > 1:
            map_sum = self.sum_over_axes()
            return map_sum.to_wcs(sum_bands=False, normalize=normalize, proj=proj,
                                  oversample=oversample)

        # FIXME: Check whether the old mapping is still valid and reuse it
        self.make_wcs_mapping(oversample=oversample, proj=proj)

        # FIXME: Need a function to extract a valid shape from npix property

        if sum_bands:
            hpx_data = np.apply_over_axes(np.sum, self.data,
                                          axes=np.arange(self.data.ndim - 1))
            hpx_data = np.squeeze(hpx_data)
            wcs_shape = tuple([t.flat[0] for t in self._hpx2wcs.npix])
            wcs_data = np.zeros(wcs_shape).T
            wcs = self.hpx.make_wcs(proj=proj,
                                    oversample=oversample,
                                    drop_axes=True)
        else:
            hpx_data = self.data
            wcs_shape = tuple([t.flat[0] for t in
                               self._hpx2wcs.npix]) + self.hpx._shape
            wcs_data = np.zeros(wcs_shape).T
            wcs = self.hpx.make_wcs(proj=proj,
                                    oversample=oversample,
                                    drop_axes=False)

        # FIXME: Should reimplement instantiating map first and fill data array

        self._hpx2wcs.fill_wcs_map_from_hpx_data(hpx_data, wcs_data, normalize)
        return WcsMapND(wcs, wcs_data)

    def get_pixel_skydirs(self):
        """Get a list of sky coordinates for the centers of every pixel. """
        return self._hpx.get_skydirs()

    def iter_by_image(self):
        for idx in np.ndindex(self.geom.shape):
            yield self.data[idx[::-1]], idx

    def iter_by_pix(self, buffersize=1):
        pix = list(self.geom.get_pixels())
        vals = self.data[np.isfinite(self.data)]
        return unpack_seq(np.nditer([vals] + pix,
                                    flags=['external_loop', 'buffered'],
                                    buffersize=buffersize))

    def iter_by_coords(self, buffersize=1):
        coords = list(self.geom.get_coords())
        vals = self.data[np.isfinite(self.data)]
        return unpack_seq(np.nditer([vals] + coords,
                                    flags=['external_loop', 'buffered'],
                                    buffersize=buffersize))

    def sum_over_axes(self):
        """Sum over all non-spatial dimensions."""

        hpx_out = self.hpx.to_image()
        map_out = self.__class__(hpx_out)

        if self.hpx.nside.size > 1:
            vals = self.get_by_idx(self.hpx.get_pixels())
            map_out.fill_by_coords(self.hpx.get_coords()[:2], vals)
        else:
            axes = np.arange(self.data.ndim - 1).tolist()
            data = np.apply_over_axes(np.sum, self.data, axes=axes)
            map_out.data = np.squeeze(data, axis=axes)

        return map_out

    def _reproject_wcs(self, geom, order=1, mode='interp'):

        map_out = WcsMapND(geom)
        axes_eq = np.all([ax0 == ax1 for ax0, ax1 in
                          zip(geom.axes, self.geom.axes)])

        for vals, idx in map_out.iter_by_image():
            pass

        return map_out

    def _reproject_hpx(self, geom, order=1, mode='interp'):

        map_out = HpxMapND(geom)
        axes_eq = np.all([ax0 == ax1 for ax0, ax1 in
                          zip(geom.axes, self.geom.axes)])

        for vals, idx in map_out.iter_by_image():
            pass

        return map_out

    def pad(self, pad_width):
        raise NotImplementedError

    def crop(self, crop_width):
        raise NotImplementedError

    def upsample(self, factor):
        raise NotImplementedError

    def downsample(self, factor):
        raise NotImplementedError

    def interp_by_coords(self, coords, interp=None):

        if interp == 'linear':
            return self._interp_by_coords(coords, interp)
        else:
            raise ValueError('Invalid interpolation method: {}'.format(interp))

    def get_by_pix(self, pix, interp=None):
        # FIXME: Support local indexing here?
        # FIXME: Support slicing?
        # FIXME: What to return for pixels outside the map

        if interp is None:
            return self.get_by_idx(pix)
        else:
            raise NotImplementedError

    def get_by_idx(self, idx):
        idx = pix_tuple_to_idx(idx)
        idx = self.hpx.global_to_local(idx)
        return self.data.T[idx]

    def _get_interp_weights(self, coords, idxs):

        import healpy as hp

        c = MapCoords.create(coords)
        coords_ctr = list(coords[:2])
        coords_ctr += [ax.pix_to_coord(t)
                       for ax, t in zip(self.geom.axes, idxs)]
        pix_ctr = pix_tuple_to_idx(self.hpx.coord_to_pix(coords_ctr))
        pix_ctr = self.hpx.global_to_local(pix_ctr)

        if np.any(pix_ctr[0] == -1):
            raise ValueError('HPX pixel index out of map bounds.')

        theta = np.array(np.pi / 2. - np.radians(c.lat), ndmin=1)
        phi = np.array(np.radians(c.lon), ndmin=1)

        if self.hpx.nside.size > 1:
            nside = self.hpx.nside[idxs]
        else:
            nside = self.hpx.nside

        pix, wts = hp.pixelfunc.get_interp_weights(nside, theta,
                                                   phi, nest=self.hpx.nest)

        if self.hpx.nside.size > 1:
            pix_local = [self.hpx.global_to_local([pix] + list(idxs))[0]]
        else:
            pix_local = [self.hpx[pix]]

        m = pix_local[0] == -1
        pix_local[0][m] = (pix_ctr[0] * np.ones(pix.shape, dtype=int))[m]

        return pix_local + list(idxs), wts

    def _interp_by_coords(self, coords, interp):
        """Linearly interpolate map values."""
        import healpy as hp

        c = MapCoords.create(coords)
        pix, wts = self._get_interp_weights(coords,
                                            self.hpx.coord_to_idx(c)[1:])

        if self.hpx.ndim == 2:
            return np.sum(self.data.T[pix] * wts, axis=0)

        val = np.zeros(pix[0].shape[1:])
        # Loop over function values at corners
        for i, t in enumerate(range(2 ** len(self.hpx.axes))):

            pix_i = []
            wt = np.ones(pix[0].shape[1:])[None, ...]
            for j, ax in enumerate(self.hpx.axes):

                idx = coord_to_idx(ax.center[:-1],
                                   c[2 + j], bounded=True)  # [None, ...]

                w = ax.center[idx + 1] - ax.center[idx]
                if (i & (1 << j)):
                    wt *= (c[2 + j] - ax.center[idx]) / w
                    pix_i += [1 + idx]
                else:
                    wt *= (1.0 - (c[2 + j] - ax.center[idx]) / w)
                    pix_i += [idx]

            if self.hpx.nside.size > 1:
                pix, wts = self._get_interp_weights(coords, pix_i)

            val += np.sum(wts * wt * self.data.T[pix[:1] + pix_i], axis=0)

        return val

    def fill_by_idx(self, idx, weights=None):

        idx = pix_tuple_to_idx(idx)
        idx_local = list(self.hpx.global_to_local(idx))
        msk = idx_local[0] >= 0
        idx_local = [t[msk] for t in idx_local]
        if weights is not None:
            weights = weights[msk]
        idx_local = np.ravel_multi_index(idx_local, self.data.T.shape)
        idx_local, idx_inv = np.unique(idx_local, return_inverse=True)
        weights = np.bincount(idx_inv, weights=weights)
        self.data.T.flat[idx_local] += weights

    def set_by_idx(self, idx, vals):

        idx = pix_tuple_to_idx(idx)
        idx_local = (self.hpx[idx[0]],) + tuple(idx[1:])
        self.data.T[idx_local] = vals

    def to_swapped_scheme(self):

        import healpy as hp
        hpx_out = self.hpx.to_swapped()
        map_out = self.__class__(hpx_out)
        idx = list(self.hpx.get_pixels())
        vals = self.get_by_idx(idx)
        msk = vals > 0
        idx = [t[msk] for t in idx]
        vals = vals[msk]

        if self.hpx.nside.size > 1:
            nside = self.hpx.nside[idx[1:]]
        else:
            nside = self.hpx.nside

        if self.hpx.nest:
            idx_new = tuple([hp.nest2ring(nside, idx[0])] + idx[1:])
        else:
            idx_new = tuple([hp.ring2nest(nside, idx[0])] + idx[1:])

        map_out.set_by_pix(idx_new, vals)
        return map_out

    def to_ud_graded(self, nside, preserve_counts=False):

        # FIXME: For partial sky maps we should ensure that a higher
        # order map fully encompasses the lower order map

        # FIXME: For higher order maps we may want the option to split
        # the pixel amplitude among all subpixels

        import healpy as hp
        order = nside_to_order(nside)
        new_hpx = self.hpx.ud_graded_hpx(order)
        map_out = self.__class__(new_hpx)

        idx = list(self.hpx.get_pixels())
        coords = self.hpx.get_coords()
        vals = self.get_by_idx(idx)
        msk = vals > 0
        coords = [t[msk] for t in coords]
        vals = vals[msk]

        map_out.fill_by_coords(coords, vals)

        if not preserve_counts:
            fact = (2 ** order) ** 2 / (2 ** self.hpx.order) ** 2
            if self.hpx.nside.size > 1:
                fact = fact[..., None]
            map_out.data *= fact

        return map_out

    def plot(self, ax=None, normalize=False, proj='AIT', oversample=4, method='raster'):
        """Quickplot method.

        This will generate a visualization of the map by converting to
        a rasterized WCS image (method='raster') or drawing polygons
        for each pixel (method='poly').

        Parameters
        ----------
        proj : string, optional
            Any valid WCS projection type.
        oversample : int
            Oversampling factor to use when generating the rasterized
            image.
        method : str
            Method for mapping HEALPix pixels to a two-dimensional
            image.  Can be set to 'raster' (rasterization to cartesian
            image plane) or 'poly' (explicit polygons for each pixel).
            WARNING: The 'poly' method is much slower than 'raster'
            and only suitable for maps with less than ~10k pixels.

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure object.

        ax : `~astropy.visualization.wcsaxes.WCSAxes`
            WCS axis object

        im : `~matplotlib.image.AxesImage` or `~matplotlib.collections.PatchCollection`
            Image object.

        """

        if method == 'raster':
            m = self.to_wcs(sum_bands=True,
                            normalize=normalize,
                            proj=proj, oversample=oversample)
            return m.plot(ax)
        elif method == 'poly':
            return self._plot_poly(proj=proj, ax=ax)
        else:
            raise ValueError('Invalid method: {}'.format(method))

    def _plot_poly(self, proj='AIT', step=1, ax=None):
        """Plot the map using a collection of polygons.

        Parameters
        ----------
        proj : string, optional
            Any valid WCS projection type.
        step : int
            Set the number vertices that will be computed for each
            pixel in multiples of 4.

        """

        # FIXME: At the moment this only works for all-sky maps if the
        # projection is centered at (0,0)

        # FIXME: Figure out how to force a square aspect-ratio like imshow

        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        import healpy as hp

        wcs = self.geom.make_wcs(proj=proj, oversample=1)
        if ax is None:
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection=wcs.wcs)

        wcs_lonlat = wcs.center_coord[:2]
        pix = self.geom.get_pixels()
        vtx = hp.boundaries(self.geom.nside, pix[0],
                            nest=self.geom.nest, step=step)
        theta, phi = hp.pixelfunc.vec2ang(np.rollaxis(vtx, 2))
        theta = theta.reshape((4 * step, -1)).T
        phi = phi.reshape((4 * step, -1)).T

        patches = []
        data = []

        def get_angle(x, t):
            return 180. - (180. - x + t) % 360.

        for i, (x, y) in enumerate(zip(phi, theta)):

            lon, lat = np.degrees(x), np.degrees(np.pi / 2. - y)
            # Add a small ofset to avoid vertices wrapping to the
            # other size of the projection
            if get_angle(np.median(lon), wcs_lonlat[0]) > 0.0:
                pix = wcs.coord_to_pix((lon - 1E-4, lat))
            else:
                pix = wcs.coord_to_pix((lon + 1E-4, lat))

            dist = np.max(np.abs(pix[0][0] - pix[0]))

            # Split pixels that wrap around the edges of the projection
            if(dist > wcs.npix[0] / 1.5):

                lon, lat = np.degrees(x), np.degrees(np.pi / 2. - y)
                lon0 = lon - 1E-4
                lon1 = lon + 1E-4
                pix0 = wcs.coord_to_pix((lon0, lat))
                pix1 = wcs.coord_to_pix((lon1, lat))

                idx0 = np.argsort(pix0[0])
                idx1 = np.argsort(pix1[0])

                pix0 = (pix0[0][idx0][:3], pix0[1][idx0][:3])
                pix1 = (pix1[0][idx1][1:], pix1[1][idx1][1:])

                patches.append(Polygon(np.vstack((pix0[0], pix0[1])).T, True))
                patches.append(Polygon(np.vstack((pix1[0], pix1[1])).T, True))
                data.append(self.data[i])
                data.append(self.data[i])

            else:
                polygon = Polygon(np.vstack((pix[0], pix[1])).T, True)
                patches.append(polygon)
                data.append(self.data[i])

        p = PatchCollection(patches, linewidths=0, edgecolors='None')
        p.set_array(np.array(data))
        ax.add_collection(p)
        ax.autoscale_view()
        ax.coords.grid(color='w', linestyle=':', linewidth=0.5)

        return fig, ax, p
