# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.io import fits
from astropy.units import Quantity
from gammapy.utils.units import unit_from_fits_image_hdu
from .geom import MapCoord, pix_tuple_to_idx
from .hpx import HPX_FITS_CONVENTIONS, HpxConv, HpxGeom, HpxToWcsMapping, nside_to_order
from .hpxmap import HpxMap
from .utils import INVALID_INDEX, interp_to_order

__all__ = ["HpxNDMap"]


class HpxNDMap(HpxMap):
    """HEALPix map with any number of non-spatial dimensions.

    This class uses a N+1D numpy array to represent the sequence of
    HEALPix image planes.  Following the convention of WCS-based maps
    this class uses a column-wise ordering for the data array with the
    spatial dimension being tied to the last index of the array.

    Parameters
    ----------
    geom : `~gammapy.maps.HpxGeom`
        HEALPIX geometry object.
    data : `~numpy.ndarray`
        HEALPIX data array.
        If none then an empty array will be allocated.
    meta : `dict`
        Dictionary to store meta data.
    unit : str or `~astropy.units.Unit`
        The map unit
    """

    def __init__(self, geom, data=None, dtype="float32", meta=None, unit=""):
        data_shape = geom.data_shape

        if data is None:
            data = self._make_default_data(geom, data_shape, dtype)

        super().__init__(geom, data, meta, unit)

    @staticmethod
    def _make_default_data(geom, shape_np, dtype):
        if geom.npix.size > 1:
            data = np.full(shape_np, np.nan, dtype=dtype)
            idx = geom.get_idx(local=True)
            data[idx[::-1]] = 0
        else:
            data = np.zeros(shape_np, dtype=dtype)

        return data

    @classmethod
    def from_hdu(cls, hdu, hdu_bands=None, format=None):
        """Make a HpxNDMap object from a FITS HDU.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            The FITS HDU
        hdu_bands  : `~astropy.io.fits.BinTableHDU`
            The BANDS table HDU
        format : str, optional
            FITS convention. If None the format is guessed. The following
            formats are supported:

                - "gadf"
                - "fgst-ccube"
                - "fgst-ltcube"
                - "fgst-bexpcube"
                - "fgst-srcmap"
                - "fgst-template"
                - "fgst-srcmap-sparse"
                - "galprop"
                - "galprop2"

        Returns
        -------
        map : `HpxMap`
            HEALPix map

        """
        if format is None:
            format = HpxConv.identify_hpx_format(hdu.header)

        geom = HpxGeom.from_header(hdu.header, hdu_bands, format=format)

        hpx_conv = HPX_FITS_CONVENTIONS[format]

        shape = geom.axes.shape[::-1]

        # TODO: Should we support extracting slices?

        meta = cls._get_meta_from_header(hdu.header)
        unit = unit_from_fits_image_hdu(hdu.header)
        map_out = cls(geom, None, meta=meta, unit=unit)

        colnames = hdu.columns.names
        cnames = []
        if hdu.header.get("INDXSCHM", None) == "SPARSE":
            pix = hdu.data.field("PIX")
            vals = hdu.data.field("VALUE")
            if "CHANNEL" in hdu.data.columns.names:
                chan = hdu.data.field("CHANNEL")
                chan = np.unravel_index(chan, shape)
                idx = chan + (pix,)
            else:
                idx = (pix,)

            map_out.set_by_idx(idx[::-1], vals)
        else:
            for c in colnames:
                if c.find(hpx_conv.colstring) == 0:
                    cnames.append(c)
            nbin = len(cnames)
            if nbin == 1:
                map_out.data = hdu.data.field(cnames[0])
            else:
                for idx, cname in enumerate(cnames):
                    idx = np.unravel_index(idx, shape)
                    map_out.data[idx + (slice(None),)] = hdu.data.field(cname)

        return map_out

    def to_wcs(
        self,
        sum_bands=False,
        normalize=True,
        proj="AIT",
        oversample=2,
        width_pix=None,
        hpx2wcs=None,
    ):
        from .wcsnd import WcsNDMap

        if sum_bands and self.geom.nside.size > 1:
            map_sum = self.sum_over_axes()
            return map_sum.to_wcs(
                sum_bands=False,
                normalize=normalize,
                proj=proj,
                oversample=oversample,
                width_pix=width_pix,
            )

        # FIXME: Check whether the old mapping is still valid and reuse it
        if hpx2wcs is None:
            geom_wcs_image = self.geom.to_wcs_geom(
                proj=proj, oversample=oversample, width_pix=width_pix, drop_axes=True
            )

            hpx2wcs = HpxToWcsMapping.create(self.geom, geom_wcs_image)

        # FIXME: Need a function to extract a valid shape from npix property

        if sum_bands:
            axes = np.arange(self.data.ndim - 1)
            hpx_data = np.apply_over_axes(np.sum, self.data, axes=axes)
            hpx_data = np.squeeze(hpx_data)
            wcs_shape = tuple([t.flat[0] for t in hpx2wcs.npix])
            wcs_data = np.zeros(wcs_shape).T
            wcs = hpx2wcs.wcs.to_image()
        else:
            hpx_data = self.data
            wcs_shape = tuple([t.flat[0] for t in hpx2wcs.npix]) + self.geom.shape_axes
            wcs_data = np.zeros(wcs_shape).T
            wcs = hpx2wcs.wcs.to_cube(self.geom.axes)

        # FIXME: Should reimplement instantiating map first and fill data array
        hpx2wcs.fill_wcs_map_from_hpx_data(hpx_data, wcs_data, normalize)
        return WcsNDMap(wcs, wcs_data, unit=self.unit)

    def pad(self, pad_width, mode="constant", cval=0, order=1):
        geom = self.geom.pad(pad_width)
        map_out = self._init_copy(geom=geom, data=None)
        map_out.coadd(self)
        coords = geom.get_coord(flat=True)
        m = self.geom.contains(coords)
        coords = tuple([c[~m] for c in coords])

        if mode == "constant":
            map_out.set_by_coord(coords, cval)
        elif mode == "interp":
            # FIXME: These modes don't work at present because
            # interp_by_coord doesn't support extrapolation
            vals = self.interp_by_coord(coords, interp=order)
            map_out.set_by_coord(coords, vals)
        else:
            raise ValueError(f"Unrecognized pad mode: {mode!r}")

        return map_out

    def crop(self, crop_width):
        geom = self.geom.crop(crop_width)
        map_out = self._init_copy(geom=geom, data=None)
        map_out.coadd(self)
        return map_out

    def upsample(self, factor, preserve_counts=True):
        geom = self.geom.upsample(factor)
        coords = geom.get_coord()
        data = self.get_by_coord(coords)

        if preserve_counts:
            data /= factor ** 2

        return self._init_copy(geom=geom, data=data)

    def downsample(self, factor, preserve_counts=True):
        geom = self.geom.downsample(factor)
        coords = self.geom.get_coord()
        vals = self.get_by_coord(coords)

        map_out = self._init_copy(geom=geom, data=None)
        map_out.fill_by_coord(coords, vals)

        if not preserve_counts:
            map_out.data /= factor ** 2

        return map_out

    def interp_by_coord(self, coords, interp=1):
        # inherited docstring
        coords = MapCoord.create(coords, frame=self.geom.frame)

        order = interp_to_order(interp)
        if order == 1:
            return self._interp_by_coord(coords, order)
        else:
            raise ValueError(f"Invalid interpolation order: {order!r}")

    def interp_by_pix(self, pix, interp=None):
        """Interpolate map values at the given pixel coordinates.
        """
        raise NotImplementedError

    def get_by_idx(self, idx):
        # inherited docstring
        idx = pix_tuple_to_idx(idx)
        idx = self.geom.global_to_local(idx)
        return self.data.T[idx]

    def _get_interp_weights(self, coords, idxs=None):
        import healpy as hp

        if idxs is None:
            idxs = self.geom.coord_to_idx(coords, clip=True)[1:]

        theta, phi = coords.theta, coords.phi

        m = ~np.isfinite(theta)
        theta[m] = 0
        phi[m] = 0

        if not self.geom.is_regular:
            nside = self.geom.nside[tuple(idxs)]
        else:
            nside = self.geom.nside

        pix, wts = hp.get_interp_weights(nside, theta, phi, nest=self.geom.nest)
        wts[:, m] = 0
        pix[:, m] = INVALID_INDEX.int

        if not self.geom.is_regular:
            pix_local = [self.geom.global_to_local([pix] + list(idxs))[0]]
        else:
            pix_local = [self.geom[pix]]

        # If a pixel lies outside of the geometry set its index to the center pixel
        m = pix_local[0] == INVALID_INDEX.int
        if m.any():
            coords_ctr = [coords.lon, coords.lat]
            coords_ctr += [ax.pix_to_coord(t) for ax, t in zip(self.geom.axes, idxs)]
            idx_ctr = self.geom.coord_to_idx(coords_ctr)
            idx_ctr = self.geom.global_to_local(idx_ctr)
            pix_local[0][m] = (idx_ctr[0] * np.ones(pix.shape, dtype=int))[m]

        pix_local += [np.broadcast_to(t, pix_local[0].shape) for t in idxs]
        return pix_local, wts

    def _interp_by_coord(self, coords, order):
        """Linearly interpolate map values."""
        pix, wts = self._get_interp_weights(coords)

        if self.geom.is_image:
            return np.sum(self.data.T[tuple(pix)] * wts, axis=0)

        val = np.zeros(pix[0].shape[1:])

        # Loop over function values at corners
        for i in range(2 ** len(self.geom.axes)):
            pix_i = []
            wt = np.ones(pix[0].shape[1:])[np.newaxis, ...]
            for j, ax in enumerate(self.geom.axes):
                idx = ax.coord_to_idx(coords[ax.name])
                idx = np.clip(idx, 0, len(ax.center) - 2)

                w = ax.center[idx + 1] - ax.center[idx]
                c = Quantity(coords[ax.name], ax.center.unit, copy=False).value

                if i & (1 << j):
                    wt *= (c - ax.center[idx].value) / w.value
                    pix_i += [idx + 1]
                else:
                    wt *= 1.0 - (c - ax.center[idx].value) / w.value
                    pix_i += [idx]

            if not self.geom.is_regular:
                pix, wts = self._get_interp_weights(coords, pix_i)

            wts[pix[0] == INVALID_INDEX.int] = 0
            wt[~np.isfinite(wt)] = 0
            val += np.nansum(wts * wt * self.data.T[tuple(pix[:1] + pix_i)], axis=0)

        return val

    def fill_by_idx(self, idx, weights=None):
        idx = pix_tuple_to_idx(idx)
        msk = np.all(np.stack([t != INVALID_INDEX.int for t in idx]), axis=0)
        if weights is not None:
            weights = weights[msk]
        idx = [t[msk] for t in idx]

        idx_local = list(self.geom.global_to_local(idx))
        msk = idx_local[0] >= 0
        idx_local = [t[msk] for t in idx_local]
        if weights is not None:
            if isinstance(weights, Quantity):
                weights = weights.to_value(self.unit)
            weights = weights[msk]

        idx_local = np.ravel_multi_index(idx_local, self.data.T.shape)
        idx_local, idx_inv = np.unique(idx_local, return_inverse=True)
        weights = np.bincount(idx_inv, weights=weights)
        self.data.T.flat[idx_local] += weights

    def set_by_idx(self, idx, vals):
        idx = pix_tuple_to_idx(idx)
        idx_local = self.geom.global_to_local(idx)
        self.data.T[idx_local] = vals

    def _make_cols(self, header, conv):
        shape = self.data.shape
        cols = []

        if header["INDXSCHM"] == "SPARSE":
            data = self.data.copy()
            data[~np.isfinite(data)] = 0
            nonzero = np.where(data > 0)
            value = data[nonzero].astype(float)
            pix = self.geom.local_to_global(nonzero[::-1])[0]
            if len(shape) == 1:
                cols.append(fits.Column("PIX", "J", array=pix))
                cols.append(fits.Column("VALUE", "E", array=value))
            else:
                channel = np.ravel_multi_index(nonzero[:-1], shape[:-1])
                cols.append(fits.Column("PIX", "J", array=pix))
                cols.append(fits.Column("CHANNEL", "I", array=channel))
                cols.append(fits.Column("VALUE", "E", array=value))
        elif len(shape) == 1:
            name = conv.colname(indx=conv.firstcol)
            array = self.data.astype(float)
            cols.append(fits.Column(name, "E", array=array))
        else:
            for i, idx in enumerate(np.ndindex(shape[:-1])):
                name = conv.colname(indx=i + conv.firstcol)
                array = self.data[idx].astype(float)
                cols.append(fits.Column(name, "E", array=array))

        return cols

    def to_swapped(self):
        import healpy as hp

        hpx_out = self.geom.to_swapped()
        map_out = self._init_copy(geom=hpx_out, data=None)
        idx = self.geom.get_idx(flat=True)
        vals = self.get_by_idx(idx)
        if self.geom.nside.size > 1:
            nside = self.geom.nside[idx[1:]]
        else:
            nside = self.geom.nside

        if self.geom.nest:
            idx_new = tuple([hp.nest2ring(nside, idx[0])]) + idx[1:]
        else:
            idx_new = tuple([hp.ring2nest(nside, idx[0])]) + idx[1:]

        map_out.set_by_idx(idx_new, vals)
        return map_out

    def to_ud_graded(self, nside, preserve_counts=False):
        # FIXME: Should we remove/deprecate this method?

        order = nside_to_order(nside)
        new_hpx = self.geom.to_ud_graded(order)
        map_out = self._init_copy(geom=new_hpx, data=None)

        if np.all(order <= self.geom.order):
            # Downsample
            idx = self.geom.get_idx(flat=True)
            coords = self.geom.pix_to_coord(idx)
            vals = self.get_by_idx(idx)
            map_out.fill_by_coord(coords, vals)
        else:
            # Upsample
            idx = new_hpx.get_idx(flat=True)
            coords = new_hpx.pix_to_coord(idx)
            vals = self.get_by_coord(coords)
            m = np.isfinite(vals)
            map_out.fill_by_coord([c[m] for c in coords], vals[m])

        if not preserve_counts:
            fact = (2 ** order) ** 2 / (2 ** self.geom.order) ** 2
            if self.geom.nside.size > 1:
                fact = fact[..., None]
            map_out.data *= fact

        return map_out

    def plot(
        self,
        method="raster",
        ax=None,
        normalize=False,
        proj="AIT",
        oversample=2,
        width_pix=1000,
        **kwargs,
    ):
        """Quickplot method.

        This will generate a visualization of the map by converting to
        a rasterized WCS image (method='raster') or drawing polygons
        for each pixel (method='poly').

        Parameters
        ----------
        method : {'raster','poly'}
            Method for mapping HEALPix pixels to a two-dimensional
            image.  Can be set to 'raster' (rasterization to cartesian
            image plane) or 'poly' (explicit polygons for each pixel).
            WARNING: The 'poly' method is much slower than 'raster'
            and only suitable for maps with less than ~10k pixels.
        proj : string, optional
            Any valid WCS projection type.
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
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.imshow`.

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure object.
        ax : `~astropy.visualization.wcsaxes.WCSAxes`
            WCS axis object
        im : `~matplotlib.image.AxesImage` or `~matplotlib.collections.PatchCollection`
            Image object.

        """
        if method == "raster":
            m = self.to_wcs(
                sum_bands=True,
                normalize=normalize,
                proj=proj,
                oversample=oversample,
                width_pix=width_pix,
            )
            return m.plot(ax, **kwargs)
        elif method == "poly":
            return self._plot_poly(proj=proj, ax=ax)
        else:
            raise ValueError(f"Invalid method: {method!r}")

    def _plot_poly(self, proj="AIT", step=1, ax=None):
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

        wcs = self.geom.to_wcs_geom(proj=proj, oversample=1)
        if ax is None:
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection=wcs.wcs, aspect="equal")

        wcs_lonlat = wcs.center_coord[:2]
        idx = self.geom.get_idx()
        vtx = hp.boundaries(self.geom.nside, idx[0], nest=self.geom.nest, step=step)
        theta, phi = hp.vec2ang(np.rollaxis(vtx, 2))
        theta = theta.reshape((4 * step, -1)).T
        phi = phi.reshape((4 * step, -1)).T

        patches = []
        data = []

        def get_angle(x, t):
            return 180.0 - (180.0 - x + t) % 360.0

        for i, (x, y) in enumerate(zip(phi, theta)):

            lon, lat = np.degrees(x), np.degrees(np.pi / 2.0 - y)
            # Add a small ofset to avoid vertices wrapping to the
            # other size of the projection
            if get_angle(np.median(lon), wcs_lonlat[0].to_value("deg")) > 0:
                idx = wcs.coord_to_pix((lon - 1e-4, lat))
            else:
                idx = wcs.coord_to_pix((lon + 1e-4, lat))

            dist = np.max(np.abs(idx[0][0] - idx[0]))

            # Split pixels that wrap around the edges of the projection
            if dist > wcs.npix[0] / 1.5:
                lon, lat = np.degrees(x), np.degrees(np.pi / 2.0 - y)
                lon0 = lon - 1e-4
                lon1 = lon + 1e-4
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
                polygon = Polygon(np.vstack((idx[0], idx[1])).T, True)
                patches.append(polygon)
                data.append(self.data[i])

        p = PatchCollection(patches, linewidths=0, edgecolors="None")
        p.set_array(np.array(data))
        ax.add_collection(p)
        ax.autoscale_view()
        ax.coords.grid(color="w", linestyle=":", linewidth=0.5)

        return fig, ax, p
