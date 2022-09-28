# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for dealing with HEALPix projections and mappings."""
import copy
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.units import Quantity
from gammapy.utils.array import is_power2
from ..axes import MapAxes
from ..coord import MapCoord, skycoord_to_lonlat
from ..geom import Geom, pix_tuple_to_idx
from ..utils import INVALID_INDEX, coordsys_to_frame, frame_to_coordsys
from .io import HPX_FITS_CONVENTIONS, HpxConv
from .utils import (
    coords_to_vec,
    get_nside_from_pix_size,
    get_pix_size_from_nside,
    get_subpixels,
    get_superpixels,
    match_hpx_pix,
    nside_to_order,
    parse_hpxregion,
    ravel_hpx_index,
    unravel_hpx_index,
)

# Not sure if we should expose this in the docs or not:
# HPX_FITS_CONVENTIONS, HpxConv
__all__ = ["HpxGeom"]


class HpxGeom(Geom):
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
        non-spatial axes. If nest is True, nside must be a power of 2,
        less than 2**30.
    nest : bool
        True -> 'NESTED', False -> 'RING' indexing scheme
    frame : str
        Coordinate system, "icrs" | "galactic"
    region : str or tuple
        Spatial geometry for partial-sky maps.  If none the map will
        encompass the whole sky.  String input will be parsed
        according to HPX_REG header keyword conventions.  Tuple
        input can be used to define an explicit list of pixels
        encompassed by the geometry.
    axes : list
        Axes for non-spatial dimensions.
    """

    is_hpx = True
    is_region = False

    def __init__(self, nside, nest=True, frame="icrs", region=None, axes=None):
        from healpy.pixelfunc import check_nside

        check_nside(nside, nest=nest)

        self._nside = np.array(nside, ndmin=1)
        self._axes = MapAxes.from_default(axes, n_spatial_axes=1)

        if self.nside.size > 1 and self.nside.shape != self.shape_axes:
            raise ValueError(
                "Wrong dimensionality for nside. nside must "
                "be a scalar or have a dimensionality consistent "
                "with the axes argument."
            )

        self._frame = frame
        self._nest = nest
        self._ipix = None
        self._region = region
        self._create_lookup(region)
        self._npix = self._npix * np.ones(self.shape_axes, dtype=int)

    def _create_lookup(self, region):
        """Create local-to-global pixel lookup table."""
        if isinstance(region, str):
            ipix = [
                self.get_index_list(nside, self._nest, region)
                for nside in self._nside.flat
            ]

            self._ipix = [
                ravel_hpx_index((p, i * np.ones_like(p)), np.ravel(self.npix_max))
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

            self._ipix = ravel_hpx_index(region, self.npix_max)
            self._ipix = np.unique(self._ipix)
            region = unravel_hpx_index(self._ipix, self.npix_max)
            self._region = "explicit"
            self._indxschm = "EXPLICIT"
            if len(region) == 1:
                self._npix = np.array([len(region[0])])
            else:
                self._npix = np.zeros(self.shape_axes, dtype=int)
                idx = np.ravel_multi_index(region[1:], self.shape_axes)
                cnt = np.unique(idx, return_counts=True)
                self._npix.flat[cnt[0]] = cnt[1]

        elif region is None:
            self._region = None
            self._indxschm = "IMPLICIT"
            self._npix = self.npix_max

        else:
            raise ValueError(f"Invalid region string: {region!r}")

    def local_to_global(self, idx_local):
        """Compute a local index (partial-sky) from a global (all-sky) index.

        Returns
        -------
        idx_global : tuple
            A tuple of pixel index vectors with global HEALPIX pixel indices
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

        idx_global = unravel_hpx_index(self._ipix[idx], self.npix_max)
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
        if (
            isinstance(idx_global, int)
            or (isinstance(idx_global, tuple) and isinstance(idx_global[0], int))
            or isinstance(idx_global, np.ndarray)
        ):
            idx_global = unravel_hpx_index(np.array(idx_global, ndmin=1), self.npix_max)

        if self.nside.size == 1:
            idx = np.array(idx_global[0], ndmin=1)
        else:
            idx = ravel_hpx_index(idx_global, self.npix_max)

        if self._ipix is not None:
            retval = np.full(idx.size, -1, "i")
            m = np.isin(idx.flat, self._ipix)
            retval[m] = np.searchsorted(self._ipix, idx.flat[m])
            retval = retval.reshape(idx.shape)
        else:
            retval = idx

        if self.nside.size == 1:
            idx_local = tuple([retval] + list(idx_global[1:]))
        else:
            idx_local = unravel_hpx_index(retval, self._npix)

        m = np.any(np.stack([t == INVALID_INDEX.int for t in idx_local]), axis=0)
        for i, t in enumerate(idx_local):
            idx_local[i][m] = INVALID_INDEX.int

        if not ravel:
            return idx_local
        else:
            return ravel_hpx_index(idx_local, self.npix)

    def cutout(self, position, width, **kwargs):
        """Create a cutout around a given position.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Center position of the cutout region.
        width : `~astropy.coordinates.Angle` or `~astropy.units.Quantity`
            Diameter of the circular cutout region.

        Returns
        -------
        cutout : `~gammapy.maps.WcsNDMap`
            Cutout map
        """
        if not self.is_regular:
            raise ValueError("Can only do a cutout from a regular map.")

        width = u.Quantity(width, "deg").value
        return self.create(
            nside=self.nside,
            nest=self.nest,
            width=width,
            skydir=position,
            frame=self.frame,
            axes=self.axes,
        )

    def coord_to_pix(self, coords):
        import healpy as hp

        coords = MapCoord.create(
            coords, frame=self.frame, axis_names=self.axes.names
        ).broadcasted
        theta, phi = coords.theta, coords.phi

        if self.axes:
            idxs = self.axes.coord_to_idx(coords, clip=True)
            bins = self.axes.coord_to_pix(coords)

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
            pix = tuple([pix]) + bins
            if np.any(m):
                for p in pix:
                    p[m] = INVALID_INDEX.int
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
            m = ipix == INVALID_INDEX.int
            ipix[m] = 0
            theta, phi = hp.pix2ang(nside, ipix, nest=self.nest)
            coords = [np.degrees(phi), np.degrees(np.pi / 2.0 - theta)]
            coords = tuple(coords + vals)
            if np.any(m):
                for c in coords:
                    c[m] = INVALID_INDEX.float
        else:
            ipix = np.round(pix[0]).astype(int)
            theta, phi = hp.pix2ang(self.nside, ipix, nest=self.nest)
            coords = (np.degrees(phi), np.degrees(np.pi / 2.0 - theta))

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

    @property
    def axes(self):
        """List of non-spatial axes."""
        return self._axes

    @property
    def axes_names(self):
        """All axes names"""
        return ["skycoord"] + self.axes.names

    @property
    def shape_axes(self):
        """Shape of non-spatial axes."""
        return self.axes.shape

    @property
    def data_shape(self):
        """Shape of the Numpy data array matching this geometry."""
        npix_shape = tuple([np.max(self.npix)])
        return (npix_shape + self.axes.shape)[::-1]

    @property
    def data_shape_axes(self):
        """Shape of data of the non-spatial axes and unit spatial axes."""
        return self.axes.shape[::-1] + (1,)

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
        """ORDER in each band (``NSIDE = 2 ** ORDER``).

        Set to -1 for bands with NSIDE that is not a power of 2.
        """
        return nside_to_order(self.nside)

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
    def npix_max(self):
        """Max. number of pixels"""
        maxpix = 12 * self.nside**2
        return maxpix * np.ones(self.shape_axes, dtype=int)

    @property
    def frame(self):
        return self._frame

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
        return self._region is None

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
        lon, lat, frame = skycoord_to_lonlat(self.center_skydir)
        return tuple([lon, lat]) + self.axes.center_coord

    @property
    def center_pix(self):
        """Pixel coordinates of the center of the geometry (tuple)."""
        return self.coord_to_pix(self.center_coord)

    @property
    def center_skydir(self):
        """Sky coordinate of the center of the geometry.

        Returns
        -------
        center : `~astropy.coordinates.SkyCoord`
            Center position
        """
        # TODO: simplify
        import healpy as hp

        if self.is_allsky:
            lon, lat = 0.0, 0.0
        elif self.region == "explicit":
            idx = unravel_hpx_index(self._ipix, self.npix_max)
            nside = self._get_nside(idx)
            vec = hp.pix2vec(nside, idx[0], nest=self.nest)
            vec = np.array([np.mean(t) for t in vec])
            lonlat = hp.vec2ang(vec, lonlat=True)
            lon, lat = lonlat[0], lonlat[1]
        else:
            tokens = parse_hpxregion(self.region)
            if tokens[0] in ["DISK", "DISK_INC"]:
                lon, lat = float(tokens[1]), float(tokens[2])
            elif tokens[0] == "HPX_PIXEL":
                nside_pix = int(tokens[2])
                ipix_pix = int(tokens[3])
                if tokens[1] == "NESTED":
                    nest_pix = True
                elif tokens[1] == "RING":
                    nest_pix = False
                else:
                    raise ValueError(f"Invalid ordering scheme: {tokens[1]!r}")
                theta, phi = hp.pix2ang(nside_pix, ipix_pix, nest_pix)
                lat = np.degrees((np.pi / 2) - theta)
                lon = np.degrees(phi)

        return SkyCoord(lon, lat, frame=self.frame, unit="deg")

    @property
    def pixel_scales(self):
        self.angle_ = """Pixel scale.

        Returns
        -------
        angle: `~astropy.coordinates.Angle`
        """
        return get_pix_size_from_nside(self.nside) * u.deg

    def interp_weights(self, coords, idxs=None):
        """Get interpolation weights for given coords

        Parameters
        ----------
        coords : `MapCoord` or dict
            Input coordinates
        idxs : `~numpy.ndarray`
            Indices for non-spatial axes.

        Returns
        -------
        weights : `~numpy.ndarray`
            Interpolation weights
        """
        import healpy as hp

        coords = MapCoord.create(coords, frame=self.frame).broadcasted

        if idxs is None:
            idxs = self.coord_to_idx(coords, clip=True)[1:]

        theta, phi = coords.theta, coords.phi

        m = ~np.isfinite(theta)
        theta[m] = 0
        phi[m] = 0

        if not self.is_regular:
            nside = self.nside[tuple(idxs)]
        else:
            nside = self.nside

        pix, wts = hp.get_interp_weights(nside, theta, phi, nest=self.nest)
        wts[:, m] = 0
        pix[:, m] = INVALID_INDEX.int

        if not self.is_regular:
            pix_local = [self.global_to_local([pix] + list(idxs))[0]]
        else:
            pix_local = [self.global_to_local(pix, ravel=True)]

        # If a pixel lies outside of the geometry set its index to the center pixel
        m = pix_local[0] == INVALID_INDEX.int
        if m.any():
            coords_ctr = [coords.lon, coords.lat]
            coords_ctr += [ax.pix_to_coord(t) for ax, t in zip(self.axes, idxs)]
            idx_ctr = self.coord_to_idx(coords_ctr)
            idx_ctr = self.global_to_local(idx_ctr)
            pix_local[0][m] = (idx_ctr[0] * np.ones(pix.shape, dtype=int))[m]

        pix_local += [np.broadcast_to(t, pix_local[0].shape) for t in idxs]
        return pix_local, wts

    @property
    def ipix(self):
        """HEALPIX pixel and band indices for every pixel in the map."""
        return self.get_idx()

    def is_aligned(self, other):
        """Check if HEALPIx geoms and extra axes are aligned.

        Parameters
        ----------
        other : `HpxGeom`
            Other geom.

        Returns
        -------
        aligned : bool
            Whether geometries are aligned
        """
        for axis, otheraxis in zip(self.axes, other.axes):
            if axis != otheraxis:
                return False

        if not self.nside == other.nside:
            return False
        elif not self.frame == other.frame:
            return False
        elif not self.nest == other.nest:
            return False
        else:
            return True

    def to_nside(self, nside):
        """Upgrade or downgrade the reoslution to a given nside

        Parameters
        ----------
        nside : int
            Nside

        Returns
        -------
        geom : `~HpxGeom`
            A HEALPix geometry object.
        """
        if not self.is_regular:
            raise ValueError("Upgrade and degrade only implemented for standard maps")

        axes = copy.deepcopy(self.axes)
        return self.__class__(
            nside=nside, nest=self.nest, frame=self.frame, region=self.region, axes=axes
        )

    def to_binsz(self, binsz):
        """Change pixel size of the geometry.

        Parameters
        ----------
        binsz : float or `~astropy.units.Quantity`
            New pixel size. A float is assumed to be in degree.

        Returns
        -------
        geom : `WcsGeom`
            Geometry with new pixel size.
        """
        binsz = u.Quantity(binsz, "deg").value

        if self.is_allsky:
            return self.create(
                binsz=binsz,
                frame=self.frame,
                axes=copy.deepcopy(self.axes),
            )
        else:
            return self.create(
                skydir=self.center_skydir,
                binsz=binsz,
                width=self.width.to_value("deg"),
                frame=self.frame,
                axes=copy.deepcopy(self.axes),
            )

    def separation(self, center):
        """Compute sky separation wrt a given center.

        Parameters
        ----------
        center : `~astropy.coordinates.SkyCoord`
            Center position

        Returns
        -------
        separation : `~astropy.coordinates.Angle`
            Separation angle array (1D)
        """
        coord = self.to_image().get_coord()
        return center.separation(coord.skycoord)

    def to_swapped(self):
        """Geometry copy with swapped ORDERING (NEST->RING or vice versa).

        Returns
        -------
        geom : `~HpxGeom`
            A HEALPix geometry object.
        """
        axes = copy.deepcopy(self.axes)
        return self.__class__(
            self.nside,
            not self.nest,
            frame=self.frame,
            region=self.region,
            axes=axes,
        )

    def to_image(self):
        return self.__class__(
            np.max(self.nside), self.nest, frame=self.frame, region=self.region
        )

    def to_cube(self, axes):
        axes = copy.deepcopy(self.axes) + axes
        return self.__class__(
            np.max(self.nside),
            self.nest,
            frame=self.frame,
            region=self.region,
            axes=axes,
        )

    def _get_neighbors(self, idx):
        import healpy as hp

        nside = self._get_nside(idx)
        idx_nb = (hp.get_all_neighbours(nside, idx[0], nest=self.nest),)
        idx_nb += tuple([t[None, ...] * np.ones_like(idx_nb[0]) for t in idx[1:]])

        return idx_nb

    def _pad_spatial(self, pad_width):
        if self.is_allsky:
            raise ValueError("Cannot pad an all-sky map.")

        idx = self.get_idx(flat=True)
        idx_r = ravel_hpx_index(idx, self.npix_max)

        # TODO: Pre-filter indices to find those close to the edge
        idx_nb = self._get_neighbors(idx)
        idx_nb = ravel_hpx_index(idx_nb, self.npix_max)

        for _ in range(pad_width):
            mask_edge = np.isin(idx_nb, idx_r, invert=True)
            idx_edge = idx_nb[mask_edge]
            idx_edge = np.unique(idx_edge)
            idx_r = np.sort(np.concatenate((idx_r, idx_edge)))
            idx_nb = unravel_hpx_index(idx_edge, self.npix_max)
            idx_nb = self._get_neighbors(idx_nb)
            idx_nb = ravel_hpx_index(idx_nb, self.npix_max)

        idx = unravel_hpx_index(idx_r, self.npix_max)
        return self.__class__(
            self.nside.copy(),
            self.nest,
            frame=self.frame,
            region=idx,
            axes=copy.deepcopy(self.axes),
        )

    def crop(self, crop_width):
        if self.is_allsky:
            raise ValueError("Cannot crop an all-sky map.")

        idx = self.get_idx(flat=True)
        idx_r = ravel_hpx_index(idx, self.npix_max)

        # TODO: Pre-filter indices to find those close to the edge
        idx_nb = self._get_neighbors(idx)
        idx_nb = ravel_hpx_index(idx_nb, self.npix_max)

        for _ in range(crop_width):
            # Mask of pixels that have at least one neighbor not
            # contained in the geometry
            mask_edge = np.any(np.isin(idx_nb, idx_r, invert=True), axis=0)
            idx_r = idx_r[~mask_edge]
            idx_nb = idx_nb[:, ~mask_edge]

        idx = unravel_hpx_index(idx_r, self.npix_max)
        return self.__class__(
            self.nside.copy(),
            self.nest,
            frame=self.frame,
            region=idx,
            axes=copy.deepcopy(self.axes),
        )

    def upsample(self, factor):
        if not is_power2(factor):
            raise ValueError("Upsample factor must be a power of 2.")

        if self.is_allsky:
            return self.__class__(
                self.nside * factor,
                self.nest,
                frame=self.frame,
                region=self.region,
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
            frame=self.frame,
            region=tuple(idx),
            axes=copy.deepcopy(self.axes),
        )

    def downsample(self, factor, axis_name=None):
        if not is_power2(factor):
            raise ValueError("Downsample factor must be a power of 2.")
        if axis_name is not None:
            raise ValueError("Currently the only valid axis name is None.")

        if self.is_allsky:
            return self.__class__(
                self.nside // factor,
                self.nest,
                frame=self.frame,
                region=self.region,
                axes=copy.deepcopy(self.axes),
            )

        idx = list(self.get_idx(flat=True))
        nside = self._get_nside(idx)
        idx_new = get_superpixels(idx[0], nside, nside // factor, nest=self.nest)
        idx[0] = idx_new
        return self.__class__(
            self.nside // factor,
            self.nest,
            frame=self.frame,
            region=tuple(idx),
            axes=copy.deepcopy(self.axes),
        )

    @classmethod
    def create(
        cls,
        nside=None,
        binsz=None,
        nest=True,
        frame="icrs",
        region=None,
        axes=None,
        skydir=None,
        width=None,
    ):
        """Create an HpxGeom object.

        Parameters
        ----------
        nside : int or `~numpy.ndarray`
            HEALPix NSIDE parameter.  This parameter sets the size of
            the spatial pixels in the map. If nest is True, nside must be a
            power of 2, less than 2**30.
        binsz : float or `~numpy.ndarray`
            Approximate pixel size in degrees.  An NSIDE will be
            chosen that corresponds to a pixel size closest to this
            value.  This option is superseded by nside.
        nest : bool
            True for HEALPIX "NESTED" indexing scheme, False for "RING" scheme
        frame : {"icrs", "galactic"}, optional
            Coordinate system, either Galactic ("galactic") or Equatorial ("icrs").
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

        Returns
        -------
        geom : `~HpxGeom`
            A HEALPix geometry object.

        Examples
        --------
        >>> from gammapy.maps import HpxGeom, MapAxis
        >>> axis = MapAxis.from_bounds(0,1,2)
        >>> geom = HpxGeom.create(nside=16) # doctest: +SKIP
        >>> geom = HpxGeom.create(binsz=0.1, width=10.0) # doctest: +SKIP
        >>> geom = HpxGeom.create(nside=64, width=10.0, axes=[axis]) # doctest: +SKIP
        >>> geom = HpxGeom.create(nside=[32,64], width=10.0, axes=[axis]) # doctest: +SKIP
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
            lon, lat, frame = skycoord_to_lonlat(skydir, frame=frame)
        else:
            raise ValueError(f"Invalid type for skydir: {type(skydir)!r}")

        if region is None and width is not None:
            region = f"DISK({lon},{lat},{width/2})"

        return cls(nside, nest=nest, frame=frame, region=region, axes=axes)

    @classmethod
    def from_header(cls, header, hdu_bands=None, format=None):
        """Create an HPX object from a FITS header.

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            The FITS header
        hdu_bands : `~astropy.io.fits.BinTableHDU`
            The BANDS table HDU.
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
                - "healpy"

        Returns
        -------
        hpx : `~HpxGeom`
            HEALPix geometry.
        """
        if format is None:
            format = HpxConv.identify_hpx_format(header)

        conv = HPX_FITS_CONVENTIONS[format]

        axes = MapAxes.from_table_hdu(hdu_bands, format=format)

        if header["PIXTYPE"] != "HEALPIX":
            raise ValueError(
                f"Invalid header PIXTYPE: {header['PIXTYPE']} (must be HEALPIX)"
            )

        if header["ORDERING"] == "RING":
            nest = False
        elif header["ORDERING"] == "NESTED":
            nest = True
        else:
            raise ValueError(
                f"Invalid header ORDERING: {header['ORDERING']} (must be RING or NESTED)"
            )

        if hdu_bands is not None and "NSIDE" in hdu_bands.columns.names:
            nside = hdu_bands.data.field("NSIDE").reshape(axes.shape).astype(int)
        elif "NSIDE" in header:
            nside = header["NSIDE"]
        elif "ORDER" in header:
            nside = 2 ** header["ORDER"]
        else:
            raise ValueError("Failed to extract NSIDE or ORDER.")

        try:
            frame = coordsys_to_frame(header[conv.frame])
        except KeyError:
            frame = header.get("COORDSYS", "icrs")

        try:
            region = header["HPX_REG"]
        except KeyError:
            try:
                region = header["HPXREGION"]
            except KeyError:
                region = None

        return cls(nside, nest, frame=frame, region=region, axes=axes)

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

    def to_header(self, format="gadf", **kwargs):
        """Build and return FITS header for this HEALPIX map."""
        header = fits.Header()
        format = kwargs.get("format", HPX_FITS_CONVENTIONS[format])

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

        if "FGST" in format.convname.upper():
            header["TELESCOP"] = "GLAST"
            header["INSTRUME"] = "LAT"

        header[format.frame] = frame_to_coordsys(self.frame)
        header["PIXTYPE"] = "HEALPIX"
        header["ORDERING"] = self.ordering
        header["INDXSCHM"] = indxschm
        header["ORDER"] = np.max(self.order)
        header["NSIDE"] = np.max(self.nside)
        header["FIRSTPIX"] = 0
        header["LASTPIX"] = np.max(self.npix_max) - 1
        header["HPX_CONV"] = format.convname.upper()

        if self.frame == "icrs":
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
                raise ValueError(f"Invalid ordering scheme: {tokens[1]!r}")
            ilist = match_hpx_pix(nside, nest, nside_pix, ipix_ring)
        else:
            raise ValueError(f"Invalid region type: {reg_type!r}")

        return ilist

    @property
    def width(self):
        """Width of the map"""
        # TODO: simplify
        import healpy as hp

        if self.is_allsky:
            width = 180.0
        elif self.region == "explicit":
            idx = unravel_hpx_index(self._ipix, self.npix_max)
            nside = self._get_nside(idx)
            ang = hp.pix2ang(nside, idx[0], nest=self.nest, lonlat=True)
            dirs = SkyCoord(ang[0], ang[1], unit="deg", frame=self.frame)
            width = np.max(dirs.separation(self.center_skydir))
        else:
            tokens = parse_hpxregion(self.region)
            if tokens[0] in {"DISK", "DISK_INC"}:
                width = float(tokens[3])
            elif tokens[0] == "HPX_PIXEL":
                pix_size = get_pix_size_from_nside(int(tokens[2]))
                width = 2.0 * pix_size

        return u.Quantity(width, "deg")

    def _get_nside(self, idx):
        if self.nside.size > 1:
            return self.nside[tuple(idx[1:])]
        else:
            return self.nside

    def to_wcs_geom(self, proj="AIT", oversample=2, width_pix=None):
        """Make a WCS projection appropriate for this HPX pixelization.

        Parameters
        ----------
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
        from gammapy.maps import WcsGeom

        pix_size = get_pix_size_from_nside(self.nside)
        binsz = np.min(pix_size) / oversample
        width = 2.0 * self.width.to_value("deg") + np.max(pix_size)

        if width_pix is not None and int(width / binsz) > width_pix:
            binsz = width / width_pix

        if width > 90.0:
            width = min(360.0, width), min(180.0, width)

        axes = copy.deepcopy(self.axes)

        return WcsGeom.create(
            width=width,
            binsz=binsz,
            frame=self.frame,
            axes=axes,
            skydir=self.center_skydir,
            proj=proj,
        )

    def to_wcs_tiles(self, nside_tiles=4, margin="0 deg"):
        """Create WCS tiles geometries from HPX geometry with given nside.

        The HEALPix geom is divide into superpixels defined by nside_tiles,
        which are then represented by a WCS geometry using a tangential
        projection. The number of WCS tiles is given by the number of pixels
        for the given nside_tiles.

        Parameters
        ----------
        nside_tiles : int
            Nside for super pixel tiles. Usually nsi
        margin : Angle
            Width margin of the wcs tile

        Returns
        -------
        wcs_tiles : list
            List of WCS tile geoms.
        """
        import healpy as hp
        from gammapy.maps import WcsGeom

        margin = u.Quantity(margin)

        if nside_tiles >= self.nside:
            raise ValueError(f"nside_tiles must be < {self.nside}")

        if not self.is_allsky:
            raise ValueError("to_wcs_tiles() is only supported for all sky geoms")

        binsz = np.degrees(hp.nside2resol(self.nside)) * u.deg

        hpx = self.to_image().to_nside(nside=nside_tiles)
        wcs_tiles = []

        for pix in range(int(hpx.npix)):
            skydir = hpx.pix_to_coord([pix])
            vtx = hp.boundaries(nside=hpx.nside, pix=pix, nest=hpx.nest, step=1)

            lon, lat = hp.vec2ang(vtx.T, lonlat=True)
            boundaries = SkyCoord(lon * u.deg, lat * u.deg, frame=hpx.frame)

            # Compute maximum separation between all pairs of boundaries and take it
            # as width
            width = boundaries.separation(boundaries[:, np.newaxis]).max()

            wcs_tile_geom = WcsGeom.create(
                skydir=(float(skydir[0]), float(skydir[1])),
                width=width + margin,
                binsz=binsz,
                frame=hpx.frame,
                proj="TAN",
                axes=self.axes,
            )
            wcs_tiles.append(wcs_tile_geom)

        return wcs_tiles

    def get_idx(
        self,
        idx=None,
        local=False,
        flat=False,
        sparse=False,
        mode="center",
        axis_name=None,
    ):
        # TODO: simplify this!!!
        if idx is not None and np.any(np.array(idx) >= np.array(self.shape_axes)):
            raise ValueError(f"Image index out of range: {idx!r}")

        # Regular all- and partial-sky maps
        if self.is_regular:
            pix = [np.arange(np.max(self._npix))]
            if idx is None:
                for ax in self.axes:
                    if mode == "edges" and ax.name == axis_name:
                        pix += [np.arange(-0.5, ax.nbin, dtype=float)]
                    else:
                        pix += [np.arange(ax.nbin, dtype=int)]
            else:
                pix += [t for t in idx]

            pix = np.meshgrid(*pix[::-1], indexing="ij", sparse=sparse)[::-1]
            pix = self.local_to_global(pix)

        # Non-regular all-sky
        elif self.is_allsky and not self.is_regular:

            shape = (np.max(self.npix),)
            if idx is None:
                shape = shape + self.shape_axes
            else:
                shape = shape + (1,) * len(self.axes)
            pix = [np.full(shape, -1, dtype=int) for i in range(1 + len(self.axes))]
            for idx_img in np.ndindex(self.shape_axes):

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
                idx_ravel = np.ravel_multi_index(idx, self.shape_axes)
                s = slice(npix_sum[idx_ravel], npix_sum[idx_ravel + 1])
            else:
                s = slice(None)
            pix_flat = unravel_hpx_index(self._ipix[s], self.npix_max)

            shape = (np.max(self.npix),)
            if idx is None:
                shape = shape + self.shape_axes
            else:
                shape = shape + (1,) * len(self.axes)
            pix = [np.full(shape, -1, dtype=int) for _ in range(1 + len(self.axes))]

            for idx_img in np.ndindex(self.shape_axes):

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
            pix = tuple([p[p != INVALID_INDEX.int] for p in pix])

        return pix

    def region_mask(self, regions):
        """Create a mask from a given list of regions

        The mask is filled such that a pixel inside the region is filled with
        "True". To invert the mask, e.g. to create a mask with exclusion regions
        the tilde (~) operator can be used (see example below).

        Parameters
        ----------
        regions : str, `~regions.Region` or list of `~regions.Region`
            Region or list of regions (pixel or sky regions accepted).
            A region can be defined as a string ind DS9 format as well.
            See http://ds9.si.edu/doc/ref/region.html for details.

        Returns
        -------
        mask_map : `~gammapy.maps.WcsNDMap` of boolean type
            Boolean region mask

        """
        from gammapy.maps import Map, RegionGeom

        if not self.is_regular:
            raise ValueError("Multi-resolution maps not supported yet")

        # TODO: use spatial coordinates only...
        geom = RegionGeom.from_regions(regions)
        coords = self.get_coord()
        mask = geom.contains(coords)
        return Map.from_geom(self, data=mask)

    def get_coord(
        self, idx=None, flat=False, sparse=False, mode="center", axis_name=None
    ):
        if mode == "edges" and axis_name is None:
            raise ValueError("Mode 'edges' requires axis name to be defined")

        pix = self.get_idx(
            idx=idx, flat=flat, sparse=sparse, mode=mode, axis_name=axis_name
        )
        data = self.pix_to_coord(pix)

        coords = MapCoord.create(
            data=data, frame=self.frame, axis_names=self.axes.names
        )

        return coords

    def contains(self, coords):
        idx = self.coord_to_idx(coords)
        return np.all(np.stack([t != INVALID_INDEX.int for t in idx]), axis=0)

    def solid_angle(self):
        """Solid angle array (`~astropy.units.Quantity` in ``sr``).

        The array has the same dimensionality as ``map.nside``
        since all pixels have the same solid angle.
        """
        import healpy as hp

        return Quantity(hp.nside2pixarea(self.nside), "sr")

    def __repr__(self):
        lon, lat = self.center_skydir.data.lon.deg, self.center_skydir.data.lat.deg
        return (
            f"{self.__class__.__name__}\n\n"
            f"\taxes       : {self.axes_names}\n"
            f"\tshape      : {self.data_shape[::-1]}\n"
            f"\tndim       : {self.ndim}\n"
            f"\tnside      : {self.nside[0]}\n"
            f"\tnested     : {self.nest}\n"
            f"\tframe      : {self.frame}\n"
            f"\tprojection : {self.projection}\n"
            f"\tcenter     : {lon:.1f} deg, {lat:.1f} deg\n"
        )

    def is_allclose(self, other, rtol_axes=1e-6, atol_axes=1e-6):
        """Compare two data IRFs for equivalency

        Parameters
        ----------
        other :  `HpxGeom`
            Geom to compare against
        rtol_axes : float
            Relative tolerance for the axes comparison.
        atol_axes : float
            Relative tolerance for the axes comparison.

        Returns
        -------
        is_allclose : bool
            Whether the geometry is all close.
        """
        if not isinstance(other, self.__class__):
            return TypeError(f"Cannot compare {type(self)} and {type(other)}")

        if self.is_allsky and not other.is_allsky:
            return False

        if self.data_shape != other.data_shape:
            return False

        axes_eq = self.axes.is_allclose(other.axes, rtol=rtol_axes, atol=atol_axes)

        hpx_eq = (
            self.nside == other.nside
            and self.frame == other.frame
            and self.order == other.order
            and self.nest == other.nest
        )

        return axes_eq and hpx_eq

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.is_allclose(other=other)

    def __ne__(self, other):
        return not self.__eq__(other)
