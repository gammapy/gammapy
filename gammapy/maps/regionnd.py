import numpy as np
from astropy import units as u
from .base import Map
from .geom import pix_tuple_to_idx
from .utils import INVALID_INDEX
from .utils import interp_to_order
from astropy.visualization import quantity_support
from gammapy.utils.interpolation import ScaledRegularGridInterpolator


class RegionNDMap(Map):
    """Region ND map

    Parameters
    ----------
    geom : `~gammapy.maps.RegionGeom`
        WCS geometry object.
    data : `~numpy.ndarray`
        Data array. If none then an empty array will be allocated.
    dtype : str, optional
        Data type, default is float32
    meta : `dict`
        Dictionary to store meta data.
    unit : str or `~astropy.units.Unit`
        The map unit
    """

    def __init__(self, geom, data=None, dtype="float32", meta=None, unit=""):
        if data is None:
            data = np.zeros(geom.data_shape, dtype=dtype)

        self.geom = geom
        self.data = data
        self.meta = meta
        self.unit = u.Unit(unit)

    def plot(self, ax=None):
        """Plot map.
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()

        if len(self.geom.axes) > 1:
            raise TypeError("Use `.plot_interactive()` if more the one extra axis is present.")

        axis = self.geom.axes[0]
        with quantity_support():
            ax.plot(axis.center, self.quantity.squeeze())

        if axis.interp == "log":
            ax.set_xscale("log")

    @classmethod
    def create(cls, region, **kwargs):
        """
        """
        if isinstance(region, str):
            region = None

        return cls(region, **kwargs)

    def downsample(self, factor, axis=None):
        raise NotImplementedError

    def fill_by_idx(self, idx, weights=None):
        idx = pix_tuple_to_idx(idx)

        msk = np.all(np.stack([t != INVALID_INDEX.int for t in idx]), axis=0)
        idx = [t[msk] for t in idx]

        if weights is not None:
            if isinstance(weights, u.Quantity):
                weights = weights.to_value(self.unit)
            weights = weights[msk]

        idx = np.ravel_multi_index(idx, self.data.T.shape)
        idx, idx_inv = np.unique(idx, return_inverse=True)
        weights = np.bincount(idx_inv, weights=weights).astype(self.data.dtype)
        self.data.T.flat[idx] += weights

    def get_by_idx(self, idxs):
        return self.data[idxs[::-1]]

    def interp_by_coord(self):
        raise NotImplementedError

    def interp_by_pix(self, pix, interp=None, fill_value=None):
        method_lookup = {0: "nearest", 1: "linear"}
        order = interp_to_order(interp)
        try:
            method = method_lookup[order]
        except KeyError:
            raise ValueError(f"Invalid interpolation order: {order!r}")

        grid_pix = [np.arange(n, dtype=float) for n in self.data.shape[::-1]]

        if np.any(np.isfinite(self.data)):
            data = self.data.copy().T
            data[~np.isfinite(data)] = 0.0
        else:
            data = self.data.T

        fn = ScaledRegularGridInterpolator(
            grid_pix, data, fill_value=fill_value, bounds_error=False, method=method
        )
        return fn(tuple(pix), clip=False)

    def set_by_idx(self, idx, value):
        self.data[idx[::-1]] = value

    def upsample(self, factor, axis=None):
        raise NotImplementedError

    @staticmethod
    def read(cls, filename):
        pass

    def write(self, filename):
        pass

    def to_hdulist(self):
        pass

    @classmethod
    def from_hdulist(cls):
        pass

    def crop(self):
        raise NotImplementedError("Crop is not supported by RegionNDMap")

    def pad(self):
        raise NotImplementedError("Pad is not supported by RegionNDMap")

    def sum_over_axes(self):
        axis = tuple(range(self.data.ndim - 2))
        geom = self.geom.to_image()
        if keepdims:
            for ax in self.geom.axes:
                geom = geom.to_cube([ax.squash()])
        data = np.nansum(self.data, axis=axis, keepdims=keepdims)
        # TODO: summing over the axis can change the unit, handle this correctly
        return self._init_copy(geom=geom, data=data)
        raise NotImplementedError

    def get_image_by_coord(self):
        raise NotImplementedError

    def get_image_by_idx(self):
        raise NotImplementedError

    def get_image_by_pix(self):
        raise NotImplementedError

    def stack(self, other):
        self.data += other.data