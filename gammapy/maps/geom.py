# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import copy
import inspect
import logging
import numpy as np
from astropy import units as u
from astropy.io import fits
from .io import find_bands_hdu, find_hdu
from .utils import INVALID_INDEX

__all__ = ["Geom"]

log = logging.getLogger(__name__)


def get_shape(param):
    if param is None:
        return tuple()

    if not isinstance(param, tuple):
        param = [param]

    return max([np.array(p, ndmin=1).shape for p in param])


def pix_tuple_to_idx(pix):
    """Convert a tuple of pixel coordinate arrays to a tuple of pixel indices.

    Pixel coordinates are rounded to the closest integer value.

    Parameters
    ----------
    pix : tuple
        Tuple of pixel coordinates with one element for each dimension

    Returns
    -------
    idx : `~numpy.ndarray`
        Array of pixel indices
    """
    idx = []
    for p in pix:
        p = np.array(p, ndmin=1)
        if np.issubdtype(p.dtype, np.integer):
            idx += [p]
        else:
            p_idx = np.rint(p).astype(int)
            p_idx[~np.isfinite(p)] = INVALID_INDEX.int
            idx += [p_idx]

    return tuple(idx)


class Geom(abc.ABC):
    """Map geometry base class.

    See also: `~gammapy.maps.WcsGeom` and `~gammapy.maps.HpxGeom`
    """

    # workaround for the lru_cache pickle issue
    # see e.g. https://github.com/cloudpipe/cloudpickle/issues/178
    def __getstate__(self):
        state = self.__dict__.copy()
        for key, value in state.items():
            func = getattr(value, "__wrapped__", None)
            if func is not None:
                state[key] = func
        return state

    @property
    @abc.abstractmethod
    def data_shape(self):
        """Shape of the Numpy data array matching this geometry."""
        pass

    def data_nbytes(self, dtype="float32"):
        """Estimate memory usage in megabytes of the Numpy data array
        matching this geometry depending on the given type.

        Parameters
        ----------
        dtype : data-type
            The desired data-type for the array. Default is "float32"

        Returns
        -------
        memory : `~astropy.units.Quantity`
            Estimated memory usage in megabytes (MB)
        """
        return (np.empty(self.data_shape, dtype).nbytes * u.byte).to("MB")

    @property
    @abc.abstractmethod
    def is_allsky(self):
        pass

    @property
    @abc.abstractmethod
    def center_coord(self):
        pass

    @property
    @abc.abstractmethod
    def center_pix(self):
        pass

    @property
    @abc.abstractmethod
    def center_skydir(self):
        pass

    @classmethod
    def from_hdulist(cls, hdulist, hdu=None, hdu_bands=None):
        """Load a geometry object from a FITS HDUList.

        Parameters
        ----------
        hdulist :  `~astropy.io.fits.HDUList`
            HDU list containing HDUs for map data and bands.
        hdu : str
            Name or index of the HDU with the map data.
        hdu_bands : str
            Name or index of the HDU with the BANDS table.  If not
            defined this will be inferred from the FITS header of the
            map HDU.

        Returns
        -------
        geom : `~Geom`
            Geometry object.
        """
        if hdu is None:
            hdu = find_hdu(hdulist)
        else:
            hdu = hdulist[hdu]

        if hdu_bands is None:
            hdu_bands = find_bands_hdu(hdulist, hdu)

        if hdu_bands is not None:
            hdu_bands = hdulist[hdu_bands]

        return cls.from_header(hdu.header, hdu_bands)

    def to_bands_hdu(self, hdu_bands=None, format="gadf"):
        table_hdu = self.axes.to_table_hdu(format=format, hdu_bands=hdu_bands)
        cols = table_hdu.columns.columns
        cols.extend(self._make_bands_cols())
        return fits.BinTableHDU.from_columns(
            cols, header=table_hdu.header, name=table_hdu.name
        )

    @abc.abstractmethod
    def _make_bands_cols(self):
        pass

    @abc.abstractmethod
    def get_idx(self, idx=None, local=False, flat=False):
        """Get tuple of pixel indices for this geometry.

        Returns all pixels in the geometry by default. Pixel indices
        for a single image plane can be accessed by setting ``idx``
        to the index tuple of a plane.

        Parameters
        ----------
        idx : tuple, optional
            A tuple of indices with one index for each non-spatial
            dimension.  If defined only pixels for the image plane with
            this index will be returned.  If none then all pixels
            will be returned.
        local : bool
            Flag to return local or global pixel indices.  Local
            indices run from 0 to the number of pixels in a given
            image plane.
        flat : bool, optional
            Return a flattened array containing only indices for
            pixels contained in the geometry.

        Returns
        -------
        idx : tuple
            Tuple of pixel index vectors with one vector for each
            dimension.
        """
        pass

    @abc.abstractmethod
    def get_coord(self, idx=None, flat=False):
        """Get the coordinate array for this geometry.

        Returns a coordinate array with the same shape as the data
        array.  Pixels outside the geometry are set to NaN.
        Coordinates for a single image plane can be accessed by
        setting ``idx`` to the index tuple of a plane.

        Parameters
        ----------
        idx : tuple, optional
            A tuple of indices with one index for each non-spatial
            dimension.  If defined only coordinates for the image
            plane with this index will be returned.  If none then
            coordinates for all pixels will be returned.
        flat : bool, optional
            Return a flattened array containing only coordinates for
            pixels contained in the geometry.

        Returns
        -------
        coords : tuple
            Tuple of coordinate vectors with one vector for each
            dimension.
        """
        pass

    @abc.abstractmethod
    def coord_to_pix(self, coords):
        """Convert map coordinates to pixel coordinates.

        Parameters
        ----------
        coords : tuple
            Coordinate values in each dimension of the map.  This can
            either be a tuple of numpy arrays or a MapCoord object.
            If passed as a tuple then the ordering should be
            (longitude, latitude, c_0, ..., c_N) where c_i is the
            coordinate vector for axis i.

        Returns
        -------
        pix : tuple
            Tuple of pixel coordinates in image and band dimensions.
        """
        pass

    def coord_to_idx(self, coords, clip=False):
        """Convert map coordinates to pixel indices.

        Parameters
        ----------
        coords : tuple or `~MapCoord`
            Coordinate values in each dimension of the map.  This can
            either be a tuple of numpy arrays or a MapCoord object.
            If passed as a tuple then the ordering should be
            (longitude, latitude, c_0, ..., c_N) where c_i is the
            coordinate vector for axis i.
        clip : bool
            Choose whether to clip indices to the valid range of the
            geometry.  If false then indices for coordinates outside
            the geometry range will be set -1.

        Returns
        -------
        pix : tuple
            Tuple of pixel indices in image and band dimensions.
            Elements set to -1 correspond to coordinates outside the
            map.
        """
        pix = self.coord_to_pix(coords)
        return self.pix_to_idx(pix, clip=clip)

    @abc.abstractmethod
    def pix_to_coord(self, pix):
        """Convert pixel coordinates to map coordinates.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinates.

        Returns
        -------
        coords : tuple
            Tuple of map coordinates.
        """
        pass

    @abc.abstractmethod
    def pix_to_idx(self, pix, clip=False):
        """Convert pixel coordinates to pixel indices.

        Returns -1 for pixel coordinates that lie outside of the map.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinates.
        clip : bool
            Choose whether to clip indices to the valid range of the
            geometry.  If false then indices for coordinates outside
            the geometry range will be set -1.

        Returns
        -------
        idx : tuple
            Tuple of pixel indices.
        """
        pass

    @abc.abstractmethod
    def contains(self, coords):
        """Check if a given map coordinate is contained in the geometry.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.MapCoord`
            Tuple of map coordinates.

        Returns
        -------
        containment : `~numpy.ndarray`
            Bool array.
        """
        pass

    def contains_pix(self, pix):
        """Check if a given pixel coordinate is contained in the geometry.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinates.

        Returns
        -------
        containment : `~numpy.ndarray`
            Bool array.
        """
        idx = self.pix_to_idx(pix)
        return np.all(np.stack([t != INVALID_INDEX.int for t in idx]), axis=0)

    def slice_by_idx(self, slices):
        """Create a new geometry by slicing the non-spatial axes.

        Parameters
        ----------
        slices : dict
            Dict of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.

        Returns
        -------
        geom : `~Geom`
            Sliced geometry.
        """
        axes = self.axes.slice_by_idx(slices)
        return self._init_copy(axes=axes)

    def rename_axes(self, names, new_names):
        """Rename axes contained in the geometry

        Parameters
        ----------
        names : list or str
            Names of the axes.
        new_names : list or str
            New names of the axes (list must be of same length than `names`).

        Returns
        -------
        geom : `~Geom`
            Renamed geometry.
        """
        axes = self.axes.rename_axes(names=names, new_names=new_names)
        return self._init_copy(axes=axes)

    @property
    def as_energy_true(self):
        """If the geom contains an energy axis rename it to energy true"""
        return self.rename_axes(names="energy", new_names="energy_true")

    @property
    def has_energy_axis(self):
        """Whether geom has an energy axis"""
        return ("energy" in self.axes.names) ^ ("energy_true" in self.axes.names)

    @abc.abstractmethod
    def to_image(self):
        """Create 2D image geometry (drop non-spatial dimensions).

        Returns
        -------
        geom : `~Geom`
            Image geometry.
        """
        pass

    @abc.abstractmethod
    def to_cube(self, axes):
        """Append non-spatial axes to create a higher-dimensional geometry.

        This will result in a new geometry with
        N+M dimensions where N is the number of current dimensions and
        M is the number of axes in the list.

        Parameters
        ----------
        axes : list
            Axes that will be appended to this geometry.

        Returns
        -------
        geom : `~Geom`
            Map geometry.
        """
        pass

    def squash(self, axis_name):
        """Squash geom axis.

        Parameters
        ----------
        axis_name : str
            Axis to squash.

        Returns
        -------
        geom : `Geom`
            Geom with squashed axis.
        """
        axes = self.axes.squash(axis_name=axis_name)
        return self.to_image().to_cube(axes=axes)

    def drop(self, axis_name):
        """Drop an axis from the geom.

        Parameters
        ----------
        axis_name : str
            Name of the axis to remove.

        Returns
            -------
        geom : `Geom`
            New geom with the axis removed.
        """
        axes = self.axes.drop(axis_name=axis_name)
        return self.to_image().to_cube(axes=axes)

    def pad(self, pad_width, axis_name):
        """
        Pad the geometry at the edges.

        Parameters
        ----------
        pad_width : {sequence, array_like, int}
            Number of values padded to the edges of each axis.
        axis_name : str
            Name of the axis to pad.

        Returns
        -------
        geom : `~Geom`
            Padded geometry.
        """
        if axis_name is None:
            return self._pad_spatial(pad_width)
        else:
            axes = self.axes.pad(axis_name=axis_name, pad_width=pad_width)
            return self.to_image().to_cube(axes)

    @abc.abstractmethod
    def _pad_spatial(self, pad_width):
        pass

    @abc.abstractmethod
    def crop(self, crop_width):
        """
        Crop the geometry at the edges.

        Parameters
        ----------
        crop_width : {sequence, array_like, int}
            Number of values cropped from the edges of each axis.

        Returns
        -------
        geom : `~Geom`
            Cropped geometry.
        """
        pass

    @abc.abstractmethod
    def downsample(self, factor, axis_name):
        """Downsample the spatial dimension of the geometry by a given factor.

        Parameters
        ----------
        factor : int
            Downsampling factor.
        axis_name : str
            Axis to downsample.

        Returns
        -------
        geom : `~Geom`
            Downsampled geometry.

        """
        pass

    @abc.abstractmethod
    def upsample(self, factor, axis_name=None):
        """Upsample the spatial dimension of the geometry by a given factor.

        Parameters
        ----------
        factor : int
            Upsampling factor.
        axis_name : str
            Axis to upsample.

        Returns
        -------
        geom : `~Geom`
            Upsampled geometry.

        """
        pass

    def resample_axis(self, axis):
        """Resample geom to a new axis binning.

        This method groups the existing bins into a new binning.

        Parameters
        ----------
        axis : `MapAxis`
            New map axis.

        Returns
        -------
        map : `Geom`
            Geom with resampled axis.
        """
        axes = self.axes.resample(axis=axis)
        return self._init_copy(axes=axes)

    def replace_axis(self, axis):
        """Replace axis with a new one.

        Parameters
        ----------
        axis : `MapAxis`
            New map axis.

        Returns
        -------
        map : `Geom`
            Geom with replaced axis.
        """
        axes = self.axes.replace(axis=axis)
        return self._init_copy(axes=axes)

    @abc.abstractmethod
    def solid_angle(self):
        """Solid angle (`~astropy.units.Quantity` in ``sr``)."""
        pass

    @property
    def is_image(self):
        """Whether the geom is an image without extra dimensions."""
        if self.axes is None:
            return True
        return len(self.axes) == 0

    @property
    def is_flat(self):
        """Whether the geom non spatial axes have length 1, equivalent to an image."""
        if self.is_image:
            return True
        else:
            valid = True
            for axis in self.axes:
                valid = valid and (axis.nbin == 1)
            return valid

    def _init_copy(self, **kwargs):
        """Init map geom instance by copying missing init arguments from self."""
        argnames = inspect.getfullargspec(self.__init__).args
        argnames.remove("self")

        for arg in argnames:
            value = getattr(self, "_" + arg)
            kwargs.setdefault(arg, copy.deepcopy(value))

        return self.__class__(**kwargs)

    def copy(self, **kwargs):
        """Copy and overwrite given attributes.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to overwrite in the map geometry constructor.

        Returns
        -------
        copy : `Geom`
            Copied map geometry.
        """
        return self._init_copy(**kwargs)

    def energy_mask(self, energy_min=None, energy_max=None, round_to_edge=False):
        """Create a mask for a given energy range.

        The energy bin must be fully contained to be included in the mask.

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy range

        Returns
        -------
        mask : `~numpy.ndarray`
            Energy mask
        """
        from . import Map

        # get energy axes and values
        energy_axis = self.axes["energy"]

        if round_to_edge:
            energy_min, energy_max = energy_axis.round([energy_min, energy_max])

        # TODO: make this more general
        shape = (-1, 1) if self.is_hpx else (-1, 1, 1)
        energy_edges = energy_axis.edges.reshape(shape)

        # set default values
        energy_min = energy_min if energy_min is not None else energy_edges[0]
        energy_max = energy_max if energy_max is not None else energy_edges[-1]

        mask = (energy_edges[:-1] >= energy_min) & (energy_edges[1:] <= energy_max)
        data = np.broadcast_to(mask, shape=self.data_shape)
        return Map.from_geom(geom=self, data=data, dtype=data.dtype)
