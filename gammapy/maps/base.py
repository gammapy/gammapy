# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import json
import numpy as np
from collections import OrderedDict
from ..extern import six
from astropy.utils.misc import InheritDocstrings
from astropy.io import fits
from .geom import pix_tuple_to_idx, MapCoord

__all__ = [
    'Map',
]


class MapMeta(InheritDocstrings, abc.ABCMeta):
    pass


@six.add_metaclass(MapMeta)
class Map(object):
    """Abstract map class.

    This can represent WCS- or HEALPIX-based maps
    with 2 spatial dimensions and N non-spatial dimensions.

    Parameters
    ----------
    geom : `~gammapy.maps.MapGeom`
        Geometry
    data : `~numpy.ndarray`
        Data array
    meta : `~collections.OrderedDict`
        Dictionary to store meta data.
    """

    def __init__(self, geom, data, meta=None):
        self._geom = geom
        self._data = data
        if meta is None:
            self.meta = OrderedDict()
        else:
            self.meta = OrderedDict(meta)

    @property
    def data(self):
        """Data array (`~numpy.ndarray`)"""
        return self._data

    @data.setter
    def data(self, val):
        if val.shape != self.data.shape:
            raise ValueError('Wrong shape.')
        self._data = val

    @property
    def geom(self):
        """Map geometry (`~gammapy.maps.MapGeom`)"""
        return self._geom

    @classmethod
    def create(cls, **kwargs):
        """Create an empty map object.

        This method accepts generic options
        listed below as well as options for `~HpxMap` and `~WcsMap`
        objects (see `~HpxMap.create` and `~WcsMap.create` for WCS-
        and HPX-specific options).

        Parameters
        ----------
        coordsys : str
            Coordinate system, either Galactic ('GAL') or Equatorial
            ('CEL').
        map_type : {'wcs', 'wcs-sparse', 'hpx', 'hpx-sparse'}
            Map type.  Selects the class that will be used to
            instantiate the map.
        binsz : float or `~numpy.ndarray`
            Pixel size in degrees.
        skydir : `~astropy.coordinates.SkyCoord`
            Coordinate of map center.
        axes : list
            List of `~MapAxis` objects for each non-spatial dimension.
            If None then the map will be a 2D image.
        dtype : str
            Data type, default is ``float32``
        unit : str or `~astropy.units.Unit`
            Data unit.
        meta : `~collections.OrderedDict`
            Dictionary to store meta data.

        Returns
        -------
        map : `~Map`
            Empty map object.
        """
        from .hpxmap import HpxMap
        from .wcsmap import WcsMap

        map_type = kwargs.setdefault('map_type', 'wcs')
        if 'wcs' in map_type.lower():
            return WcsMap.create(**kwargs)
        elif 'hpx' in map_type.lower():
            return HpxMap.create(**kwargs)
        else:
            raise ValueError('Unrecognized map type: {}'.format(map_type))

    @classmethod
    def read(cls, filename, hdu=None, hdu_bands=None, map_type='auto'):
        """Read a map from a FITS file.

        Parameters
        ----------
        filename : str
            Name of the FITS file.
        hdu : str
            Name or index of the HDU with the map data.
        hdu_bands : str
            Name or index of the HDU with the BANDS table.  If not
            defined this will be inferred from the FITS header of the
            map HDU.
        map_type : {'wcs', 'wcs-sparse', 'hpx', 'hpx-sparse', 'auto'}
            Map type.  Selects the class that will be used to
            instantiate the map.  The map type should be consistent
            with the format of the input file.  If map_type is 'auto'
            then an appropriate map type will be inferred from the
            input file.

        Returns
        -------
        map_out : `~Map`
            Map object
        """
        with fits.open(filename) as hdulist:
            map_out = cls.from_hdu_list(hdulist, hdu, hdu_bands, map_type)

        return map_out

    @classmethod
    def from_geom(cls, geom, meta=None, map_type='auto'):
        """Generate an empty map from a `~Geom` instance.

        Parameters
        ----------
        geom : `~MapGeom`
            Map geometry.

        meta : `~collections.OrderedDict`
            Dictionary to store meta data.

        map_type : {'wcs', 'wcs-sparse', 'hpx', 'hpx-sparse', 'auto'}        
            Map type.  Selects the class that will be used to
            instantiate the map.  The map type should be consistent
            with the geometry.  If map_type is 'auto' then an
            appropriate map type will be inferred from type of
            ``geom``.

        Returns
        -------
        map_out : `~Map`
            Map object

        """
        if map_type == 'auto':

            from .hpx import HpxGeom
            from .wcs import WcsGeom
            if isinstance(geom, HpxGeom):
                map_type = 'hpx'
            elif isinstance(geom, WcsGeom):
                map_type = 'wcs'
            else:
                raise ValueError('Unrecognized geom type.')

        cls_out = cls._get_map_cls(map_type)
        map_out = cls_out(geom, meta=meta)
        return map_out

    @classmethod
    def from_hdu_list(cls, hdulist, hdu=None, hdu_bands=None, map_type='auto'):
        if map_type == 'auto':
            map_type = cls._get_map_type(hdulist, hdu)
        cls_out = cls._get_map_cls(map_type)
        map_out = cls_out.from_hdulist(hdulist, hdu=hdu, hdu_bands=hdu_bands)
        return map_out

    @staticmethod
    def _get_meta_from_header(header):
        """Load meta data from a FITS header."""
        if 'META' in header:
            meta = json.loads(header['META'], object_pairs_hook=OrderedDict)
        else:
            meta = {}
        return meta

    @staticmethod
    def _get_map_type(hdu_list, hdu_name):
        """Infer map type from a FITS HDU.

        Only read header, never data, to have good performance.
        """
        if hdu_name is None:
            # Find the header of the first non-empty HDU
            header = hdu_list[0].header
            if header['NAXIS'] == 0:
                header = hdu_list[1].header
        else:
            header = hdu_list[hdu_name].header

        if ('PIXTYPE' in header) and (header['PIXTYPE'] == 'HEALPIX'):
            return 'hpx'
        else:
            return 'wcs'

    @staticmethod
    def _get_map_cls(map_type):
        """Get map class for given `map_type` string.

        This should probably be a registry dict so that users
        can add supported map types to the `gammapy.maps` I/O
        (see e.g. the Astropy table format I/O registry),
        but that's non-trivial to implement without avoiding circular imports.
        """
        if map_type == 'wcs':
            from .wcsnd import WcsNDMap
            return WcsNDMap
        elif map_type == 'wcs-sparse':
            raise NotImplementedError()
        elif map_type == 'hpx':
            from .hpxnd import HpxNDMap
            return HpxNDMap
        elif map_type == 'hpx-sparse':
            from .hpxsparse import HpxSparseMap
            return HpxSparseMap
        else:
            raise ValueError('Unrecognized map type: {!r}'.format(map_type))

    def write(self, filename, **kwargs):
        """Write to a FITS file.

        Parameters
        ----------
        filename : str
            Output file name.
        hdu : str
            Set the name of the image extension.  By default this will
            be set to SKYMAP (for BINTABLE HDU) or PRIMARY (for IMAGE
            HDU).
        hdu_bands : str
            Set the name of the bands table extension.  By default this will
            be set to BANDS.
        conv : str        
            FITS format convention.  By default files will be written
            to the gamma-astro-data-formats (GADF) format.  This
            option can be used to write files that are compliant with
            format conventions required by specific software (e.g. the
            Fermi Science Tools).  Supported conventions are 'gadf',
            'fgst-ccube', 'fgst-ltcube', 'fgst-bexpcube',
            'fgst-template', 'fgst-srcmap', 'fgst-srcmap-sparse',
            'galprop', and 'galprop2'.            
        sparse : bool        
            Sparsify the map by dropping pixels with zero amplitude.
            This option is only compatible with the 'gadf' format.
        """
        hdulist = self.to_hdulist(**kwargs)
        overwrite = kwargs.get('overwrite', True)
        hdulist.writeto(filename, overwrite=overwrite)

    @abc.abstractmethod
    def iter_by_image(self):
        """Iterate over image planes of the map returning a tuple with the image
        array and image plane index.

        Returns
        -------
        val : `~numpy.ndarray`
            Array of image plane values.
        idx : tuple
            Index of image plane.
        """
        pass

    @abc.abstractmethod
    def iter_by_pix(self, buffersize=1):
        """Iterate over elements of the map returning a tuple with values and
        pixel coordinates.

        Parameters
        ----------
        buffersize : int
            Set the size of the buffer.  The map will be returned in
            chunks of the given size.

        Returns
        -------
        val : `~numpy.ndarray`
            Map values.
        pix : tuple
            Tuple of pixel coordinates.
        """
        pass

    @abc.abstractmethod
    def iter_by_coord(self, buffersize=1):
        """Iterate over elements of the map returning a tuple with values and
        map coordinates.

        Parameters
        ----------
        buffersize : int
            Set the size of the buffer.  The map will be returned in
            chunks of the given size.

        Returns
        -------
        val : `~numpy.ndarray`
            Map values.
        coords : tuple
            Tuple of map coordinates.
        """
        pass

    @abc.abstractmethod
    def sum_over_axes(self):
        """Reduce to a 2D image by summing over non-spatial dimensions."""
        pass

    def coadd(self, map_in):
        """Add the contents of ``map_in`` to this map.  This method can be
        used to combine maps containing integral quantities (e.g. counts)
        or differential quantities if the maps have the same binning.

        Parameters
        ----------
        map_in : `~Map`
            Input map.

        """
        # TODO: Check whether geometries are aligned and if so sum the
        # data vectors directly
        idx = map_in.geom.get_idx()
        coords = map_in.geom.get_coord()
        vals = map_in.get_by_idx(idx)
        self.fill_by_coord(coords, vals)

    def reproject(self, geom, order=1, mode='interp'):
        """Reproject this map to a different geometry.

        Parameters
        ----------
        geom : `~MapGeom`
            Geometry of projection.
        mode : {'interp', 'exact'}
            Method for reprojection.  'interp' method interpolates at pixel
            centers.  'exact' method integrates over intersection of pixels.
        order : int or str
            Order of interpolating polynomial (0 = nearest-neighbor, 1 =
            linear, 2 = quadratic, 3 = cubic).

        Returns
        -------
        map : `~Map`
            Reprojected map.
        """
        if geom.ndim == 2 and self.geom.ndim > 2:
            geom = geom.to_cube(self.geom.axes)
        elif geom.ndim != self.geom.ndim:
            raise ValueError('Projection geometry must be 2D or have the '
                             'same number of dimensions as the map.')

        if geom.projection == 'HPX':
            return self._reproject_hpx(geom, mode=mode, order=order)
        else:
            return self._reproject_wcs(geom, mode=mode, order=order)

    @abc.abstractmethod
    def pad(self, pad_width, mode='constant', cval=0, order=1):
        """Pad the spatial dimension of the map by extending the edge of the
        map by the given number of pixels.

        Parameters
        ----------
        pad_width : {sequence, array_like, int}
            Number of pixels padded to the edges of each axis.
        mode : {'edge', 'constant', 'interp'}
            Padding mode.  'edge' pads with the closest edge value.
            'constant' pads with a constant value. 'interp' pads with
            an extrapolated value.
        cval : float
            Padding value when mode='consant'.
        order : int
            Order of interpolation when mode='constant' (0 =
            nearest-neighbor, 1 = linear, 2 = quadratic, 3 = cubic).

        Returns
        -------
        map : `~Map`
            Padded map.

        """
        pass

    @abc.abstractmethod
    def crop(self, crop_width):
        """Crop the spatial dimension of the map by removing a number of
        pixels from the edge of the map.

        Parameters
        ----------
        crop_width : {sequence, array_like, int}
            Number of pixels cropped from the edges of each axis.
            Defined analogously to `pad_with` from `~numpy.pad`.

        Returns
        -------
        map : `~Map`
            Cropped map.

        """
        pass

    @abc.abstractmethod
    def downsample(self, factor, preserve_counts=True):
        """Downsample the spatial dimension of the map by a given factor. 

        Parameters
        ----------
        factor : int
            Downsampling factor.
        preserve_counts : bool
            Preserve the integral over each bin.  This should be true
            if the map is an integral quantity (e.g. counts) and false if
            the map is a differential quantity (e.g. intensity).

        Returns
        -------
        map : `~Map`
            Downsampled map.
        """
        pass

    @abc.abstractmethod
    def upsample(self, factor, order=0, preserve_counts=True):
        """Upsample the spatial dimension of the map by a given factor. 

        Parameters
        ----------
        factor : int
            Upsampling factor.
        order : int
            Order of the interpolation used for upsampling.
        preserve_counts : bool
            Preserve the integral over each bin.  This should be true
            if the map is an integral quantity (e.g. counts) and false if
            the map is a differential quantity (e.g. intensity).

        Returns
        -------
        map : `~Map`
            Upsampled map.

        """
        pass

    def get_by_coord(self, coords):
        """Return map values at the given map coordinates.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.MapCoord`
            `~gammapy.maps.MapCoord` object or tuple of
            coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.

        Returns
        -------
        vals : `~numpy.ndarray`
           Values of pixels in the map.  np.nan used to flag coords
           outside of map.
        """
        coords = MapCoord.create(coords, coordsys=self.geom.coordsys)
        msk = self.geom.contains(coords)
        vals = np.empty(coords.shape, dtype=self.data.dtype)
        coords = coords.apply_mask(msk)
        idx = self.geom.coord_to_idx(coords)
        vals[msk] = self.get_by_idx(idx)
        vals[~msk] = np.nan
        return vals

    def get_by_pix(self, pix):
        """Return map values at the given pixel coordinates.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.
            Pixel indices can be either float or integer type. 

        Returns
        ----------
        vals : `~numpy.ndarray`
           Array of pixel values.  np.nan used to flag coordinates
           outside of map
        """
        # FIXME: Support local indexing here?
        # FIXME: Support slicing?
        pix = [np.array(p, copy=False, ndmin=1) for p in pix]
        pix = np.broadcast_arrays(*pix)
        msk = self.geom.contains_pix(pix)
        vals = np.empty(pix[0].shape, dtype=self.data.dtype)
        pix = tuple([p[msk] for p in pix])
        idx = self.geom.pix_to_idx(pix)
        vals[msk] = self.get_by_idx(idx)
        vals[~msk] = np.nan
        return vals

    @abc.abstractmethod
    def get_by_idx(self, idx):
        """Return map values at the given pixel indices.

        Parameters
        ----------
        idx : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.

        Returns
        ----------
        vals : `~numpy.ndarray`
           Array of pixel values.
           np.nan used to flag coordinate outside of map
        """
        pass

    @abc.abstractmethod
    def interp_by_coord(self, coords, interp=None):
        """Interpolate map values at the given map coordinates.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.MapCoord`
            `~gammapy.maps.MapCoord` object or tuple of
            coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.

        interp : {None, 'nearest', 'linear', 'cubic', 0, 1, 2, 3}
            Method to interpolate data values.  By default no
            interpolation is performed and the return value will be
            the amplitude of the pixel encompassing the given
            coordinate.  Integer values can be used in lieu of strings
            to choose the interpolation method of the given order
            (0='nearest', 1='linear', 2='quadratic', 3='cubic').  Note
            that only 'nearest' and 'linear' methods are supported for
            all map types.

        Returns
        -------
        vals : `~numpy.ndarray`
            Interpolated pixel values.
        """
        pass

    @abc.abstractmethod
    def interp_by_pix(self, pix, interp=None):
        """Interpolate map values at the given pixel coordinates.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinate arrays for each dimension of the
            map.  Tuple should be ordered as (p_lon, p_lat, p_0, ...,
            p_n) where p_i are pixel coordinates for non-spatial
            dimensions of the map.

        interp : {None, 'nearest', 'linear', 'cubic', 0, 1, 2, 3}
            Method to interpolate data values.  By default no
            interpolation is performed and the return value will be
            the amplitude of the pixel encompassing the given
            coordinate.  Integer values can be used in lieu of strings
            to choose the interpolation method of the given order
            (0='nearest', 1='linear', 2='quadratic', 3='cubic').  Note
            that only 'nearest' and 'linear' methods are supported for
            all map types.

        Returns
        -------
        vals : `~numpy.ndarray`
            Interpolated pixel values.
        """
        pass

    def fill_by_coord(self, coords, weights=None):
        """Fill pixels at the given map coordinates with values in `weights`
        vector.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.MapCoord`
            `~gammapy.maps.MapCoord` object or tuple of
            coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.

        weights : `~numpy.ndarray`
            Weights vector. If None then a unit weight will be assumed
            for each element in `coords`.
        """
        idx = self.geom.coord_to_idx(coords)
        self.fill_by_idx(idx, weights)

    def fill_by_pix(self, pix, weights=None):
        """Fill pixels at the given pixel coordinates with values in `weights`
        vector.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.
            Pixel indices can be either float or integer type.  Float
            indices will be rounded to the nearest integer.

        weights : `~numpy.ndarray`
            Weights vector. If None then a unit weight will be assumed
            for each element in `pix`.
        """
        idx = pix_tuple_to_idx(pix)
        return self.fill_by_idx(idx, weights=weights)

    @abc.abstractmethod
    def fill_by_idx(self, idx, weights=None):
        """Fill pixels at the given pixel indices with values in `weights`
        vector.

        Parameters
        ----------
        idx : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.

        weights : `~numpy.ndarray`
            Weights vector. If None then a unit weight will be assumed
            for each element in `idx`.
        """
        pass

    def set_by_coord(self, coords, vals):
        """Set pixels at the given map coordinates to the values in `vals`
        vector.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.MapCoord`
            `~gammapy.maps.MapCoord` object or tuple of
            coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.

        vals : `~numpy.ndarray`
            Values vector.  Pixels at `coords` will be set to these values.
        """
        idx = self.geom.coord_to_pix(coords)
        self.set_by_pix(idx, vals)

    def set_by_pix(self, pix, vals):
        """Set pixels at the given pixel coordinates to the values in `vals`
        vector.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.
            Pixel indices can be either float or integer type.  Float
            indices will be rounded to the nearest integer.

        vals : `~numpy.ndarray`
            Values vector. Pixels at `pix` will be set to these values.
        """
        idx = pix_tuple_to_idx(pix)
        return self.set_by_idx(idx, vals)

    @abc.abstractmethod
    def set_by_idx(self, idx, vals):
        """Set pixels at the given pixel indices to the values in `vals`
        vector.

        Parameters
        ----------
        idx : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.

        vals : `~numpy.ndarray`
            Values vector. Pixels at `idx` will be set to these values.
        """
        pass
