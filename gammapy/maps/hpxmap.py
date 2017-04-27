# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import numpy as np
from astropy.extern import six
from astropy.io import fits
from astropy.coordinates import SkyCoord
from .hpx import HpxToWcsMapping

__all__ = [
    'HpxMap',
]


@six.add_metaclass(abc.ABCMeta)
class HpxMap(object):
    """Base class for HEALPIX map classes.

    Parameters
    ----------
    data : `~numpy.ndarray`
        TODO
    """

    def __init__(self, hpx, data):
        self._data = data
        self._hpx = hpx
        self._wcs2d = None
        self._hpx2wcs = None

    @property
    def hpx(self):
        """TODO"""
        return self._hpx

    @property
    def data(self):
        """TODO"""
        return self._data

    @classmethod
    def read(cls, filename, **kwargs):
        """Read from a FITS file.

        Parameters
        ----------
        filename : str
            File name
        hdu : str
            Name of the HDU with the map data
        ebounds : str
            Name of the HDU with the energy bin data

        Returns
        -------
        hpx_map : `HpxMap`
            Map object
        """
        hdulist = fits.open(filename)
        return cls.from_hdulist(hdulist, **kwargs)

    @classmethod
    def from_hdulist(cls, hdulist, **kwargs):
        """Make a HpxMap object from a FITS HDUList.

        Parameters
        ----------
        hdulist :  `~astropy.io.fits.HDUList`
            HDU list containing HDUs for map data and bands/ebounds
        hdu : str
            Name of the HDU with the map data
        ebounds : str
            Name of the HDU with the energy bin data

        Returns
        -------
        hpx_map : `HpxMap`
            Map object
        """
        extname = kwargs.get('hdu', 'SKYMAP')
        ebins = fits_utils.find_and_read_ebins(hdulist)
        return cls.from_hdu(hdulist[extname], ebins)

    def to_image_hdu(self, name=None, **kwargs):
        """TODO"""
        kwargs['extname'] = name
        return self.hpx.make_hdu(self.counts, **kwargs)

    def make_wcs_from_hpx(self, sum_ebins=False, proj='CAR', oversample=2,
                          normalize=True):
        """Make a WCS object and convert HEALPIX data into WCS projection.

        NOTE: this re-calculates the mapping, if you have already
        calculated the mapping it is much faster to use
        convert_to_cached_wcs() instead

        Parameters
        ----------
        sum_ebins  : bool
           sum energy bins over energy bins before reprojecting
        proj : str
           WCS-projection
        oversample : int
           Oversampling factor for WCS map
        normalize  : bool
           True -> preserve integral by splitting HEALPIX values between bins

        Returns
        -------
        wcs : `~gammapy.maps.wcs.WCSGeom`
            WCS geometry
        wcs_data : `~numpy.ndarray`
            WCS data
        """
        self._wcs_proj = proj
        self._wcs_oversample = oversample
        self._wcs_2d = self.hpx.make_wcs(2, proj=proj, oversample=oversample)
        self._hpx2wcs = HpxToWcsMapping(self.hpx, self._wcs_2d)
        wcs, wcs_data = self.convert_to_cached_wcs(self.counts, sum_ebins,
                                                   normalize)
        return wcs, wcs_data

    @abc.abstractmethod
    def to_cached_wcs(self, hpx_in, sum_ebins=False, normalize=True):
        """Make a WCS object and convert HEALPIX data into WCS projection.

        Parameters
        ----------
        hpx_in : `~numpy.ndarray`
            HEALPIX input data
        sum_ebins : bool
            Sum energy bins over energy bins before reprojecting
        normalize : bool
            True -> preserve integral by splitting HEALPIX values between bins

        Returns
        -------
        (WCS object, np.ndarray() with reprojected data)
        """
        pass

    def get_pixel_skydirs(self):
        """Get a list of sky coordinates for the centers of every pixel. """
        sky_coords = self._hpx.get_sky_coords()
        frame = 'galactic' if self.hpx.coordsys == 'GAL' else 'icrs'
        return SkyCoord(sky_coords[0], sky_coords[1], frame=frame, unit='deg')

    @abc.abstractmethod
    def sum_over_axes(self):
        """Reduce to a map by dropping non-spatial dimensions."""
        pass

    @abc.abstractmethod
    def get_by_coord(self, coords, interp=None):
        """Return map values at the given map coordinates.

        Parameters
        ----------
        coords : tuple
            TODO

        Returns
        -------
        vals : `~numpy.ndarray`
           Values of pixels in the flattened map.
           np.nan used to flag coords outside of map
        """
        pass

    @abc.abstractmethod
    def get_by_pix(self, coords, interp=None):
        """Return map values at the given pixel coordinates.

        Parameters
        ----------
        coords : tuple
            TODO
        Returns
        ----------
        vals : `~numpy.ndarray`
           Values of pixels in the flattened map
           np.nan used to flag coords outside of map
        """
        pass

    def swap_scheme(self):
        """TODO.
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
        return HpxMap(data_out, hpx_out)

    def ud_grade(self, order, preserve_counts=False):
        """TODO.
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
            new_data = hp.pixelfunc.ud_grade(self.counts,
                                             nside_out=new_hpx.nside,
                                             order_in=new_hpx.ordering,
                                             order_out=ew_hpx.ordering,
                                             power=power)
        else:
            new_data = np.vstack([hp.pixelfunc.ud_grade(self.counts[i],
                                                        nside_out=new_hpx.nside,
                                                        order_in=new_hpx.ordering,
                                                        order_out=new_hpx.ordering,
                                                        power=power) for i in range(shape[0])])
        return HpxMap(new_data, new_hpx)

    def make_hdu(self, **kwargs):
        """Make a FITS HDU with input data.

        Parameters
        ----------
        extname : str
            The HDU extension name.
        colbase : str
            The prefix for column names
        """
        data = self.data
        shape = data.shape
        extname = kwargs.get('extname', self.conv.extname)
        convname = kwargs.get('convname', self.conv.convname)
        header = self.hpx.make_header()

        if shape[-1] != self._npix:
            raise ValueError('Size of data array does not match number of pixels')
        cols = []
        if self._region:
            header['INDXSCHM'] = 'EXPLICIT'
            cols.append(fits.Column('PIX', 'J', array=self._ipix))
        else:
            header['INDXSCHM'] = 'IMPLICIT'

        if convname == 'FGST_SRCMAP_SPARSE':
            nonzero = data.nonzero()
            nfilled = len(nonzero[0])
            if len(shape) == 1:
                nonzero = nonzero[0]
                cols.append(fits.Column(
                    'KEY', '{}J'.format(nfilled),
                    array=nonzero.reshape(1, nfilled)),
                )
                cols.append(fits.Column(
                    'VALUE', '{}E'.format(nfilled),
                    array=data[nonzero].astype(float).reshape(1, nfilled)),
                )
            elif len(shape) == 2:
                nonzero = self._npix * nonzero[0] + nonzero[1]
                cols.append(fits.Column(
                    'KEY', '{}J'.format(nfilled),
                    array=nonzero.reshape(1, nfilled)),
                )
                cols.append(fits.Column(
                    'VALUE', '{}E'.format(nfilled),
                    array=data.flat[nonzero].astype(float).reshape(1, nfilled)),
                )
            else:
                raise Exception('HPX.write_fits only handles 1D and 2D maps')

        else:
            if len(shape) == 1:
                cols.append(fits.Column(self.conv.colname(
                    indx=i + self.conv.firstcol), 'E', array=data.astype(float)))
            elif len(shape) == 2:
                for i in range(shape[0]):
                    cols.append(fits.Column(self.conv.colname(
                        indx=i + self.conv.firstcol), 'E', array=data[i].astype(float)))
            else:
                raise Exception('HPX.write_fits only handles 1D and 2D maps')

        hdu = fits.BinTableHDU.from_columns(cols, header=header, name=extname)

        return hdu
