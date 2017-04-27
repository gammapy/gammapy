# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import abc
import numpy as np
import healpy as hp
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.interpolation import map_coordinates
from astropy.extern import six
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from .wcs import wcs_to_coords, wcs_to_axes
from .hpx import HPXGeom, HpxToWcsMapping


@six.add_metaclass(abc.ABCMeta)
class HpxMap(object):
    """Base class for HEALPix map classes.

    Parameters
    ----------
    data : `~numpy.ndarray`
    """
    
    def __init__(self, hpx, data):
        print('here')
        #MapBase.__init__(self, data)
        self._data = data
        self._hpx = hpx
        self._wcs2d = None
        self._hpx2wcs = None

    @property
    def hpx(self):
        return self._hpx

    @property
    def data(self):
        return self._data

    @classmethod
    def read(cls, fitsfile, **kwargs):
        """Read from a FITS file.

        Parameters
        ----------
        filename : str
            File name.

        hdu : str
            The name of the HDU with the map data.

        ebounds : str
            The name of the HDU with the energy bin data.

        Returns
        -------
        hpx_map : `HpxMap`
            Map object.
        """
        hdulist = fits.open(fitsfile)
        return cls.from_hdulist(hdulist, **kwargs)

    @classmethod
    def from_hdulist(cls, hdulist, **kwargs):
        """Creates and returns an HpxMap object from a FITS HDUList.

        Parameters
        ----------
        hdulist :  `~astropy.io.fits.HDUList`
            An HDUList containing HDUs for map data and bands/ebounds.

        hdu : str
            The name of the HDU with the map data.

        ebounds : str
            The name of the HDU with the energy bin data

        Returns
        -------
        hpx_map : `HpxMap`
            Map object.
        """
        extname = kwargs.get('hdu', 'SKYMAP')
        ebins = fits_utils.find_and_read_ebins(hdulist)
        return cls.from_hdu(hdulist[extname], ebins)
    
    def to_image_hdu(self, name=None, **kwargs):
        kwargs['extname'] = name
        return self.hpx.make_hdu(self.counts, **kwargs)

    def make_wcs_from_hpx(self, sum_ebins=False, proj='CAR', oversample=2,
                          normalize=True):
        """Make a WCS object and convert HEALPix data into WCS projection

        NOTE: this re-calculates the mapping, if you have already
        calculated the mapping it is much faster to use
        convert_to_cached_wcs() instead

        Parameters
        ----------
        sum_ebins  : bool
           sum energy bins over energy bins before reprojecting

        proj       : str
           WCS-projection

        oversample : int
           Oversampling factor for WCS map

        normalize  : bool
           True -> perserve integral by splitting HEALPix values between bins

        Returns
        -------
        wcs : `~gammapy.maps.wcs.WCSGeom`
        
        wcs_data : `~numpy.ndarray`
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
        """ Make a WCS object and convert HEALPix data into WCS projection

        Parameters
        ----------
        hpx_in     : `~numpy.ndarray`
           HEALPix input data
        sum_ebins  : bool
           sum energy bins over energy bins before reprojecting
        normalize  : bool
           True -> perserve integral by splitting HEALPix values between bins

        returns (WCS object, np.ndarray() with reprojected data)
        """
        return

    def get_pixel_skydirs(self):
        """Get a list of sky coordinates for the centers of every pixel. """
        sky_coords = self._hpx.get_sky_coords()
        if self.hpx.coordsys == 'GAL':
            frame = Galactic
        else:
            frame = ICRS
        return SkyCoord(sky_coords[0], sky_coords[1], frame=frame, unit='deg')

    @abc.abstractmethod
    def sum_over_axes(self):
        """ Reduce to a map by droppping non-spatial dimensions."""
        return

    @abc.abstractmethod
    def get_by_coord(self, coords, interp=None):
        """Return map values at the given map coordinates.

        Parameters
        ----------
        coords : tuple

        Returns
        ----------
        vals : `~numpy.ndarray`
           Values of pixels in the flattened map, np.nan used to flag
           coords outside of map

        """
        return

    @abc.abstractmethod
    def get_by_pix(self, coords, interp=None):
        """Return map values at the given pixel coordinates.

        Parameters
        ----------
        coords : tuple

        Returns
        ----------
        vals : `~numpy.ndarray`
           Values of pixels in the flattened map, np.nan used to flag
           coords outside of map

        """
        return

    @abc.abstractmethod
    def interpolate(self, coords):
        """Interpolate map values at the given coordinates.

        Parameters
        ----------
        coords : tuple

        """
        return

    def swap_scheme(self):
        """
        """
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
        """
        """
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
