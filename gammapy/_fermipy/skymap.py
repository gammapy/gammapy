# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import copy
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from . import wcs_utils
from . import hpx_utils
from . import fits_utils
from .hpx_utils import HPX, HpxToWcsMapping


def make_coadd_map(maps, proj, shape):
    if isinstance(proj, WCS):
        return make_coadd_wcs(maps, proj, shape)
    elif isinstance(proj, HPX):
        return make_coadd_hpx(maps, proj, shape)
    else:
        raise Exception("Can't co-add map of unknown type %s" % type(proj))


def make_coadd_wcs(maps, wcs, shape):
    data = np.zeros(shape)
    axes = wcs_utils.wcs_to_axes(wcs, shape)

    for m in maps:
        c = wcs_utils.wcs_to_coords(m.wcs, m.counts.shape)
        o = np.histogramdd(c.T, bins=axes[::-1], weights=np.ravel(m.counts))[0]
        data += o

    return Map(data, copy.deepcopy(wcs))


def make_coadd_hpx(maps, hpx, shape):
    data = np.zeros(shape)
    axes = hpx_utils.hpx_to_axes(hpx, shape)
    for m in maps:
        c = hpx_utils.hpx_to_coords(m.hpx, m.counts.shape)
        o = np.histogramdd(c.T, bins=axes, weights=np.ravel(m.counts))[0]
        data += o
    return HpxMap(data, copy.deepcopy(hpx))


def read_map_from_fits(fitsfile, extname=None):
    """
    """
    proj, f, hdu = fits_utils.read_projection_from_fits(fitsfile, extname)
    if isinstance(proj, WCS):
        m = Map(hdu.data, proj)
    elif isinstance(proj, HPX):
        m = HpxMap.create_from_hdu(hdu, proj.ebins)
    else:
        raise Exception("Did not recognize projection type %s" % type(proj))
    return m, f


class Map_Base(object):
    """ Abstract representation of a 2D or 3D counts map."""

    def __init__(self, counts):
        self._counts = counts

    @property
    def counts(self):
        return self._counts

    def get_pixel_indices(self, lats, lons):
        raise NotImplementedError("MapBase.get_pixel_indices)")


class Map(Map_Base):
    """ Representation of a 2D or 3D counts map using WCS. """

    def __init__(self, counts, wcs):
        """
        Parameters
        ----------
        counts : `~numpy.ndarray`
          Counts array.
        """
        Map_Base.__init__(self, counts)
        self._wcs = wcs

        self._npix = counts.shape

        if len(self._npix) == 3:
            xindex = 2
            yindex = 1
        elif len(self._npix) == 2:
            xindex = 1
            yindex = 0
        else:
            raise Exception('Wrong number of dimensions for Map object.')

        self._width = \
            np.array([np.abs(self.wcs.wcs.cdelt[0]) * self._npix[xindex],
                      np.abs(self.wcs.wcs.cdelt[1]) * self._npix[yindex]])
        self._pix_center = np.array([(self._npix[xindex] - 1.0) / 2.,
                                     (self._npix[yindex] - 1.0) / 2.])
        self._pix_size = np.array([np.abs(self.wcs.wcs.cdelt[0]),
                                   np.abs(self.wcs.wcs.cdelt[1])])

        self._skydir = SkyCoord.from_pixel(self._pix_center[0],
                                           self._pix_center[1],
                                           self.wcs)

    @property
    def wcs(self):
        return self._wcs

    @property
    def skydir(self):
        """Return the sky coordinate of the image center."""
        return self._skydir

    @property
    def width(self):
        """Return the dimensions of the image."""
        return self._width

    @property
    def pix_size(self):
        """Return the pixel size along the two image dimensions."""
        return self._pix_size

    @property
    def pix_center(self):
        """Return the ROI center in pixel coordinates."""
        return self._pix_center

    @staticmethod
    def create_from_hdu(hdu, wcs):
        return Map(hdu.data.T, wcs)

    @staticmethod
    def create_from_fits(fitsfile, **kwargs):
        hdu = kwargs.get('hdu', 0)

        hdulist = fits.open(fitsfile)
        header = hdulist[hdu].header
        data = hdulist[hdu].data
        header = fits.Header.fromstring(header.tostring())
        wcs = WCS(header)
        return Map(data, wcs)

    @staticmethod
    def create(skydir, cdelt, npix, coordsys='CEL', projection='AIT'):
        crpix = np.array([n / 2. + 0.5 for n in npix])
        wcs = wcs_utils.create_wcs(skydir, coordsys, projection,
                                   cdelt, crpix)
        return Map(np.zeros(npix), wcs)

    def create_image_hdu(self, name=None):
        return fits.ImageHDU(self.counts, header=self.wcs.to_header(),
                             name=name)

    def create_primary_hdu(self):
        return fits.PrimaryHDU(self.counts, header=self.wcs.to_header())

    def sum_over_energy(self):
        """ Reduce a 3D counts cube to a 2D counts map
        """
        # Note that the array is using the opposite convention from WCS
        # so we sum over axis 0 in the array, but drop axis 2 in the WCS object
        return Map(self.counts.sum(0), self.wcs.dropaxis(2))

    def xy_pix_to_ipix(self, xypix, colwise=False):
        """ Return the pixel index from the pixel xy coordinates

        if colwise is True (False) this uses columnwise (rowwise) indexing
        """
        if colwise:
            return np.where((xypix[0] < self._wcs._naxis1) *
                            (xypix[1] < self._wcs._naxis2),
                            xypix[0] * self._wcs._naxis2 + xypix[1], -1)
        else:
            return np.where((xypix[0] < self._wcs._naxis2) *
                            (xypix[1] < self._wcs._naxis1),
                            xypix[1] * self._wcs._naxis1 + xypix[0], -1)

    def ipix_to_xypix(self, ipix, colwise=False):
        """ Return the pixel xy coordinates from the pixel index

        if colwise is True (False) this uses columnwise (rowwise) indexing
        """
        if colwise:
            return (int(ipix / self._wcs._naxis2), ipix % self._wcs._naxis2)
        else:
            return (ipix % self._wcs._naxis1, int(ipix / self._wcs._naxis1))

    def ipix_swap_axes(self, ipix, colwise=False):
        """ Return the transposed pixel index from the pixel xy coordinates

        if colwise is True (False) this assumes the original index was
        in column wise scheme
        """
        xy = self.ipix_to_xypix(ipix, colwise)
        return self.xy_pix_to_ipix(xy, not colwise)

    def get_pixel_skydirs(self):

        xpix = np.linspace(0, self.counts.shape[0] - 1., self.counts.shape[0])
        ypix = np.linspace(0, self.counts.shape[1] - 1., self.counts.shape[1])
        xypix = np.meshgrid(xpix, ypix, indexing='ij')
        return SkyCoord.from_pixel(np.ravel(xypix[0]),
                                   np.ravel(xypix[1]), self.wcs)

    def get_pixel_indices(self, lons, lats):
        """Return the indices in the flat array corresponding to a set of coordinates

        Parameters
        ----------
        lons  : array-like
           'Longitudes' (RA or GLON)

        lats  : array-like
           'Latitidues' (DEC or GLAT)

        Returns
        ----------
        idxs : numpy.ndarray((n),'i')
           Indices of pixels in the flattened map, -1 used to flag
           coords outside of map
        """
        if len(lats) != len(lons):
            raise RuntimeError('Map.get_pixel_indices, input lengths '
                               'do not match %i %i' % (len(lons), len(lats)))
        pix_x, pix_y = self._wcs.wcs_world2pix(lons, lats, 1)
        pixcrd = [np.floor(pix_x).astype(int), np.floor(pix_y).astype(int)]
        idxs = self.xy_pix_to_ipix(pixcrd, colwise=False)
        return idxs

    def get_map_values(self, lons, lats):
        """Return the indices in the flat array corresponding to a set of coordinates

        Parameters
        ----------
        lons  : array-like
           'Longitudes' (RA or GLON)

        lats  : array-like
           'Latitidues' (DEC or GLAT)

        Returns
        ----------
        vals : numpy.ndarray((n))
           Values of pixels in the flattened map, np.nan used to flag
           coords outside of map
        """
        pix_idxs = self.get_pixel_indices(lons, lats)
        vals = np.where(pix_idxs > 0, self.counts.flat[pix_idxs], np.nan)
        return vals


class HpxMap(Map_Base):
    """ Representation of a 2D or 3D counts map using HEALPix. """

    def __init__(self, counts, hpx):
        """ C'tor, fill with a counts vector and a HPX object """
        Map_Base.__init__(self, counts)
        self._hpx = hpx
        self._wcs2d = None
        self._hpx2wcs = None

    @property
    def hpx(self):
        return self._hpx

    @staticmethod
    def create_from_hdu(hdu, ebins):
        """ Creates and returns an HpxMap object from a FITS HDU.

        hdu    : The FITS
        ebins  : Energy bin edges [optional]
        """
        hpx = HPX.create_from_header(hdu.header, ebins)
        colnames = hdu.columns.names
        nebin = 0
        for c in colnames:
            if c.find("CHANNEL") == 0:
                nebin += 1
            pass
        data = np.ndarray((nebin, hpx.npix))
        for i in range(nebin):
            cname = "CHANNEL%i" % (i + 1)
            data[i, 0:] = hdu.data.field(cname)
            pass
        return HpxMap(data, hpx)

    @staticmethod
    def create_from_hdulist(hdulist, extname="SKYMAP", ebounds="EBOUNDS"):
        """ Creates and returns an HpxMap object from a FITS HDUList

        extname : The name of the HDU with the map data
        ebounds : The name of the HDU with the energy bin data
        """
        if ebounds is not None:
            try:
                ebins = fits_utils.read_energy_bounds(hdulist[ebounds])
            except:
                ebins = None
        else:
            ebins = None

        hpxMap = HpxMap.create_from_hdu(hdulist[extname], ebins)
        return hpxMap

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

        returns (WCS object, np.ndarray() with reprojected data)

        """
        self._wcs_proj = proj
        self._wcs_oversample = oversample
        self._wcs_2d = self.hpx.make_wcs(2, proj=proj, oversample=oversample)
        self._hpx2wcs = HpxToWcsMapping(self.hpx, self._wcs_2d)
        wcs, wcs_data = self.convert_to_cached_wcs(self.counts, sum_ebins,
                                                   normalize)
        return wcs, wcs_data

    def convert_to_cached_wcs(self, hpx_in, sum_ebins=False, normalize=True):
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
                hpx_data = hpx_in.sum(1)
                loop_ebins = False
            else:
                wcs_data = np.ndarray((self.counts.shape[0],
                                       self._hpx2wcs.npix[0],
                                       self._hpx2wcs.npix[1]))
                hpx_data = hpx_in
                loop_ebins = True
        else:
            raise Exception('Wrong dimension for HpxMap %i' %
                            len(hpx_in.shape))

        if loop_ebins:
            for i in range(hpx_data.shape[0]):
                self._hpx2wcs.fill_wcs_map_from_hpx_data(
                    hpx_data[i], wcs_data[i], normalize)
                pass
            wcs_data.reshape((self.counts.shape[0], self._hpx2wcs.npix[
                0], self._hpx2wcs.npix[1]))
            # replace the WCS with a 3D one
            wcs = self.hpx.make_wcs(3, proj=self._wcs_proj,
                                    energies=self.hpx.ebins,
                                    oversample=self._wcs_oversample)
        else:
            self._hpx2wcs.fill_wcs_map_from_hpx_data(
                hpx_data, wcs_data, normalize)
            wcs_data.reshape(self._hpx2wcs.npix)
            wcs = self._wcs_2d

        return wcs, wcs_data
