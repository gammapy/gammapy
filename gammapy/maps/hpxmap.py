# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import re
import numpy as np
from astropy.extern import six
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.utils.misc import InheritDocstrings
from .hpx import HpxToWcsMapping
from .geom import MapAxis

__all__ = [
    'HpxMap',
]


def find_and_read_bands(hdulist, extname=None):
    """  Reads and returns the energy bin edges.

    This works for both the CASE where the energies are in the ENERGIES HDU
    and the case where they are in the EBOUND HDU
    """
    axes = []
    axis_cols = []
    if extname is not None and extname in hdulist:
        hdu = hdulist[extname]
        for i in range(5):
            if 'AXCOLS%i' % i in hdu.header:
                axis_cols += [hdu.header['AXCOLS%i' % i].split(',')]
            else:
                break
    elif 'ENERGIES' in hdulist:
        hdu = hdulist['ENERGIES']
        axis_cols = [['ENERGY']]
    elif 'EBOUNDS' in hdulist:
        hdu = hdulist['EBOUNDS']
        axis_cols = [['E_MIN', 'E_MAX']]

    for i, cols in enumerate(axis_cols):

        if 'ENERGY' in cols or 'E_MIN' in cols:
            name = 'energy'
        elif re.search('(.+)_MIN', cols[0]):
            name = re.search('(.+)_MIN', cols[0]).group(1)
        else:
            name = cols[0]

        if len(cols) == 2:
            xmin = np.unique(hdu.data.field(cols[0]))  # / 1E3
            xmax = np.unique(hdu.data.field(cols[1]))  # / 1E3
            axes += [MapAxis(np.append(xmin, xmax[-1]), name=name)]
        else:
            x = np.unique(hdu.data.field(cols[0]))
            axes += [MapAxis.from_nodes(x, name=name)]

    return axes


class HpxMeta(InheritDocstrings, abc.ABCMeta):
    pass


@six.add_metaclass(HpxMeta)
class HpxMap(object):
    """Base class for HEALPIX map classes.

    Parameters
    ----------
    hpx : `~gammapy.maps.hpx.HPXGeom`
        HEALPix geometry object.

    data : `~numpy.ndarray`
        Data array.
    """

    def __init__(self, hpx, data):
        self._data = data
        self._hpx = hpx
        self._wcs2d = None
        self._hpx2wcs = None

    @property
    def hpx(self):
        """HEALPix geometry object."""
        return self._hpx

    @property
    def data(self):
        """Array of data values."""
        return self._data

    @classmethod
    def read(cls, filename, **kwargs):
        """Read from a FITS file.

        Parameters
        ----------
        filename : str
            Name of the FITS file.
        hdu : str
            Name or index of the HDU with the map data.
        hdu_bands : str
            Name or index of the HDU with the BANDS table.

        Returns
        -------
        hpx_map : `~HpxMap`
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
            HDU list containing HDUs for map data and bands.
        hdu : str
            Name or index of the HDU with the map data.
        hdu_bands : str
            Name or index of the HDU with the BANDS table.

        Returns
        -------
        hpx_map : `HpxMap`
            Map object
        """
        extname = kwargs.get('hdu', 'SKYMAP')

        hdu = hdulist[extname]
        extname_bands = kwargs.get('hdu_bands', None)
        if 'BANDSHDU' in hdu.header and extname_bands is None:
            extname_bands = hdu.header['BANDSHDU']

        axes = find_and_read_bands(hdulist, extname=extname_bands)
        return cls.from_hdu(hdu, axes)

    def write(self, filename, **kwargs):
        """Write to a FITS file.

        Parameters
        ----------
        filename : str
            Output file name.
        extname : str
            Set the name of the image extension.  By default this will
            be set to SKYMAP.
        extname_bands : str
            Set the name of the binning extension.  By default this will
            be set to BANDS.
        hpxconv : str
            HEALPix format convention.  This option can be used to
            write files that are compliant with non-standard HEALPix
            conventions.
        sparse : bool
            Sparsify the map by dropping pixels with zero amplitude.

        """
        hdulist = self.to_hdulist(**kwargs)
        overwrite = kwargs.get('overwrite', True)
        hdulist.writeto(filename, overwrite=overwrite)

    def to_hdulist(self, **kwargs):

        extname = kwargs.get('extname', 'SKYMAP')
        extname_bands = kwargs.get('extname_bands', self.hpx.conv.bands_hdu)
        hdulist = [fits.PrimaryHDU(), self.make_hdu(**kwargs)]
        if self.hpx.axes:
            hdulist += [self.hpx.make_bands_hdu(extname=extname_bands)]
        return fits.HDUList(hdulist)

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

        # FIXME: This should just return a WCSMap object

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

    def get_skydirs(self):
        """Get a list of sky coordinates for the centers of every pixel. """
        return self.hpx.get_skydirs()

    @abc.abstractmethod
    def sum_over_axes(self):
        """Reduce to a 2D image by dropping non-spatial dimensions."""
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
        extname_bands : str
            The HDU extension name for BANDS table.
        colbase : str
            The prefix for column names
        sparse : bool
            Set INDXSCHM to SPARSE and sparsify the map by only
            writing pixels with non-zero amplitude.
        """

        # FIXME: Should this be a method of HpxCube?
        # FIXME: Should we assign extname in this method?

        data = self.data
        shape = data.shape
        extname = kwargs.get('extname', self.hpx.conv.extname)
        extname_bands = kwargs.get('extname_bands', self.hpx.conv.bands_hdu)
        convname = kwargs.get('convname', self.hpx.conv.convname)
        sparse = kwargs.get('sparse', False)
        header = self.hpx.make_header()
        conv = self.hpx.conv
        header['BANDSHDU'] = extname_bands

        if sparse:
            header['INDXSCHM'] = 'SPARSE'

        # if shape[-1] != self._npix:
        #    raise ValueError('Size of data array does not match number of pixels')
        cols = []
        if header['INDXSCHM'] == 'EXPLICIT':
            cols.append(fits.Column('PIX', 'J', array=self.hpx._ipix))

        if header['INDXSCHM'] == 'SPARSE':
            nonzero = data.nonzero()
            if len(shape) == 1:
                cols.append(fits.Column('PIX', 'J', array=nonzero[0]))
                cols.append(fits.Column('VALUE', 'E',
                                        array=data[nonzero].astype(float)))
            else:
                channel = np.ravel_multi_index(nonzero[:-1], shape[:-1])
                cols.append(fits.Column('PIX', 'J', array=nonzero[-1]))
                cols.append(fits.Column('CHANNEL', 'I', array=channel))
                cols.append(fits.Column('VALUE', 'E',
                                        array=data[nonzero].astype(float)))

        else:
            if len(shape) == 1:
                cols.append(fits.Column(conv.colname(indx=conv.firstcol),
                                        'E', array=data.astype(float)))
            else:
                for i, idx in enumerate(np.ndindex(shape[:-1])):
                    cols.append(fits.Column(conv.colname(indx=i + conv.firstcol), 'E',
                                            array=data[idx].astype(float)))

        hdu = fits.BinTableHDU.from_columns(cols, header=header, name=extname)
        return hdu
