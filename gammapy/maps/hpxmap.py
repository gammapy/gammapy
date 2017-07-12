# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import re
import numpy as np
from astropy.io import fits
from .base import MapBase
from .hpx import HpxGeom
from .geom import MapAxis

__all__ = [
    'HpxMap',
]


def find_and_read_bands(hdulist, extname=None):
    """Read and returns the energy bin edges.

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


class HpxMap(MapBase):
    """Base class for HEALPIX map classes.

    Parameters
    ----------
    hpx : `~gammapy.maps.hpx.HpxGeom`
        HEALPix geometry object.

    data : `~numpy.ndarray`
        Data array.
    """

    def __init__(self, hpx, data):
        MapBase.__init__(self, hpx, data)
        self._wcs2d = None
        self._hpx2wcs = None

    @property
    def hpx(self):
        """HEALPix geometry object."""
        return self.geom

    @classmethod
    def create(cls, nside=None, binsz=None, nest=True, map_type=None, coordsys='CEL',
               data=None, skydir=None, width=None, dtype='float32',
               region=None, axes=None):
        """Factory method to create an empty HEALPix map.

        Parameters
        ----------
        nside : int or `~numpy.ndarray`
            HEALPix NSIDE parameter.  This parameter sets the size of
            the spatial pixels in the map.
        binsz : float or `~numpy.ndarray`
            Approximate pixel size in degrees.  An NSIDE will be
            chosen that correponds to a pixel size closest to this
            value.  This option is superseded by nside.
        nest : bool
            True for HEALPix "NESTED" indexing scheme, False for "RING" scheme.
        coordsys : {'CEL', 'GAL'}, optional
            Coordinate system, either Galactic ('GAL') or Equatorial ('CEL').
        skydir : tuple or `~astropy.coordinates.SkyCoord`
            Sky position of map center.  Can be either a SkyCoord
            object or a tuple of longitude and latitude in deg in the
            coordinate system of the map.
        map_type : str
            Internal map representation.  Valid types are `HpxMapND`/`hpx` and
            `HpxMapSparse`/`hpx-sparse`.
        width : float
            Diameter of the map in degrees.  If None then an all-sky
            geometry will be created.
        axes : list
            List of `~MapAxis` objects for each non-spatial dimension.
        """
        from .hpxcube import HpxMapND
        from .hpxsparse import HpxMapSparse

        hpx = HpxGeom.create(nside=nside, binsz=binsz,
                             nest=nest, coordsys=coordsys, region=region,
                             conv=None, axes=axes, skydir=skydir, width=width)
        if map_type in [None,'hpx','HpxMapND']:
            return HpxMapND(hpx, dtype=dtype)
        elif map_type in ['hpx-sparse','HpxMapSparse']:
            return HpxMapSparse(hpx, dtype=dtype)
        else:
            raise ValueError('Unregnized Map type: {}'.format(map_type))

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
        with fits.open(filename) as hdulist:
            hpx_map = cls.from_hdulist(hdulist, **kwargs)
        return hpx_map

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

    @abc.abstractmethod
    def to_wcs(self, sum_bands=False, normalize=True):
        """Make a WCS object and convert HEALPIX data into WCS projection.

        Parameters
        ----------
        sum_bands : bool
            Sum over non-spatial axes before reprojecting.  If False
            then the WCS map will have the same dimensionality as the
            HEALPix one.
        normalize : bool
            True -> preserve integral by splitting HEALPIX values between bins

        Returns
        -------
        wcs : `~astropy.wcs.WCS`
            WCS object
        data : `~numpy.ndarray`
            Reprojected data

        """
        pass

    @abc.abstractmethod
    def to_swapped_scheme(self):
        """Return a new map with the opposite scheme (ring or nested).
        """
        pass

    @abc.abstractmethod
    def to_ud_graded(self, order, preserve_counts=False):
        """Upgrade or downgrade the resolution of the map to the chosen order.
        """
        pass

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
        # FIXME: Should this be a method of HpxMapND?
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
