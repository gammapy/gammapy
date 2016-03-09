# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from subprocess import call
from tempfile import NamedTemporaryFile

import numpy as np

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.units import Quantity

from ..extern.bunch import Bunch
from ..image.utils import make_header
from ..utils.wcs import get_wcs_ctype

__all__ = ['SkyMap', 'SkyMapCollection']

log = logging.getLogger(__name__)


# It might an option to inherit from `~astropy.nddata.NDData` later, but as
# astropy.nddata is still in development, I decided to not inherit for now.

# The class provides Fits I/O and generic methods, that are not specific to the
# data it contains. Special data classes, such as an ExclusionMap or FluxMap should
# inherit from this class and implement special, data related methods themselves. 


class SkyMap(object):
    """
    Base class to represent sky maps.

    Parameters
    ----------
    name : str
        Name of the sky map.
    data : `~numpy.ndarray`
        Data array.
    wcs : `~astropy.wcs.WCS`
        WCS transformation object.
    unit : str
        String specifying the data units.
    meta : dict
        Dictionary to store meta data.
    """
    def __init__(self, name, data, wcs, unit=None, meta=None):
        self.name = name
        self.data = data
        self.wcs = wcs
        self.meta = meta
        self.unit = unit

    @classmethod
    def read(cls, filename, *args, **kwargs):
        """
        Read sky map from Fits file.

        Parameters
        ----------
        filename : str
            Name of the Fits file.
        """
        data = fits.getdata(filename, *args, **kwargs)
        header = fits.getheader(filename, *args, **kwargs)
        wcs = WCS(header)
        meta = header
        name = header.get('EXTNAME')
        name = header.get('HDUNAME')
        try:
            unit = header['BUNIT']
        except KeyError:
            unit = None
            log.warn('No units found for extension {}'.format(name))

        return cls(name, data, wcs, unit, meta)

    @classmethod
    def empty(cls, name=None, nxpix=200, nypix=200, binsz=0.02, xref=0, yref=0, fill=0,
              proj='CAR', coordsys='GAL', xrefpix=None, yrefpix=None, dtype='float64',
              unit=None, meta=None):
        """
        Create an empty sky map from scratch.

        Uses the same parameter names as the Fermi tool gtbin
        (see http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/help/gtbin.txt).

        If no reference pixel position is given it is assumed to be
        at the center of the image.

        Parameters
        ----------
        name : str
            Name of the sky map.
        nxpix : int, optional
            Number of pixels in x axis. Default is 200.
        nypix : int, optional
            Number of pixels in y axis. Default is 200.
        binsz : float, optional
            Bin size for x and y axes in units of degrees. Default is 0.02.
        xref : float, optional
            Coordinate system value at reference pixel for x axis. Default is 0.
        yref : float, optional
            Coordinate system value at reference pixel for y axis. Default is 0.
        fill : float or 'checkerboard', optional
            Creates checkerboard image or uniform image of any float
        proj : string, optional
            Any valid WCS projection type. Default is 'CAR' (cartesian).
        coordsys : {'CEL', 'GAL'}, optional
            Coordinate system, either Galactic ('GAL') or Equatorial ('CEL').
            Default is 'GAL' (Galactic).
        xrefpix : float, optional
            Coordinate system reference pixel for x axis. Default is None.
        yrefpix: float, optional
            Coordinate system reference pixel for y axis. Default is None.
        dtype : str, optional
            Data type, default is float32
        unit : str
            Data unit.
        meta : dict
            Meta data attached to the sky map.

        Returns
        -------
        skymap : `~gammapy.data.SkyMap`
            Empty sky map.
        """
        header = make_header(nxpix, nypix, binsz, xref, yref,
                         proj, coordsys, xrefpix, yrefpix)
        data = fill * np.ones((nypix, nxpix), dtype=dtype)
        wcs = WCS(header)
        return cls(name, data, wcs, unit, meta)

    def write(self, filename, *args, **kwargs):
        """
        Write sky map to Fits file.
        
        Parameters
        ----------
        filename : str
            Name of the Fits file.
        *args : list
            Arguments passed to `~astropy.fits.ImageHDU.writeto`.
        **kwargs : dict
            Keyword arguments passed to `~astropy.fits.ImageHDU.writeto`.
        """
        hdu = self.to_image_hdu()
        hdu.writeto(filename, *args, **kwargs)

    def coordinates(self, origin=0, mode='center'):
        """
        Sky coordinate images.

        Parameters
        ----------
        origin : {0, 1}
            Pixel coordinate origin.
        mode : {'center', 'edges'}
            Return coordinate values at the pixels edges or pixel centers.
        """
        if mode == 'center':
            y, x = np.indices(self.data.shape)
        elif mode == 'edges':
            raise NotImplementedError
        else:
            raise ValueError('Invalid mode to compute coordinates.')

        return self.wcs.wcs_pix2world(x, y, origin)

    def lookup(self, position, interpolation=None, origin=0):
        """
        Lookup value at given sky position.
        
        Parameters
        ----------
        position : tuple or `~astropy.coordinates.SkyCoord`
            Position on the sky. Can be either an instance of
            `~astropy.coordinates.SkyCoord` or a tuple of `~numpy.ndarray`
            of the form (lon, lat) or (ra, dec), depending on the WCS
            transformation that is set for the sky map.
        interpolation : {'None'}
            Interpolation mode.
        origin : {0, 1}
            Pixel coordinate origin.
        """
        if isinstance(position, SkyCoord):
            if get_wcs_ctype(self.wcs) == 'galactic':
                xsky, ysky = position.galactic.l.value, position.galactic.b.value
            else:
                xsky, ysky = position.icrs.ra.value, position.icrs.dec.value
        elif isinstance(position, (tuple, list)):
            xsky, ysky = position[0], position[1]

        x, y = self.wcs.wcs_world2pix(xsky, ysky, origin)
        return self.data[y.astype('int'), x.astype('int')]

    def to_quantity(self):
        """
        Convert sky map to `~astropy.units.Quantity`.
        """
        return Quantity(self.data, self.unit)

    def to_sherpa_data2d(self, dstype='Data2D'):
        """
        Convert sky map to `~sherpa.data.Data2D` or `~sherpa.data.Data2DInt` class.

        Parameter
        ---------
        dstype : {'Data2D', 'Data2DInt'}
            Sherpa data type.
        """
        from sherpa.data import Data2D, Data2DInt
        
        if dstype == 'Data2D':
            x, y = self.coordinates(mode='center')
            return Data2D(self.name, x.ravel(), y.ravel(), self.data.ravel(),
                          self.data.shape)
        elif dstype == 'Data2DInt':
            x, y = self.coordinates(mode='edges')
            xlo, xhi = x[:-1], x[1:]
            ylo, yhi = y[:-1], y[1:]
            return Data2DInt(self.name, xlo.ravel(), xhi.ravel(),
                             ylo.ravel(), yhi.ravel(), self.data.ravel(),
                             self.data.shape)
        else:
            raise ValueError('Invalid sherpa data type.')

    def to_image_hdu(self):
        """
        Convert sky map to '~astropy.fits.ImageHDU'.
        """
        header = self.wcs.to_header()

        # Add meta data
        header.update(self.meta)
        header['BUNIT'] = self.unit
        return fits.ImageHDU(data=self.data, header=header, name=self.name) 

    def reproject(self, refheader, mode='interp', *args, **kwargs):
        """
        Reproject sky map to given reference header.

        Parameters
        ----------
        refheader : `~astropy.fits.Header`
            Reference header to reproject the data on. 
        mode : {'interp', 'exact'}
            Interpolation mode.
        """
        from reproject import reproject_interp, reproject_exact
        raise NotImplementedError

    def show(self, viewer='mpl', **kwargs):
        """
        Show sky map in image viewer.

        Parameters
        ----------
        viewer : {'mpl', 'ds9'}
            Which image viewer to use. Option 'ds9' requires ds9 to be installed.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.imshow`.
        """
        if viewer == 'mpl':
            # TODO: replace by better MPL or web based image viewer 
            import matplotlib.pyplot as plt    
            fig = plt.figure()
            axes = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=self.wcs)
            self.plot(axes, **kwargs)
            plt.show()
        elif viewer == 'ds9':
            with NamedTemporaryFile() as f:
                self.write(f)
                call(['ds9', f.name])
        else:
            raise ValueError('Invalid image viewer option.')

    def plot(self, axes, **kwargs):
        """
        Plot sky map on matplotlib WCS axes.
        """
        caxes = axes.imshow(self.data, **kwargs)
        cbar = axes._figure.colorbar(caxes, label='{0} ({1})'.format(self.name, self.unit))
        try:
            axes.coords['glon'].set_axislabel('Galactic Longitude')
            axes.coords['glat'].set_axislabel('Galactic Latitude')
        except KeyError:
            axes.coords['ra'].set_axislabel('Right Ascension')
            axes.coords['dec'].set_axislabel('Declination')


    def info(self):
        """
        Print summary info about the sky map.
        """
        info = "Name: {}\n".format(self.name)
        info += "Data shape: {}\n".format(self.data.shape)
        info += "Data type: {}\n".format(self.data.dtype)
        info += "Data unit: {}\n".format(self.unit)
        info += "Data mean: {:.3e}\n".format(np.nanmean(self.data))
        info += "WCS type: {}\n".format(self.wcs.wcs.ctype)
        print(info)


class SkyMapCollection(Bunch):
    """
    Bunch with Fits I/O.
    
    Here's an example:

    .. code-block:: python
    
        from gammapy.data import MapsBunch
        maps = MapsBunch.read('$GAMMAPY_EXTRA/datasets/fermi_survey/all.fits.gz')

    Then try tab completion on the ``maps`` object.
    """
    @classmethod
    def read(cls, filename):
        """
        Create Bunch of maps from Fits file.

        Parameters
        ----------
        filename : str
            Fits file name.
        """
        hdulist = fits.open(filename)
        kwargs = {}
        _map_names = []  # list of map names to save order in fits file
        kwargs['_ref_header'] = hdulist[0].header

        for hdu in hdulist:
            name =  hdu.header.get('HDUNAME', hdu.name.lower())
            kwargs[name] = hdu.data
            _map_names.append(name)
        kwargs['_map_names'] = _map_names
        return cls(**kwargs)


    def write(self, filename=None, header=None, **kwargs):
        """
        Write Bunch of maps to Fits file.

        Parameters
        ----------
        filename : str
            Fits file name.
        header : `~astropy.io.fits.Header`
            Reference header to be used for all maps. 
        """
        hdulist = fits.HDUList()
        header = header or self.get('_ref_header')
        for name in self.get('_map_names', sorted(self)):
            hdu = fits.ImageHDU(data=self[name], header=header, name=name)
            hdulist.append(hdu)
        hdulist.writeto(filename, **kwargs)