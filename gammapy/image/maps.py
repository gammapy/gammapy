# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from subprocess import call
from tempfile import NamedTemporaryFile

import numpy as np

from astropy.io import fits
from astropy.coordinates import SkyCoord, Longitude, Latitude
from astropy.wcs import WCS
from astropy.units import Quantity, Unit
from astropy.extern import six

from ..extern.bunch import Bunch
from ..image.utils import make_header, _bin_events_in_cube
from ..utils.wcs import get_wcs_ctype
from ..utils.scripts import make_path

__all__ = ['SkyMap', 'SkyMapCollection']

log = logging.getLogger(__name__)


# It might be a good option to inherit from `~astropy.nddata.NDData` later, but
# as astropy.nddata is still in development, I decided to not inherit for now.

# The class provides Fits I/O and generic methods, that are not specific to the
# data it contains. Special data classes, such as an ExclusionMap, FluxMap or
# CountsMap should inherit from this class and implement special, data related
# methods themselves. 

# Actually this class is generic enough to be rather in astropy and not in
# gammapy and should maybe be moved to there...

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

    def __init__(self, name=None, data=None, wcs=None, unit=None, meta=None):
        # TODO: validate inputs
        self.name = name
        self.data = data
        self.wcs = wcs
        self.meta = meta
        self.unit = unit

    @classmethod
    def read(cls, fobj, *args, **kwargs):
        """
        Read sky map from Fits file.

        Parameters
        ----------
        fobj : file like object or `~astropy.io.fits.ImageHDU` or `~astropy.io.fits.PrimaryHDU`
            Name of the Fits file or ImageHDU object.
        *args : list
            Arguments passed `~astropy.io.fits.gedata`.
        **kwargs : dict
            Keyword arguments passed `~astropy.io.fits.gedata`.
        """
        if isinstance(fobj, (fits.ImageHDU, fits.PrimaryHDU)):
            data, header = fobj.data, fobj.header
        else:
            if isinstance(fobj, six.string_types):
                fobj = str(make_path(fobj))
            data = fits.getdata(fobj, *args, **kwargs)
            header = fits.getheader(fobj, *args, **kwargs)
        wcs = WCS(header)
        meta = header

        name = header.get('HDUNAME')
        if name is None:
            name = header.get('EXTNAME')
        try:
            # Valitade unit string
            unit = Unit(header['BUNIT'], format='fits').to_string()
        except (KeyError, ValueError):
            unit = None
            log.warn('No valid units found for extension {}'.format(name))

        return cls(name, data, wcs, unit, meta)

    @classmethod
    def empty(cls, name=None, nxpix=200, nypix=200, binsz=0.02, xref=0, yref=0,
              fill=0, proj='CAR', coordsys='GAL', xrefpix=None, yrefpix=None,
              dtype='float64', unit=None, meta=None):
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
        fill : float, optional
            Fill sky map with constant value. Default is 0.
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
        return cls(name=name, data=data, wcs=wcs, unit=unit, meta=header)

    @classmethod
    def empty_like(cls, skymap, name=None, unit=None, fill=0, meta=None):
        """
        Create an empty sky map with the same WCS specification as given sky map. 
        
        Parameters
        ----------
        skymap : `~gammapy.image.SkyMap` or `~astropy.io.fits.ImageHDU`
            Instance of `~gammapy.image.SkyMap`.
        fill : float, optional
            Fill sky map with constant value. Default is 0.
        name : str
            Name of the sky map.
        unit : str
            String specifying the data units.
        meta : dict
            Dictionary to store meta data.            
        """
        if isinstance(skymap, SkyMap):
            wcs = skymap.wcs.copy()
        elif isinstance(skymap, (fits.ImageHDU, fits.PrimaryHDU)):
            wcs = WCS(skymap.header)
        else:
            raise TypeError("Can't create sky map from type {}".format(type(skymap)))
        data = fill * np.ones_like(skymap.data)
        return cls(name, data, wcs, unit, meta=wcs.to_header())

    def fill(self, events, origin=0):
        """
        Fill sky map with events.

        Parameters
        ----------
        events : `~astropy.table.Table`
            Event list table
        origin : {0, 1}
            Pixel coordinate origin.

        """
        self.data = _bin_events_in_cube(events, self.wcs, self.data.shape, origin=origin).sum(axis=0)

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

    def coordinates(self, coord_type='world', origin=0, mode='center'):
        """
        Sky coordinate images.

        Parameters
        ----------
        coord_type : {'world', 'pix', 'skycoord'}
            Which type of coordinates to return.
        origin : {0, 1}
            Pixel coordinate origin.
        mode : {'center', 'edges'}
            Return coordinate values at the pixels edges or pixel centers.
        """
        if mode == 'center':
            y, x = np.indices(self.data.shape)
        elif mode == 'edges':
            shape = self.data.shape[0] + 1, self.data.shape[1] + 1
            y, x = np.indices(shape)
            y, x = y - 0.5, x - 0.5
        else:
            raise ValueError('Invalid mode to compute coordinates.')

        if coord_type == 'pix':
            return x, y
        else:
            xsky, ysky = self.wcs.wcs_pix2world(x, y, origin)
            l, b = Longitude(xsky, unit='deg'), Latitude(ysky, unit='deg')
            l = l.wrap_at('180d')
            if coord_type == 'world':
                return l.degree, b.degree
            elif coord_type == 'skycoord':
                return l, b
            else:
                raise ValueError("Not a valid coordinate type. Choose either"
                                 " 'world', 'pix' or 'skycoord'.")

    def solid_angle(self):
        """
        Solid angle image
        """
        xsky, ysky = self.coordinates(mode='edges')
        omega = -np.diff(xsky, axis=1)[1:, :] * np.diff(ysky, axis=0)[:, 1:]
        return Quantity(omega, 'deg2').to('sr')

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
        return self.data[np.round(y).astype('int'), np.round(x).astype('int')]

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
        Convert sky map to `~astropy.fits.ImageHDU`.
        """
        header = self.wcs.to_header()

        # Add meta data
        header.update(self.meta)
        header['BUNIT'] = self.unit
        return fits.ImageHDU(data=self.data, header=header, name=self.name)

    def reproject(self, refheader, mode='interp', *args, **kwargs):
        """
        Reproject sky map to given reference header.

        Calls `reproject.reproject.interp`, passing along ``args`` and ``kwargs``.

        TODO: ``mode`` isn't use yet ... should dispatch to correct method.

        Parameters
        ----------
        refheader : `~astropy.fits.Header`
            Reference header to reproject the data on. 
        mode : {'interp', 'exact'}
            Interpolation mode.

        Returns
        -------
        skymap : `SkyMap`
            Skymap reprojected onto ``refheader``.
        """
        from reproject import reproject_interp
        out = reproject_interp(input_data=self.to_image_hdu(),
                               output_projection=refheader, *args, **kwargs)
        map = SkyMap(name=self.name, data=out[0], wcs=self.wcs, unit=self.unit,
                     meta=self.meta)
        return map

    def show(self, viewer='mpl', **kwargs):
        """
        Show sky map in image viewer.

        Parameters
        ----------
        viewer : {'mpl', 'ds9'}
            Which image viewer to use. Option 'ds9' requires ds9 to be installed.
        options : list, optional
            List of options passed to ds9. E.g. ['-cmap', 'heat', '-scale', 'log'].
            Any valid ds9 command line option can be passed. 
            See http://ds9.si.edu/doc/ref/command.html for details. 
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.imshow` or ds9.
        """
        if viewer == 'mpl':
            # TODO: replace by better MPL or web based image viewer 
            import matplotlib.pyplot as plt
            fig = plt.figure()
            axes = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=self.wcs)
            self.plot(axes, fig, **kwargs)
            plt.show()
        elif viewer == 'ds9':
            with NamedTemporaryFile() as f:
                self.write(f)
                call(['ds9', f.name, '-cmap', 'bb'] + kwargs.get('options', []))
        else:
            raise ValueError("Invalid image viewer option, choose either"
                             " 'mpl' or 'ds9'.")

    def plot(self, axes=None, fig=None, **kwargs):
        """
        Plot sky map on matplotlib WCS axes.
        
        Parameters
        ----------
        ax : `~astropy.wcsaxes.WCSAxes`, optional
            WCS axis object to plot on.
        """
        caxes = axes.imshow(self.data, **kwargs)
        unit = self.unit or 'A.E.'
        cbar = fig.colorbar(caxes, label='{0} ({1})'.format(self.name, unit))
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
        print(repr(self))

    def __repr__(self):
        """
        String representation of the class.
        """
        info = "Name: {}\n".format(self.name)
        if not self.data is None:
            info += "Data shape: {}\n".format(self.data.shape)
            info += "Data type: {}\n".format(self.data.dtype)
            info += "Data unit: {}\n".format(self.unit)
            info += "Data mean: {:.3e}\n".format(np.nanmean(self.data))
        if not self.wcs is None:
            info += "WCS type: {}\n".format(self.wcs.wcs.ctype)
        return info

    def __array__(self):
        """
        Array representation of sky map.
        """
        return self.data

    def binarize(self, threshold=0.5):
        """Make binary mask (`~gammapy.image.ExclusionMask`).

        TODO: this isn't working yet.
        The ``wcs`` attribute isn't copied over!?

        TODO: some docs and example
        """
        # TODO: Add test.
        from .mask import ExclusionMask
        map = ExclusionMask(self)
        map.data = np.where(self.data > threshold, 0, 1)

        return map


class SkyMapCollection(Bunch):
    """
    Container for a collection of sky maps.

    This class bundles as set of SkyMaps in single data container and provides
    convenience methods for Fits I/O and `~gammapy.extern.bunch.Bunch` like 
    handling of the data members. 
    
    Here's an example how to use it:

    .. code-block:: python
    
        from gammapy.image import SkyMapCollection
        skymaps = SkyMapCollection.read('$GAMMAPY_EXTRA/datasets/fermi_survey/all.fits.gz')

    Then try tab completion on the ``skymaps`` object.
    """

    @classmethod
    def read(cls, filename):
        """
        Create collection of sky maps from Fits file.

        Parameters
        ----------
        filename : str
            Fits file name.
        """
        hdulist = fits.open(str(make_path(filename)))
        kwargs = {}
        _map_names = []  # list of map names to save order in fits file

        for hdu in hdulist:
            skymap = SkyMap.read(hdu)

            # This forces lower case map names, but only on the collection object
            # When writing to fits again the skymap.name attribute is used.
            name = skymap.name.lower()
            kwargs[name] = skymap
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
        for name in self.get('_map_names', sorted(self)):
            hdu = self[name].to_image_hdu()
            hdulist.append(hdu)
        hdulist.writeto(filename, **kwargs)

    def set_reference_wcs(self):
        """
        Set reference WCS for all sky maps.
        """
        raise NotImplementedError

    def __repr__(self):
        """
        String representation of the sky map collection.
        """
        info = ''
        for name in self.get('_map_names', sorted(self)):
            info += self[name].__repr__()
            info += '\n'
        return info
