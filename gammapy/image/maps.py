# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from subprocess import call
from tempfile import NamedTemporaryFile
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates.angle_utilities import angular_separation
from astropy.wcs import WCS, WcsError
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.units import Quantity, Unit
from astropy.nddata.utils import Cutout2D
from ..extern.bunch import Bunch
from ..utils.scripts import make_path
from ..image.utils import make_header, _bin_events_in_cube
from ..data import EventList

__all__ = ['SkyImage', 'SkyImageCollection']

log = logging.getLogger(__name__)


# It might be a good option to inherit from `~astropy.nddata.NDData` later, but
# as astropy.nddata is still in development, I decided to not inherit for now.

# The class provides Fits I/O and generic methods, that are not specific to the
# data it contains. Special data classes, such as an ExclusionMap, FluxMap or
# CountsMap should inherit from this class and implement special, data related
# methods themselves.

# Actually this class is generic enough to be rather in astropy and not in
# gammapy and should maybe be moved to there...

class SkyImage(object):
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
    meta : `~collections.OrderedDict`
        Dictionary to store meta data.
    """
    wcs_origin = 0

    def __init__(self, name=None, data=None, wcs=None, unit=None, meta=None):
        # TODO: validate inputs
        self.name = name
        self.data = data
        self.wcs = wcs
        self.meta = meta
        self.unit = unit

    @classmethod
    def read(cls, filename, **kwargs):
        """Read sky map from FITS file.

        Parameters
        ----------
        filename : str
            FITS file name
        **kwargs : dict
            Keyword arguments passed `~astropy.io.fits.getdata`.
        """
        filename = str(make_path(filename))
        data = fits.getdata(filename, **kwargs)
        header = fits.getheader(filename, **kwargs)
        image_hdu = fits.ImageHDU(data, header)
        return cls.from_image_hdu(image_hdu)

    @classmethod
    def from_image_hdu(cls, image_hdu):
        """
        Create sky map from ImageHDU.

        Parameters
        ----------
        image_hdu : `astropy.io.fits.ImageHDU`
            Source image HDU.

        Examples
        --------
        >>> from astropy.io import fits
        >>> from gammapy.image import SkyImage
        >>> hdu_list = fits.open('data.fits')
        >>> sky_map = SkyImage.from_image_hdu(hdu_list['myimage'])
        """
        data = image_hdu.data
        header = image_hdu.header
        wcs = WCS(image_hdu.header)

        name = header.get('HDUNAME')
        if name is None:
            name = header.get('EXTNAME')
        try:
            # Validate unit string
            unit = Unit(header['BUNIT'], format='fits').to_string()
        except (KeyError, ValueError):
            unit = None

        meta = OrderedDict(header)

        # parse astropy.io.fits.header._HeaderCommentaryCards as strings
        if 'HISTORY' in meta:
            meta['HISTORY'] = str(meta['HISTORY'])
        if 'COMMENT' in meta:
            meta['COMMENT'] = str(meta['COMMENT'])

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
        meta : `~collections.OrderedDict`
            Meta data attached to the sky map.

        Returns
        -------
        skymap : `~gammapy.image.SkyImage`
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
        skymap : `~gammapy.image.SkyImage` or `~astropy.io.fits.ImageHDU`
            Instance of `~gammapy.image.SkyImage`.
        fill : float, optional
            Fill sky map with constant value. Default is 0.
        name : str
            Name of the sky map.
        unit : str
            String specifying the data units.
        meta : `~collections.OrderedDict`
            Dictionary to store meta data.
        """
        if isinstance(skymap, SkyImage):
            wcs = skymap.wcs.copy()
        elif isinstance(skymap, (fits.ImageHDU, fits.PrimaryHDU)):
            wcs = WCS(skymap.header)
        else:
            raise TypeError("Can't create sky map from type {}".format(type(skymap)))
        data = fill * np.ones_like(skymap.data)
        return cls(name, data, wcs, unit, meta=wcs.to_header())

    def fill(self, value):
        """
        Fill sky map with events.

        Parameters
        ----------
        value : float or `~gammapy.data.EventList`
             Value to fill in the map. If an event list is given, events will be
             binned in the map.
        """
        if isinstance(value, EventList):
            counts = _bin_events_in_cube(value, self.wcs, self.data.shape,
                                         origin=self.wcs_origin).sum(axis=0)
            self.data = counts.value
            self.unit = 'ct'
        elif np.isscalar(value):
            self.data.fill(value)
        else:
            raise TypeError("Can't fill value of type {}".format(type(value)))


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

    def coordinates_pix(self, mode='center'):
        """
        Pixel sky coordinate images.

        Parameters
        ----------
        mode : {'center', 'edges'}
            Return coordinate values at the pixels edges or pixel centers.

        Returns
        -------
        x, y : tuple
            Return arrays representing the coordinates of a sky grid.
        """
        if mode == 'center':
            y, x = np.indices(self.data.shape)
        elif mode == 'edges':
            shape = self.data.shape[0] + 1, self.data.shape[1] + 1
            y, x = np.indices(shape)
            y, x = y - 0.5, x - 0.5
        else:
            raise ValueError('Invalid mode to compute coordinates.')
        return x, y

    def coordinates(self, mode='center'):
        """
        Sky coordinate images.

        Parameters
        ----------
        mode : {'center', 'edges'}
            Return coordinate values at the pixels edges or pixel centers.

        Returns
        -------
        coordinates : `~astropy.coordinates.SkyCoord`
            Position on the sky.
        """
        x, y = self.coordinates_pix(mode=mode)
        coordinates = pixel_to_skycoord(x, y, self.wcs, self.wcs_origin)
        return coordinates

    def contains(self, position):
        """
        Check if given position on the sky is contained in the sky map.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position on the sky. 

        Returns
        -------
        containment : array
            Bool array
        """
        ny, nx = self.data.shape
        x, y = skycoord_to_pixel(position, self.wcs, self.wcs_origin)
        return (x >= 0.5) & (x <= nx + 0.5) & (y >= 0.5) & (y <= ny + 0.5)

    def _get_boundaries(self, skymap_ref, skymap, wcs_check):
        """
        Get boundary coordinates of one sky map in the pixel coordinate system
        of another reference sky map.
        """
        ymax, xmax = skymap.data.shape
        ymax_ref, xmax_ref = skymap_ref.data.shape

        # transform boundaries in world coordinates
        bounds = skymap.wcs.wcs_pix2world([0, xmax], [0, ymax], skymap.wcs_origin)

        # transform to pixel coordinats in the reference skymap
        bounds_ref = skymap_ref.wcs.wcs_world2pix(bounds[0], bounds[1], skymap_ref.wcs_origin)

        # round to nearest integer and clip at the boundaries
        xlo, xhi = np.rint(np.clip(bounds_ref[0], 0, xmax_ref))
        ylo, yhi = np.rint(np.clip(bounds_ref[1], 0, ymax_ref))

        if wcs_check:
            if not np.allclose(bounds_ref, np.rint(bounds_ref)):
                raise WcsError('World coordinate systems not aligned. Try to call'
                               ' .reproject() on one of the maps first.')
        return xlo, xhi, ylo, yhi

    def paste(self, skymap, method='sum', wcs_check=True):
        """
        Paste smaller skymap into sky map. 

        WCS specifications of both skymaps must be aligned. If not call
        `SkyImage.reproject()` on one of the maps first. See :ref:`skymap-cutpaste`
        more for information how to cut and paste sky images.

        Parameters
        ----------
        skymap : `~gammapy.image.SkyImage`
            Smaller sky map to paste.
        method : {'sum', 'replace'}, optional
            Sum or replace total values with cutout values.
        wcs_check : bool
            Check if both WCS are aligned. Raises `~astropy.wcs.WcsError` if not.
            Disable for performance critical computations.
        """
        xlo, xhi, ylo, yhi = self._get_boundaries(self, skymap, wcs_check)
        xlo_c, xhi_c, ylo_c, yhi_c = self._get_boundaries(skymap, self, wcs_check)

        if method == 'sum':
            self.data[ylo:yhi, xlo:xhi] += skymap.data[ylo_c:yhi_c, xlo_c:xhi_c]
        elif method == 'replace':
            self.data[ylo:yhi, xlo:xhi] = skymap.data[ylo_c:yhi_c, xlo_c:xhi_c]
        else:
            raise ValueError('Invalid method: {}'.format(method))


    def cutout(self, position, size):
        """
        Cut out rectangular piece of a sky map.

        See :ref:`skymap-cutpaste` for more information how to cut and paste
        sky images.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position of the center of the sky map to cut out.
        size : int, array-like, `~astropy.units.Quantity`
            The size of the cutout array along each axis.  If ``size``
            is a scalar number or a scalar `~astropy.units.Quantity`,
            then a square cutout of ``size`` will be created.  If
            ``size`` has two elements, they should be in ``(ny, nx)``
            order.  Scalar numbers in ``size`` are assumed to be in
            units of pixels.  ``size`` can also be a
            `~astropy.units.Quantity` object or contain
            `~astropy.units.Quantity` objects.  Such
            `~astropy.units.Quantity` objects must be in pixel or
            angular units.  For all cases, ``size`` will be converted to
            an integer number of pixels, rounding the the nearest
            integer.  See the ``mode`` keyword for additional details on
            the final cutout size.

            .. note::
                If ``size`` is in angular units, the cutout size is
                converted to pixels using the pixel scales along each
                axis of the image at the ``CRPIX`` location.  Projection
                and other non-linear distortions are not taken into
                account.

        Returns
        -------
        cutout : `~gammapy.image.SkyImage`
            Cut out sky map.
        """
        cutout = Cutout2D(self.data, position=position, wcs=self.wcs, size=size,
                          copy=True)
        skymap = SkyImage(data=cutout.data, wcs=cutout.wcs, unit=self.unit)
        return skymap

    def pad(self, mode, pad_to_factor=None, pad_width=None, **kwargs):
        """
        Pad image to the nearest larger shape, that is divisible by the given factor in both axis.

        Calls `numpy.pad`, passing ``mode`` and ``kwargs`` to it.

        Parameters
        ----------
        mode : str
            Padding mode, passed to `numpy.pad`.
        pad_to_factor : int
            Factor used for output shape computation
        pad_width: {sequence, array_like, int}
            Number of values padded to the edges of each axis, passed to `numpy.pad`

        Returns
        -------
        image : `~gammapy.image.SkyImage`
            Padded image

        Examples
        --------

        >>> from gammapy.image import SkyImage
        >>> image = SkyImage.empty(nxpix=10, nypix=13)
        >>> print(image.data.shape)
        (13, 10)
        >>> image2 = image.pad(pad_to_factor=4, mode='reflect')
        >>> image2.data.shape
        (16, 12)
        """
        if pad_to_factor is not None and pad_width is not None:
            raise ValueError('Indicate only one parameter: '
                             'either "pad_width" or "pad_to_factor"')
        if pad_to_factor is None and pad_width is None:
            raise ValueError('One parameter must be indicated: '
                             'either "pad_width" or "pad_to_factor"')

        if pad_to_factor is not None:
            pad_width = pad_to_factor - (np.array(self.data.shape) % pad_to_factor)
            pad_width = [(0, pad_width[0]), (0, pad_width[1])]

        # converting from unicode to ascii string as a workaround
        # for https://github.com/numpy/numpy/issues/7112
        mode = str(mode)

        data = np.pad(self.data, pad_width=pad_width, mode=mode, **kwargs)

        # We don't have to adjust WCS here, because we only pad on the
        # right and top, and for this change, the CRPIX doesn't change.

        return SkyImage(data=data, wcs=self.wcs)

    def lookup_max(self, region=None):
        """
        Find position of maximum in a skymap.

        Parameters
        ----------
        region : `~regions.SkyRegion` (optional)
            Limit lookup of maximum to that given sky region.

        Returns
        -------
        (position, value): `~astropy.coordinates.SkyCoord`, float
            Position and value of the maximum.
        """
        if region:
            mask = region.contains(self.coordinates())
        else:
            mask = np.ones_like(self.data)

        idx = np.nanargmax(self.data * mask)
        y, x = np.unravel_index(idx, self.data.shape)
        pos = pixel_to_skycoord(x, y, self.wcs, self.wcs_origin)
        return pos, self.data[y, x]

    def solid_angle(self):
        """
        Solid angle image (2-dim `~astropy.units.Quantity` in `sr`).
        """

        coordinates = self.coordinates(mode='edges')
        lon = coordinates.data.lon.radian
        lat = coordinates.data.lat.radian

        # Compute solid angle using the approximation that it's
        # the product between angular separation of pixel corners.
        # First index is "y", second index is "x"
        ylo_xlo = lon[:-1, :-1], lat[:-1, :-1]
        ylo_xhi = lon[:-1, 1:], lat[:-1, 1:]
        yhi_xlo = lon[1:, :-1], lat[1:, :-1]

        dx = angular_separation(*(ylo_xlo + ylo_xhi))
        dy = angular_separation(*(ylo_xlo + yhi_xlo))
        omega = Quantity(dx * dy, 'sr')
        return omega

    def center(self):
        """
        Center sky coordinates of the sky map.
        """
        return SkyCoord.from_pixel((self.data.shape[0] - 1) / 2.,
                                   (self.data.shape[1] - 1) / 2.,
                                    self.wcs, origin=self.wcs_origin)

    def lookup(self, position, interpolation=None):
        """
        Lookup value at given sky position.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position on the sky. 
        interpolation : {'None'}
            Interpolation mode.
        """
        x, y = skycoord_to_pixel(position, self.wcs, self.wcs_origin)
        return self.data[np.rint(y).astype('int'), np.rint(x).astype('int')]

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
            coordinates = self.coordinates(mode='center')
            x = coordinates.data.lon
            y = coordinates.data.lat
            return Data2D(self.name, x.ravel(), y.ravel(), self.data.ravel(),
                          self.data.shape)
        elif dstype == 'Data2DInt':
            coordinates = self.coordinates(mode='edges')
            x = coordinates.data.lon
            y = coordinates.data.lat
            xlo, xhi = x[:-1], x[1:]
            ylo, yhi = y[:-1], y[1:]
            return Data2DInt(self.name, xlo.ravel(), xhi.ravel(),
                             ylo.ravel(), yhi.ravel(), self.data.ravel(),
                             self.data.shape)
        else:
            raise ValueError('Invalid sherpa data type.')

    def copy(self):
        """
        Copy sky map.
        """
        return deepcopy(self)

    def to_image_hdu(self):
        """
        Convert sky map to `~astropy.fits.PrimaryHDU`.
        
        Returns
        -------
        primaryhdu : `~astropy.fits.PrimaryHDU`
            Primary image hdu object.
        """
        if self.wcs is not None:
            header = self.wcs.to_header()
        else:
            header = fits.Header()

        # Add meta data
        header.update(self.meta)
        if self.unit is not None:
            header['BUNIT'] = Unit(self.unit).to_string('fits')
        if self.name is not None:
            header['EXTNAME'] = self.name
            header['HDUNAME'] = self.name
        return fits.PrimaryHDU(data=self.data, header=header)

    def reproject(self, reference, mode='interp', *args, **kwargs):
        """
        Reproject sky map to given reference.

        Parameters
        ----------
        reference : `~astropy.fits.Header`, or `~gammapy.image.SkyImage`
            Reference map specification to reproject the data on.
        mode : {'interp', 'exact'}
            Interpolation mode.
        *args : list
            Arguments passed to `~reproject.reproject_interp` or
            `~reproject.reproject_exact`.
        **kwargs : dict
            Keyword arguments passed to `~reproject.reproject_interp` or
            `~reproject.reproject_exact`.

        Returns
        -------
        skymap : `~gammapy.image.SkyImage`
            Skymap reprojected onto ``reference``.
        """

        from reproject import reproject_interp, reproject_exact

        if isinstance(reference, SkyImage):
            wcs_reference = reference.wcs
            shape_out = reference.data.shape
        elif isinstance(reference, fits.Header):
            wcs_reference = WCS(reference)
            shape_out = (reference['NAXIS2'], reference['NAXIS1'])
        else:
            raise TypeError("Invalid reference map must be either instance"
                            "of `Header`, `WCS` or `SkyImage`.")

        if mode == 'interp':
            out = reproject_interp((self.data, self.wcs), wcs_reference,
                                    shape_out=shape_out, *args, **kwargs)
        elif mode == 'exact':
            out = reproject_exact((self.data, self.wcs), wcs_reference,
                                   shape_out=shape_out, *args, **kwargs)
        else:
            raise TypeError("Invalid reprojection mode, either choose 'interp' or 'exact'")

        return SkyImage(name=self.name, data=out[0], wcs=wcs_reference,
                        unit=self.unit, meta=self.meta)

    def show(self, viewer='mpl', ds9options=None, **kwargs):
        """
        Show sky map in image viewer.

        Parameters
        ----------
        viewer : {'mpl', 'ds9'}
            Which image viewer to use. Option 'ds9' requires ds9 to be installed.
        ds9options : list, optional
            List of options passed to ds9. E.g. ['-cmap', 'heat', '-scale', 'log'].
            Any valid ds9 command line option can be passed.
            See http://ds9.si.edu/doc/ref/command.html for details.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.imshow`.
        """
        if viewer == 'mpl':
            # TODO: replace by better MPL or web based image viewer
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            axes = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=self.wcs)
            self.plot(axes, fig, **kwargs)
            plt.show()
        elif viewer == 'ds9':
            ds9options = ds9options or []
            with NamedTemporaryFile() as f:
                self.write(f)
                call(['ds9', f.name, '-cmap', 'bb'] + ds9options)
        else:
            raise ValueError("Invalid image viewer option, choose either"
                             " 'mpl' or 'ds9'.")

    def plot(self, ax=None, fig=None, **kwargs):
        """
        Plot sky map on matplotlib WCS axes.

        Parameters
        ----------
        ax : `~astropy.wcsaxes.WCSAxes`, optional
            WCS axis object to plot on.
        """
        import matplotlib.pyplot as plt

        if fig is None and ax is None:
            fig = plt.gcf()
            ax = fig.add_subplot(1,1,1, projection=self.wcs)

        kwargs['origin'] = kwargs.get('origin', 'lower')
        caxes = ax.imshow(self.data, **kwargs)
        unit = self.unit or 'A.U.'
        if unit == 'ct':
            quantity = 'counts'
        elif unit is 'A.U.':
            quantity = 'Unknown'
        else:
            quantity = Unit(unit).physical_type
        cbar = fig.colorbar(caxes, label='{0} ({1})'.format(quantity, unit))
        try:
            ax.coords['glon'].set_axislabel('Galactic Longitude')
            ax.coords['glat'].set_axislabel('Galactic Latitude')
        except KeyError:
            ax.coords['ra'].set_axislabel('Right Ascension')
            ax.coords['dec'].set_axislabel('Declination')
        except AttributeError:
            log.info("Can't set coordinate axes. No WCS information available.")

    def info(self):
        """
        Print summary info about the sky map.
        """
        print(str(self))

    def __str__(self):
        """
        String representation of the class.
        """
        info = "Name: {}\n".format(self.name)
        if self.data is not None:
            info += "Data shape: {}\n".format(self.data.shape)
            info += "Data type: {}\n".format(self.data.dtype)
            info += "Data unit: {}\n".format(self.unit)
            info += "Data mean: {:.3e}\n".format(np.nanmean(self.data))
        if self.wcs is not None:
            info += "WCS type: {}\n".format(self.wcs.wcs.ctype)
        return info

    def __array__(self):
        """
        Array representation of sky map.
        """
        return self.data

    def threshold(self, threshold):
        """
        Threshold sykmap data to create a `~gammapy.image.ExclusionMask`.

        Parameters
        ----------
        threshold : float
            Threshold value.

        Returns
        -------
        mask : `~gammapy.image.ExclusionMask`
            Exclusion mask object.

        TODO: some more docs and example
        """
        from .mask import ExclusionMask
        mask = ExclusionMask.empty_like(self)
        mask.data = np.where(self.data > threshold, 0, 1)
        return mask


class SkyImageCollection(Bunch):
    """
    Container for a collection of sky maps.

    This class bundles as set of SkyMaps in single data container and provides
    convenience methods for Fits I/O and `~gammapy.extern.bunch.Bunch` like
    handling of the data members.

    Here's an example how to use it:

    .. code-block:: python

        from gammapy.image import SkyImageCollection
        skymaps = SkyImageCollection.read('$GAMMAPY_EXTRA/datasets/fermi_survey/all.fits.gz')

    Then try tab completion on the ``skymaps`` object.
    """
    # Real class attributes have to be defined here
    _map_names = []
    name = None
    meta = None
    wcs = None

    def __init__(self, name=None, wcs=None, meta=None, **kwargs):
        # Set real class attributes
        self.name = name
        self.wcs = wcs
        self.meta = meta

        # Everything else is stored as dict entries
        for key in kwargs:
            self[key] = kwargs[key]

    def __setitem__(self, key, item):
        """
        Overwrite __setitem__ operator to remember order the sky maps are added
        to the collection, by storing it in the _map_names list.
        """
        if isinstance(item, np.ndarray):
            item = SkyImage(name=key, data=item, wcs=self.wcs)
        if isinstance(item, SkyImage):
            self._map_names.append(key)
        super(SkyImageCollection, self).__setitem__(key, item)

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
            skymap = SkyImage.from_image_hdu(hdu)

            # This forces lower case map names, but only on the collection object
            # When writing to fits again the skymap.name attribute is used.
            name = skymap.name.lower()
            kwargs[name] = skymap
            _map_names.append(name)
        _ = cls(**kwargs)
        _._map_names = _map_names
        return _

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
            if isinstance(self[name], SkyImage):
                hdu = self[name].to_image_hdu()

                # For now add common collection meta info to the single map headers
                hdu.header.update(self.meta)
                hdu.name = name
                hdulist.append(hdu)
            else:
                log.warn("Can't save {} to file, not a sky map.".format(name))
        hdulist.writeto(filename, **kwargs)

    def info(self):
        """
        Print summary info about the sky map collection.
        """
        print(str(self))

    def __str__(self):
        """
        String representation of the sky map collection.
        """
        info = ''
        for name in self.get('_map_names', sorted(self)):
            info += self[name].__str__()
            info += '\n'
        return info
