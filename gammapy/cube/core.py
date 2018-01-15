# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
import astropy.units as u
from astropy.units import Quantity
from astropy.table import Table
from astropy.wcs import WCS
from astropy.utils import lazyproperty
from ..utils.scripts import make_path
from ..utils.energy import EnergyBounds, Energy
from ..utils.fits import SmartHDUList, fits_header_to_meta_dict, table_to_fits_table
from ..image.core import SkyImage, MapBase
from ..spectrum import LogEnergyAxis
from ..spectrum.utils import _trapz_loglog

__all__ = [
    'SkyCube',
]


class SkyCube(MapBase):
    """Sky cube with dimensions lon, lat and energy.

    .. note::

        A new set of map and cube classes is being developed in `gammapy.maps`
        and long-term will replace the existing `gammapy.image.SkyImage` and
        `gammapy.cube.SkyCube` classes. Please consider trying out `gammapy.maps`
        and changing your scripts to use those new classes. See :ref:`maps`.

    The order of the sky cube axes is defined as following:

    * The ``data`` array axis order is ``(energy, lat, lon)``.
    * The ``wcs`` object is a two dimensional celestial WCS with axis order ``(lon, lat)``.

    For further information, see :ref:`cube`.

    Parameters
    ----------
    name : str
        Name of the cube.
    data : `~numpy.ndarray`
        Data array.
    wcs : `~astropy.wcs.WCS`
        WCS transformation object.
    energy_axis : `~gammapy.spectrum.LogEnergyAxis`
        Energy axis object, defining the energy transformation.
    meta : `~collections.OrderedDict`
        Dictionary to store meta data.
    """

    def __init__(self, name=None, data=None, wcs=None, energy_axis=None, meta=None):
        # TODO: check validity of inputs
        self.name = name
        # TODO: In gammapy SkyCube is used sometimes with ndim = 2 for cubes
        # with a single energy band
        if not data.ndim > 1:
            raise ValueError('Dimension of the data must be ndim = 3, but is '
                             'ndim = {}'.format(data.ndim))
        self.data = data
        self.wcs = wcs
        self.meta = meta
        self.energy_axis = energy_axis

    @lazyproperty
    def _interpolate_data(self):
        """Interpolate data using `~scipy.interpolate.RegularGridInterpolator`."""
        from scipy.interpolate import RegularGridInterpolator

        # set up log interpolation
        unit = self.data.unit
        # TODO: how to deal with zero values?
        log_data = np.log(self.data.value)
        z = np.arange(self.data.shape[0])
        y = np.arange(self.data.shape[1])
        x = np.arange(self.data.shape[2])

        f = RegularGridInterpolator((z, y, x), log_data, fill_value=None,
                                    bounds_error=False)

        def interpolate(z, y, x, method='linear'):
            shape = z.shape
            coords = np.column_stack([z.flat, y.flat, x.flat])
            val = f(coords, method=method)
            return Quantity(np.exp(val).reshape(shape), unit)

        return interpolate

    @classmethod
    def read_hdu(cls, hdu_list):
        """Read sky cube from HDU list.

        Parameters
        ----------
        object_hdu : `~astropy.io.fits.ImageHDU`
            Image HDU object to be read
        energy_table_hdu : `~astropy.io.fits.TableHDU`
            Table HDU object giving energies of each slice
            of the Image HDU object_hdu

        Returns
        -------
        sky_cube : `SkyCube`
            Sky cube
        """
        object_hdu = hdu_list[0]
        energy_table_hdu = hdu_list[1]
        data = object_hdu.data
        data = Quantity(data, '1 / (cm2 MeV s sr)')
        # Note: the energy axis of the FITS cube is unusable.
        # We only use proj for LON, LAT and call ENERGY from the table extension
        header = object_hdu.header
        wcs = WCS(header)
        energy = energy_table_hdu.data['Energy']
        energy = Quantity(energy, 'MeV')
        meta = OrderedDict(header)
        return cls(data=data, wcs=wcs, energy=energy, meta=meta)

    @classmethod
    def read(cls, filename, format='fermi-counts'):
        """Read sky cube from FITS file.

        Parameters
        ----------
        filename : str
            File name
        format : {'fermi-counts', 'fermi-background', 'fermi-exposure'}
            Fits file format.

        Returns
        -------
        sky_cube : `SkyCube`
            Sky cube
        """
        filename = str(make_path(filename))
        hdu_list = SmartHDUList.open(filename)
        hdu = hdu_list.get_hdu(hdu_type='image')

        header = hdu.header
        wcs = WCS(header).celestial
        meta = fits_header_to_meta_dict(header)
        data = hdu.data

        # TODO: check and give reference for fermi data units
        # TODO: choose format automatically
        if format == 'fermi-background':
            energy = Table.read(filename, 'ENERGIES')['Energy']
            energy_axis = LogEnergyAxis(Quantity(energy, 'MeV'), mode='center')
            data = Quantity(data, '1 / (cm2 MeV s sr)')
            name = 'flux'
        elif format == 'fermi-counts':
            energy = EnergyBounds.from_ebounds(fits.open(filename)['EBOUNDS'])
            energy_axis = LogEnergyAxis(energy, mode='edges')
            data = Quantity(data, 'count')
            name = 'counts'
        elif format == 'fermi-exposure':
            energy = Table.read(filename, 'ENERGIES')['Energy']
            energy_axis = LogEnergyAxis(Quantity(energy, 'MeV'), mode='center')
            data = Quantity(data, 'cm2 s')
            name = 'exposure'
        else:
            raise ValueError('Not a valid cube fits format')

        obj = cls(name=name, data=data, wcs=wcs, energy_axis=energy_axis, meta=meta)
        obj._header = header
        return obj

    def fill_events(self, events, weights=None):
        """Fill events (modifies ``data`` attribute).

        Parameters
        ----------
        events : `~gammapy.data.EventList`
            Event list
        weights : str, optional
            Column to use as weights (none by default)
        """
        if weights is not None:
            weights = events[weights]
        xx, yy = self.sky_image_ref.wcs_skycoord_to_pixel(events.radec)
        zz = events.energy
        bins = self.energies(mode='edges'), self.sky_image_ref._bins_pix[0], self.sky_image_ref._bins_pix[1]

        data = np.histogramdd([zz, yy, xx], bins, weights=weights)[0]

        self.data = self.data + data

    @classmethod
    def empty(cls, emin=0.5, emax=100, enumbins=10, eunit='TeV', mode='edges', **kwargs):
        """Create empty sky cube with log equal energy binning from the scratch.

        Parameters
        ----------
        emin : float
            Minimum energy.
        emax : float
            Maximum energy.
        enumbins : int
            Number of energy bins.
        eunit : str
            Energy unit.
        mode : {'edges', 'center'}
            Whether emin and emax denote the bin centers or edges.
        kwargs : dict
            Keyword arguments passed to `~gammapy.image.SkyImage.empty` to create
            the spatial part of the cube.

        Examples
        --------
        Create an empty sky cube::

            from gammapy.cube import SkyCube
            cube = SkyCube.empty(nxpix=11, nypix=7, enumbins=3, mode='center',
                            emin=1, emax=100, eunit='TeV')

        Returns
        -------
        empty_cube : `SkyCube`
            Empty sky cube object.
        """
        image = SkyImage.empty(**kwargs)

        if mode == 'edges':
            energy = EnergyBounds.equal_log_spacing(emin, emax, enumbins, eunit)
        elif mode == 'center':
            energy = Energy.equal_log_spacing(emin, emax, enumbins, eunit)
        else:
            raise ValueError("Not a valid mode. Choose either 'center' or 'edges'.")

        energy_axis = LogEnergyAxis(energy, mode=mode)
        data = Quantity(image.data * np.ones(enumbins).reshape((-1, 1, 1)), image.unit)
        return cls(data=data, wcs=image.wcs, energy_axis=energy_axis)

    @classmethod
    def empty_like(cls, reference, energies=None, unit='', fill=0):
        """Create an empty sky cube with a given WCS and energy specification.

        Parameters
        ----------
        reference : `~gammapy.cube.SkyCube` or `~gammapy.image.SkyImage`
            Reference sky cube or image.
        energies : `~gammapy.utils.energy.Energy` or `~gammapy.utils.energy.EnergyBounds` (optional)
            Reference energies, mandatory when a `~gammapy.image.SkyImage` is passed.
        unit : str
            String specifying the data units.
        fill : float
            Value to fill the data array with.

        Examples
        --------
        Create an empty sky cube from an image and energy center specification::

            from astropy import units as u
            from gammapy.image import SkyImage
            from gammapy.cube import SkyCube
            from gammapy.utils.energy import Energy, EnergyBounds

            # define reference image
            image = SkyImage.empty(nxpix=11, nypix=7)

            # define energy binning centers
            energies = Energy.equal_log_spacing(1 * u.TeV, 100 * u.TeV, 3)
            cube = SkyCube.empty_like(reference=image, energies=energies)

            # define energy binning bounds
            energies = EnergyBounds.equal_log_spacing(1 * u.TeV, 100 * u.TeV, 3)
            cube = SkyCube.empty_like(reference=image, energies=energies)

        Returns
        -------
        empty_cube : `SkyCube`
            Empty sky cube object.
        """
        wcs = reference.wcs.copy()

        if isinstance(reference, SkyImage):
            if type(energies) == Energy:
                mode = 'center'
                enumbins = len(energies)
            elif type(energies) == EnergyBounds:
                mode = 'edges'
                enumbins = len(energies) - 1
            else:
                raise ValueError("'energies' must be instance of Energy or EnergyBounds, "
                                 "but {} was given.".format(type(energies)))
            energy_axis = LogEnergyAxis(energies, mode=mode)
            data = np.ones(reference.data.shape)
            data = data * np.ones(enumbins).reshape((-1, 1, 1))

        elif isinstance(reference, SkyCube):
            energy_axis = reference.energy_axis
            data = np.ones(reference.data.shape)

        else:
            raise ValueError("'reference' must be instance of SkyImage or SkyCube")

        return cls(data=data * fill * u.Unit(unit), wcs=wcs,
                   energy_axis=energy_axis)

    def energies(self, mode='center'):
        """Energy coordinate vector.

        Parameters
        ----------
        mode : {'center', 'edges'}
            Return coordinate values at the pixels edges or pixel centers.

        Returns
        -------
        coordinates : `~astropy.units.Quantity`
            Energy
        """
        if mode == 'center':
            z = np.arange(self.data.shape[0])
        elif mode == 'edges':
            z = np.arange(self.data.shape[0] + 1) - 0.5
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

        return self.energy_axis.wcs_pix2world(z)

    @property
    def energy_width(self):
        """Energy bin width vector (`~astropy.units.Quantity`)"""
        ebounds = self.energies(mode='edges')
        return ebounds[1:] - ebounds[:-1]

    @property
    def bin_size(self):
        """Sky cube element bin size (`~astropy.units.Quantity`)

        This is a convenience method which computes this::

            cube.energy_width * cube.sky_image_ref.solid_angle()

        Units could be "TeV" (or whatever ``energy_width`` returns) times "sr"
        """
        solid_angle = self.sky_image_ref.solid_angle()
        de = self.energy_width
        return solid_angle * de[:, np.newaxis, np.newaxis]

    def cutout(self, position, size):
        """Cut out rectangular piece of a cube.

        See `~gammapy.image.SkyImage.cutout()` for details.
        """
        out = []
        for energy in self.energies():
            image = self.sky_image(energy)
            cutout = image.cutout(position=position, size=size)
            out.append(cutout.data)

        data = Quantity(np.stack(out, axis=0), self.data.unit)
        wcs = cutout.wcs.copy()
        return self.__class__(name=self.name, data=data, wcs=wcs, meta=self.meta,
                              energy_axis=self.energy_axis)

    def wcs_skycoord_to_pixel(self, position, energy):
        """Convert world to pixel coordinates.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position on the sky.
        energy : `~astropy.units.Quantity`
            Energy

        Returns
        -------
        (x, y, z) : tuple
            Tuple of x, y, z coordinates.
        """
        x, y = self.sky_image_ref.wcs_skycoord_to_pixel(position)
        z = self.energy_axis.wcs_world2pix(energy)
        # TODO: check order, so that it corresponds to data axis order
        return x, y, z

    def wcs_pixel_to_skycoord(self, x, y, z):
        """Convert pixel to world coordinates.

        Parameters
        ----------
        x : `~numpy.ndarry`
            x coordinate array
        y : `~numpy.ndarry`
            y coordinate array
        z : `~numpy.ndarry`
            z coordinate array

        Returns
        -------
        (position, energy) : tuple
            Tuple of (`~astropy.coordinates.SkyCoord`, `~astropy.units.Quantity`).
        """
        position = self.sky_image_ref.wcs_pixel_to_skycoord(x, y)
        energy = self.energy_axis.wcs_pix2world(z)
        return position, energy

    def to_sherpa_data3d(self, dstype='Data3D'):
        """Convert sky cube to sherpa `Data3D` or `Data3DInt` object.

        Parameters
        ----------
        dstype : {'Data3D', 'Data3DInt'}
            Sherpa data type.
        """
        from .sherpa_ import Data3D, Data3DInt
        energies = self.energies(mode='edges').to("TeV").value
        elo = energies[:-1]
        ehi = energies[1:]
        n_ebins = len(elo)
        if dstype == 'Data3DInt':
            coordinates = self.sky_image_ref.coordinates(mode="edges")
            ra = coordinates.data.lon.degree
            dec = coordinates.data.lat.degree
            ra_cube_hi = np.tile(ra[0:-1, 0:-1], (n_ebins, 1, 1))
            ra_cube_lo = np.tile(ra[0:-1, 1:], (n_ebins, 1, 1))
            dec_cube_hi = np.tile(dec[1:, 0:-1], (n_ebins, 1, 1))
            dec_cube_lo = np.tile(dec[0:-1, 0:-1], (n_ebins, 1, 1))
            elo_cube = elo.reshape(n_ebins, 1, 1) * np.ones_like(ra[0:-1, 0:-1]) * u.TeV
            ehi_cube = ehi.reshape(n_ebins, 1, 1) * np.ones_like(ra[0:-1, 0:-1]) * u.TeV
            return Data3DInt('', elo_cube.ravel(), ra_cube_lo.ravel(), dec_cube_lo.ravel(), ehi_cube.ravel(),
                             ra_cube_hi.ravel(), dec_cube_hi.ravel(), self.data.value.ravel(),
                             self.data.value.ravel().shape)

        elif dstype == 'Data3D':
            coordinates = self.sky_image_ref.coordinates()
            ra = coordinates.data.lon.degree
            dec = coordinates.data.lat.degree
            ra_cube = np.tile(ra, (n_ebins, 1, 1))
            dec_cube = np.tile(dec, (n_ebins, 1, 1))
            elo_cube = elo.reshape(n_ebins, 1, 1) * np.ones_like(ra) * u.TeV
            ehi_cube = ehi.reshape(n_ebins, 1, 1) * np.ones_like(ra) * u.TeV
            return Data3D('', elo_cube.ravel(), ehi_cube.ravel(), ra_cube.ravel(),
                          dec_cube.ravel(), self.data.value.ravel(),
                          self.data.value.ravel().shape)

        else:
            raise ValueError('Invalid sherpa data type.')

    def sky_image(self, energy, interpolation=None):
        """Slice a 2-dim `~gammapy.image.SkyImage` from the cube at a given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy value
        interpolation : {None, 'linear', 'nearest'}
            Interpolate data values between energies. None corresponds to
            'nearest', but might have advantages in performance, because
            no interpolator is set up.

        Returns
        -------
        image : `~gammapy.image.SkyImage`
            2-dim sky image
        """
        # TODO: should we pass something in SkyImage (we speak about meta)?
        z = self.energy_axis.wcs_world2pix(energy)

        if interpolation:
            y = np.arange(self.data.shape[1])
            x = np.arange(self.data.shape[2])
            z, y, x = np.meshgrid(z, y, x, indexing='ij')
            data = self._interpolate_data(z, y, x)[0]
        else:
            data = self.data[int(np.rint(z))].copy()
        wcs = self.wcs.deepcopy() if self.wcs else None
        return SkyImage(name=self.name, data=data, wcs=wcs)

    def sky_image_idx(self, idx):
        """Slice a 2-dim `~gammapy.image.SkyImage` from the cube at a given index.

        Parameters
        ----------
        idx : int
            Index of the sky image.

        Returns
        -------
        image : `~gammapy.image.SkyImage`
            2-dim sky image
        """
        data = self.data[idx].copy()
        wcs = self.wcs.deepcopy() if self.wcs else None
        return SkyImage(name=self.name, data=data, wcs=wcs)

    @lazyproperty
    def sky_image_ref(self):
        """Empty reference `~gammapy.image.SkyImage`.

        Examples
        --------
        Can be used to access the spatial information of the cube:

        >>> from gammapy.cube import SkyCube
        >>> cube = SkyCube.empty()
        >>> coords = cube.sky_image_ref.coordinates()
        >>> solid_angle = cube.sky_image_ref.solid_angle()
        """
        wcs = self.wcs.celestial
        data = np.zeros_like(self.data[0])
        return SkyImage(name=self.name, data=data, wcs=wcs)

    def lookup(self, position, energy, interpolation=None):
        """Lookup value in the cube at given sky position and energy.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position on the sky.
        energy : `~astropy.units.Quantity`
            Energy
        interpolation : {None, 'linear', 'nearest'}
            Interpolate data values between energies.

        Returns
        -------
        value : `~astropy.units.Quantity`
            Value at the given sky position and energy.
        """
        # TODO: add interpolation option using NDDataArray

        x, y, z = self.wcs_skycoord_to_pixel(position, energy)

        if interpolation:
            vals = self._interpolate_data(z, y, x)
        else:
            vals = self.data[np.rint(z).astype('int'), np.rint(y).astype('int'),
                             np.rint(x).astype('int')]
        return vals

    def show(self, viewer='mpl', ds9options=None, **kwargs):
        """Show sky cube in image viewer.

        Parameters
        ----------
        viewer : {'mpl', 'ds9'}
            Which image viewer to use. Option 'ds9' requires ds9 to be installed.
        ds9options : list, optional
            List of options passed to ds9. E.g. ['-cmap', 'heat', '-scale', 'log'].
            Any valid ds9 command line option can be passed.
            See http://ds9.si.edu/doc/ref/command.html for details.
        **kwargs : dict
            Keyword arguments passed to `matplotlib.pyplot.imshow`.
        """
        from ipywidgets import interact

        if viewer == 'mpl':
            max_ = self.data.shape[0] - 1

            def show_image(idx):
                energy = self.energy_axis.wcs_pix2world(idx)
                image = self.sky_image(energy)
                image.data = image.data.value
                try:
                    norm = kwargs['norm']
                    norm.vmax = np.nanmax(image.data)
                except KeyError:
                    pass
                image.show(**kwargs)

            return interact(show_image, idx=(0, max_, 1))
        elif viewer == 'ds9':
            raise NotImplementedError
        else:
            raise ValueError('Invalid viewer: {}'.format(viewer))

    def plot_rgb(self, ax=None, fig=None, **kwargs):
        """Plot sky cube as RGB image.

        Parameters
        ----------
        ax : `~astropy.visualization.wcsaxes.WCSAxes`, optional
            WCS axis object to plot on.
        fig : `~matplotlib.figure.Figure`, optional
            Figure

        Returns
        -------
        ax : `~astropy.visualization.wcsaxes.WCSAxes`
            WCS axis object
        """
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.gcf()

        if ax is None:
            ax = fig.add_subplot(1, 1, 1, projection=self.wcs)

        kwargs['origin'] = kwargs.get('origin', 'lower')
        kwargs['interpolation'] = kwargs.get('interpolation', 'None')

        ne = self.data.shape[0]
        if ne != 3:
            raise ValueError("Energy axes must be length 3, but is {}".format(ne))

        data_rgb = []
        for idx in range(3):
            data_rgb.append(self.data[idx, :, :])

        ax.imshow(np.dstack(data_rgb), **kwargs)

        try:
            ax.coords['glon'].set_axislabel('Galactic Longitude')
            ax.coords['glat'].set_axislabel('Galactic Latitude')
        except KeyError:
            ax.coords['ra'].set_axislabel('Right Ascension')
            ax.coords['dec'].set_axislabel('Declination')

        # without this the axis limits are changed when calling scatter
        ax.autoscale(enable=False)

        return ax

    def sky_image_integral(self, emin, emax, nbins=10, per_decade=False, interpolation='linear'):
        """Integrate cube along the energy axes using the log-log trapezoidal rule.

        Parameters
        ----------
        emin : `~astropy.units.Quantity`
            Integration range minimum.
        emax : `~astropy.units.Quantity`
            Integration range maximum.
        nbins : int, optional
            Number of grid points used for the integration.
        per_decade : bool
            Whether nbins is per decade.
        interpolation : {None, 'linear', 'nearest'}
            Interpolate data values between energies.

        Returns
        -------
        image : `~gammapy.image.SkyImage`
            Integral image.
        """
        y, x = np.indices(self.data.shape[1:])

        if interpolation:
            energy = Energy.equal_log_spacing(emin, emax, nbins, per_decade=per_decade)
            z = self.energy_axis.wcs_world2pix(energy).reshape(-1, 1, 1)
            y = np.arange(self.data.shape[1])
            x = np.arange(self.data.shape[2])
            z, y, x = np.meshgrid(z, y, x, indexing='ij')
            data = self._interpolate_data(z, y, x)
        else:
            energy_slice = self._get_energy_slice_for_energy_range(emin, emax)
            energy = self.energies()[energy_slice]
            data = self.data[energy_slice]

        integral = _trapz_loglog(data, energy, axis=0)
        name = 'integrated {}'.format(self.name)
        return SkyImage(name=name, data=integral, wcs=self.wcs.celestial)

    # TODO: is this the rounding we want? document what it does!
    def _get_energy_slice_for_energy_range(self, emin, emax):
        idx_min = np.rint(self.energy_axis.wcs_world2pix(emin)).astype('int')
        idx_max = np.rint(self.energy_axis.wcs_world2pix(emax)).astype('int')
        return slice(idx_min, idx_max)

    def sky_image_sum(self, emin, emax):
        """Sum cube along the energy axis.

        Similar to the ``sky_image_integral`` method,
        but not doing interpolation / integration.

        Just selects a subset of energy bins and sums those.
        This is useful for counts.
        """
        sky_image = self.sky_image_ref()
        energy_slice = self._get_energy_slice_for_energy_range(emin, emax)
        sky_image.data = self.data[energy_slice].sum(axis=1)
        return sky_image

    def reproject(self, reference, mode='interp', *args, **kwargs):
        """Reproject spatial dimensions onto a reference image.

        Parameters
        ----------
        reference : `~astropy.io.fits.Header`, `~gammapy.image.SkyImage` or `SkyCube`
            Reference wcs specification to reproject the data on.
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
        reprojected_cube : `SkyCube`
            Cube spatially reprojected to the reference.
        """
        if isinstance(reference, SkyCube):
            reference = reference.sky_image_ref

        out = []
        for energy in self.energies():
            image = self.sky_image(energy)
            image_out = image.reproject(reference, mode=mode, *args, **kwargs)
            out.append(image_out.data)

        data = Quantity(np.stack(out, axis=0), self.data.unit)
        wcs = image_out.wcs.copy()
        return self.__class__(name=self.name, data=data, wcs=wcs, meta=self.meta,
                              energy_axis=self.energy_axis)

    def convolve(self, kernels, **kwargs):
        """Convolve cube with a given set of kernels.

        Parameters
        ----------
        kernels : list or `~numpy.ndarray`
            List of 2D convolution kernels or 3D array. The energy axis
            must correspond to array axis=0.

        Returns
        -------
        convolved : `SkyCube`
            Convolved cube.
        """
        data = []
        if not len(kernels) == self.data.shape[0]:
            raise ValueError('Number of kernels must match size of energy axis'
                             ' of the cube.')

        for idx, kernel in enumerate(kernels):
            image = self.sky_image_idx(idx)
            data.append(image.convolve(kernel, **kwargs).data)

        convolved = u.Quantity(data)
        wcs = self.wcs.deepcopy() if self.wcs else None
        return self.__class__(name=self.name, data=convolved, wcs=wcs,
                              energy_axis=self.energy_axis)

    def to_fits(self, format):
        """Write to FITS HDU list.

        Parameters
        ----------
        format : {'fermi-counts', 'fermi-background', 'fermi-exposure'}
            Fits file format.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            * hdu_list[0] : `~astropy.io.fits.PrimaryHDU`
                Image array of data
            * hdu_list[1] : `~astropy.io.fits.BinTableHDU`
                Table of energies
        """
        image_hdu = fits.PrimaryHDU(self.data.value, self.wcs.to_header())
        image_hdu.header['SPECUNIT'] = '{0.unit:FITS}'.format(self.data)

        if format == 'fermi-counts':
            energies = self.energies(mode='edges')
            # for BinTableHDU's the data must be added via a Table object
            energy_table = Table()
            energy_table['E_MIN'] = energies[:-1]
            energy_table['E_MAX'] = energies[1:]
            energy_table.meta['name'] = 'EBOUNDS'
            energy_hdu = table_to_fits_table(energy_table)
        elif format in ['fermi-exposure', 'fermi-background']:
            # for BinTableHDU's the data must be added via a Table object
            energy_table = Table()
            energy_table['Energy'] = self.energies()
            energy_table.meta['name'] = 'ENERGIES'
            energy_hdu = table_to_fits_table(energy_table)
        else:
            raise ValueError('Not a valid cube fits format')

        hdu_list = fits.HDUList([image_hdu, energy_hdu])
        return hdu_list

    def to_images(self):
        """Convert to `~gammapy.cube.SkyCubeImages`."""
        from .images import SkyCubeImages
        energies = self.energies(mode='center')
        images = [self.sky_image(energy) for energy in energies]
        return SkyCubeImages(self.name, images, self.wcs, energies)

    def spectrum(self, region):
        """Extract spectrum in a given sky region.

        Parameters
        ----------
        region : `~regions.SkyRegion`
            Sky region to extract the spectrum from.

        Returns
        -------
        spectrum : `~astropy.table.Table`
            Summed spectrum of pixels in the mask.
        """
        spectrum = Table()

        # store energy binning
        energies = self.energies('edges')
        e_ref = self.energies('center')
        spectrum['e_min'] = energies[:-1]
        spectrum['e_max'] = energies[1:]
        spectrum['e_ref'] = e_ref

        # mask region and sum
        mask = self.region_mask(region)
        value = (self.data * mask.data).sum(-1).sum(-1)
        spectrum['value'] = value
        return spectrum

    def region_mask(self, region):
        """Create a boolean cube mask for a region.

        The mask is:

        - ``True`` for pixels inside the region
        - ``False`` for pixels outside the region

        Parameters
        ----------
        region : `~regions.PixelRegion` or `~regions.SkyRegion`
            Region in pixel or sky coordinates.

        Returns
        -------
        mask : `SkyCube`
            A boolean sky cube mask.
        """
        mask = self.sky_image_ref.region_mask(region)
        data = mask.data * np.ones(self.data.shape, dtype='bool') * u.Unit('')
        wcs = self.wcs.deepcopy() if self.wcs else None
        return self.__class__(name=self.name, data=data.astype('bool'), wcs=wcs,
                              energy_axis=self.energy_axis)

    def write(self, filename, format='fermi-counts', **kwargs):
        """Write to FITS file.

        Parameters
        ----------
        filename : str
            Filename
        format : {'fermi-counts', 'fermi-background', 'fermi-exposure'}
            Fits file format.
        """
        filename = str(make_path(filename))
        self.to_fits(format).writeto(filename, **kwargs)

    def __str__(self):
        ss = "Sky cube {} with shape={}".format(self.name, self.data.shape)
        if self.data.unit is u.dimensionless_unscaled:
            ss += ":\n"
        else:
            ss += " and unit={}:\n".format(self.data.unit)

        ss += " n_lon:    {:5d}  type_lon:    {:15s}  unit_lon:    {}\n".format(
            self.data.shape[2], self.wcs.wcs.ctype[0], self.wcs.wcs.cunit[0])
        ss += " n_lat:    {:5d}  type_lat:    {:15s}  unit_lat:    {}\n".format(
            self.data.shape[1], self.wcs.wcs.ctype[1], self.wcs.wcs.cunit[1])
        ss += " n_energy: {:5d}  unit_energy: {}\n".format(
            self.data.shape[0], self.energy_axis._eunit)

        return ss

    def info(self):
        """Print summary info about the cube."""
        print(repr(self))

    @staticmethod
    def assert_allclose(cube1, cube2):
        from ..utils.testing import assert_wcs_allclose
        assert cube1.name == cube2.name
        assert_allclose(cube1.data, cube2.data)

        # TODO: add check_unit option, just like SkyImage has it.

        assert_allclose(cube1.energies(), cube2.energies())
        assert_wcs_allclose(cube1.wcs, cube2.wcs)

    def to_wcs_nd_map(self, energy_axis_mode='center'):
        """Convert to a `gammapy.maps.WcsNDMap`.

        There is no copy of the ``data`` or ``wcs`` object, this conversion is cheap.

        This is meant to help migrate code using `SkyCube`
        over to the new maps classes.
        """
        from gammapy.maps import WcsNDMap, WcsGeom, MapAxis

        if energy_axis_mode == 'center':
            energy = self.energies(mode='center')
            energy_axis = MapAxis.from_nodes(energy.value, unit=energy.unit)
        elif energy_axis_mode == 'edges':
            energy = self.energies(mode='edges')
            energy_axis = MapAxis.from_edges(energy.value, unit=energy.unit)
        else:
            raise ValueError('Invalid energy_axis_mode: {}'.format(energy_axis_mode))

        # Axis order in SkyCube: energy, lat, lon
        npix = (self.data.shape[2], self.data.shape[1])

        geom = WcsGeom(wcs=self.wcs, npix=npix, axes=[energy_axis])

        # TODO: change maps and SkyCube to have a unit attribute
        # For now, SkyCube is a mix of numpy array and quantity in `data`
        # and we just strip the unit here
        data = np.asarray(self.data)
        # unit = getattr(self.data, 'unit', None)

        return WcsNDMap(geom=geom, data=data)

    @classmethod
    def from_wcs_nd_map(cls, wcs_map_nd):
        """Create from a `gammapy.maps.WcsNDMap`.

        There is no copy of the ``data`` or ``wcs`` object, this conversion is cheap.

        This is meant to help migrate code using `SkyCube`
        over to the new maps classes.
        """
        geom_axis = wcs_map_nd.geom.axes[0]

        if geom_axis.node_type == 'center':
            energy = geom_axis.center * geom_axis.unit
            energy_axis = LogEnergyAxis(energy, mode='center')
        elif geom_axis.node_type == 'edges':
            energy = geom_axis.edges * geom_axis.unit
            energy_axis = LogEnergyAxis(energy, mode='edges')
        else:
            raise ValueError('Not supported: node_type: {}'.format(geom_axis.node_type))

        data = wcs_map_nd.data
        # TODO: copy unit once it's added to
        # if wcs_map_nd.unit is not None:
        #     data = data * wcs_map_nd.unit

        return cls(
            data=data,
            wcs=wcs_map_nd.geom.wcs,
            energy_axis=energy_axis,
        )
