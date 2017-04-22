# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.io import fits
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from astropy.convolution.utils import discretize_oversample_2D
from ..image.models import Gauss2DPDF
from ..utils.array import array_stats_str
from ..utils.energy import Energy


__all__ = [
    'TablePSF',
    'EnergyDependentTablePSF',
]

# Default PSF spline keyword arguments
# TODO: test and document
DEFAULT_PSF_SPLINE_KWARGS = dict(k=1, s=0)


class TablePSF(object):
    r"""Radially-symmetric table PSF.

    This PSF represents a :math:`PSF(\theta)=dP / d\Omega(\theta)`
    spline interpolation curve for a given set of offset :math:`\theta`
    and :math:`PSF` points.

    Uses `scipy.interpolate.UnivariateSpline`.

    Parameters
    ----------
    offset : `~astropy.units.Quantity` with angle units
        Offset angle array
    dp_domega : `~astropy.units.Quantity` with sr^-1 units
        PSF value array
    spline_kwargs : dict
        Keyword arguments passed to `~scipy.interpolate.UnivariateSpline`

    Notes
    -----
    * This PSF class works well for model PSFs of arbitrary shape (represented by a table),
      but might give unstable results if the PSF has noise.
      E.g. if ``dp_domega`` was estimated from histograms of real or simulated event data
      with finite statistics, it will have noise and it is your responsibility
      to check that the interpolating spline is reasonable.
    * To customize the spline, pass keyword arguments to `~scipy.interpolate.UnivariateSpline`
      in ``spline_kwargs``. E.g. passing ``dict(k=1)`` changes from the default cubic to
      linear interpolation.
    * TODO: evaluate spline for ``(log(offset), log(PSF))`` for numerical stability?
    * TODO: merge morphology.theta class functionality with this class.
    * TODO: add FITS I/O methods
    * TODO: add ``normalize`` argument to ``__init__`` with default ``True``?
    * TODO: ``__call__`` doesn't show up in the html API docs, but it should:
      https://github.com/astropy/astropy/pull/2135
    """

    def __init__(self, offset, dp_domega, spline_kwargs=DEFAULT_PSF_SPLINE_KWARGS):

        self._offset = Angle(offset).to('radian')
        self._dp_domega = Quantity(dp_domega).to('sr^-1')

        assert self._offset.ndim == self._dp_domega.ndim == 1
        assert self._offset.shape == self._dp_domega.shape

        # Store input arrays as quantities in default internal units
        self._dp_dtheta = (2 * np.pi * self._offset * self._dp_domega).to('radian^-1')
        self._spline_kwargs = spline_kwargs

        self._compute_splines(spline_kwargs)

    @classmethod
    def from_shape(cls, shape, width, offset):
        """Make TablePSF objects with commonly used shapes.

        This function is mostly useful for examples and testing.

        Parameters
        ----------
        shape : {'disk', 'gauss'}
            PSF shape.
        width : `~astropy.units.Quantity` with angle units
            PSF width angle (radius for disk, sigma for Gauss).
        offset : `~astropy.units.Quantity` with angle units
            Offset angle

        Returns
        -------
        psf : `TablePSF`
            Table PSF

        Examples
        --------
        >>> import numpy as np
        >>> from astropy.coordinates import Angle
        >>> from gammapy.irf import TablePSF
        >>> TablePSF.from_shape(shape='gauss', width=Angle(0.2, 'deg'),
                             offset=Angle(np.linspace(0, 0.7, 100), 'deg'))
        """
        width = Angle(width)
        offset = Angle(offset)

        if shape == 'disk':
            amplitude = 1 / (np.pi * width.radian ** 2)
            psf_value = np.where(offset < width, amplitude, 0)
        elif shape == 'gauss':
            gauss2d_pdf = Gauss2DPDF(sigma=width.radian)
            psf_value = gauss2d_pdf(offset.radian)
        else:
            raise ValueError('Invalid shape: disk or gauss. Input was: {}'.format(shape))

        psf_value = Quantity(psf_value, 'sr^-1')

        return cls(offset, psf_value)

    def info(self):
        """Print basic info."""
        ss = array_stats_str(self._offset.degree, 'offset')
        ss += 'integral = {0}\n'.format(self.integral())

        for containment in [50, 68, 80, 95]:
            radius = self.containment_radius(0.01 * containment)
            ss += ('containment radius {0} deg for {1}%\n'
                   .format(radius.degree, containment))

        return ss

    # TODO: remove because it's not flexible enough?
    def __call__(self, lon, lat):
        """Evaluate PSF at a 2D position.

        The PSF is centered on ``(0, 0)``.

        Parameters
        ----------
        lon, lat : `~astropy.coordinates.Angle`
            Longitude / latitude position

        Returns
        -------
        psf_value : `~astropy.units.Quantity`
            PSF value
        """
        center = SkyCoord(0, 0, unit='radian')
        point = SkyCoord(lon, lat)
        offset = center.separation(point)
        return self.evaluate(offset)

    def kernel(self, reference):
        """
        Make a 2-dimensional kernel image.

        The kernel image is evaluated on a cartesian grid defined by the
        reference sky image.

        Parameters
        ----------
        reference : `~gammapy.image.SkyImage` or `~gammapy.cube.SkyCube`
            Reference sky image or sky cube defining the spatial grid.
        offset_max : `~astropy.coordinates.Angle`
             maximal radius up to which the PSF is evaluated.
        normalize : bool
            Whether to normalize the kernel.

        Returns
        -------
        kernel : `~astropy.units.Quantity`
            Kernel 2D image of Quantities

        """
        from ..cube import SkyCube
        from ..background import fill_acceptance_image
        if isinstance(reference, SkyCube):
            reference = reference.sky_image_ref
        header = reference.to_image_hdu().header
        kernel=fill_acceptance_image(header, reference.center, self._offset.to("deg"),
                                                         self._dp_domega, self._offset.to("deg")[-1]).data
        return kernel
    def evaluate(self, offset, quantity='dp_domega'):
        r"""Evaluate PSF.

        The following PSF quantities are available:

        * 'dp_domega': PDF per 2-dim solid angle :math:`\Omega` in sr^-1

            .. math:: \frac{dP}{d\Omega}

        * 'dp_dtheta': PDF per 1-dim offset :math:`\theta` in radian^-1

            .. math:: \frac{dP}{d\theta} = 2 \pi \theta \frac{dP}{d\Omega}

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            Offset angle
        quantity : {'dp_domega', 'dp_dtheta'}
            Which PSF quantity?

        Returns
        -------
        psf_value : `~astropy.units.Quantity`
            PSF value
        """
        offset = Angle(offset)

        shape = offset.shape
        x = np.array(offset.radian).flat

        if quantity == 'dp_domega':
            y = self._dp_domega_spline(x)
            unit = 'sr^-1'
        elif quantity == 'dp_dtheta':
            y = self._dp_dtheta_spline(x)
            unit = 'radian^-1'
        else:
            ss = 'Invalid quantity: {0}\n'.format(quantity)
            ss += "Choose one of: 'dp_domega', 'dp_dtheta'"
            raise ValueError(ss)

        y = np.clip(a=y, a_min=0, a_max=None)
        return Quantity(y, unit).reshape(shape)

    def integral(self, offset_min=None, offset_max=None):
        """Compute PSF integral, aka containment fraction.

        Parameters
        ----------
        offset_min, offset_max : `~astropy.units.Quantity` with angle units
            Offset angle range

        Returns
        -------
        integral : float
            PSF integral
        """
        if offset_min is None:
            offset_min = self._offset[0]
        else:
            offset_min = Angle(offset_min)

        if offset_max is None:
            offset_max = self._offset[-1]
        else:
            offset_max = Angle(offset_max)

        offset_min = self._offset_clip(offset_min)
        offset_max = self._offset_clip(offset_max)

        cdf_min = self._cdf_spline(offset_min)
        cdf_max = self._cdf_spline(offset_max)

        return cdf_max - cdf_min

    def containment_radius(self, fraction):
        """Containment radius.

        Parameters
        ----------
        fraction : array_like
            Containment fraction (range 0 .. 1)

        Returns
        -------
        radius : `~astropy.coordinates.Angle`
            Containment radius angle
        """
        radius = self._ppf_spline(fraction)
        return Angle(radius, 'radian').to('deg')

    def normalize(self):
        """Normalize PSF to unit integral.

        Computes the total PSF integral via the :math:`dP / d\theta` spline
        and then divides the :math:`dP / d\theta` array.
        """
        integral = self.integral()

        self._dp_dtheta /= integral

        # Don't divide by 0
        EPS = 1e-6
        offset = np.clip(self._offset.radian, EPS, None)
        offset = Quantity(offset, 'radian')
        self._dp_domega = self._dp_dtheta / (2 * np.pi * offset)
        self._compute_splines(self._spline_kwargs)

    def broaden(self, factor, normalize=True):
        r"""Broaden PSF by scaling the offset array.

        For a broadening factor :math:`f` and the offset
        array :math:`\theta`, the offset array scaled
        in the following way:

        .. math::
            \theta_{new} = f \times \theta_{old}
            \frac{dP}{d\theta}(\theta_{new}) = \frac{dP}{d\theta}(\theta_{old})

        Parameters
        ----------
        factor : float
            Broadening factor
        normalize : bool
            Normalize PSF after broadening
        """
        self._offset *= factor
        # We define broadening such that self._dp_domega remains the same
        # so we only have to re-compute self._dp_dtheta and the slines here.
        self._dp_dtheta = (2 * np.pi * self._offset * self._dp_domega).to('radian^-1')
        self._compute_splines(self._spline_kwargs)

        if normalize:
            self.normalize()

    def plot_psf_vs_theta(self, ax=None, quantity='dp_domega', **kwargs):
        """Plot PSF vs offset.

        TODO: describe PSF ``quantity`` argument in a central place and link to it from here.
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        x = self._offset.to('deg')
        y = self.evaluate(self._offset, quantity)

        ax.plot(x.value, y.value, **kwargs)
        ax.loglog()
        ax.set_xlabel('Offset ({0})'.format(x.unit))
        ax.set_ylabel('PSF ({0})'.format(y.unit))

    def _compute_splines(self, spline_kwargs=DEFAULT_PSF_SPLINE_KWARGS):
        """Compute two splines representing the PSF.

        * `_dp_domega_spline` is used to evaluate the 2D PSF.
        * `_dp_dtheta_spline` is not really needed for most applications,
          but is available via `eval`.
        * `_cdf_spline` is used to compute integral and for normalisation.
        * `_ppf_spline` is used to compute containment radii.
        """
        from scipy.interpolate import UnivariateSpline

        # Compute spline and normalize.
        x, y = self._offset.value, self._dp_domega.value
        self._dp_domega_spline = UnivariateSpline(x, y, **spline_kwargs)

        x, y = self._offset.value, self._dp_dtheta.value
        self._dp_dtheta_spline = UnivariateSpline(x, y, **spline_kwargs)

        # We use the terminology for scipy.stats distributions
        # http://docs.scipy.org/doc/scipy/reference/tutorial/stats.html#common-methods

        # cdf = "cumulative distribution function"
        self._cdf_spline = self._dp_dtheta_spline.antiderivative()

        # ppf = "percent point function" (inverse of cdf)
        # Here's a discussion on methods to compute the ppf
        # http://mail.scipy.org/pipermail/scipy-user/2010-May/025237.html
        x = self._offset.value
        y = self.integral(Angle(0, 'rad'), self._offset)

        # This is a hack to stabilize the univariate spline. Only use the first
        # i entries, where the integral is srictly increasing, to build the spline.
        i = (np.diff(y) <= 0).argmax()
        i = len(y) if i == 0 else i
        self._ppf_spline = UnivariateSpline(y[:i], x[:i], **spline_kwargs)

    def _offset_clip(self, offset):
        """Clip to offset support range, because spline extrapolation is unstable."""
        offset = Angle(offset, 'radian').radian
        offset = np.clip(offset, 0, self._offset[-1].radian)
        return offset


class EnergyDependentTablePSF(object):
    """Energy-dependent radially-symmetric table PSF (``gtpsf`` format).

    TODO: add references and explanations.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy (1-dim)
    offset : `~astropy.units.Quantity` with angle units
        Offset angle (1-dim)
    exposure : `~astropy.units.Quantity`
        Exposure (1-dim)
    psf_value : `~astropy.units.Quantity`
        PSF (2-dim with axes: psf[energy_index, offset_index]
    """

    def __init__(self, energy, offset, exposure=None, psf_value=None):

        self.energy = Quantity(energy).to('GeV')
        self.offset = Quantity(offset).to('radian')
        if not exposure:
            self.exposure = Quantity(np.ones(len(energy)), 'cm^2 s')
        else:
            self.exposure = Quantity(exposure).to('cm^2 s')

        if not psf_value:
            self.psf_value = Quantity(np.zeros(len(energy), len(offset)), 'sr^-1')
        else:
            self.psf_value = Quantity(psf_value).to('sr^-1')

        # Cache for TablePSF at each energy ... only computed when needed
        self._table_psf_cache = [None] * len(self.energy)

    @classmethod
    def from_fits(cls, hdu_list):
        """Create `EnergyDependentTablePSF` from ``gtpsf`` format HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``THETA`` and ``PSF`` extensions.
        """
        offset = Angle(hdu_list['THETA'].data['Theta'], 'deg')
        energy = Quantity(hdu_list['PSF'].data['Energy'], 'MeV')
        exposure = Quantity(hdu_list['PSF'].data['Exposure'], 'cm^2 s')
        psf_value = Quantity(hdu_list['PSF'].data['PSF'], 'sr^-1')

        return cls(energy, offset, exposure, psf_value)

    def to_fits(self):
        """Convert to FITS HDU list format.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        # TODO: write HEADER keywords as gtpsf

        data = self.offset
        theta_hdu = fits.BinTableHDU(data=data, name='Theta')

        data = [self.energy, self.exposure, self.psf_value]
        psf_hdu = fits.BinTableHDU(data=data, name='PSF')

        hdu_list = fits.HDUList([theta_hdu, psf_hdu])
        return hdu_list

    @classmethod
    def read(cls, filename):
        """Create `EnergyDependentTablePSF` from ``gtpsf``-format FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        hdu_list = fits.open(filename)
        return cls.from_fits(hdu_list)

    def write(self, *args, **kwargs):
        """Write to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(*args, **kwargs)

    def evaluate(self, energy=None, offset=None,
                 interp_kwargs=None):
        """Interpolate the value of the `EnergyOffsetArray` at a given offset and Energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            energy value
        offset : `~astropy.coordinates.Angle`
            offset value
        interp_kwargs : dict
            option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        values : `~astropy.units.Quantity`
            Interpolated value
        """
        if not interp_kwargs:
            interp_kwargs = dict(bounds_error=False, fill_value=None)

        from scipy.interpolate import RegularGridInterpolator
        if energy is None:
            energy = self.energy
        if offset is None:
            offset = self.offset
        energy = Energy(energy).to('TeV')
        offset = Angle(offset).to('deg')
        energy_bin = self.energy.to('TeV')
        offset_bin = self.offset.to('deg')
        points = (energy_bin, offset_bin)
        interpolator = RegularGridInterpolator(points, self.psf_value, **interp_kwargs)
        ee, off = np.meshgrid(energy.value, offset.value, indexing='ij')
        shape = ee.shape
        pix_coords = np.column_stack([ee.flat, off.flat])
        data_interp = interpolator(pix_coords)
        return Quantity(data_interp.reshape(shape), self.psf_value.unit)

    def table_psf_at_energy(self, energy, interp_kwargs=None, **kwargs):
        """Evaluate the `EnergyOffsetArray` at one given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        interp_kwargs : dict
            Option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        table : `~astropy.table.Table`
            Table with two columns: offset, value
        """
        psf_value = self.evaluate(energy, None, interp_kwargs)[0, :]
        table_psf = TablePSF(self.offset, psf_value, **kwargs)
        return table_psf

    def kernels(self, cube, **kwargs):
        """
        Make a set of 2D kernel images, representing the PSF at different energies.

        The kernel image is evaluated on the spatial and energy grid defined by
        the reference sky cube.

        Parameters
        ----------
        cube : `~gammapy.cube.SkyCube`
            Reference sky cube.
        kwargs : dict
            Keyword arguments passed to `EnergyDependentTablePSF.table_psf_in_energy_band()`.

        Returns
        -------
        kernels : list of `~numpy.ndarray`
            List of 2D convolution kernels.
        """
        energies = cube.energies(mode='edges')

        kernels = []
        for emin, emax in zip(energies[:-1], energies[1:]):
            energy_band = Quantity([emin, emax])
            psf = self.table_psf_in_energy_band(energy_band, **kwargs)
            kernel = psf.kernel(cube.sky_image_ref)
            kernels.append(kernel)
        return kernels

    def cube_kernels(self, cube, **kwargs):
        """
        Make a set of 2D kernel images, representing the PSF at different energies.

        The kernel image is evaluated on the spatial and energy grid defined by
        the reference sky cube.

        Parameters
        ----------
        cube : `~gammapy.cube.SkyCube`
            Reference sky cube.
        kwargs : dict
            Keyword arguments passed to `EnergyDependentTablePSF.table_psf_in_energy_band()`.

        Returns
        -------
        kernels : list of `~numpy.ndarray`
            List of 2D convolution kernels.
        """
        from gammapy.cube import SkyCube
        psf_cube = SkyCube.empty_like(cube)
        energies = cube.energies(mode='edges')
        for iE, (emin, emax) in enumerate(zip(energies[:-1], energies[1:])):
            energy_band = Quantity([emin, emax])
            psf = self.table_psf_in_energy_band(energy_band, **kwargs)
            psf_cube.data[iE,:,:] = psf.kernel(cube.sky_image_ref)

        return psf_cube

    def table_psf_in_energy_band(self, energy_band, spectral_index=2,
                                 spectrum=None, **kwargs):
        """Average PSF in a given energy band.

        Expected counts in sub energy bands given the given exposure
        and spectrum are used as weights.

        Parameters
        ----------
        energy_band : `~astropy.units.Quantity`
            Energy band
        spectral_index : float
            Power law spectral index (used if spectrum=None).
        spectrum : callable
            Spectrum (callable with energy as parameter).

        Returns
        -------
        psf : `TablePSF`
            Table PSF
        """
        if spectrum is None:
            def spectrum(energy):
                return (energy / energy_band[0]) ** (-spectral_index)

        # TODO: warn if `energy_band` is outside available data.
        energy_idx_min, energy_idx_max = self._energy_index(energy_band)

        # TODO: extract this into a utility function `npred_weighted_mean()`

        # Compute weights for energy bins
        weights = np.zeros_like(self.energy.value, dtype=np.float64)
        for idx in range(energy_idx_min, energy_idx_max - 1):
            energy_min = self.energy[idx]
            energy_max = self.energy[idx + 1]
            exposure = self.exposure[idx]

            flux = spectrum(energy_min)
            weights[idx] = (exposure * flux * (energy_max - energy_min)).value

        # Normalize weights to sum to 1
        weights = weights / weights.sum()

        # Compute weighted PSF value array
        total_psf_value = np.zeros_like(self._get_1d_psf_values(0), dtype=np.float64)
        for idx in range(energy_idx_min, energy_idx_max - 1):
            psf_value = self._get_1d_psf_values(idx)
            total_psf_value += weights[idx] * psf_value

        # TODO: add version that returns `total_psf_value` without
        # making a `TablePSF`.
        return TablePSF(self.offset, total_psf_value, **kwargs)

    def containment_radius(self, energies, fraction, interp_kwargs=None):
        """Containment radius.

        Parameters
        ----------
        energies : `~astropy.units.Quantity`
            Energy
        fraction : float
            Containment fraction in %

        Returns
        -------
        radius : `~astropy.units.Quantity`
            Containment radius in deg
        """
        # TODO: figure out if there's a more efficient implementation to support
        # arrays of energy
        energies = np.atleast_1d(energies)
        psfs = [self.table_psf_at_energy(energy, interp_kwargs) for energy in energies]
        radii = [psf.containment_radius(fraction) for psf in psfs]
        return Quantity(radii)

    def integral(self, energy, offset_min, offset_max):
        """Containment fraction.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        offset_min, offset_max : `~astropy.coordinates.Angle`
            Offset

        Returns
        -------
        fraction : array_like
            Containment fraction (in range 0 .. 1)
        """
        # TODO: useless at the moment ... support array inputs or remove!

        psf = self.table_psf_at_energy(energy)
        return psf.integral(offset_min, offset_max)

    def __str__(self):
        ss = 'EnergyDependentTablePSF\n'
        ss += '-----------------------\n'
        ss += '\nAxis info:\n'
        ss += '  ' + array_stats_str(self.offset.to('deg'), 'offset')
        ss += '  ' + array_stats_str(self.energy, 'energy')
        ss += '  ' + array_stats_str(self.exposure, 'exposure')

        # ss += 'integral = {0}\n'.format(self.integral())

        ss += '\nContainment info:\n'
        # Print some example containment radii
        fractions = [0.68, 0.95]
        energies = Quantity([10, 100], 'GeV')
        for fraction in fractions:
            radii = self.containment_radius(energies=energies, fraction=fraction)
            for energy, radius in zip(energies, radii):
                ss += '  ' + '{0}% containment radius at {1:3.0f}: {2:.2f}\n'.format(100 * fraction, energy, radius)
        return ss


    def info(self):
        """Print basic info"""
        print(self.__str__)

    def plot_psf_vs_theta(self, filename=None, energies=[1e4, 1e5, 1e6]):
        """Plot PSF vs theta.

        Parameters
        ----------
        TODO
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))

        for energy in energies:
            energy_index = self._energy_index(energy)
            psf = self.psf_value[energy_index, :]
            label = '{0} GeV'.format(1e-3 * energy)
            x = np.hstack([-self.theta[::-1], self.theta])
            y = 1e-6 * np.hstack([psf[::-1], psf])
            plt.plot(x, y, lw=2, label=label)
        # plt.semilogy()
        # plt.loglog()
        plt.legend()
        plt.xlim(-0.2, 0.5)
        plt.xlabel('Offset (deg)')
        plt.ylabel('PSF (1e-6 sr^-1)')
        plt.tight_layout()

        if filename != None:
            plt.savefig(filename)

    def plot_containment_vs_energy(self, ax=None, fractions=[0.63, 0.8, 0.95], **kwargs):
        """Plot containment versus energy."""
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = Energy.equal_log_spacing(
            self.energy.min(), self.energy.max(), 10)

        for fraction in fractions:
            radius = self.containment_radius(energy, fraction)
            label = '{:.1f}% Containment'.format(100 * fraction)
            ax.plot(energy.value, radius.value, label=label, **kwargs)

        ax.semilogx()
        ax.legend(loc='best')
        ax.set_xlabel('Energy (GeV)')
        ax.set_ylabel('Containment radius (deg)')

    def plot_exposure_vs_energy(self, filename=None):
        """Plot exposure versus energy."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 3))
        plt.plot(self.energy, self.exposure, color='black', lw=3)
        plt.semilogx()
        plt.xlabel('Energy (MeV)')
        plt.ylabel('Exposure (cm^2 s)')
        plt.xlim(1e4 / 1.3, 1.3 * 1e6)
        plt.ylim(0, 1.5e11)
        plt.tight_layout()

        if filename != None:
            plt.savefig(filename)

    def _energy_index(self, energy):
        """Find energy array index.
        """
        # TODO: test with array input
        return np.searchsorted(self.energy, energy)

    def _get_1d_psf_values(self, energy_index):
        """Get 1-dim PSF value array.

        Parameters
        ----------
        energy_index : int
            Energy index

        Returns
        -------
        psf_values : `~astropy.units.Quantity`
            PSF value array
        """
        psf_values = self.psf_value[energy_index, :].flatten().copy()
        return psf_values

    def _get_1d_table_psf(self, energy_index, **kwargs):
        """Get 1-dim TablePSF (cached).

        Parameters
        ----------
        energy_index : int
            Energy index

        Returns
        -------
        table_psf : `TablePSF`
            Table PSF
        """
        # TODO: support array_like `energy_index` here?
        if self._table_psf_cache[energy_index] is None:
            psf_value = self._get_1d_psf_values(energy_index)
            table_psf = TablePSF(self.offset, psf_value, **kwargs)
            self._table_psf_cache[energy_index] = table_psf

        return self._table_psf_cache[energy_index]
