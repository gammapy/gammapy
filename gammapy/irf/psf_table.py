# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from astropy.convolution.utils import discretize_oversample_2D
from astropy import log
from ..morphology import Gauss2DPDF
from ..utils.scripts import make_path
from ..utils.array import array_stats_str
from ..utils.energy import Energy, EnergyBounds
from ..utils.fits import table_to_fits_table

__all__ = [
    'TablePSF',
    'EnergyDependentTablePSF',
    'PSF3D'
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
    offset : `~astropy.coordinates.Angle`
        Offset angle array
    dp_domega : `~astropy.units.Quantity`
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

        if not isinstance(offset, Angle):
            raise ValueError("offset must be an Angle object.")
        if not isinstance(dp_domega, Quantity):
            raise ValueError("dp_domega must be a Quantity object.")

        assert offset.ndim == dp_domega.ndim == 1
        assert offset.shape == dp_domega.shape

        # Store input arrays as quantities in default internal units
        self._offset = offset.to('radian')
        self._dp_domega = dp_domega.to('sr^-1')
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
        width : `~astropy.coordinates.Angle`
            PSF width angle (radius for disk, sigma for Gauss).
        offset : `~astropy.coordinates.Angle`
            Offset angle

        Returns
        -------
        psf : `TablePSF`
            Table PSF

        Examples
        --------
        >>> import numpy as np
        >>> from astropy.coordinates import Angle
        >>> from gammapy.irf import make_table_psf
        >>> make_table_psf(shape='gauss', width=Angle(0.2, 'deg'),
        ...                offset=Angle(np.linspace(0, 0.7, 100), 'deg'))
        """
        if not isinstance(width, Angle):
            raise ValueError("width must be an Angle object.")
        if not isinstance(offset, Angle):
            raise ValueError("offset must be an Angle object.")

        if shape == 'disk':
            amplitude = 1 / (np.pi * width.radian ** 2)
            psf_value = np.where(offset < width, amplitude, 0)
        elif shape == 'gauss':
            gauss2d_pdf = Gauss2DPDF(sigma=width.radian)
            psf_value = gauss2d_pdf(offset.radian)

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

    def kernel(self, pixel_size, offset_max=None, normalize=True,
               discretize_model_kwargs=dict(factor=10)):
        """Make a 2-dimensional kernel image.

        The kernel image is evaluated on a cartesian
        grid with ``pixel_size`` spacing, not on the sphere.

        Calls `astropy.convolution.discretize_model`,
        allowing for accurate discretization.

        Parameters
        ----------
        pixel_size : `~astropy.coordinates.Angle`
            Kernel pixel size
        discretize_model_kwargs : dict
            Keyword arguments passed to
            `astropy.convolution.discretize_model`

        Returns
        -------
        kernel : `~astropy.units.Quantity`
            Kernel 2D image of Quantities

        Notes
        -----
        * In the future, `astropy.modeling.Fittable2DModel` and
          `astropy.convolution.Model2DKernel` could be used to construct
          the kernel.
        """
        if not isinstance(pixel_size, Angle):
            raise ValueError("pixel_size must be an Angle object.")

        if offset_max is None:
            offset_max = self._offset.max()

        def _model(x, y):
            """Model in the appropriate format for discretize_model."""
            offset = np.sqrt(x * x + y * y) * pixel_size
            return self.evaluate(offset)

        npix = int(offset_max.radian / pixel_size.radian)
        pix_range = (-npix, npix + 1)

        # FIXME: Using `discretize_model` is currently very cumbersome due to these issue:
        # https://github.com/astropy/astropy/issues/2274
        # https://github.com/astropy/astropy/issues/1763#issuecomment-39552900
        # from astropy.modeling import Fittable2DModel
        #
        # class TempModel(Fittable2DModel):
        #    @staticmethod
        #    def evaluate(x, y):
        #        return 42 temp_model_function(x, y)
        #
        # temp_model = TempModel()

        array = discretize_oversample_2D(_model,
                                         x_range=pix_range, y_range=pix_range,
                                         **discretize_model_kwargs)
        if normalize:
            return array / array.value.sum()
        else:
            return array

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
        if not isinstance(offset, Angle):
            raise ValueError("offset must be an Angle object.")

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
        offset_min, offset_max : `~astropy.coordinates.Angle`
            Offset angle range

        Returns
        -------
        integral : float
            PSF integral
        """
        if offset_min is None:
            offset_min = self._offset[0]
        else:
            if not isinstance(offset_min, Angle):
                raise ValueError("offset_min must be an Angle object.")

        if offset_max is None:
            offset_max = self._offset[-1]
        else:
            if not isinstance(offset_max, Angle):
                raise ValueError("offset_max must be an Angle object.")

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

    def plot_psf_vs_theta(self, quantity='dp_domega'):
        """Plot PSF vs offset.

        TODO: describe PSF ``quantity`` argument in a central place and link to it from here.
        """
        import matplotlib.pyplot as plt

        x = self._offset.to('deg')
        y = self.evaluate(self._offset, quantity)

        plt.plot(x.value, y.value, lw=2)
        plt.semilogy()
        plt.loglog()
        plt.xlabel('Offset ({0})'.format(x.unit))
        plt.ylabel('PSF ({0})'.format(y.unit))

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
    offset : `~astropy.coordinates.Angle`
        Offset angle (1-dim)
    exposure : `~astropy.units.Quantity`
        Exposure (1-dim)
    psf_value : `~astropy.units.Quantity`
        PSF (2-dim with axes: psf[energy_index, offset_index]
    """

    def __init__(self, energy, offset, exposure=None, psf_value=None):

        # Default for exposure
        exposure = exposure or Quantity(np.ones(len(energy)), 'cm^2 s')

        if not isinstance(energy, Quantity):
            raise ValueError("energy must be a Quantity object.")
        if not isinstance(offset, Angle):
            raise ValueError("offset must be an Angle object.")
        if not isinstance(exposure, Quantity):
            raise ValueError("exposure must be a Quantity object.")
        if not isinstance(psf_value, Quantity):
            raise ValueError("psf_value must be a Quantity object.")

        self.energy = energy.to('GeV')
        self.offset = offset.to('radian')
        self.exposure = exposure.to('cm^2 s')
        self.psf_value = psf_value.to('sr^-1')

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

    def table_psf_in_energy_band(self, energy_band, spectral_index=2, spectrum=None, **kwargs):
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

    def containment_radius(self, energy, fraction, interp_kwargs=None):
        """Containment radius.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        fraction : float
            Containment fraction in %

        Returns
        -------
        radius : `~astropy.units.Quantity`
            Containment radius in deg
        """
        # TODO: useless at the moment ... support array inputs or remove!
        psf = self.table_psf_at_energy(energy, interp_kwargs)
        return psf.containment_radius(fraction)

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

    def info(self):
        """Print basic info."""
        # Summarise data members
        ss = array_stats_str(self.offset.to('deg'), 'offset')
        ss += array_stats_str(self.energy, 'energy')
        ss += array_stats_str(self.exposure, 'exposure')

        # ss += 'integral = {0}\n'.format(self.integral())

        # Print some example containment radii
        fractions = [0.68, 0.95]
        energies = Quantity([10, 100], 'GeV')
        for energy in energies:
            for fraction in fractions:
                radius = self.containment_radius(energy=energy, fraction=fraction)
                ss += '{0}% containment radius at {1}: {2}\n'.format(100 * fraction, energy, radius)
        return ss

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

    def plot_containment_vs_energy(self, filename=None):
        """Plot containment versus energy."""
        raise NotImplementedError
        import matplotlib.pyplot as plt
        plt.clf()

        if filename != None:
            plt.savefig(filename)

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

class PSF3D(object):
    """Table PSF.

    Parameters
    ----------
    
    """
    
    def __init__(self, energy_lo, energy_hi, offset, rad_lo, rad_hi, psf_value, energy_thresh_lo=Quantity(0.1, 'TeV'),
                 energy_thresh_hi=Quantity(100, 'TeV')):
        self.energy_lo = energy_lo.to('TeV')
        self.energy_hi = energy_hi.to('TeV')
        self.offset = Angle(offset)
        self.rad_lo = Angle(rad_lo)
        self.rad_hi = Angle(rad_hi)
        self.psf_value = psf_value.to('sr^-1')
        self.energy_thresh_lo = energy_thresh_lo.to('TeV')
        self.energy_thresh_hi = energy_thresh_hi.to('TeV')

    def info(self):
        """Print some basic info.
        """
        ss = "\nSummary PSF3D info\n"
        ss += "---------------------\n"
        ss += array_stats_str(self.energy_lo, 'energy_lo')
        ss += array_stats_str(self.energy_hi, 'energy_hi')
        ss += array_stats_str(self.offset, 'offset')
        ss += array_stats_str(self.rad_lo, 'rad_lo')
        ss += array_stats_str(self.rad_hi, 'rad_hi')
        ss += array_stats_str(self.psf_value, 'psf_value')

        # TODO: should quote containment values also

        return ss

    def energy_logcenter(self):
        """Get logcenters of energy bins.
        
        Returns
        -------
        energies : `~astropy.units.Quantity`
            Logcenters of energy bins
        """

        return 10**((np.log10(self.energy_hi/Quantity(1, self.energy_hi.unit))
                     + np.log10(self.energy_lo/Quantity(1, self.energy_lo.unit))) / 2) * Quantity(1, self.energy_lo.unit)

    def rad_center(self):
        """Get centers of rad bins.
        
        Returns
        -------
        rad : `~astropy.coordinates.Angle`
            Centers of rad bins
        """
        
        return ((self.rad_hi + self.rad_lo) / 2).to('deg')
        
    @classmethod
    def read(cls, filename, hdu='PSF_2D_TABLE'):
        """Create `PSF3D` from FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        filename = str(make_path(filename))
        # TODO: implement it so that HDUCLASS is used
        # http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/hdu_index/index.html

        table = Table.read(filename, hdu=hdu)
        return cls.from_table(table)

    @classmethod
    def from_table(cls, table):
        """Create `PSF3D` from `~astropy.table.Table`.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table Table-PSF info.
        """
        theta_lo = table['THETA_LO'].squeeze()
        theta_hi = table['THETA_HI'].squeeze()
        offset = (theta_hi + theta_lo) / 2
        offset = Angle(offset, unit=table['THETA_LO'].unit)

        energy_lo = table['ENERG_LO'].squeeze()
        energy_hi = table['ENERG_HI'].squeeze()
        energy_lo = Energy(energy_lo, unit=table['ENERG_LO'].unit)
        energy_hi = Energy(energy_hi, unit=table['ENERG_HI'].unit)

        rad_lo = Quantity(table['RAD_LO'].squeeze(), table['RAD_LO'].unit)
        rad_hi = Quantity(table['RAD_HI'].squeeze(), table['RAD_HI'].unit)

        psf_value = Quantity(table['RPSF'].squeeze(), table['RPSF'].unit)

        try:
            energy_thresh_lo = Quantity(table.meta['LO_THRES'], 'TeV')
            energy_thresh_hi = Quantity(table.meta['HI_THRES'], 'TeV')
            return cls(energy_lo, energy_hi, offset, rad_lo, rad_hi, psf_value, energy_thresh_lo, energy_thresh_hi)
        except KeyError:
            log.warning('No safe energy thresholds found. Setting to default')
            return cls(energy_lo, energy_hi, offset, rad_lo, rad_hi, psf_value)

    def to_fits(self):
        """
        Convert psf table data to FITS hdu list.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        # Set up data
        names = ['ENERG_LO', 'ENERG_HI', 'THETA_LO', 'THETA_HI',
                 'RAD_LO', 'RAD_HI', 'RPSF']
        units = ['TeV', 'TeV', 'deg', 'deg',
                 'deg', 'deg', 'sr^-1']
        data = [self.energy_lo, self.energy_hi, self.offset, self.offset,
                self.rad_lo, self.rad_hi, self.psf_value]

        table = Table()
        for name_, data_, unit_ in zip(names, data, units):
            table[name_] = [data_]
            table[name_].unit = unit_

        hdu = table_to_fits_table(table)
        hdu.header['LO_THRES'] = self.energy_thresh_lo.value
        hdu.header['HI_THRES'] = self.energy_thresh_hi.value

        return fits.HDUList([fits.PrimaryHDU(), hdu])

    def write(self, filename, *args, **kwargs):
        """Write PSF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(filename, *args, **kwargs)

    def evaluate(self, energy=None, offset=None, rad=None,
                 interp_kwargs=None):
        """Interpolate the value of the `EnergyOffsetArray` at a given offset and Energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            energy value
        offset : `~astropy.coordinates.Angle`
            offset value
        rad : `~astropy.coordinates.Angle`
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
            energy = self.energy_logcenter()
        if offset is None:
            offset = self.offset
        if rad is None:
            rad = self.rad_center()

        energy = Energy(energy).to('TeV')
        offset = Angle(offset).to('deg')
        rad = Angle(rad).to('deg')

        energy_bin = self.energy_logcenter()
                     
        offset_bin = self.offset.to('deg')
        rad_bin = self.rad_center()
        points = (rad_bin, offset_bin, energy_bin)
        interpolator = RegularGridInterpolator(points, self.psf_value, **interp_kwargs)
        rr, off, ee = np.meshgrid(rad.value, offset.value, energy.value, indexing='ij')
        shape = ee.shape
        pix_coords = np.column_stack([rr.flat, off.flat, ee.flat])
        data_interp = interpolator(pix_coords)
        return Quantity(data_interp.reshape(shape), self.psf_value.unit)

    def to_energy_dependent_table_psf(self, theta=None, exposure=None):
        """
        Convert PSF3D in EnergyDependentTablePSF.

        Parameters
        ----------
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        exposure : `~astropy.units.Quantity`
            Energy dependent exposure. Should be in units equivalent to 'cm^2 s'.
            Default exposure = 1.

        Returns
        -------
        table_psf : `~gammapy.irf.EnergyDependentTablePSF`
            Instance of `EnergyDependentTablePSF`.
        """
        energies = self.energy_logcenter()

        # Defaults
        theta = theta or Angle(0, 'deg')
        offset = self.rad_center()
        psf_value = self.evaluate(offset=theta).squeeze().T

        return EnergyDependentTablePSF(energy=energies, offset=offset,
                                       exposure=exposure, psf_value=psf_value)
    
    def to_table_psf(self, energy, theta=None, interp_kwargs=None, **kwargs):
        """Evaluate the `EnergyOffsetArray` at one given energy.
        
        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        interp_kwargs : dict
            Option for interpolation for `~scipy.interpolate.RegularGridInterpolator`
            
        Returns
        -------
        table : `~astropy.table.Table`
            Table with two columns: offset, value
        """
        
        # Defaults
        theta = theta or Angle(0, 'deg')
        
        psf_value = self.evaluate(energy, theta, interp_kwargs=interp_kwargs).squeeze()
        for v in psf_value.value:
            if v != v or v == 0:
                return None
        table_psf = TablePSF(self.rad_center(), psf_value, **kwargs)
        
        return table_psf

    def containment_radius(self, energy, theta=None, fraction=0.68, interp_kwargs=None):
        """Containment radius.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        fraction : float
            Containment fraction. Default fraction = 0.68

        Returns
        -------
        radius : `~astropy.units.Quantity`
            Containment radius in deg
        """

        # Defaults
        theta = theta or Angle(0, 'deg')
        if energy.ndim == 0:
            energy = Quantity([energy.value], energy.unit)
        if theta.ndim == 0:
            theta = Quantity([theta.value], theta.unit)
        
        unit = None
        radius = np.zeros((energy.size, theta.size))
        for e in range(energy.size):
            for t in range(theta.size):
                psf = self.to_table_psf(energy[e], theta[t], interp_kwargs)
                if psf == None:
                    radius[e, t] = np.nan
                    continue
                r = psf.containment_radius(fraction)
                radius[e, t] = r.value
                unit = r.unit
        return Quantity(radius.squeeze(), unit)
        
    def plot_containment_vs_energy(self, fractions=[0.68, 0.95],
                                   thetas=Angle([0, 1], 'deg'), ax=None, **kwargs):
        """Plot containment fraction as a function of energy.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = Energy.equal_log_spacing(
            self.energy_lo[0], self.energy_hi[-1], 100)

        for theta in thetas:
            for fraction in fractions:
                radius = self.containment_radius(energy, theta, fraction).squeeze()
                label = '{} deg, {:.1f}%'.format(theta, 100 * fraction)
                ax.plot(energy.value, radius.value, label=label)

        ax.semilogx()
        ax.legend(loc='best')
        ax.set_xlabel('Energy (TeV)')
        ax.set_ylabel('Containment radius (deg)')

    def plot_psf_vs_rad(self, filename=None, theta=Angle(0, 'deg'), energy=Quantity(1, 'TeV')):
        """Plot PSF vs rad.

        Parameters
        ----------
        TODO
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))


        psf = self.evaluate(energy, theta).squeeze()
        #label = '{0} GeV'.format(1e-3 * energy)
        #x = np.hstack([-self.theta[::-1], self.theta])
        plt.plot(self.rad_center(), psf, lw=2)
        
        # plt.semilogy()
        # plt.loglog()
        #plt.legend()
        plt.xlim(0.0, self.rad_center()[-1].degree)
        plt.xlabel('Offset (deg)')
        plt.ylabel('PSF (1e-6 sr^-1)')
        plt.tight_layout()

        if filename != None:
            plt.savefig(filename)

        plt.show()


    def plot_containment(self, fraction=0.68, ax=None, show_safe_energy=False,
                         add_cbar=True, **kwargs):
        """
        Plot containment image with energy and theta axes.

        Parameters
        ----------
        fraction : float
            Containment fraction between 0 and 1.
        add_cbar : bool
            Add a colorbar
        """
        from matplotlib.colors import PowerNorm
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        kwargs.setdefault('cmap', 'afmhot')
        kwargs.setdefault('norm', PowerNorm(gamma=0.5))
        kwargs.setdefault('origin', 'lower')
        kwargs.setdefault('interpolation', 'nearest')
        # kwargs.setdefault('vmin', 0.1)
        # kwargs.setdefault('vmax', 0.2)

        # Set up and compute data
        containment = self.containment_radius(self.energy_logcenter(), self.offset, fraction)
        print(containment)

        extent = [
            self.offset[0].value, self.offset[-1].value,
            self.energy_lo[0].value, self.energy_hi[-1].value,
        ]

        # Plotting
        ax.imshow(containment.value, extent=extent, **kwargs)

        if show_safe_energy:
            # Log scale transformation for position of energy threshold
            e_min = self.energy_hi.value.min()
            e_max = self.energy_hi.value.max()
            e = (self.energy_thresh_lo.value - e_min) / (e_max - e_min)
            x = (np.log10(e * (e_max / e_min - 1) + 1) / np.log10(e_max / e_min)
                 * (len(self.energy_hi) + 1))
            ax.vlines(x, -0.5, len(self.theta) - 0.5)
            ax.text(x + 0.5, 0, 'Safe energy threshold: {0:3.2f}'.format(self.energy_thresh_lo))

        # Axes labels and ticks, colobar
        ax.semilogy()
        ax.set_xlabel('Offset (deg)')
        ax.set_ylabel('Energy (TeV)')

        if add_cbar:
            ax_cbar = plt.colorbar(fraction=0.1, pad=0.01, shrink=0.9,
                                   mappable=ax.images[0], ax=ax)
            label = 'Containment radius R{0:.0f} (deg)'.format(100 * fraction)
            ax_cbar.set_label(label)

        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

        self.plot_containment(fraction=0.68, ax=axes[0])
        self.plot_containment(fraction=0.95, ax=axes[1])
        self.plot_containment_vs_energy(ax=axes[2])

        # TODO: implement this plot
        # psf = self.psf_at_energy_and_theta(energy='1 TeV', theta='1 deg')
        # psf.plot_components(ax=axes[2])

        plt.tight_layout()
        plt.show()
        return fig
