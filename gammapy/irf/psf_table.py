# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy.io import fits
from astropy.units import Quantity
from astropy.coordinates import Angle, ICRS
#from astropy.convolution import discretize_model
from astropy.convolution.utils import discretize_oversample_2D
from ..morphology import Gauss2DPDF

__all__ = ['TablePSF',
           'EnergyDependentTablePSF',
           ]


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
    def __init__(self, offset, dp_domega, spline_args=dict(k=1)):

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
        self._spline_args = spline_args

        self._compute_splines(spline_args)

    @staticmethod
    def from_shape(shape, width, offset):
        """Make TablePSF objects with commonly used shapes.
        
        This function is mostly useful for examples and testing. 
        
        Parameters
        ----------
        shape : {'disk', 'gauss'}
            PSF shape.
        width : `~astropy.unit.Angle`
            PSF width angle (radius for disk, sigma for Gauss).
        offset : `~astropy.units.Angle`
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
        >>> make_table_psf(shape='gauss', width=Angle(0.2, 'degree'),
        ...                offset=Angle(np.linspace(0, 0.7, 100), 'degree'))
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
    
        return TablePSF(offset, psf_value)

    def info(self):
        """Print basic info."""
        x = self._offset.degree
        ss = ('offset: min = {0} deg, max = {1} deg, n_points = {2}\n'
              .format(x.min(), x.max(), len(x)))

        ss += 'integral = {0}\n'.format(self.integral())

        for containment in [50, 68, 80, 95]:
            radius = self.containment_radius(0.01 * containment)
            ss += ('containment radius {0} deg for {1}%\n'
                   .format(radius.degree, containment))

        return ss

    # TODO: remove because it's not flexible enough?
    def __call__(self, lon, lat):
        """Evaluate PSF at a 2D position.
        
        The PSF is centered on `(0, 0)`.
        
        Parameters
        ----------
        lon, lat : `~astropy.coordinates.Angle`
            Longitude / latitude position

        Returns
        -------
        psf_value : `~astropy.units.Quantity`
            PSF value
        """
        # TODO: It should not be necessary to use ICRS coordinates here.
        center = ICRS(0, 0, unit=('radian', 'radian'))
        point = ICRS(lon, lat)
        offset = center.separation(point)
        return self.eval(offset)

    def kernel(self, pixel_size, offset_max=None,
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
        kernel : `numpy.array`
            Kernel 2D image
        
        Notes
        -----
        * In the future, `astropy.modeling.Parametric2DModel` and
          `astropy.convolution.Model2DKernel` could be used to construct
          the kernel.
        """
        if not isinstance(pixel_size, Angle):
            raise ValueError("pixel_size must be an Angle object.")

        if offset_max == None:
            offset_max = self._offset.max()

        def _model(x, y):
            """Model in the appropriate format for discretize_model."""
            offset = np.sqrt(x * x + y * y) * pixel_size
            return self.eval(offset)

        npix = int(offset_max.radian / pixel_size.radian)
        pix_range = (-npix, npix)

        # FIXME: Using `discretize_model` is currently very cumbersome due to these issue:
        # https://github.com/astropy/astropy/issues/2274
        # https://github.com/astropy/astropy/issues/1763#issuecomment-39552900
        #from astropy.modeling import Parametric2DModel
        #
        #class TempModel(Parametric2DModel):
        #    @staticmethod
        #    def eval(x, y):
        #        return 42 temp_model_function(x, y)
        #
        #temp_model = TempModel()
        
        #import IPython; IPython.embed()
        array = discretize_oversample_2D(_model,
                                         x_range=pix_range, y_range=pix_range,
                                         **discretize_model_kwargs)
        return array


    def eval(self, offset, quantity='dp_domega'):
        r"""Evaluate PSF.

        The following PSF quantities are available:
        
        * 'dp_domega': PDF per 2-dim solid angle :math:`\Omega` in sr^-1
        
            .. math:: \frac{dP}{d\Omega}
        
        * 'dp_dtheta': PDF per 1-dim offset :math:`\theta` in radian^-1
        
            .. math:: \frac{dP}{d\theta} = 2 \pi \theta \frac{dP}{d\Omega}

        Parameters
        ----------
        offset : `~astropy.units.Angle`
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
            return Quantity(y, 'sr^-1').reshape(shape)
        elif quantity == 'dp_dtheta':
            y = self._dp_dtheta_spline(x)
            return Quantity(y, 'radian^-1').reshape(shape)
        else:
            ss = 'Invalid quantity: {0}\n'.format(quantity)
            ss += "Choose one of: 'dp_domega', 'dp_dtheta'"
            raise ValueError(ss)

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
        if offset_min == None:
            offset_min = self._offset[0]
        else:
            if not isinstance(offset_min, Angle):
                raise ValueError("offset_min must be an Angle object.")
        
        if offset_max == None:
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
        return Angle(radius, 'radian').to('degree')

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
        self._compute_splines(self._spline_args)

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
        self._compute_splines(self._spline_args)

        if normalize:
            self.normalize()

    def plot_psf_vs_theta(self, quantity='dp_domega'):
        """Plot PSF vs offset.
        
        TODO: describe PSF `quantity` argument in a central place and link to it from here.
        """
        import matplotlib.pyplot as plt

        x = self._offset.to('degree')
        y = self.eval(self._offset, quantity)

        plt.plot(x.value, y.value, lw=2)
        plt.semilogy()
        plt.loglog()
        plt.xlabel('Offset ({0})'.format(x.unit))
        plt.ylabel('PSF ({0})'.format(y.unit))

    def _compute_splines(self, spline_args={}):
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
        self._dp_domega_spline = UnivariateSpline(x, y, **spline_args)

        x, y = self._offset.value, self._dp_dtheta.value 
        self._dp_dtheta_spline = UnivariateSpline(x, y, **spline_args)

        # We use the terminology for scipy.stats distributions
        # http://docs.scipy.org/doc/scipy/reference/tutorial/stats.html#common-methods

        # cdf = "cumulative distribution function"
        self._cdf_spline = self._dp_dtheta_spline.antiderivative()

        # ppf = "percent point function" (inverse of cdf)
        # Here's a discussion on methods to compute the ppf
        # http://mail.scipy.org/pipermail/scipy-user/2010-May/025237.html
        x = self._offset.value
        y = self._cdf_spline(x)
        self._ppf_spline = UnivariateSpline(y, x, **spline_args)
        
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
    def __init__(self, energy, offset, exposure, psf_value):
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


    @staticmethod
    def from_fits(hdu_list):
        """Create EnergyDependentTablePSF from ``gtpsf`` format HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with `THETA` and `PSF` extensions.

        Returns
        -------
        psf : `EnergyDependentTablePSF`
            PSF object.
        """
        offset = Angle(hdu_list['THETA'].data['Theta'], 'degree')
        energy = Quantity(hdu_list['PSF'].data['Energy'], 'GeV')
        exposure = Quantity(hdu_list['PSF'].data['Exposure'], 'cm^2 s')
        psf_value = Quantity(hdu_list['PSF'].data['PSF'], 'sr^-1')

        return EnergyDependentTablePSF(energy, offset, exposure, psf_value)

    def to_fits(self):
        """Convert PSF to FITS HDU list format.

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

    @staticmethod
    def read(filename):
        """Read FITS format PSF file (``gtpsf`` output).
        
        Parameters
        ----------
        filename : str
            File name
        
        Returns
        -------
        psf : `EnergyDependentTablePSF`
            PSF object.
        """
        hdu_list = fits.open(filename)
        return EnergyDependentTablePSF.from_fits(hdu_list)

    def write(self, *args, **kwargs):
        """Write PSF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(*args, **kwargs)

    def psf_at_energy(self, energy, **kwargs):
        """PSF at a given energy.
        
        Extra `kwargs` are passed to the `~gammapy.irf.TablePSF` constructor.
        
        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy

        Returns
        -------
        psf : `~gammapy.irf.TablePSF`
            PSF
        """
        if not isinstance(energy, Quantity):
            raise ValueError("energy must be a Quantity object.")

        psf_value = self._psf(energy)

        return TablePSF(self.offset, psf_value, **kwargs)


    def psf_in_energy_band(self, energy_band, spectrum=None,
                           spectral_index=2):
        """Average PSF in a given energy band.
        
        Parameters
        ----------
        spectrum : callable
            Spectrum (callable with energy as parameter)
        energy_band : `~astropy.units.Quantity`
            Energy band
        """
        energy_indices = self._energy_indices(energy_band)
        energies = self.energy[energy_indices]
        weights = spectrum(energies)
        psfs = []
        for energy, weight in zip(energies, weights):
            psf = self.table_psf(energy)

    def containment_radius(self, energy, fraction):
        """Containment radius.
        
        Parameters
        ----------
        energy : float
            Energy (GeV)
        fraction : float
            Containment fraction in %
        
        Returns
        -------
        radius : float
            Containment radius in deg
        """
        # psf = self._psf(energy)
        # radius = 
        # return radius
        pass

    def containment_fraction(self, energy, offset):
        """Containment fraction.
        
        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        offset : `~astropy.units.Quantity`
            Offset
        
        Returns
        -------
        fraction : array_like
            Containment fraction (in range 0 .. 1)
        """
        psf = self._psf(energy)
        offset_max = self._offset_index(offset)
        t = np.radians(self.theta)
        fraction_per_bin = 2 * np.pi * t[:-1] * psf[:-1] * np.diff(t) 
        fraction = fraction_per_bin[0:offset_max].sum()
        return fraction

    def info(self):
        """Print basic info."""
        r68 = self.containment_radius(energy=10, fraction=0.68)
        ss = '68% containment radius at 10 GeV: {0}'.format(r68)
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
        return np.searchsorted(self.energy, energy)

    def _offset_index(self, offset):
        """Find offset array index.
        """
        return np.searchsorted(self.offset, offset)

    def _psf(self, energy):
        """PSF values.
        TODO: describe better
        """
        energy_index = self._energy_index(energy)
        psf = self.psf_value[energy_index, :]
        return psf

    def _get_1d_psf(self, energy):
        """TODO.
        """
        energy_index = self._energy_index(energy)

        for idx in energy_index:
            if self._1d_psf_cache[idx] == None:
                self._1d_psf_cache[idx] = self.make_psf_at_energy(energy[idx])

        return self._1d_psf_cache[energy_index] 
