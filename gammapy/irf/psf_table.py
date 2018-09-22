# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.io import fits
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from ..utils.gauss import Gauss2DPDF
from ..utils.scripts import make_path
from ..utils.array import array_stats_str
from ..utils.energy import Energy

__all__ = ["TablePSF", "EnergyDependentTablePSF"]

log = logging.getLogger(__name__)

# Default PSF spline keyword arguments
# TODO: test and document
DEFAULT_PSF_SPLINE_KWARGS = dict(k=1, s=0)


class TablePSF(object):
    r"""Radially-symmetric table PSF.

    This PSF represents a :math:`PSF(r)=dP / d\Omega(r)`
    spline interpolation curve for a given set of offset :math:`r`
    and :math:`PSF` points.

    Uses `scipy.interpolate.UnivariateSpline`.

    Parameters
    ----------
    rad : `~astropy.units.Quantity` with angle units
        Offset wrt source position
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
    * TODO: evaluate spline for ``(log(rad), log(PSF))`` for numerical stability?
    * TODO: merge morphology.theta class functionality with this class.
    * TODO: add FITS I/O methods
    * TODO: add ``normalize`` argument to ``__init__`` with default ``True``?
    * TODO: ``__call__`` doesn't show up in the html API docs, but it should:
      https://github.com/astropy/astropy/pull/2135
    """

    def __init__(self, rad, dp_domega, spline_kwargs=DEFAULT_PSF_SPLINE_KWARGS):

        self._rad = Angle(rad).to("radian")
        self._dp_domega = Quantity(dp_domega).to("sr^-1")

        assert self._rad.ndim == self._dp_domega.ndim == 1
        assert self._rad.shape == self._dp_domega.shape

        # Store input arrays as quantities in default internal units
        self._dp_dr = (2 * np.pi * self._rad * self._dp_domega).to("radian^-1")
        self._spline_kwargs = spline_kwargs

        self._compute_splines(spline_kwargs)

    @classmethod
    def from_shape(cls, shape, width, rad):
        """Make TablePSF objects with commonly used shapes.

        This function is mostly useful for examples and testing.

        Parameters
        ----------
        shape : {'disk', 'gauss'}
            PSF shape.
        width : `~astropy.units.Quantity` with angle units
            PSF width angle (radius for disk, sigma for Gauss).
        rad : `~astropy.units.Quantity` with angle units
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
        >>> TablePSF.from_shape(shape='gauss', width='0.2 deg',
        ...                     rad=Angle(np.linspace(0, 0.7, 100), 'deg'))
        """
        width = Angle(width)
        rad = Angle(rad)

        if shape == "disk":
            amplitude = 1 / (np.pi * width.radian ** 2)
            psf_value = np.where(rad < width, amplitude, 0)
        elif shape == "gauss":
            gauss2d_pdf = Gauss2DPDF(sigma=width.radian)
            psf_value = gauss2d_pdf(rad.radian)
        else:
            raise ValueError("Invalid shape: {}".format(shape))

        psf_value = Quantity(psf_value, "sr^-1")

        return cls(rad, psf_value)

    def info(self):
        """Print basic info."""
        ss = array_stats_str(self._rad.degree, "offset")
        ss += "integral = {}\n".format(self.integral())

        for containment in [50, 68, 80, 95]:
            radius = self.containment_radius(0.01 * containment)
            ss += "containment radius {} deg for {}%\n".format(
                radius.degree, containment
            )

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
        center = SkyCoord(0, 0, unit="radian")
        point = SkyCoord(lon, lat)
        rad = center.separation(point)
        return self.evaluate(rad)

    def evaluate(self, rad, quantity="dp_domega"):
        r"""Evaluate PSF.

        The following PSF quantities are available:

        * 'dp_domega': PDF per 2-dim solid angle :math:`\Omega` in sr^-1

            .. math:: \frac{dP}{d\Omega}

        * 'dp_dr': PDF per 1-dim offset :math:`r` in radian^-1

            .. math:: \frac{dP}{dr} = 2 \pi r \frac{dP}{d\Omega}

        Parameters
        ----------
        rad : `~astropy.coordinates.Angle`
            Offset wrt source position
        quantity : {'dp_domega', 'dp_dr'}
            Which PSF quantity?

        Returns
        -------
        psf_value : `~astropy.units.Quantity`
            PSF value
        """
        rad = Angle(rad)

        shape = rad.shape
        x = np.array(rad.radian).flat

        if quantity == "dp_domega":
            y = self._dp_domega_spline(x)
            unit = "sr^-1"
        elif quantity == "dp_dr":
            y = self._dp_dr_spline(x)
            unit = "radian^-1"
        else:
            ss = "Invalid quantity: {}\n".format(quantity)
            ss += "Choose one of: 'dp_domega', 'dp_dr'"
            raise ValueError(ss)

        y = np.clip(a=y, a_min=0, a_max=None)
        return Quantity(y, unit).reshape(shape)

    def integral(self, rad_min=None, rad_max=None):
        """Compute PSF integral, aka containment fraction.

        Parameters
        ----------
        rad_min, rad_max : `~astropy.units.Quantity` with angle units
            Offset angle range

        Returns
        -------
        integral : float
            PSF integral
        """
        if rad_min is None:
            rad_min = self._rad[0]
        else:
            rad_min = Angle(rad_min)

        if rad_max is None:
            rad_max = self._rad[-1]
        else:
            rad_max = Angle(rad_max)

        rad_min = self._rad_clip(rad_min)
        rad_max = self._rad_clip(rad_max)

        cdf_min = self._cdf_spline(rad_min)
        cdf_max = self._cdf_spline(rad_max)

        return cdf_max - cdf_min

    def containment_radius(self, fraction):
        """Containment radius.

        Parameters
        ----------
        fraction : array_like
            Containment fraction (range 0 .. 1)

        Returns
        -------
        rad : `~astropy.coordinates.Angle`
            Containment radius angle
        """
        rad = self._ppf_spline(fraction)
        return Angle(rad, "radian").to("deg")

    def normalize(self):
        """Normalize PSF to unit integral.

        Computes the total PSF integral via the :math:`dP / dr` spline
        and then divides the :math:`dP / dr` array.
        """
        integral = self.integral()

        self._dp_dr /= integral

        # Clip to small positive number to avoid divide by 0
        rad = np.clip(self._rad.radian, 1e-6, None)

        rad = Quantity(rad, "radian")
        self._dp_domega = self._dp_dr / (2 * np.pi * rad)
        self._compute_splines(self._spline_kwargs)

    def broaden(self, factor, normalize=True):
        r"""Broaden PSF by scaling the offset array.

        For a broadening factor :math:`f` and the offset
        array :math:`r`, the offset array scaled
        in the following way:

        .. math::
            r_{new} = f \times r_{old}
            \frac{dP}{dr}(r_{new}) = \frac{dP}{dr}(r_{old})

        Parameters
        ----------
        factor : float
            Broadening factor
        normalize : bool
            Normalize PSF after broadening
        """
        self._rad *= factor
        # We define broadening such that self._dp_domega remains the same
        # so we only have to re-compute self._dp_dr and the slines here.
        self._dp_dr = (2 * np.pi * self._rad * self._dp_domega).to("radian^-1")
        self._compute_splines(self._spline_kwargs)

        if normalize:
            self.normalize()

    def plot_psf_vs_rad(self, ax=None, quantity="dp_domega", **kwargs):
        """Plot PSF vs radius.

        TODO: describe PSF ``quantity`` argument in a central place and link to it from here.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        x = self._rad.to("deg")
        y = self.evaluate(self._rad, quantity)

        ax.plot(x.value, y.value, **kwargs)
        ax.loglog()
        ax.set_xlabel("Radius ({})".format(x.unit))
        ax.set_ylabel("PSF ({})".format(y.unit))

    def _compute_splines(self, spline_kwargs=DEFAULT_PSF_SPLINE_KWARGS):
        """Compute two splines representing the PSF.

        * `_dp_domega_spline` is used to evaluate the 2D PSF.
        * `_dp_dr_spline` is not really needed for most applications,
          but is available via `eval`.
        * `_cdf_spline` is used to compute integral and for normalisation.
        * `_ppf_spline` is used to compute containment radii.
        """
        from scipy.interpolate import UnivariateSpline

        # Compute spline and normalize.
        x, y = self._rad.value, self._dp_domega.value
        self._dp_domega_spline = UnivariateSpline(x, y, **spline_kwargs)

        x, y = self._rad.value, self._dp_dr.value
        self._dp_dr_spline = UnivariateSpline(x, y, **spline_kwargs)

        # We use the terminology for scipy.stats distributions
        # http://docs.scipy.org/doc/scipy/reference/tutorial/stats.html#common-methods

        # cdf = "cumulative distribution function"
        self._cdf_spline = self._dp_dr_spline.antiderivative()

        # ppf = "percent point function" (inverse of cdf)
        # Here's a discussion on methods to compute the ppf
        # http://mail.scipy.org/pipermail/scipy-user/2010-May/025237.html
        y = self._rad.value
        x = self.integral(Angle(0, "rad"), self._rad)

        # Since scipy 1.0 the UnivariateSpline requires that x is strictly increasing
        # So only keep nodes where this is the case (and always keep the first one):
        x, idx = np.unique(x, return_index=True)
        y = y[idx]

        # Dummy values, for cases where one really doesn't have a valid PSF.
        if len(x) < 4:
            x = [0, 1, 2, 3]
            y = [0, 0, 0, 0]

        self._ppf_spline = UnivariateSpline(x, y, **spline_kwargs)

    def _rad_clip(self, rad):
        """Clip to radius support range, because spline extrapolation is unstable."""
        rad = Angle(rad, "radian").radian
        rad = np.clip(rad, 0, self._rad[-1].radian)
        return rad


class EnergyDependentTablePSF(object):
    """Energy-dependent radially-symmetric table PSF (``gtpsf`` format).

    TODO: add references and explanations.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy (1-dim)
    rad : `~astropy.units.Quantity` with angle units
        Offset angle wrt source position (1-dim)
    exposure : `~astropy.units.Quantity`
        Exposure (1-dim)
    psf_value : `~astropy.units.Quantity`
        PSF (2-dim with axes: psf[energy_index, offset_index]
    """

    def __init__(self, energy, rad, exposure=None, psf_value=None):

        self.energy = Quantity(energy).to("GeV")
        self.rad = Quantity(rad).to("radian")
        if exposure is None:
            self.exposure = Quantity(np.ones(len(energy)), "cm^2 s")
        else:
            self.exposure = Quantity(exposure).to("cm^2 s")

        if psf_value is None:
            self.psf_value = Quantity(np.zeros(len(energy), len(rad)), "sr^-1")
        else:
            self.psf_value = Quantity(psf_value).to("sr^-1")

        # Cache for TablePSF at each energy ... only computed when needed
        self._table_psf_cache = [None] * len(self.energy)

    def __str__(self):
        ss = "EnergyDependentTablePSF\n"
        ss += "-----------------------\n"
        ss += "\nAxis info:\n"
        ss += "  " + array_stats_str(self.rad.to("deg"), "rad")
        ss += "  " + array_stats_str(self.energy, "energy")
        # ss += '  ' + array_stats_str(self.exposure, 'exposure')

        # ss += 'integral = {}\n'.format(self.integral())

        ss += "\nContainment info:\n"
        # Print some example containment radii
        fractions = [0.68, 0.95]
        energies = Quantity([10, 100], "GeV")
        for fraction in fractions:
            rads = self.containment_radius(energies=energies, fraction=fraction)
            for energy, rad in zip(energies, rads):
                ss += "  " + "{}% containment radius at {:3.0f}: {:.2f}\n".format(
                    100 * fraction, energy, rad
                )
        return ss

    @classmethod
    def from_fits(cls, hdu_list):
        """Create `EnergyDependentTablePSF` from ``gtpsf`` format HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``THETA`` and ``PSF`` extensions.
        """
        rad = Angle(hdu_list["THETA"].data["Theta"], "deg")
        energy = Quantity(hdu_list["PSF"].data["Energy"], "MeV")
        exposure = Quantity(hdu_list["PSF"].data["Exposure"], "cm^2 s")
        psf_value = Quantity(hdu_list["PSF"].data["PSF"], "sr^-1")

        return cls(energy, rad, exposure, psf_value)

    def to_fits(self):
        """Convert to FITS HDU list format.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        # TODO: write HEADER keywords as gtpsf

        data = self.rad
        theta_hdu = fits.BinTableHDU(data=data, name="Theta")

        data = [self.energy, self.exposure, self.psf_value]
        psf_hdu = fits.BinTableHDU(data=data, name="PSF")

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
        filename = str(make_path(filename))
        with fits.open(filename, memmap=False) as hdulist:
            psf = cls.from_fits(hdulist)

        return psf

    def write(self, *args, **kwargs):
        """Write to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(*args, **kwargs)

    def evaluate(self, energy=None, rad=None, interp_kwargs=None):
        """Evaluate the PSF at a given energy and offset

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            energy value
        rad : `~astropy.coordinates.Angle`
            Offset wrt source position
        interp_kwargs : dict
            option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        values : `~astropy.units.Quantity`
            Interpolated value
        """
        if interp_kwargs is None:
            interp_kwargs = dict(bounds_error=False, fill_value=None)

        from scipy.interpolate import RegularGridInterpolator

        if energy is None:
            energy = self.energy
        if rad is None:
            rad = self.rad
        energy = Energy(energy).to("TeV")
        rad = Angle(rad).to("deg")
        energy_bin = self.energy.to("TeV")
        rad_bin = self.rad.to("deg")
        points = (energy_bin, rad_bin)
        interpolator = RegularGridInterpolator(
            points, self.psf_value.value, **interp_kwargs
        )
        energy_grid, rad_grid = np.meshgrid(energy.value, rad.value, indexing="ij")
        shape = energy_grid.shape
        pix_coords = np.column_stack([energy_grid.flat, rad_grid.flat])
        data_interp = interpolator(pix_coords)
        return Quantity(data_interp.reshape(shape), self.psf_value.unit)

    def table_psf_at_energy(self, energy, interp_kwargs=None, **kwargs):
        """Create `~gammapy.irf.TablePSF` at one given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        interp_kwargs : dict
            Option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        psf : `~gammapy.irf.TablePSF`
            Table PSF
        """
        psf_value = self.evaluate(energy, None, interp_kwargs)[0, :]
        return TablePSF(self.rad, psf_value, **kwargs)

    def table_psf_in_energy_band(
        self, energy_band, spectral_index=2, spectrum=None, **kwargs
    ):
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
            # This is a false positive error from pylint
            # See https://github.com/PyCQA/pylint/issues/2410#issuecomment-415026690
            def spectrum(energy):  # pylint:disable=function-redefined
                return (energy / energy_band[0]) ** (-spectral_index)

        # TODO: warn if `energy_band` is outside available data.
        energy_idx_min, energy_idx_max = self._energy_index(energy_band)

        # TODO: improve this, probably by evaluating the PSF (i.e. interpolating in energy) onto a new energy grid
        # This is a bit of a hack, but makes sure that a PSF is given, by forcing at least one slice:
        if energy_idx_max - energy_idx_min < 2:
            # log.warning('Dubious case of PSF energy binning')
            # Note that below always range stop of `energy_idx_max - 1` is used!
            # That's why we put +2 here to make sure we have at least one bin.
            energy_idx_max = max(energy_idx_min + 2, energy_idx_max)
            # Make sure we don't step out of the energy array (doesn't help much)
            energy_idx_max = min(energy_idx_max, len(self.energy))

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
        return TablePSF(self.rad, total_psf_value, **kwargs)

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
        rad : `~astropy.units.Quantity`
            Containment radius in deg
        """
        # TODO: figure out if there's a more efficient implementation to support
        # arrays of energy
        energies = np.atleast_1d(energies)
        psfs = [self.table_psf_at_energy(energy, interp_kwargs) for energy in energies]
        rad = [psf.containment_radius(fraction) for psf in psfs]
        return Quantity(rad)

    def integral(self, energy, rad_min, rad_max):
        """Containment fraction.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        rad_min, rad_max : `~astropy.coordinates.Angle`
            Offset

        Returns
        -------
        fraction : array_like
            Containment fraction (in range 0 .. 1)
        """
        # TODO: useless at the moment ... support array inputs or remove!

        psf = self.table_psf_at_energy(energy)
        return psf.integral(rad_min, rad_max)

    def info(self):
        """Print basic info"""
        print(str(self))

    def plot_psf_vs_rad(self, energies=[1e4, 1e5, 1e6]):
        """Plot PSF vs radius.

        Parameters
        ----------
        TODO
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))

        for energy in energies:
            energy_index = self._energy_index(energy)
            psf = self.psf_value[energy_index, :]
            label = "{} GeV".format(1e-3 * energy)
            x = np.hstack([-self.rad[::-1], self.rad])
            y = 1e-6 * np.hstack([psf[::-1], psf])
            plt.plot(x, y, lw=2, label=label)
        # plt.semilogy()
        # plt.loglog()
        plt.legend()
        plt.xlim(-0.2, 0.5)
        plt.xlabel("Offset (deg)")
        plt.ylabel("PSF (1e-6 sr^-1)")
        plt.tight_layout()

    def plot_containment_vs_energy(
        self, ax=None, fractions=[0.63, 0.8, 0.95], **kwargs
    ):
        """Plot containment versus energy."""
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = Energy.equal_log_spacing(self.energy.min(), self.energy.max(), 10)

        for fraction in fractions:
            rad = self.containment_radius(energy, fraction)
            label = "{:.1f}% Containment".format(100 * fraction)
            ax.plot(energy.value, rad.value, label=label, **kwargs)

        ax.semilogx()
        ax.legend(loc="best")
        ax.set_xlabel("Energy (GeV)")
        ax.set_ylabel("Containment radius (deg)")

    def plot_exposure_vs_energy(self):
        """Plot exposure versus energy."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 3))
        plt.plot(self.energy, self.exposure, color="black", lw=3)
        plt.semilogx()
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Exposure (cm^2 s)")
        plt.xlim(1e4 / 1.3, 1.3 * 1e6)
        plt.ylim(0, 1.5e11)
        plt.tight_layout()

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
        # When the PSF Table is not filled (with nan), the psf estimation at a given energy crashes
        psf_values[np.isnan(psf_values)] = 0
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
            table_psf = TablePSF(self.rad, psf_value, **kwargs)
            self._table_psf_cache[energy_index] = table_psf

        return self._table_psf_cache[energy_index]
