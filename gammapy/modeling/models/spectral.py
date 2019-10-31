# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy."""
import operator
import numpy as np
import scipy.optimize
import scipy.special
import astropy.units as u
from astropy.table import Table
from gammapy.modeling import Model, Parameter, Parameters
from gammapy.utils.energy import energy_logspace
from gammapy.utils.integrate import integrate_spectrum
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from gammapy.utils.scripts import make_path


class SpectralModel(Model):
    """Spectral model base class."""

    def __call__(self, energy):
        kwargs = {}
        for par in self.parameters:
            quantity = par.quantity
            if quantity.unit.physical_type == "energy":
                quantity = quantity.to(energy.unit)
            kwargs[par.name] = quantity

        return self.evaluate(energy, **kwargs)

    def __add__(self, model):
        if not isinstance(model, SpectralModel):
            model = ConstantSpectralModel(const=model)
        return CompoundSpectralModel(self, model, operator.add)

    def __radd__(self, model):
        return self.__add__(model)

    def __sub__(self, model):
        if not isinstance(model, SpectralModel):
            model = ConstantSpectralModel(const=model)
        return CompoundSpectralModel(self, model, operator.sub)

    def __rsub__(self, model):
        return self.__sub__(model)

    def _parse_uarray(self, uarray):
        from uncertainties import unumpy

        values = unumpy.nominal_values(uarray)
        errors = unumpy.std_devs(uarray)
        return values, errors

    def _convert_energy(self, energy):
        if "reference" in self.parameters.names:
            return energy.to(self.parameters["reference"].unit)
        elif "emin" in self.parameters.names:
            return energy.to(self.parameters["emin"].unit)
        else:
            return energy

    def evaluate_error(self, energy):
        """Evaluate spectral model with error propagation.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy at which to evaluate

        Returns
        -------
        flux, flux_error : tuple of `~astropy.units.Quantity`
            Tuple of flux and flux error.
        """
        energy = self._convert_energy(energy)

        unit = self(energy).unit
        upars = self.parameters._ufloats
        uarray = self.evaluate(energy.value, **upars)
        return self._parse_uarray(uarray) * unit

    def integral(self, emin, emax, **kwargs):
        r"""Integrate spectral model numerically.

        .. math::
            F(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}} \phi(E) dE

        If array input for ``emin`` and ``emax`` is given you have to set
        ``intervals=True`` if you want the integral in each energy bin.

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        **kwargs : dict
            Keyword arguments passed to :func:`~gammapy.utils.integrate.integrate_spectrum`
        """
        return integrate_spectrum(self, emin, emax, **kwargs)

    def integral_error(self, emin, emax, **kwargs):
        """Integrate spectral model numerically with error propagation.

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower adn upper  bound of integration range.
        **kwargs : dict
            Keyword arguments passed to func:`~gammapy.utils.integrate.integrate_spectrum`

        Returns
        -------
        integral, integral_error : tuple of `~astropy.units.Quantity`
            Tuple of integral flux and integral flux error.
        """
        emin = self._convert_energy(emin)
        emax = self._convert_energy(emax)
        unit = self.integral(emin, emax, **kwargs).unit
        upars = self.parameters._ufloats

        def f(x):
            return self.evaluate(x, **upars)

        uarray = integrate_spectrum(f, emin.value, emax.value, **kwargs)
        return self._parse_uarray(uarray) * unit

    def energy_flux(self, emin, emax, **kwargs):
        r"""Compute energy flux in given energy range.

        .. math::
            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}} E \phi(E) dE

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        **kwargs : dict
            Keyword arguments passed to func:`~gammapy.utils.integrate.integrate_spectrum`
        """

        def f(x):
            return x * self(x)

        return integrate_spectrum(f, emin, emax, **kwargs)

    def energy_flux_error(self, emin, emax, **kwargs):
        r"""Compute energy flux in given energy range with error propagation.

        .. math::
            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}} E \phi(E) dE

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower bound of integration range.
        **kwargs : dict
            Keyword arguments passed to :func:`~gammapy.utils.integrate.integrate_spectrum`

        Returns
        -------
        energy_flux, energy_flux_error : tuple of `~astropy.units.Quantity`
            Tuple of energy flux and energy flux error.
        """
        emin = self._convert_energy(emin)
        emax = self._convert_energy(emax)

        unit = self.energy_flux(emin, emax, **kwargs).unit
        upars = self.parameters._ufloats

        def f(x):
            return x * self.evaluate(x, **upars)

        uarray = integrate_spectrum(f, emin.value, emax.value, **kwargs)
        return self._parse_uarray(uarray) * unit

    def plot(
        self,
        energy_range,
        ax=None,
        energy_unit="TeV",
        flux_unit="cm-2 s-1 TeV-1",
        energy_power=0,
        n_points=100,
        **kwargs,
    ):
        """Plot spectral model curve.

        kwargs are forwarded to `matplotlib.pyplot.plot`

        By default a log-log scaling of the axes is used, if you want to change
        the y axis scaling to linear you can use::

            from gammapy.modeling.models import ExpCutoffPowerLawSpectralModel
            from astropy import units as u

            pwl = ExpCutoffPowerLawSpectralModel()
            ax = pwl.plot(energy_range=(0.1, 100) * u.TeV)
            ax.set_yscale('linear')

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy_range : `~astropy.units.Quantity`
            Plot range
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        energy_power : int, optional
            Power of energy to multiply flux axis with
        n_points : int, optional
            Number of evaluation nodes

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        emin, emax = energy_range
        energy = energy_logspace(emin, emax, n_points, energy_unit)

        # evaluate model
        flux = self(energy).to(flux_unit)

        y = self._plot_scale_flux(energy, flux, energy_power)

        ax.plot(energy.value, y.value, **kwargs)

        self._plot_format_ax(ax, energy, y, energy_power)
        return ax

    def plot_error(
        self,
        energy_range,
        ax=None,
        energy_unit="TeV",
        flux_unit="cm-2 s-1 TeV-1",
        energy_power=0,
        n_points=100,
        **kwargs,
    ):
        """Plot spectral model error band.

        .. note::

            This method calls ``ax.set_yscale("log", nonposy='clip')`` and
            ``ax.set_xscale("log", nonposx='clip')`` to create a log-log representation.
            The additional argument ``nonposx='clip'`` avoids artefacts in the plot,
            when the error band extends to negative values (see also
            https://github.com/matplotlib/matplotlib/issues/8623).

            When you call ``plt.loglog()`` or ``plt.semilogy()`` explicitely in your
            plotting code and the error band extends to negative values, it is not
            shown correctly. To circumvent this issue also use
            ``plt.loglog(nonposx='clip', nonposy='clip')``
            or ``plt.semilogy(nonposy='clip')``.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy_range : `~astropy.units.Quantity`
            Plot range
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        energy_power : int, optional
            Power of energy to multiply flux axis with
        n_points : int, optional
            Number of evaluation nodes
        **kwargs : dict
            Keyword arguments forwarded to `matplotlib.pyplot.fill_between`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("facecolor", "black")
        kwargs.setdefault("alpha", 0.2)
        kwargs.setdefault("linewidth", 0)

        emin, emax = energy_range
        energy = energy_logspace(emin, emax, n_points, energy_unit)

        flux, flux_err = self.evaluate_error(energy).to(flux_unit)

        y_lo = self._plot_scale_flux(energy, flux - flux_err, energy_power)
        y_hi = self._plot_scale_flux(energy, flux + flux_err, energy_power)

        where = (energy >= energy_range[0]) & (energy <= energy_range[1])
        ax.fill_between(energy.value, y_lo.value, y_hi.value, where=where, **kwargs)

        self._plot_format_ax(ax, energy, y_lo, energy_power)
        return ax

    @staticmethod
    def _plot_format_ax(ax, energy, y, energy_power):
        ax.set_xlabel(f"Energy [{energy.unit}]")
        if energy_power > 0:
            ax.set_ylabel(f"E{energy_power} * Flux [{y.unit}]")
        else:
            ax.set_ylabel(f"Flux [{y.unit}]")

        ax.set_xscale("log", nonposx="clip")
        ax.set_yscale("log", nonposy="clip")

    @staticmethod
    def _plot_scale_flux(energy, flux, energy_power):
        try:
            eunit = [_ for _ in flux.unit.bases if _.physical_type == "energy"][0]
        except IndexError:
            eunit = energy.unit
        y = flux * np.power(energy, energy_power)
        return y.to(flux.unit * eunit ** energy_power)

    def spectral_index(self, energy, epsilon=1e-5):
        """Compute spectral index at given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy at which to estimate the index
        epsilon : float
            Fractional energy increment to use for determining the spectral index.

        Returns
        -------
        index : float
            Estimated spectral index.
        """
        f1 = self(energy)
        f2 = self(energy * (1 + epsilon))
        return np.log(f1 / f2) / np.log(1 + epsilon)

    def inverse(self, value, emin=0.1 * u.TeV, emax=100 * u.TeV):
        """Return energy for a given function value of the spectral model.

        Calls the `scipy.optimize.brentq` numerical root finding method.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Function value of the spectral model.
        emin : `~astropy.units.Quantity`
            Lower bracket value in case solution is not unique.
        emax : `~astropy.units.Quantity`
            Upper bracket value in case solution is not unique.

        Returns
        -------
        energy : `~astropy.units.Quantity`
            Energies at which the model has the given ``value``.
        """
        eunit = "TeV"

        energies = []
        for val in np.atleast_1d(value):

            def f(x):
                # scale by 1e12 to achieve better precision
                energy = u.Quantity(x, eunit, copy=False)
                y = self(energy).to_value(value.unit)
                return 1e12 * (y - val.value)

            energy = scipy.optimize.brentq(
                f, emin.to_value(eunit), emax.to_value(eunit)
            )
            energies.append(energy)

        return u.Quantity(energies, eunit, copy=False)


class ConstantSpectralModel(SpectralModel):
    r"""Constant model.

    .. math:: \phi(E) = k

    Parameters
    ----------
    const : `~astropy.units.Quantity`
        :math:`k`
    """

    tag = "ConstantSpectralModel"
    const = Parameter("const", "1e-12 cm-2 s-1 TeV-1")

    @staticmethod
    def evaluate(energy, const):
        """Evaluate the model (static function)."""
        return np.ones(np.atleast_1d(energy).shape) * const


class CompoundSpectralModel(SpectralModel):
    """Arithmetic combination of two spectral models.

    Itself again a spectral model.
    """

    tag = "CompoundSpectralModel"

    def __init__(self, model1, model2, operator):
        self.model1 = model1
        self.model2 = model2
        self.operator = operator
        super().__init__()

    @property
    def parameters(self):
        return self.model1.parameters + self.model2.parameters

    def __str__(self):
        return (
            f"{self.__class__.__name__}\n"
            f"    Component 1 : {self.model1}\n"
            f"    Component 2 : {self.model2}\n"
            f"    Operator : {self.operator}\n"
        )

    def __call__(self, energy):
        val1 = self.model1(energy)
        val2 = self.model2(energy)
        return self.operator(val1, val2)

    def to_dict(self):
        return {
            "model1": self.model1.to_dict(),
            "model2": self.model2.to_dict(),
            "operator": self.operator,
        }


class PowerLawSpectralModel(SpectralModel):
    r"""Spectral power-law model.

    .. math::
        \phi(E) = \phi_0 \cdot \left( \frac{E}{E_0} \right)^{-\Gamma}

    Parameters
    ----------
    index : `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    """

    tag = "PowerLawSpectralModel"
    index = Parameter("index", 2.0)
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1")
    reference = Parameter("reference", "1 TeV", frozen=True)

    @staticmethod
    def evaluate(energy, index, amplitude, reference):
        """Evaluate the model (static function)."""
        return amplitude * np.power((energy / reference), -index)

    @staticmethod
    def evaluate_integral(emin, emax, index, amplitude, reference):
        """Evaluate the model integral (static function)."""
        val = -1 * index + 1

        prefactor = amplitude * reference / val
        upper = np.power((emax / reference), val)
        lower = np.power((emin / reference), val)
        integral = prefactor * (upper - lower)

        mask = np.isclose(val, 0)

        if mask.any():
            integral[mask] = (amplitude * reference * np.log(emax / emin))[mask]

        return integral

    @staticmethod
    def evaluate_energy_flux(emin, emax, index, amplitude, reference):
        """Evaluate the energy flux (static function)"""
        val = -1 * index + 2

        prefactor = amplitude * reference ** 2 / val
        upper = (emax / reference) ** val
        lower = (emin / reference) ** val
        energy_flux = prefactor * (upper - lower)

        mask = np.isclose(val, 0)

        if mask.any():
            # see https://www.wolframalpha.com/input/?i=a+*+x+*+(x%2Fb)+%5E+(-2)
            # for reference
            energy_flux[mask] = amplitude * reference ** 2 * np.log(emax / emin)[mask]

        return energy_flux

    def integral(self, emin, emax, **kwargs):
        r"""Integrate power law analytically.

        .. math::
            F(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}\phi(E)dE = \left.
            \phi_0 \frac{E_0}{-\Gamma + 1} \left( \frac{E}{E_0} \right)^{-\Gamma + 1}
            \right \vert _{E_{min}}^{E_{max}}

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower and upper bound of integration range
        """
        emin = self._convert_energy(emin)
        emax = self._convert_energy(emax)
        kwargs = {par.name: par.quantity for par in self.parameters}
        return self.evaluate_integral(emin=emin, emax=emax, **kwargs)

    def integral_error(self, emin, emax, **kwargs):
        r"""Integrate power law analytically with error propagation.

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.

        Returns
        -------
        integral, integral_error : tuple of `~astropy.units.Quantity`
            Tuple of integral flux and integral flux error.
        """
        # kwargs are passed to this function but not used
        # this is to get a consistent API with SpectralModel.integral()
        emin = self._convert_energy(emin)
        emax = self._convert_energy(emax)

        unit = self.integral(emin, emax, **kwargs).unit
        upars = self.parameters._ufloats

        if np.isclose(upars["index"].nominal_value, 1):
            prefactor = upars["amplitude"] * upars["reference"]
            upper = np.log(emax.value)
            lower = np.log(emin.value)
        else:
            val = -1 * upars["index"] + 1
            prefactor = upars["amplitude"] * upars["reference"] / val
            upper = np.power((emax.value / upars["reference"]), val)
            lower = np.power((emin.value / upars["reference"]), val)

        uarray = prefactor * (upper - lower)
        return self._parse_uarray(uarray) * unit

    def energy_flux(self, emin, emax):
        r"""Compute energy flux in given energy range analytically.

        .. math::
            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}E \phi(E)dE = \left.
            \phi_0 \frac{E_0^2}{-\Gamma + 2} \left( \frac{E}{E_0} \right)^{-\Gamma + 2}
            \right \vert _{E_{min}}^{E_{max}}

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        """
        emin = self._convert_energy(emin)
        emax = self._convert_energy(emax)
        kwargs = {par.name: par.quantity for par in self.parameters}
        return self.evaluate_energy_flux(emin=emin, emax=emax, **kwargs)

    def energy_flux_error(self, emin, emax, **kwargs):
        r"""Compute energy flux in given energy range analytically with error propagation.

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.

        Returns
        -------
        energy_flux, energy_flux_error : tuple of `~astropy.units.Quantity`
            Tuple of energy flux and energy flux error.
        """
        emin = self._convert_energy(emin)
        emax = self._convert_energy(emax)

        unit = self.energy_flux(emin, emax, **kwargs).unit
        upars = self.parameters._ufloats

        val = -1 * upars["index"] + 2

        if np.isclose(val.nominal_value, 0):
            # see https://www.wolframalpha.com/input/?i=a+*+x+*+(x%2Fb)+%5E+(-2)
            # for reference
            temp = upars["amplitude"] * upars["reference"] ** 2
            uarray = temp * np.log(emax.value / emin.value)
        else:
            prefactor = upars["amplitude"] * upars["reference"] ** 2 / val
            upper = (emax.value / upars["reference"]) ** val
            lower = (emin.value / upars["reference"]) ** val
            uarray = prefactor * (upper - lower)

        return self._parse_uarray(uarray) * unit

    def inverse(self, value):
        """Return energy for a given function value of the spectral model.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Function value of the spectral model.
        """
        p = self.parameters
        base = value / p["amplitude"].quantity
        return p["reference"].quantity * np.power(base, -1.0 / p["index"].value)

    @property
    def pivot_energy(self):
        r"""The decorrelation energy is defined as:

        .. math::

            E_D = E_0 * \exp{cov(\phi_0, \Gamma) / (\phi_0 \Delta \Gamma^2)}

        Formula (1) in https://arxiv.org/pdf/0910.4881.pdf
        """
        index_err = self.parameters.error("index")
        reference = self.reference.quantity
        amplitude = self.amplitude.quantity
        cov_index_ampl = self.parameters.covariance[0, 1] * amplitude.unit
        return reference * np.exp(cov_index_ampl / (amplitude * index_err ** 2))


class PowerLaw2SpectralModel(SpectralModel):
    r"""Spectral power-law model with integral as amplitude parameter.

    See also: https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html

    .. math::
        \phi(E) = F_0 \cdot \frac{\Gamma + 1}{E_{0, max}^{-\Gamma + 1}
         - E_{0, min}^{-\Gamma + 1}} \cdot E^{-\Gamma}

    Parameters
    ----------
    index : `~astropy.units.Quantity`
        Spectral index :math:`\Gamma`
    amplitude : `~astropy.units.Quantity`
        Integral flux :math:`F_0`.
    emin : `~astropy.units.Quantity`
        Lower energy limit :math:`E_{0, min}`.
    emax : `~astropy.units.Quantity`
        Upper energy limit :math:`E_{0, max}`.
    """

    tag = "PowerLaw2SpectralModel"

    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1")
    index = Parameter("index", 2)
    emin = Parameter("emin", "0.1 TeV", frozen=True)
    emax = Parameter("emax", "100 TeV", frozen=True)

    @staticmethod
    def evaluate(energy, amplitude, index, emin, emax):
        """Evaluate the model (static function)."""
        top = -index + 1

        # to get the energies dimensionless we use a modified formula
        bottom = emax - emin * (emin / emax) ** (-index)
        return amplitude * (top / bottom) * np.power(energy / emax, -index)

    def integral(self, emin, emax, **kwargs):
        r"""Integrate power law analytically.

        .. math::
            F(E_{min}, E_{max}) = F_0 \cdot \frac{E_{max}^{\Gamma + 1} \
                                - E_{min}^{\Gamma + 1}}{E_{0, max}^{\Gamma + 1} \
                                - E_{0, min}^{\Gamma + 1}}

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        """
        pars = self.parameters

        temp1 = np.power(emax, -pars["index"].value + 1)
        temp2 = np.power(emin, -pars["index"].value + 1)
        top = temp1 - temp2

        temp1 = np.power(pars["emax"].quantity, -pars["index"].value + 1)
        temp2 = np.power(pars["emin"].quantity, -pars["index"].value + 1)
        bottom = temp1 - temp2

        return pars["amplitude"].quantity * top / bottom

    def integral_error(self, emin, emax, **kwargs):
        r"""Integrate power law analytically with error propagation.

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.

        Returns
        -------
        integral, integral_error : tuple of `~astropy.units.Quantity`
            Tuple of integral flux and integral flux error.
        """
        emin = self._convert_energy(emin)
        emax = self._convert_energy(emax)

        unit = self.integral(emin, emax, **kwargs).unit
        upars = self.parameters._ufloats

        temp1 = np.power(emax.value, -upars["index"] + 1)
        temp2 = np.power(emin.value, -upars["index"] + 1)
        top = temp1 - temp2

        temp1 = np.power(upars["emax"], -upars["index"] + 1)
        temp2 = np.power(upars["emin"], -upars["index"] + 1)
        bottom = temp1 - temp2

        uarray = upars["amplitude"] * top / bottom
        return self._parse_uarray(uarray) * unit

    def inverse(self, value):
        """Return energy for a given function value of the spectral model.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Function value of the spectral model.
        """
        p = self.parameters
        amplitude, index, emin, emax = (
            p["amplitude"].quantity,
            p["index"].value,
            p["emin"].quantity,
            p["emax"].quantity,
        )

        # to get the energies dimensionless we use a modified formula
        top = -index + 1
        bottom = emax - emin * (emin / emax) ** (-index)
        term = (bottom / top) * (value / amplitude)
        return np.power(term.to_value(""), -1.0 / index) * emax


class ExpCutoffPowerLawSpectralModel(SpectralModel):
    r"""Spectral exponential cutoff power-law model.

    .. math::
        \phi(E) = \phi_0 \cdot \left(\frac{E}{E_0}\right)^{-\Gamma} \exp(-\lambda E)

    Parameters
    ----------
    index : `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    lambda_ : `~astropy.units.Quantity`
        :math:`\lambda`
    """

    tag = "ExpCutoffPowerLawSpectralModel"

    index = Parameter("index", 1.5)
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1")
    reference = Parameter("reference", "1 TeV", frozen=True)
    lambda_ = Parameter("lambda_", "0.1 TeV-1")

    @staticmethod
    def evaluate(energy, index, amplitude, reference, lambda_):
        """Evaluate the model (static function)."""
        pwl = amplitude * (energy / reference) ** (-index)
        try:
            cutoff = np.exp(-energy * lambda_)
        except (AttributeError, TypeError):
            from uncertainties.unumpy import exp

            cutoff = exp(-energy * lambda_)
        return pwl * cutoff

    @property
    def e_peak(self):
        r"""Spectral energy distribution peak energy (`~astropy.units.Quantity`).

        This is the peak in E^2 x dN/dE and is given by:

        .. math::
            E_{Peak} = (2 - \Gamma) / \lambda
        """
        p = self.parameters
        reference = p["reference"].quantity
        index = p["index"].quantity
        lambda_ = p["lambda_"].quantity
        if index >= 2:
            return np.nan * reference.unit
        else:
            return (2 - index) / lambda_


class ExpCutoffPowerLaw3FGLSpectralModel(SpectralModel):
    r"""Spectral exponential cutoff power-law model used for 3FGL.

    Note that the parametrization is different from `ExpCutoffPowerLawSpectralModel`:

    .. math::
        \phi(E) = \phi_0 \cdot \left(\frac{E}{E_0}\right)^{-\Gamma}
                  \exp \left( \frac{E_0 - E}{E_{C}} \right)

    Parameters
    ----------
    index : `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    ecut : `~astropy.units.Quantity`
        :math:`E_{C}`
    """

    tag = "ExpCutoffPowerLaw3FGLSpectralModel"
    index = Parameter("index", 1.5)
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1")
    reference = Parameter("reference", "1 TeV", frozen=True)
    ecut = Parameter("ecut", "10 TeV")

    @staticmethod
    def evaluate(energy, index, amplitude, reference, ecut):
        """Evaluate the model (static function)."""
        pwl = amplitude * (energy / reference) ** (-index)
        try:
            cutoff = np.exp((reference - energy) / ecut)
        except (AttributeError, TypeError):
            from uncertainties.unumpy import exp

            cutoff = exp((reference - energy) / ecut)
        return pwl * cutoff


class SuperExpCutoffPowerLaw3FGLSpectralModel(SpectralModel):
    r"""Spectral super exponential cutoff power-law model used for 3FGL.

    .. math::
        \phi(E) = \phi_0 \cdot \left(\frac{E}{E_0}\right)^{-\Gamma_1}
                  \exp \left( \left(\frac{E_0}{E_{C}} \right)^{\Gamma_2} -
                              \left(\frac{E}{E_{C}} \right)^{\Gamma_2}
                              \right)

    Parameters
    ----------
    index_1 : `~astropy.units.Quantity`
        :math:`\Gamma_1`
    index_2 : `~astropy.units.Quantity`
        :math:`\Gamma_2`
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    ecut : `~astropy.units.Quantity`
        :math:`E_{C}`
    """

    tag = "SuperExpCutoffPowerLaw3FGLSpectralModel"
    index_1 = Parameter("index_1", 1.5)
    index_2 = Parameter("index_2", 2)
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1")
    reference = Parameter("reference", "1 TeV", frozen=True)
    ecut = Parameter("ecut", "10 TeV")

    @staticmethod
    def evaluate(energy, amplitude, reference, ecut, index_1, index_2):
        """Evaluate the model (static function)."""
        pwl = amplitude * (energy / reference) ** (-index_1)
        try:
            cutoff = np.exp((reference / ecut) ** index_2 - (energy / ecut) ** index_2)
        except (AttributeError, TypeError):
            from uncertainties.unumpy import exp

            cutoff = exp((reference / ecut) ** index_2 - (energy / ecut) ** index_2)
        return pwl * cutoff


class SuperExpCutoffPowerLaw4FGLSpectralModel(SpectralModel):
    r"""Spectral super exponential cutoff power-law model used for 4FGL.

    This model parametrisation is very similar, but slightly different from
    `SuperExpCutoffPowerLaw3FGLSpectralModel` or `ExpCutoffPowerLaw3FGLSpectralModel`.

    See Equation (3) in https://arxiv.org/pdf/1902.10045.pdf

    .. math::
        \phi(E) = \phi_0 \cdot \left(\frac{E}{E_0}\right)^{-\Gamma_1}
                  \exp \left(
                      a \left( E_0 ^{\Gamma_2} - E^{\Gamma_2} \right)
                  \right)

    Parameters
    ----------
    index_1 : `~astropy.units.Quantity`
        :math:`\Gamma_1`
    index_2 : `~astropy.units.Quantity`
        :math:`\Gamma_2`
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    expfactor : `~astropy.units.Quantity`
        :math:`a`, given as dimensionless value but
        internally assumes unit of :math:`[E_0]` power :math:`-\Gamma_2`
    """

    tag = "SuperExpCutoffPowerLaw4FGLSpectralModel"
    index_1 = Parameter("index_1", 1.5)
    index_2 = Parameter("index_2", 2)
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1")
    reference = Parameter("reference", "1 TeV", frozen=True)
    expfactor = Parameter("expfactor", "1e-2")

    @staticmethod
    def evaluate(energy, amplitude, reference, expfactor, index_1, index_2):
        """Evaluate the model (static function)."""
        pwl = amplitude * (energy / reference) ** (-index_1)
        try:
            cutoff = np.exp(
                expfactor
                / reference.unit ** index_2
                * (reference ** index_2 - energy ** index_2)
            )
        except (AttributeError, TypeError):
            from uncertainties.unumpy import exp

            cutoff = exp(expfactor * (reference ** index_2 - energy ** index_2))
        return pwl * cutoff


class LogParabolaSpectralModel(SpectralModel):
    r"""Spectral log parabola model.

    .. math::
        \phi(E) = \phi_0 \left( \frac{E}{E_0} \right) ^ {
          - \alpha - \beta \log{ \left( \frac{E}{E_0} \right) }
        }

    Note that :math:`log` refers to the natural logarithm. This is consistent
    with the `Fermi Science Tools
    <https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html>`_
    and `ctools
    <http://cta.irap.omp.eu/ctools-devel/users/user_manual/getting_started/models.html#log-parabola>`_.
    The `Sherpa <http://cxc.harvard.edu/sherpa/ahelp/logparabola.html_
    package>`_ package, however, uses :math:`log_{10}`. If you have
    parametrization based on :math:`log_{10}` you can use the
    :func:`~gammapy.modeling.models.LogParabolaSpectralModel.from_log10` method.

    Parameters
    ----------
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    alpha : `~astropy.units.Quantity`
        :math:`\alpha`
    beta : `~astropy.units.Quantity`
        :math:`\beta`
    """

    tag = "LogParabolaSpectralModel"
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1")
    reference = Parameter("reference", "10 TeV", frozen=True)
    alpha = Parameter("alpha", 2)
    beta = Parameter("beta", 1)

    @classmethod
    def from_log10(cls, amplitude, reference, alpha, beta):
        """Construct from :math:`log_{10}` parametrization."""
        beta_ = beta / np.log(10)
        return cls(amplitude=amplitude, reference=reference, alpha=alpha, beta=beta_)

    @staticmethod
    def evaluate(energy, amplitude, reference, alpha, beta):
        """Evaluate the model (static function)."""
        try:
            xx = (energy / reference).to("")
            exponent = -alpha - beta * np.log(xx)
        except (AttributeError, TypeError):
            from uncertainties.unumpy import log

            xx = energy / reference
            exponent = -alpha - beta * log(xx)
        return amplitude * np.power(xx, exponent)

    @property
    def e_peak(self):
        r"""Spectral energy distribution peak energy (`~astropy.units.Quantity`).

        This is the peak in E^2 x dN/dE and is given by:

        .. math::
            E_{Peak} = E_{0} \exp{ (2 - \alpha) / (2 * \beta)}
        """
        p = self.parameters
        reference = p["reference"].quantity
        alpha = p["alpha"].quantity
        beta = p["beta"].quantity
        return reference * np.exp((2 - alpha) / (2 * beta))


class TemplateSpectralModel(SpectralModel):
    """A model generated from a table of energy and value arrays.

    the units returned will be the units of the values array provided at
    initialization. The model will return values interpolated in
    log-space, returning 0 for energies outside of the limits of the provided
    energy array.

    Class implementation follows closely what has been done in
    `naima.models.TemplateSpectralModel`

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Array of energies at which the model values are given
    values : array
        Array with the values of the model at energies ``energy``.
    norm : float
        Model scale that is multiplied to the supplied arrays. Defaults to 1.
    interp_kwargs : dict
        Interpolation keyword arguments pass to `scipy.interpolate.interp1d`.
        By default all values outside the interpolation range are set to zero.
        If you want to apply linear extrapolation you can pass `interp_kwargs={'fill_value':
        'extrapolate', 'kind': 'linear'}`. If you want to choose the interpolation
        scaling applied to values, you can use `interp_kwargs={"values_scale": "log"}`.
    meta : dict, optional
        Meta information, meta['filename'] will be used for serialization
    """

    tag = "TemplateSpectralModel"
    norm = Parameter("norm", 1, unit="")
    tilt = Parameter("tilt", 0, unit="", frozen=True)
    reference = Parameter("reference", "1 TeV", frozen=True)

    def __init__(
        self,
        energy,
        values,
        norm=norm.quantity,
        tilt=tilt.quantity,
        reference=reference.quantity,
        interp_kwargs=None,
        meta=None,
    ):
        self.energy = energy
        self.values = values
        self.meta = dict() if meta is None else meta
        interp_kwargs = interp_kwargs or {}
        interp_kwargs.setdefault("values_scale", "log")
        interp_kwargs.setdefault("points_scale", ("log",))

        self._evaluate = ScaledRegularGridInterpolator(
            points=(energy,), values=values, **interp_kwargs
        )

        super().__init__(norm=norm, tilt=tilt, reference=reference)

    @classmethod
    def read_xspec_model(cls, filename, param, **kwargs):
        """Read XSPEC table model.

        The input is a table containing absorbed values from a XSPEC model as a
        function of energy.

        TODO: Format of the file should be described and discussed in
        https://gamma-astro-data-formats.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        filename : str
            File containing the XSPEC model
        param : float
            Model parameter value

        Examples
        --------
        Fill table from an EBL model (Franceschini, 2008)

        >>> from gammapy.modeling.models import TemplateSpectralModel
        >>> filename = '$GAMMAPY_DATA/ebl/ebl_franceschini.fits.gz'
        >>> table_model = TemplateSpectralModel.read_xspec_model(filename=filename, param=0.3)
        """
        filename = make_path(filename)

        # Check if parameter value is in range
        table_param = Table.read(filename, hdu="PARAMETERS")
        pmin = table_param["MINIMUM"]
        pmax = table_param["MAXIMUM"]
        if param < pmin or param > pmax:
            raise ValueError(f"Out of range: param={param}, min={pmin}, max={pmax}")

        # Get energy values
        table_energy = Table.read(filename, hdu="ENERGIES")
        energy_lo = table_energy["ENERG_LO"]
        energy_hi = table_energy["ENERG_HI"]

        # set energy to log-centers
        energy = np.sqrt(energy_lo * energy_hi)

        # Get spectrum values (no interpolation, take closest value for param)
        table_spectra = Table.read(filename, hdu="SPECTRA")
        idx = np.abs(table_spectra["PARAMVAL"] - param).argmin()
        values = u.Quantity(table_spectra[idx][1], "", copy=False)  # no dimension

        kwargs.setdefault("interp_kwargs", {"values_scale": "lin"})
        return cls(energy=energy, values=values, **kwargs)

    def evaluate(self, energy, norm, tilt, reference):
        """Evaluate the model (static function)."""
        values = self._evaluate((energy,), clip=True)
        tilt_factor = np.power(energy / reference, -tilt)
        return norm * values * tilt_factor

    def to_dict(self):
        return {
            "type": self.tag,
            "parameters": self.parameters.to_dict()["parameters"],
            "energy": {
                "data": self.energy.data.tolist(),
                "unit": str(self.energy.unit),
            },
            "values": {
                "data": self.values.data.tolist(),
                "unit": str(self.values.unit),
            },
        }

    @classmethod
    def from_dict(cls, data):
        energy = u.Quantity(data["energy"]["data"], data["energy"]["unit"])
        values = u.Quantity(data["values"]["data"], data["values"]["unit"])
        model = cls(energy=energy, values=values)
        model._update_from_dict(data)
        return model


class ScaleSpectralModel(SpectralModel):
    """Wrapper to scale another spectral model by a norm factor.

    Parameters
    ----------
    model : `SpectralModel`
        Spectral model to wrap.
    norm : float
        Multiplicative norm factor for the model value.
    """

    tag = "ScaleSpectralModel"
    norm = Parameter("norm", 1, unit="")

    def __init__(self, model, norm=norm.quantity):
        self.model = model
        super().__init__(norm=norm)

    def evaluate(self, energy, norm):
        return norm * self.model(energy)


class Absorption:
    r"""Gamma-ray absorption models.

    Usually used as part of `AbsorbedSpectralModel`.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy node values
    param : `~astropy.units.Quantity`
        Parameter node values
    data : `~astropy.units.Quantity`
        Model value
    """

    tag = "Absorption"

    def __init__(self, energy, param, data, filename=None, interp_kwargs=None):
        self.data = data
        self.filename = filename
        # set values log centers
        self.param = param
        self.energy = energy

        interp_kwargs = interp_kwargs or {}
        interp_kwargs.setdefault("points_scale", ("log", "lin"))

        self._evaluate = ScaledRegularGridInterpolator(
            points=(self.param, self.energy), values=data, **interp_kwargs
        )

    def to_dict(self):
        if self.filename is None:
            return {
                "type": self.tag,
                "energy": {
                    "data": self.energy.data.tolist(),
                    "unit": str(self.energy.unit),
                },
                "param": {
                    "data": self.param.data.tolist(),
                    "unit": str(self.param.unit),
                },
                "values": {
                    "data": self.data.data.tolist(),
                    "unit": str(self.data.unit),
                },
            }
        else:
            return {"type": self.tag, "filename": self.filename}

    @classmethod
    def from_dict(cls, data):

        if "filename" in data:
            return cls.read(data["filename"])
        else:
            energy = u.Quantity(data["energy"]["data"], data["energy"]["unit"])
            param = u.Quantity(data["param"]["data"], data["param"]["unit"])
            values = u.Quantity(data["values"]["data"], data["values"]["unit"])
            return cls(energy=energy, param=param, data=values)

    @classmethod
    def read(cls, filename):
        """Build object from an XSPEC model.

        Todo: Format of XSPEC binary files should be referenced at https://gamma-astro-data-formats.readthedocs.io/en/latest/

        Parameters
        ----------
        filename : str
            File containing the model.
        """
        # Create EBL data array
        filename = make_path(filename)
        table_param = Table.read(filename, hdu="PARAMETERS")

        # TODO: for some reason the table contain duplicated values
        param, idx = np.unique(table_param[0]["VALUE"], return_index=True)

        # Get energy values
        table_energy = Table.read(filename, hdu="ENERGIES")
        energy_lo = u.Quantity(
            table_energy["ENERG_LO"], "keV", copy=False
        )  # unit not stored in file
        energy_hi = u.Quantity(
            table_energy["ENERG_HI"], "keV", copy=False
        )  # unit not stored in file
        energy = np.sqrt(energy_lo * energy_hi)

        # Get spectrum values
        table_spectra = Table.read(filename, hdu="SPECTRA")
        data = table_spectra["INTPSPEC"].data[idx, :]
        return cls(energy=energy, param=param, data=data, filename=filename)

    @classmethod
    def read_builtin(cls, name):
        """Read one of the built-in absorption models.

        Parameters
        ----------
        name : {'franceschini', 'dominguez', 'finke'}
            name of one of the available model in gammapy-data

        References
        ----------
        .. [1] Franceschini et al., "Extragalactic optical-infrared background radiation, its time evolution and the cosmic photon-photon opacity",
            `Link <https://ui.adsabs.harvard.edu/abs/2008A%26A...487..837F>`__
        .. [2] Dominguez et al., " Extragalactic background light inferred from AEGIS galaxy-SED-type fractions"
            `Link <https://ui.adsabs.harvard.edu/abs/2011MNRAS.410.2556D>`__
        .. [3] Finke et al., "Modeling the Extragalactic Background Light from Stars and Dust"
            `Link <https://ui.adsabs.harvard.edu/abs/2010ApJ...712..238F>`__
        """
        models = dict()
        models["franceschini"] = "$GAMMAPY_DATA/ebl/ebl_franceschini.fits.gz"
        models["dominguez"] = "$GAMMAPY_DATA/ebl/ebl_dominguez11.fits.gz"
        models["finke"] = "$GAMMAPY_DATA/ebl/frd_abs.fits.gz"

        return cls.read(models[name])

    def table_model(self, parameter):
        """Table model for a given parameter (`~gammapy.modeling.models.TemplateSpectralModel`).

        Parameters
        ----------
        parameter : float
            Parameter value.
        """
        energy = self.energy
        values = self.evaluate(energy=energy, parameter=parameter)
        return TemplateSpectralModel(
            energy=energy, values=values, interp_kwargs={"values_scale": "log"}
        )

    def evaluate(self, energy, parameter):
        """Evaluate model for energy and parameter value."""
        return self._evaluate((parameter, energy))


class AbsorbedSpectralModel(SpectralModel):
    r"""Spectral model with EBL absorption.

    The spectral model is evaluated, and then multiplied with an EBL
    absorption factor given by

    .. math::
        \exp{ \left ( -\alpha \times \tau(E, z) \right )}

    where :math:`\tau(E, z)` is the optical depth predicted by the model
    (`Absorption`), which depends on the energy
    of the gamma-rays and the redshift z of the source, and :math:`\alpha`
    is a scale factor (default: 1) for the optical depth.

    Parameters
    ----------
    spectral_model : `SpectralModel`
        Spectral model.
    absorption : `Absorption`
        Absorption model.
    parameter : float
        parameter value for absorption model
    parameter_name : str, optional
        parameter name
    alpha_norm: float
        Norm of the EBL model
    """

    tag = "AbsorbedSpectralModel"

    def __init__(
        self,
        spectral_model,
        absorption,
        parameter,
        parameter_name="redshift",
        alpha_norm=1.0,
    ):
        self.spectral_model = spectral_model
        self.absorption = absorption
        self.parameter = parameter
        self.parameter_name = parameter_name
        min_ = self.absorption.param.min()
        max_ = self.absorption.param.max()

        par = Parameter(parameter_name, parameter, min=min_, max=max_, frozen=True)
        alpha_norm = Parameter("alpha_norm", alpha_norm, frozen=True)
        parameters = Parameters([par, alpha_norm])

        super()._init_from_parameters(parameters)

    @property
    def parameters(self):
        return self._parameters + self.spectral_model.parameters

    def evaluate(self, energy, **kwargs):
        """Evaluate the model at a given energy."""
        # assign redshift value and remove it from dictionary
        # since it does not belong to the spectral model
        parameter = kwargs[self.parameter_name]
        del kwargs[self.parameter_name]
        del kwargs["alpha_norm"]

        dnde = self.spectral_model.evaluate(energy=energy, **kwargs)
        absorption = self.absorption.evaluate(energy=energy, parameter=parameter)
        # Power rule: (e ^ a) ^ b = e ^ (a * b)
        absorption = np.power(absorption, self.alpha_norm.value)
        return dnde * absorption

    def to_dict(self):
        return {
            "type": self.tag,
            "base_model": self.spectral_model.to_dict(),
            "absorption": self.absorption.to_dict(),
            "absorption_parameter": {
                "name": self.parameter_name,
                "value": self.parameter,
            },
            "parameters": self._parameters.to_dict()["parameters"],
        }

    @classmethod
    def from_dict(cls, data):
        from gammapy.modeling.models import SPECTRAL_MODELS

        model_class = SPECTRAL_MODELS.get_cls(data["base_model"]["type"])
        model = cls(
            spectral_model=model_class.from_dict(data["base_model"]),
            absorption=Absorption.from_dict(data["absorption"]),
            parameter=data["absorption_parameter"]["value"],
            parameter_name=data["absorption_parameter"]["name"],
        )
        model._update_from_dict(data)
        return model


class NaimaSpectralModel(SpectralModel):
    r"""A wrapper for Naima models.

    This class provides an interface with the models defined in the `~naima.models` module.
    The model accepts as a positional argument a `Naima <https://naima.readthedocs.io/en/latest/>`_
    radiative model instance, used to compute the non-thermal emission from populations of
    relativistic electrons or protons due to interactions with the ISM or with radiation and magnetic fields.

    One of the advantages provided by this class consists in the possibility of performing a maximum
    likelihood spectral fit of the model's parameters directly on observations, as opposed to the MCMC
    `fit to flux points <https://naima.readthedocs.io/en/latest/mcmc.html>`_ featured in
    Naima. All the parameters defining the parent population of charged particles are stored as
    `~gammapy.modeling.Parameter` and left free by default. In case that the radiative model is `
    ~naima.radiative.Synchrotron`, the magnetic field strength may also be fitted. Parameters can be
    freezed/unfreezed before the fit, and maximum/minimum values can be set to limit the parameters space to
    the physically interesting region.

    Parameters
    ----------
    radiative_model : `~naima.models.BaseRadiative`
        An instance of a radiative model defined in `~naima.models`
    distance : `~astropy.units.Quantity`, optional
        Distance to the source. If set to 0, the intrinsic differential
        luminosity will be returned. Default is 1 kpc
    seed : str or list of str, optional
        Seed photon field(s) to be considered for the `radiative_model` flux computation,
        in case of a `~naima.models.InverseCompton` model. It can be a subset of the
        `seed_photon_fields` list defining the `radiative_model`. Default is the whole list
        of photon fields
    """

    tag = "NaimaSpectralModel"

    def __init__(self, radiative_model, distance=1.0 * u.kpc, seed=None):
        import naima

        self.radiative_model = radiative_model
        self._particle_distribution = self.radiative_model.particle_distribution
        self.distance = u.Quantity(distance)
        self.seed = seed

        if isinstance(self._particle_distribution, naima.models.TableModel):
            param_names = ["amplitude"]
        else:
            param_names = self._particle_distribution.param_names

        parameters = []

        for name in param_names:
            value = getattr(self._particle_distribution, name)
            parameter = Parameter(name, value)
            parameters.append(parameter)

        # In case of a synchrotron radiative model, append B to the fittable parameters
        if "B" in self.radiative_model.param_names:
            value = getattr(self.radiative_model, "B")
            parameter = Parameter("B", value)
            parameters.append(parameter)

        super()._init_from_parameters(parameters)

    def evaluate_error(self, energy):
        # This method will need to be overridden here, since the radiative models in naima don't
        # support the evaluation on energy values that is performed in the base class method
        raise NotImplementedError(
            "Error evaluation for naima models currently not supported."
        )

    def evaluate(self, energy, **kwargs):
        """Evaluate the model."""
        for name, value in kwargs.items():
            setattr(self._particle_distribution, name, value)

        # Flattening the input energy list and later reshaping the flux list
        # prevents some radiative models from displaying broadcasting problems.
        if self.seed is None:
            dnde = self.radiative_model.flux(energy.flatten(), distance=self.distance)
        else:
            dnde = self.radiative_model.flux(
                energy.flatten(), seed=self.seed, distance=self.distance
            )

        dnde = dnde.reshape(energy.shape)
        unit = 1 / (energy.unit * u.cm ** 2 * u.s)
        return dnde.to(unit)


class GaussianSpectralModel(SpectralModel):
    r"""Gaussian spectral model.

    .. math::
        \phi(E) = \frac{N_0}{\sigma \sqrt{2\pi}}  \exp{ \frac{- \left( E-\bar{E} \right)^2 }{2 \sigma^2} }

    Parameters
    ----------
    norm : `~astropy.units.Quantity`
        :math:`N_0`
    mean : `~astropy.units.Quantity`
        :math:`\bar{E}`
    sigma : `~astropy.units.Quantity`
        :math:`\sigma`
    """

    tag = "GaussianSpectralModel"
    norm = Parameter("norm", 1e-12 * u.Unit("cm-2 s-1"))
    mean = Parameter("mean", 1 * u.TeV)
    sigma = Parameter("sigma", 2 * u.TeV)

    @staticmethod
    def evaluate(energy, norm, mean, sigma):
        return (
            norm
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-(energy - mean) ** 2 / (2 * sigma ** 2))
        )

    def integral(self, emin, emax, **kwargs):
        r"""Integrate Gaussian analytically.

        .. math::
            F(E_{min}, E_{max}) = \frac{N_0}{2} \left[ erf(\frac{E - \bar{E}}{\sqrt{2} \sigma})\right]_{E_{min}}^{E_{max}}

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower and upper bound of integration range
        """
        # kwargs are passed to this function but not used
        # this is to get a consistent API with SpectralModel.integral()
        pars = self.parameters
        u_min = (
            (emin - pars["mean"].quantity) / (np.sqrt(2) * pars["sigma"].quantity)
        ).to_value("")
        u_max = (
            (emax - pars["mean"].quantity) / (np.sqrt(2) * pars["sigma"].quantity)
        ).to_value("")

        return (
            pars["norm"].quantity
            / 2
            * (scipy.special.erf(u_max) - scipy.special.erf(u_min))
        )

    def energy_flux(self, emin, emax):
        r"""Compute energy flux in given energy range analytically.

        .. math::
            G(E_{min}, E_{max}) =  \frac{N_0 \sigma}{\sqrt{2*\pi}}* \left[ - \exp(\frac{E_{min}-\bar{E}}{\sqrt{2} \sigma})
            \right]_{E_{min}}^{E_{max}} + \frac{N_0 * \bar{E}}{2} \left[ erf(\frac{E - \bar{E}}{\sqrt{2} \sigma})
             \right]_{E_{min}}^{E_{max}}


        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        """
        pars = self.parameters
        u_min = (
            (emin - pars["mean"].quantity) / (np.sqrt(2) * pars["sigma"].quantity)
        ).to_value("")
        u_max = (
            (emax - pars["mean"].quantity) / (np.sqrt(2) * pars["sigma"].quantity)
        ).to_value("")
        a = pars["norm"].quantity * pars["sigma"].quantity / np.sqrt(2 * np.pi)
        b = pars["norm"].quantity * pars["mean"].quantity / 2
        return a * (np.exp(-u_min ** 2) - np.exp(-u_max ** 2)) + b * (
            scipy.special.erf(u_max) - scipy.special.erf(u_min)
        )


class LogGaussianSpectralModel(SpectralModel):
    r"""Log Gaussian spectral model.

    .. math::
        \phi(E) = \frac{N_0}{E \, \sigma \sqrt{2\pi}}
         \exp{ \frac{- \left( \ln(\frac{E}{\bar{E}}) \right)^2 }{2 \sigma^2} }

    This model was used in this CTA study for the electron spectrum: Table 3
    in https://ui.adsabs.harvard.edu/abs/2013APh....43..171B

    Parameters
    ----------
    norm : `~astropy.units.Quantity`
        :math:`N_0`
    mean : `~astropy.units.Quantity`
        :math:`\bar{E}`
    sigma : `float`
        :math:`\sigma`
    """

    tag = "LogGaussianSpectralModel"
    norm = Parameter("norm", 1e-12 * u.Unit("cm-2 s-1"))
    mean = Parameter("mean", 1 * u.TeV)
    sigma = Parameter("sigma", 2)

    @staticmethod
    def evaluate(energy, norm, mean, sigma):
        return (
            norm
            / (energy * sigma * np.sqrt(2 * np.pi))
            * np.exp(-(np.log(energy / mean)) ** 2 / (2 * sigma ** 2))
        )
