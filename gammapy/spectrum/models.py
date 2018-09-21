# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import copy
import operator
import astropy.units as u
from astropy.table import Table
from ..utils.energy import EnergyBounds
from ..utils.nddata import NDDataArray, BinnedDataAxis
from .utils import integrate_spectrum
from ..utils.scripts import make_path
from ..utils.fitting import Parameter, Parameters

__all__ = [
    "SpectralModel",
    "ConstantModel",
    "CompoundSpectralModel",
    "PowerLaw",
    "PowerLaw2",
    "ExponentialCutoffPowerLaw",
    "ExponentialCutoffPowerLaw3FGL",
    "PLSuperExpCutoff3FGL",
    "LogParabola",
    "TableModel",
    "AbsorbedSpectralModel",
    "Absorption",
]


class SpectralModel(object):
    """Spectral model base class.

    Derived classes should store their parameters as
    `~gammapy.utils.modeling.Parameters`
    See for example return pardict of
    `~gammapy.spectrum.models.PowerLaw`.
    """

    def __repr__(self):
        fmt = "{}()"
        return fmt.format(self.__class__.__name__)

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n\nParameters: \n\n\t"

        table = self.parameters.to_table()
        ss += "\n\t".join(table.pformat())

        if self.parameters.covariance is not None:
            ss += "\n\nCovariance: \n\n\t"
            covar = self.parameters.covariance_to_table()
            ss += "\n\t".join(covar.pformat())
        return ss

    def __call__(self, energy):
        """Call evaluate method of derived classes"""
        kwargs = dict()
        for par in self.parameters.parameters:
            kwargs[par.name] = par.quantity

        return self.evaluate(energy, **kwargs)

    def __mul__(self, model):
        if not isinstance(model, SpectralModel):
            model = ConstantModel(const=model)
        return CompoundSpectralModel(self, model, operator.mul)

    def __rmul__(self, model):
        # This is needed to support e.g. 5 * model
        return self.__mul__(model)

    def __add__(self, model):
        if not isinstance(model, SpectralModel):
            model = ConstantModel(const=model)
        return CompoundSpectralModel(self, model, operator.add)

    def __radd__(self, model):
        return self.__add__(model)

    def __sub__(self, model):
        if not isinstance(model, SpectralModel):
            model = ConstantModel(const=model)
        return CompoundSpectralModel(self, model, operator.sub)

    def __rsub__(self, model):
        return self.__sub__(model)

    def __truediv__(self, model):
        if not isinstance(model, SpectralModel):
            model = ConstantModel(const=model)
        return CompoundSpectralModel(self, model, operator.truediv)

    def __rtruediv__(self, model):
        return self.__div__(model)

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
        """Integrate spectral model numerically.

        .. math::

            F(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}\phi(E)dE

        If array input for ``emin`` and ``emax`` is given you have to set
        ``intervals=True`` if you want the integral in each energy bin.

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        **kwargs : dict
            Keyword arguments passed to :func:`~gammapy.spectrum.integrate_spectrum`
        """
        return integrate_spectrum(self, emin, emax, **kwargs)

    def integral_error(self, emin, emax, **kwargs):
        """Integrate spectral model numerically with error propagation.

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower adn upper  bound of integration range.
        **kwargs : dict
            Keyword arguments passed to func:`~gammapy.spectrum.integrate_spectrum`

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
        """Compute energy flux in given energy range.

        .. math::

            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}E \phi(E)dE

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        **kwargs : dict
            Keyword arguments passed to func:`~gammapy.spectrum.integrate_spectrum`
        """

        def f(x):
            return x * self(x)

        return integrate_spectrum(f, emin, emax, **kwargs)

    def energy_flux_error(self, emin, emax, **kwargs):
        """Compute energy flux in given energy range with error propagation.

        .. math::

            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}E \phi(E)dE

        Parameters
        ----------
        emin, emax : `~astropy.units.Quantity`
            Lower bound of integration range.
        **kwargs : dict
            Keyword arguments passed to :func:`~gammapy.spectrum.integrate_spectrum`

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

    def to_dict(self):
        """Convert to dict."""
        retval = self.parameters.to_dict()
        retval["name"] = self.__class__.__name__
        return retval

    @classmethod
    def from_dict(cls, val):
        """Create from dict."""
        classname = val.pop("name")
        parameters = Parameters.from_dict(val)
        model = globals()[classname]()
        model.parameters = parameters
        model.parameters.covariance = parameters.covariance
        return model

    def plot(
        self,
        energy_range,
        ax=None,
        energy_unit="TeV",
        flux_unit="cm-2 s-1 TeV-1",
        energy_power=0,
        n_points=100,
        **kwargs
    ):
        """Plot spectral model curve.

        kwargs are forwarded to `matplotlib.pyplot.plot`

        By default a log-log scaling of the axes is used, if you want to change
        the y axis scaling to linear you can use:

        .. code:

            from gammapy.spectrum.models import PowerLaw
            from astropy import units as u

            pwl = ExponentialCutoffPowerLaw()
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
        energy = EnergyBounds.equal_log_spacing(emin, emax, n_points, energy_unit)

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
        **kwargs
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
        energy = EnergyBounds.equal_log_spacing(emin, emax, n_points, energy_unit)

        flux, flux_err = self.evaluate_error(energy).to(flux_unit)

        y_lo = self._plot_scale_flux(energy, flux - flux_err, energy_power)
        y_hi = self._plot_scale_flux(energy, flux + flux_err, energy_power)

        where = (energy >= energy_range[0]) & (energy <= energy_range[1])
        ax.fill_between(energy.value, y_lo.value, y_hi.value, where=where, **kwargs)

        self._plot_format_ax(ax, energy, y_lo, energy_power)
        return ax

    def _plot_format_ax(self, ax, energy, y, energy_power):
        ax.set_xlabel("Energy [{}]".format(energy.unit))
        if energy_power > 0:
            ax.set_ylabel("E{} * Flux [{}]".format(energy_power, y.unit))
        else:
            ax.set_ylabel("Flux [{}]".format(y.unit))

        ax.set_xscale("log", nonposx="clip")
        ax.set_yscale("log", nonposy="clip")

    def _plot_scale_flux(self, energy, flux, energy_power):
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
        from scipy.optimize import brentq

        energies = []
        for val in np.atleast_1d(value):

            def f(x):
                # scale by 1e12 to achieve better precision
                y = self(x * u.TeV).to(value.unit).value
                return 1e12 * (y - val.value)

            energy = brentq(f, emin.to("TeV").value, emax.to("TeV").value)
            energies.append(energy)

        return energies * u.TeV

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)


class ConstantModel(SpectralModel):
    r"""Constant model

    .. math::

        \phi(E) = k

    Parameters
    ----------
    const : `~astropy.units.Quantity`
        :math:`k`
    """

    def __init__(self, const):
        self.parameters = Parameters([Parameter("const", const)])

    @staticmethod
    def evaluate(energy, const):
        """Evaluate the model (static function)."""
        return np.ones(len(np.atleast_1d(energy))) * const


class CompoundSpectralModel(SpectralModel):
    """Represents the algebraic combination of two
    `~gammapy.spectrum.models.SpectralModel`

    """

    def __init__(self, model1, model2, operator):
        self.model1 = model1
        self.model2 = model2
        self.operator = operator

    # TODO: Think about how to deal with covariance matrix
    @property
    def parameters(self):
        val = self.model1.parameters.parameters + self.model2.parameters.parameters
        return Parameters(val)

    @parameters.setter
    def parameters(self, parameters):
        idx = len(self.model1.parameters.parameters)
        self.model1.parameters.parameters = parameters.parameters[:idx]
        self.model2.parameters.parameters = parameters.parameters[idx:]

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n    Component 1 : {}".format(self.model1)
        ss += "\n    Component 2 : {}".format(self.model2)
        ss += "\n    Operator : {}".format(self.operator)
        return ss

    def __call__(self, energy):
        val1 = self.model1(energy)
        val2 = self.model2(energy)

        return self.operator(val1, val2)

    def to_dict(self):
        retval = dict()
        retval["model1"] = self.model1.to_dict()
        retval["model2"] = self.model2.to_dict()
        retval["operator"] = self.operator


class PowerLaw(SpectralModel):
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


    Examples
    --------
    This is how to plot the default `PowerLaw` model:

    .. code:: python

        from astropy import units as u
        from gammapy.spectrum.models import PowerLaw

        pwl = PowerLaw()
        pwl.plot(energy_range=[0.1, 100] * u.TeV)
        plt.show()

    """

    def __init__(
        self, index=2., amplitude=1E-12 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
    ):
        self.parameters = Parameters(
            [
                Parameter("index", index),
                Parameter("amplitude", amplitude),
                Parameter("reference", reference, min=0, frozen=True),
            ]
        )

    @staticmethod
    def evaluate(energy, index, amplitude, reference):
        """Evaluate the model (static function)."""
        return amplitude * np.power((energy / reference), -index)

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
        # kwargs are passed to this function but not used
        # this is to get a consistent API with SpectralModel.integral()
        pars = self.parameters

        if np.isclose(pars["index"].value, 1):
            e_unit = emin.unit
            prefactor = pars["amplitude"].quantity * pars["reference"].quantity.to(
                e_unit
            )
            upper = np.log(emax.to(e_unit).value)
            lower = np.log(emin.value)
        else:
            val = -1 * pars["index"].value + 1
            prefactor = pars["amplitude"].quantity * pars["reference"].quantity / val
            upper = np.power((emax / pars["reference"].quantity), val)
            lower = np.power((emin / pars["reference"].quantity), val)

        return prefactor * (upper - lower)

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
        pars = self.parameters
        val = -1 * pars["index"].value + 2

        if np.isclose(val, 0):
            # see https://www.wolframalpha.com/input/?i=a+*+x+*+(x%2Fb)+%5E+(-2)
            # for reference
            temp = pars["amplitude"].quantity * pars["reference"].quantity ** 2
            return temp * np.log(emax / emin)
        else:
            prefactor = (
                pars["amplitude"].quantity * pars["reference"].quantity ** 2 / val
            )
            upper = (emax / pars["reference"].quantity) ** val
            lower = (emin / pars["reference"].quantity) ** val
            return prefactor * (upper - lower)

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
        return p["reference"].quantity * np.power(base, -1. / p["index"].value)


class PowerLaw2(SpectralModel):
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

    Examples
    --------
    This is how to plot the default `PowerLaw2` model:

    .. code:: python

        from astropy import units as u
        from gammapy.spectrum.models import PowerLaw2

        pwl2 = PowerLaw2()
        pwl2.plot(energy_range=[0.1, 100] * u.TeV)
        plt.show()
    """

    def __init__(
        self,
        amplitude=1E-12 * u.Unit("cm-2 s-1"),
        index=2,
        emin=0.1 * u.TeV,
        emax=100 * u.TeV,
    ):
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude),
                Parameter("index", index),
                Parameter("emin", emin, frozen=True),
                Parameter("emax", emax, frozen=True),
            ]
        )

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
        index = p["index"].value
        top = -index + 1
        bottom = p["emax"].quantity.to("TeV").value ** (-index + 1) - p[
            "emin"
        ].quantity.to("TeV").value ** (-index + 1)
        term = (bottom / top) * (value / p["amplitude"].quantity).to("1 / TeV")
        return np.power(term.value, -1. / index) * u.TeV


class ExponentialCutoffPowerLaw(SpectralModel):
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

    Examples
    --------
    This is how to plot the default `ExponentialCutoffPowerLaw` model:

    .. code:: python

        from astropy import units as u
        from gammapy.spectrum.models import ExponentialCutoffPowerLaw

        ecpl = ExponentialCutoffPowerLaw()
        ecpl.plot(energy_range=[0.1, 100] * u.TeV)
        plt.show()
    """

    def __init__(
        self,
        index=1.5,
        amplitude=1E-12 * u.Unit("cm-2 s-1 TeV-1"),
        reference=1 * u.TeV,
        lambda_=0.1 / u.TeV,
    ):
        self.parameters = Parameters(
            [
                Parameter("index", index),
                Parameter("amplitude", amplitude),
                Parameter("reference", reference, frozen=True),
                Parameter("lambda_", lambda_),
            ]
        )

    @staticmethod
    def evaluate(energy, index, amplitude, reference, lambda_):
        """Evaluate the model (static function)."""
        pwl = amplitude * (energy / reference) ** (-index)
        try:
            cutoff = np.exp(-energy * lambda_)
        except AttributeError:
            from uncertainties.unumpy import exp

            cutoff = exp(-energy * lambda_)
        return pwl * cutoff

    @property
    def e_peak(self):
        r"""Spectral energy distribution peak energy (`~astropy.utils.Quantity`).

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


class ExponentialCutoffPowerLaw3FGL(SpectralModel):
    r"""Spectral exponential cutoff power-law model used for 3FGL.

    Note that the parametrization is different from `ExponentialCutoffPowerLaw`:

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

    Examples
    --------
    This is how to plot the default `ExponentialCutoffPowerLaw3FGL` model:

    .. code:: python

        from astropy import units as u
        from gammapy.spectrum.models import ExponentialCutoffPowerLaw3FGL

        ecpl_3fgl = ExponentialCutoffPowerLaw3FGL()
        ecpl_3fgl.plot(energy_range=[0.1, 100] * u.TeV)
        plt.show()
    """

    def __init__(
        self,
        index=1.5,
        amplitude=1E-12 * u.Unit("cm-2 s-1 TeV-1"),
        reference=1 * u.TeV,
        ecut=10 * u.TeV,
    ):
        self.parameters = Parameters(
            [
                Parameter("index", index),
                Parameter("amplitude", amplitude),
                Parameter("reference", reference, frozen=True),
                Parameter("ecut", ecut),
            ]
        )

    @staticmethod
    def evaluate(energy, index, amplitude, reference, ecut):
        """Evaluate the model (static function)."""
        pwl = amplitude * (energy / reference) ** (-index)
        try:
            cutoff = np.exp((reference - energy) / ecut)
        except AttributeError:
            from uncertainties.unumpy import exp

            cutoff = exp((reference - energy) / ecut)
        return pwl * cutoff


class PLSuperExpCutoff3FGL(SpectralModel):
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

    Examples
    --------
    This is how to plot the default `PLSuperExpCutoff3FGL` model:

    .. code:: python

        from astropy import units as u
        from gammapy.spectrum.models import PLSuperExpCutoff3FGL

        secpl_3fgl = PLSuperExpCutoff3FGL()
        secpl_3fgl.plot(energy_range=[0.1, 100] * u.TeV)
        plt.show()
    """

    def __init__(
        self,
        index_1=1.5,
        index_2=2,
        amplitude=1E-12 * u.Unit("cm-2 s-1 TeV-1"),
        reference=1 * u.TeV,
        ecut=10 * u.TeV,
    ):
        # TODO: order or parameters is different from argument list / docstring. Make uniform!
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude),
                Parameter("reference", reference, frozen=True),
                Parameter("ecut", ecut),
                Parameter("index_1", index_1),
                Parameter("index_2", index_2),
            ]
        )

    @staticmethod
    def evaluate(energy, amplitude, reference, ecut, index_1, index_2):
        """Evaluate the model (static function)."""
        pwl = amplitude * (energy / reference) ** (-index_1)
        try:
            cutoff = np.exp((reference / ecut) ** index_2 - (energy / ecut) ** index_2)
        except AttributeError:
            from uncertainties.unumpy import exp

            cutoff = exp((reference / ecut) ** index_2 - (energy / ecut) ** index_2)
        return pwl * cutoff


class LogParabola(SpectralModel):
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
    :func:`~gammapy.spectrum.models.LogParabola.from_log10` method.

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

    Examples
    --------
    This is how to plot the default `LogParabola` model:

    .. code:: python

        from astropy import units as u
        from gammapy.spectrum.models import LogParabola

        log_parabola = LogParabola()
        log_parabola.plot(energy_range=[0.1, 100] * u.TeV)
        plt.show()
    """

    def __init__(
        self,
        amplitude=1E-12 * u.Unit("cm-2 s-1 TeV-1"),
        reference=10 * u.TeV,
        alpha=2,
        beta=1,
    ):
        self.parameters = Parameters(
            [
                Parameter("amplitude", amplitude),
                Parameter("reference", reference, frozen=True),
                Parameter("alpha", alpha),
                Parameter("beta", beta),
            ]
        )

    @classmethod
    def from_log10(cls, amplitude, reference, alpha, beta):
        """Construct LogParabola from :math:`log_{10}` parametrization"""
        beta_ = beta / np.log(10)
        return cls(amplitude=amplitude, reference=reference, alpha=alpha, beta=beta_)

    @staticmethod
    def evaluate(energy, amplitude, reference, alpha, beta):
        """Evaluate the model (static function)."""
        # TODO: can this comment be removed?
        # cast dimensionless values as np.array, because of bug in Astropy < v1.2
        # https://github.com/astropy/astropy/issues/4764
        try:
            xx = (energy / reference).to("")
            exponent = -alpha - beta * np.log(xx)
        except AttributeError:
            from uncertainties.unumpy import log

            xx = energy / reference
            exponent = -alpha - beta * log(xx)
        return amplitude * np.power(xx, exponent)

    @property
    def e_peak(self):
        r"""Spectral energy distribution peak energy (`~astropy.utils.Quantity`).

        This is the peak in E^2 x dN/dE and is given by:

        .. math::

            E_{Peak} = E_{0} \exp{ (2 - \alpha) / (2 * \beta)}

        """
        p = self.parameters
        reference = p["reference"].quantity
        alpha = p["alpha"].quantity
        beta = p["beta"].quantity
        return reference * np.exp((2 - alpha) / (2 * beta))


class TableModel(SpectralModel):
    """A model generated from a table of energy and value arrays.

    the units returned will be the units of the values array provided at
    initialization. The model will return values interpolated in
    log-space, returning 0 for energies outside of the limits of the provided
    energy array.

    Class implementation follows closely what has been done in
    `naima.models.TableModel`

    Parameters
    ----------
    energy : `~astropy.units.Quantity` array
        Array of energies at which the model values are given
    values : array
        Array with the values of the model at energies ``energy``.
    norm : float
        Model scale that is multiplied to the supplied arrays. Defaults to 1.
    values_scale : {'log', 'lin', 'sqrt'}
        Interpolation scaling applied to values. If the values vary over many magnitudes
        a 'log' scaling is recommended.
    interp_kwargs : dict
        Interpolation keyword arguments pass to `scipy.interpolate.interp1d`.
        By default all values outside the interpolation range are set to zero.
        If you want to apply linear extrapolation you can pass `interp_kwargs={'fill_value':
        'extrapolate', 'kind': 'linear'}`
    meta : dict, optional
        Meta information, meta['filename'] will be used for serialization
    """

    def __init__(
        self, energy, values, norm=1, values_scale="log", interp_kwargs=None, meta=None
    ):
        from scipy.interpolate import interp1d

        self.parameters = Parameters([Parameter("norm", norm, min=0, unit="")])
        self.energy = energy
        self.values = values
        self.values_scale = values_scale
        self.meta = dict() if meta is None else meta

        interp_kwargs = interp_kwargs or {}
        interp_kwargs.setdefault("bounds_error", False)
        interp_kwargs.setdefault("kind", "cubic")

        if values_scale == "log":
            fn_0, fn_1 = np.log, np.exp
            interp_kwargs.setdefault("fill_value", -np.inf)
        elif values_scale == "lin":
            fn_0, fn_1 = lambda x: x, lambda x: x
            interp_kwargs.setdefault("fill_value", 0)
        elif values_scale == "sqrt":
            interp_kwargs.setdefault("fill_value", 0)
            fn_0, fn_1 = np.sqrt, lambda x: x ** 2
        else:
            raise ValueError("Not a valid interpolation mode.")

        with np.errstate(divide="ignore"):
            y = fn_0(values.value)
        x = np.log(energy.value)
        interpy = interp1d(x, y, **interp_kwargs)
        self._evaluate = lambda x: fn_1(interpy(x))

    @classmethod
    def read_xspec_model(cls, filename, param, **kwargs):
        """Read XSPEC table model

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

        >>> from gammapy.spectrum.models import TableModel
        >>> filename = '$GAMMAPY_DATA/ebl/ebl_franceschini.fits.gz'
        >>> table_model = TableModel.read_xspec_model(filename=filename, param=0.3)
        """
        filename = str(make_path(filename))

        # Check if parameter value is in range
        table_param = Table.read(filename, hdu="PARAMETERS")
        param_min = table_param["MINIMUM"]
        param_max = table_param["MAXIMUM"]
        if param < param_min or param > param_max:
            err = "Parameter out of range, param={}, param_min={}, param_max={}".format(
                param, param_min, param_max
            )
            raise ValueError(err)

        # Get energy values
        table_energy = Table.read(filename, hdu="ENERGIES")
        energy_lo = table_energy["ENERG_LO"]
        energy_hi = table_energy["ENERG_HI"]

        # Hack while format is not fixed, energy values are in keV
        energy_bounds = EnergyBounds.from_lower_and_upper_bounds(
            lower=energy_lo, upper=energy_hi, unit=u.keV
        )
        energy = energy_bounds.log_centers

        # Get spectrum values (no interpolation, take closest value for param)
        table_spectra = Table.read(filename, hdu="SPECTRA")
        idx = np.abs(table_spectra["PARAMVAL"] - param).argmin()
        values = table_spectra[idx][1] * u.Unit("")  # no dimension

        kwargs.setdefault("values_scale", "lin")
        return cls(energy=energy, values=values, **kwargs)

    @classmethod
    def read_fermi_isotropic_model(cls, filename, **kwargs):
        """Read Fermi isotropic diffuse model

        see `LAT Background models <https://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html>`_

        Parameters
        ----------
        filename : str
            filename
        """
        filename = str(make_path(filename))
        vals = np.loadtxt(filename)
        energy = u.Quantity(vals[:, 0], "MeV", copy=False)
        values = u.Quantity(vals[:, 1], "MeV-1 s-1 cm-2 sr-1", copy=False)
        return cls(energy=energy, values=values, **kwargs)

    def evaluate(self, energy, norm):
        """Evaluate the model (static function)."""
        x = np.log(energy.to(self.energy.unit).value)
        vals = self._evaluate(x)
        vals = np.clip(vals, 0, np.inf)
        return u.Quantity(norm.value * vals, self.values.unit, copy=False)


class Absorption(object):
    """Gamma-ray absorption models.

    Parameters
    ----------
    energy_lo, energy_hi : `~astropy.units.Quantity`
        Lower and upper bin edges of energy axis
    param_lo, param_hi : `~astropy.units.Quantity`
        Lower and upper bin edges of parameter axis
    data : `~astropy.units.Quantity`
        Model value

    Examples
    --------
    Create and plot EBL absorption models for a redshift of 0.5:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import astropy.units as u
        from gammapy.spectrum.models import Absorption

        # Load tables for z=0.5
        redshift = 0.5
        dominguez = Absorption.read_builtin('dominguez').table_model(redshift)
        franceschini = Absorption.read_builtin('franceschini').table_model(redshift)
        finke = Absorption.read_builtin('finke').table_model(redshift)

        # start customised plot
        energy_range = [0.08, 3] * u.TeV
        ax = plt.gca()
        opts = dict(energy_range=energy_range, energy_unit='TeV', ax=ax, flux_unit='')
        franceschini.plot(label='Franceschini 2008', **opts)
        finke.plot(label='Finke 2010', **opts)
        dominguez.plot(label='Dominguez 2011', **opts)

        # tune plot
        ax.set_ylabel(r'Absorption coefficient [$\exp{(-\tau(E))}$]')
        ax.set_xlim(energy_range.value)  # we get ride of units
        ax.set_ylim([1.e-4, 2.])
        ax.set_yscale('log')
        ax.set_title('EBL models (z=' + str(redshift) + ')')
        plt.grid(which='both')
        plt.legend(loc='best') # legend

        # show plot
        plt.show()
    """

    def __init__(self, energy_lo, energy_hi, param_lo, param_hi, data):
        axes = [
            BinnedDataAxis(
                param_lo, param_hi, interpolation_mode="linear", name="parameter"
            ),
            BinnedDataAxis(
                energy_lo, energy_hi, interpolation_mode="log", name="energy"
            ),
        ]

        self.data = NDDataArray(axes=axes, data=data)
        self.data.default_interp_kwargs["fill_value"] = None

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
        filename = str(make_path(filename))
        table_param = Table.read(filename, hdu="PARAMETERS")

        par_min = table_param["MINIMUM"]
        par_max = table_param["MAXIMUM"]

        par_array = table_param[0]["VALUE"]
        par_delta = np.diff(par_array) * 0.5

        param_lo, param_hi = par_array, par_array  # initialisation
        param_lo[0] = par_min - par_delta[0]
        param_lo[1:] -= par_delta
        param_hi[:-1] += par_delta
        param_hi[-1] = par_max

        # Get energy values
        table_energy = Table.read(filename, hdu="ENERGIES")
        energy_lo = table_energy["ENERG_LO"] * u.keV  # unit not stored in file
        energy_hi = table_energy["ENERG_HI"] * u.keV  # unit not stored in file

        # Energies are in keV
        energy_bounds = EnergyBounds.from_lower_and_upper_bounds(
            lower=energy_lo, upper=energy_hi, unit=u.keV
        )

        # Get spectrum values
        table_spectra = Table.read(filename, hdu="SPECTRA")
        data = table_spectra["INTPSPEC"].data

        return cls(
            energy_lo=energy_bounds.lower_bounds,
            energy_hi=energy_bounds.upper_bounds,
            param_lo=param_lo,
            param_hi=param_hi,
            data=data,
        )

    @classmethod
    def read_builtin(cls, name):
        """Read one of the built-in absorption models.

        Parameters
        ----------
        name : {'franceschini', 'dominguez', 'finke'}
            name of one of the available model in gammapy-extra

        References
        ----------
        .. [1] Franceschini et al., "Extragalactic optical-infrared background radiation, its time evolution and the cosmic photon-photon opacity",
            `Link <http://adsabs.harvard.edu/abs/2008A%26A...487..837F>`__
        .. [2] Dominguez et al., " Extragalactic background light inferred from AEGIS galaxy-SED-type fractions"
            `Link <http://adsabs.harvard.edu/abs/2011MNRAS.410.2556D>`__
        .. [3] Finke et al., "Modeling the Extragalactic Background Light from Stars and Dust"
            `Link <http://adsabs.harvard.edu/abs/2010ApJ...712..238F>`__
        """
        models = dict()
        models["franceschini"] = "$GAMMAPY_DATA/ebl/ebl_franceschini.fits.gz"
        models["dominguez"] = "$GAMMAPY_DATA/ebl/ebl_dominguez11.fits.gz"
        models["finke"] = "$GAMMAPY_DATA/ebl/frd_abs.fits.gz"

        return cls.read(models[name])

    def table_model(self, parameter, unit="TeV"):
        """Table model for a given parameter (`~gammapy.spectrum.models.TableModel`).

        Parameters
        ----------
        parameter : float
            Parameter value.
        unit : str, (optional)
            desired value for energy axis
        """
        energy_axis = self.data.axes[1]
        energy = (energy_axis.log_center()).to(unit)

        values = self.evaluate(energy=energy, parameter=parameter)

        return TableModel(energy=energy, values=values, values_scale="lin")

    def evaluate(self, energy, parameter):
        """Evaluate model for energy and parameter value."""
        return self.data.evaluate(energy=energy, parameter=parameter)


class AbsorbedSpectralModel(SpectralModel):
    """Spectral model with EBL absorption.

    Parameters
    ----------
    spectral_model : `~gammapy.spectrum.models.SpectralModel`
        Spectral model.
    absorption : `~gammapy.spectrum.models.Absorption`
        Absorption model.
    parameter : float
        parameter value for absorption model
    parameter_name : str, optional
        parameter name
    """

    def __init__(
        self, spectral_model, absorption, parameter, parameter_name="redshift"
    ):
        self.spectral_model = spectral_model
        self.absorption = absorption
        self.parameter = parameter
        self.parameter_name = parameter_name

        # initialise list parameters from spectral model
        param_list = []
        for param in spectral_model.parameters.parameters:
            param_list.append(param)

        # Add parameter to the list
        min_ = self.absorption.data.axes[0].lo[0]
        max_ = self.absorption.data.axes[0].lo[-1]
        par = Parameter(parameter_name, parameter, min=min_, max=max_, frozen=True)
        param_list.append(par)

        self.parameters = Parameters(param_list)

    def evaluate(self, energy, **kwargs):
        """Evaluate the model at a given energy."""
        # assign redshift value and remove it from dictionnary
        # since it does not belong to the spectral model
        parameter = kwargs[self.parameter_name]
        del kwargs[self.parameter_name]

        flux = self.spectral_model.evaluate(energy=energy, **kwargs)
        absorption = self.absorption.evaluate(energy=energy, parameter=parameter)
        return flux * absorption
