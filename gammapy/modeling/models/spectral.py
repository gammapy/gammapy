# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy."""
import logging
import operator
import os
import numpy as np
import scipy.optimize
import scipy.special
import astropy.units as u
from astropy import constants as const
from astropy.table import Table
from astropy.utils.decorators import classproperty
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from gammapy.maps import MapAxis, RegionNDMap
from gammapy.modeling import Parameter, Parameters
from gammapy.utils.integrate import trapz_loglog
from gammapy.utils.interpolation import (
    ScaledRegularGridInterpolator,
    interpolation_scale,
)
from gammapy.utils.roots import find_roots
from gammapy.utils.scripts import make_path
from .core import ModelBase

log = logging.getLogger(__name__)


__all__ = [
    "BrokenPowerLawSpectralModel",
    "CompoundSpectralModel",
    "ConstantSpectralModel",
    "EBLAbsorptionNormSpectralModel",
    "ExpCutoffPowerLaw3FGLSpectralModel",
    "ExpCutoffPowerLawNormSpectralModel",
    "ExpCutoffPowerLawSpectralModel",
    "GaussianSpectralModel",
    "integrate_spectrum",
    "LogParabolaNormSpectralModel",
    "LogParabolaSpectralModel",
    "NaimaSpectralModel",
    "PiecewiseNormSpectralModel",
    "PowerLaw2SpectralModel",
    "PowerLawNormSpectralModel",
    "PowerLawSpectralModel",
    "scale_plot_flux",
    "ScaleSpectralModel",
    "SmoothBrokenPowerLawSpectralModel",
    "SpectralModel",
    "SuperExpCutoffPowerLaw3FGLSpectralModel",
    "SuperExpCutoffPowerLaw4FGLDR3SpectralModel",
    "SuperExpCutoffPowerLaw4FGLSpectralModel",
    "TemplateSpectralModel",
]


def scale_plot_flux(flux, energy_power=0):
    """Scale flux to plot

    Parameters
    ----------
    flux : `Map`
        Flux map
    energy_power : int, optional
        Power of energy to multiply flux axis with

    Returns
    -------
    flux : `Map`
        Scaled flux map
    """
    energy = flux.geom.get_coord(sparse=True)["energy"]
    try:
        eunit = [_ for _ in flux.unit.bases if _.physical_type == "energy"][0]
    except IndexError:
        eunit = energy.unit
    y = flux * np.power(energy, energy_power)
    return y.to_unit(flux.unit * eunit**energy_power)


def integrate_spectrum(func, energy_min, energy_max, ndecade=100):
    """Integrate 1d function using the log-log trapezoidal rule.

    Internally an oversampling of the energy bins to "ndecade" is used.

    Parameters
    ----------
    func : callable
        Function to integrate.
    energy_min : `~astropy.units.Quantity`
        Integration range minimum
    energy_max : `~astropy.units.Quantity`
        Integration range minimum
    ndecade : int, optional
        Number of grid points per decade used for the integration.
        Default : 100
    """
    # Here we impose to duplicate the number
    num = np.maximum(np.max(ndecade * np.log10(energy_max / energy_min)), 2)
    energy = np.geomspace(energy_min, energy_max, num=int(num), axis=-1)
    integral = trapz_loglog(func(energy), energy, axis=-1)
    return integral.sum(axis=0)


class SpectralModel(ModelBase):
    """Spectral model base class."""

    _type = "spectral"

    def __call__(self, energy):
        kwargs = {par.name: par.quantity for par in self.parameters}
        kwargs = self._convert_evaluate_unit(kwargs, energy)
        return self.evaluate(energy, **kwargs)

    @classproperty
    def is_norm_spectral_model(cls):
        """Whether model is a norm spectral model"""
        return "Norm" in cls.__name__

    @staticmethod
    def _convert_evaluate_unit(kwargs_ref, energy):
        kwargs = {}
        for name, quantity in kwargs_ref.items():
            if quantity.unit.physical_type == "energy":
                quantity = quantity.to(energy.unit)
            kwargs[name] = quantity
        return kwargs

    def __add__(self, model):
        if not isinstance(model, SpectralModel):
            model = ConstantSpectralModel(const=model)
        return CompoundSpectralModel(self, model, operator.add)

    def __mul__(self, other):
        if isinstance(other, SpectralModel):
            return CompoundSpectralModel(self, other, operator.mul)
        else:
            raise TypeError(f"Multiplication invalid for type {other!r}")

    def __radd__(self, model):
        return self.__add__(model)

    def __sub__(self, model):
        if not isinstance(model, SpectralModel):
            model = ConstantSpectralModel(const=model)
        return CompoundSpectralModel(self, model, operator.sub)

    def __rsub__(self, model):
        return self.__sub__(model)

    def _propagate_error(self, epsilon, fct, **kwargs):
        """Evaluate error for a given function with uncertainty propagation.

        Parameters
        ----------
        fct : `~astropy.units.Quantity`
            Function to estimate the error.
        epsilon : float
            Step size of the gradient evaluation. Given as a
            fraction of the parameter error.
        **kwargs : dict
            Keyword argument

        Returns
        -------
        f_cov : `~astropy.units.Quantity`
            Error of the given function.
        """
        eps = np.sqrt(np.diag(self.covariance)) * epsilon

        n, f_0 = len(self.parameters), fct(**kwargs)
        shape = (n, len(np.atleast_1d(f_0)))
        df_dp = np.zeros(shape)

        for idx, parameter in enumerate(self.parameters):
            if parameter.frozen or eps[idx] == 0:
                continue

            parameter.value += eps[idx]
            df = fct(**kwargs) - f_0

            df_dp[idx] = df.value / eps[idx]
            parameter.value -= eps[idx]

        f_cov = df_dp.T @ self.covariance @ df_dp
        f_err = np.sqrt(np.diagonal(f_cov))
        return u.Quantity([f_0.value, f_err], unit=f_0.unit)

    def evaluate_error(self, energy, epsilon=1e-4):
        """Evaluate spectral model with error propagation.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy at which to evaluate
        epsilon : float
            Step size of the gradient evaluation. Given as a
            fraction of the parameter error.

        Returns
        -------
        dnde, dnde_error : tuple of `~astropy.units.Quantity`
            Tuple of flux and flux error.
        """
        return self._propagate_error(epsilon=epsilon, fct=self, energy=energy)

    def integral(self, energy_min, energy_max, **kwargs):
        r"""Integrate spectral model numerically if no analytical solution defined.

        .. math::
            F(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}} \phi(E) dE

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        **kwargs : dict
            Keyword arguments passed to :func:`~gammapy.utils.integrate.integrate_spectrum`
        """

        if hasattr(self, "evaluate_integral"):
            kwargs = {par.name: par.quantity for par in self.parameters}
            kwargs = self._convert_evaluate_unit(kwargs, energy_min)
            return self.evaluate_integral(energy_min, energy_max, **kwargs)
        else:
            return integrate_spectrum(self, energy_min, energy_max, **kwargs)

    def integral_error(self, energy_min, energy_max, epsilon=1e-4, **kwargs):
        """Evaluate the error of the integral flux of a given spectrum in
        a given energy range.

        Parameters
        ----------
        energy_min, energy_max :  `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        epsilon : float
            Step size of the gradient evaluation. Given as a
            fraction of the parameter error.


        Returns
        -------
        flux, flux_err : tuple of `~astropy.units.Quantity`
            Integral flux and flux error between energy_min and energy_max.
        """
        return self._propagate_error(
            epsilon=epsilon,
            fct=self.integral,
            energy_min=energy_min,
            energy_max=energy_max,
            **kwargs,
        )

    def energy_flux(self, energy_min, energy_max, **kwargs):
        r"""Compute energy flux in given energy range.

        .. math::
            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}} E \phi(E) dE

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        **kwargs : dict
            Keyword arguments passed to func:`~gammapy.utils.integrate.integrate_spectrum`
        """

        def f(x):
            return x * self(x)

        if hasattr(self, "evaluate_energy_flux"):
            kwargs = {par.name: par.quantity for par in self.parameters}
            kwargs = self._convert_evaluate_unit(kwargs, energy_min)
            return self.evaluate_energy_flux(energy_min, energy_max, **kwargs)
        else:
            return integrate_spectrum(f, energy_min, energy_max, **kwargs)

    def energy_flux_error(self, energy_min, energy_max, epsilon=1e-4, **kwargs):
        """Evaluate the error of the energy flux of a given spectrum in
            a given energy range.

        Parameters
        ----------
        energy_min, energy_max :  `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        epsilon : float
            Step size of the gradient evaluation. Given as a
            fraction of the parameter error.


        Returns
        -------
        energy_flux, energy_flux_err : tuple of `~astropy.units.Quantity`
            Energy flux and energy flux error between energy_min and energy_max.
        """
        return self._propagate_error(
            epsilon=epsilon,
            fct=self.energy_flux,
            energy_min=energy_min,
            energy_max=energy_max,
            **kwargs,
        )

    def reference_fluxes(self, energy_axis):
        """Get reference fluxes for a given energy axis.

        Parameters
        ----------
        energy_axis : `MapAxis`
            Energy axis

        Returns
        -------
        fluxes : dict of `~astropy.units.Quantity`
            Reference fluxes
        """
        energy = energy_axis.center
        energy_min, energy_max = energy_axis.edges_min, energy_axis.edges_max
        return {
            "e_ref": energy,
            "e_min": energy_min,
            "e_max": energy_max,
            "ref_dnde": self(energy),
            "ref_flux": self.integral(energy_min, energy_max),
            "ref_eflux": self.energy_flux(energy_min, energy_max),
            "ref_e2dnde": self(energy) * energy**2,
        }

    def _get_plot_flux(self, energy, sed_type):
        flux = RegionNDMap.create(region=None, axes=[energy])
        flux_err = RegionNDMap.create(region=None, axes=[energy])

        if sed_type in ["dnde", "norm"]:
            flux.quantity, flux_err.quantity = self.evaluate_error(energy.center)

        elif sed_type == "e2dnde":
            flux.quantity, flux_err.quantity = energy.center**2 * self.evaluate_error(
                energy.center
            )

        elif sed_type == "flux":
            flux.quantity, flux_err.quantity = self.integral_error(
                energy.edges_min, energy.edges_max
            )

        elif sed_type == "eflux":
            flux.quantity, flux_err.quantity = self.energy_flux_error(
                energy.edges_min, energy.edges_max
            )
        else:
            raise ValueError(f"Not a valid SED type: '{sed_type}'")

        return flux, flux_err

    def plot(
        self,
        energy_bounds,
        ax=None,
        sed_type="dnde",
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
            ax = pwl.plot(energy_bounds=(0.1, 100) * u.TeV)
            ax.set_yscale('linear')

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy_bounds : `~astropy.units.Quantity`
            Plot energy bounds passed to MapAxis.from_energy_bounds
        sed_type : {"dnde", "flux", "eflux", "e2dnde"}
            Evaluation methods of the model
        energy_power : int, optional
            Power of energy to multiply flux axis with
        n_points : int, optional
            Number of evaluation nodes
        **kwargs : dict
            Keyword arguments forwarded to `~matplotlib.pyplot.plot`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        """
        from gammapy.estimators.map.core import DEFAULT_UNIT

        ax = plt.gca() if ax is None else ax

        if self.is_norm_spectral_model:
            sed_type = "norm"

        energy_min, energy_max = energy_bounds
        energy = MapAxis.from_energy_bounds(
            energy_min,
            energy_max,
            n_points,
        )

        if ax.yaxis.units is None:
            ax.yaxis.set_units(DEFAULT_UNIT[sed_type] * energy.unit**energy_power)

        flux, _ = self._get_plot_flux(sed_type=sed_type, energy=energy)

        flux = scale_plot_flux(flux, energy_power=energy_power)

        with quantity_support():
            ax.plot(energy.center, flux.quantity[:, 0, 0], **kwargs)

        self._plot_format_ax(ax, energy_power, sed_type)
        return ax

    def plot_error(
        self,
        energy_bounds,
        ax=None,
        sed_type="dnde",
        energy_power=0,
        n_points=100,
        **kwargs,
    ):
        """Plot spectral model error band.

        .. note::

            This method calls ``ax.set_yscale("log", nonpositive='clip')`` and
            ``ax.set_xscale("log", nonposx='clip')`` to create a log-log representation.
            The additional argument ``nonposx='clip'`` avoids artefacts in the plot,
            when the error band extends to negative values (see also
            https://github.com/matplotlib/matplotlib/issues/8623).

            When you call ``plt.loglog()`` or ``plt.semilogy()`` explicitly in your
            plotting code and the error band extends to negative values, it is not
            shown correctly. To circumvent this issue also use
            ``plt.loglog(nonposx='clip', nonpositive='clip')``
            or ``plt.semilogy(nonpositive='clip')``.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy_bounds : `~astropy.units.Quantity`
            Plot energy bounds passed to MapAxis.from_energy_bounds
        sed_type : {"dnde", "flux", "eflux", "e2dnde"}
            Evaluation methods of the model
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
        from gammapy.estimators.map.core import DEFAULT_UNIT

        ax = plt.gca() if ax is None else ax

        if self.is_norm_spectral_model:
            sed_type = "norm"

        energy_min, energy_max = energy_bounds
        energy = MapAxis.from_energy_bounds(
            energy_min,
            energy_max,
            n_points,
        )

        kwargs.setdefault("facecolor", "black")
        kwargs.setdefault("alpha", 0.2)
        kwargs.setdefault("linewidth", 0)

        if ax.yaxis.units is None:
            ax.yaxis.set_units(DEFAULT_UNIT[sed_type] * energy.unit**energy_power)

        flux, flux_err = self._get_plot_flux(sed_type=sed_type, energy=energy)
        y_lo = scale_plot_flux(flux - flux_err, energy_power).quantity[:, 0, 0]
        y_hi = scale_plot_flux(flux + flux_err, energy_power).quantity[:, 0, 0]

        with quantity_support():
            ax.fill_between(energy.center, y_lo, y_hi, **kwargs)

        self._plot_format_ax(ax, energy_power, sed_type)
        return ax

    @staticmethod
    def _plot_format_ax(ax, energy_power, sed_type):
        ax.set_xlabel(f"Energy [{ax.xaxis.units}]")
        if energy_power > 0:
            ax.set_ylabel(f"e{energy_power} * {sed_type} [{ax.yaxis.units}]")
        else:
            ax.set_ylabel(f"{sed_type} [{ax.yaxis.units}]")

        ax.set_xscale("log", nonpositive="clip")
        ax.set_yscale("log", nonpositive="clip")

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

    def inverse(self, value, energy_min=0.1 * u.TeV, energy_max=100 * u.TeV):
        """Return energy for a given function value of the spectral model.

        Calls the `scipy.optimize.brentq` numerical root finding method.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Function value of the spectral model.
        energy_min : `~astropy.units.Quantity`
            Lower energy bound of the roots finding
        energy_max : `~astropy.units.Quantity`
            Upper energy bound of the roots finding

        Returns
        -------
        energy : `~astropy.units.Quantity`
            Energies at which the model has the given ``value``.
        """

        eunit = "TeV"
        energy_min = energy_min.to(eunit)
        energy_max = energy_max.to(eunit)

        def f(x):
            # scale by 1e12 to achieve better precision
            energy = u.Quantity(x, eunit, copy=False)
            y = self(energy).to_value(value.unit)
            return 1e12 * (y - value.value)

        roots, res = find_roots(f, energy_min, energy_max, points_scale="log")
        return roots

    def inverse_all(self, values, energy_min=0.1 * u.TeV, energy_max=100 * u.TeV):
        """Return energies for multiple function values of the spectral model.

        Calls the `scipy.optimize.brentq` numerical root finding method.

        Parameters
        ----------
        values : `~astropy.units.Quantity`
            Function values of the spectral model.
        energy_min : `~astropy.units.Quantity`
            Lower energy bound of the roots finding
        energy_max : `~astropy.units.Quantity`
            Upper energy bound of the roots finding

        Returns
        -------
        energy : list of `~astropy.units.Quantity`
            each element contain the energies at which the model
            has corresponding value of ``values``.
        """
        energies = []
        for val in np.atleast_1d(values):
            res = self.inverse(val, energy_min, energy_max)
            energies.append(res)
        return energies


class ConstantSpectralModel(SpectralModel):
    r"""Constant model.

    For more information see :ref:`constant-spectral-model`.

    Parameters
    ----------
    const : `~astropy.units.Quantity`
        :math:`k`
    """

    tag = ["ConstantSpectralModel", "const"]
    const = Parameter("const", "1e-12 cm-2 s-1 TeV-1", is_norm=True)

    @staticmethod
    def evaluate(energy, const):
        """Evaluate the model (static function)."""
        return np.ones(np.atleast_1d(energy).shape) * const


class CompoundSpectralModel(SpectralModel):
    """Arithmetic combination of two spectral models.

    For more information see :ref:`compound-spectral-model`.
    """

    tag = ["CompoundSpectralModel", "compound"]

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
            f"    Operator : {self.operator.__name__}\n"
        )

    def __call__(self, energy):
        val1 = self.model1(energy)
        val2 = self.model2(energy)
        return self.operator(val1, val2)

    def to_dict(self, full_output=False):
        dict1 = self.model1.to_dict(full_output)
        dict2 = self.model2.to_dict(full_output)
        return {
            self._type: {
                "type": self.tag[0],
                "model1": dict1["spectral"],  # for cleaner output
                "model2": dict2["spectral"],
                "operator": self.operator.__name__,
            }
        }

    @classmethod
    def from_dict(cls, data):
        from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY

        data = data["spectral"]
        model1_cls = SPECTRAL_MODEL_REGISTRY.get_cls(data["model1"]["type"])
        model1 = model1_cls.from_dict({"spectral": data["model1"]})
        model2_cls = SPECTRAL_MODEL_REGISTRY.get_cls(data["model2"]["type"])
        model2 = model2_cls.from_dict({"spectral": data["model2"]})
        op = getattr(operator, data["operator"])
        return cls(model1, model2, op)


class PowerLawSpectralModel(SpectralModel):
    r"""Spectral power-law model.

    For more information see :ref:`powerlaw-spectral-model`.

    Parameters
    ----------
    index : `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`

    See Also
    --------
    PowerLaw2SpectralModel, PowerLawNormSpectralModel
    """

    tag = ["PowerLawSpectralModel", "pl"]
    index = Parameter("index", 2.0)
    amplitude = Parameter(
        "amplitude",
        "1e-12 cm-2 s-1 TeV-1",
        scale_method="scale10",
        interp="log",
        is_norm=True,
    )
    reference = Parameter("reference", "1 TeV", frozen=True)

    @staticmethod
    def evaluate(energy, index, amplitude, reference):
        """Evaluate the model (static function)."""
        return amplitude * np.power((energy / reference), -index)

    @staticmethod
    def evaluate_integral(energy_min, energy_max, index, amplitude, reference):
        r"""Integrate power law analytically (static function).

        .. math::
            F(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}\phi(E)dE = \left.
            \phi_0 \frac{E_0}{-\Gamma + 1} \left( \frac{E}{E_0} \right)^{-\Gamma + 1}
            \right \vert _{E_{min}}^{E_{max}}

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Lower and upper bound of integration range
        """
        val = -1 * index + 1

        prefactor = amplitude * reference / val
        upper = np.power((energy_max / reference), val)
        lower = np.power((energy_min / reference), val)
        integral = prefactor * (upper - lower)

        mask = np.isclose(val, 0)

        if mask.any():
            integral[mask] = (amplitude * reference * np.log(energy_max / energy_min))[
                mask
            ]

        return integral

    @staticmethod
    def evaluate_energy_flux(energy_min, energy_max, index, amplitude, reference):
        r"""Compute energy flux in given energy range analytically (static function).

        .. math::
            G(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}}E \phi(E)dE = \left.
            \phi_0 \frac{E_0^2}{-\Gamma + 2} \left( \frac{E}{E_0} \right)^{-\Gamma + 2}
            \right \vert _{E_{min}}^{E_{max}}

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        """
        val = -1 * index + 2

        prefactor = amplitude * reference**2 / val
        upper = (energy_max / reference) ** val
        lower = (energy_min / reference) ** val
        energy_flux = prefactor * (upper - lower)

        mask = np.isclose(val, 0)

        if mask.any():
            # see https://www.wolframalpha.com/input/?i=a+*+x+*+(x%2Fb)+%5E+(-2)
            # for reference
            energy_flux[mask] = (
                amplitude * reference**2 * np.log(energy_max / energy_min)[mask]
            )

        return energy_flux

    def inverse(self, value, *args):
        """Return energy for a given function value of the spectral model.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Function value of the spectral model.
        """
        base = value / self.amplitude.quantity
        return self.reference.quantity * np.power(base, -1.0 / self.index.value)

    @property
    def pivot_energy(self):
        r"""The decorrelation energy is defined as:

        .. math::

            E_D = E_0 * \exp{cov(\phi_0, \Gamma) / (\phi_0 \Delta \Gamma^2)}

        Formula (1) in https://arxiv.org/pdf/0910.4881.pdf
        """
        index_err = self.index.error
        reference = self.reference.quantity
        amplitude = self.amplitude.quantity
        cov_index_ampl = self.covariance.data[0, 1] * amplitude.unit
        return reference * np.exp(cov_index_ampl / (amplitude * index_err**2))


class PowerLawNormSpectralModel(SpectralModel):
    r"""Spectral power-law model with normalized amplitude parameter.

    Parameters
    ----------
    tilt : `~astropy.units.Quantity`
        :math:`\Gamma`
    norm : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`

    See Also
    --------
    PowerLawSpectralModel, PowerLaw2SpectralModel
    """

    tag = ["PowerLawNormSpectralModel", "pl-norm"]
    norm = Parameter("norm", 1, unit="", interp="log", is_norm=True)
    tilt = Parameter("tilt", 0, frozen=True)
    reference = Parameter("reference", "1 TeV", frozen=True)

    @staticmethod
    def evaluate(energy, tilt, norm, reference):
        """Evaluate the model (static function)."""
        return norm * np.power((energy / reference), -tilt)

    @staticmethod
    def evaluate_integral(energy_min, energy_max, tilt, norm, reference):
        """Evaluate pwl integral."""
        val = -1 * tilt + 1

        prefactor = norm * reference / val
        upper = np.power((energy_max / reference), val)
        lower = np.power((energy_min / reference), val)
        integral = prefactor * (upper - lower)

        mask = np.isclose(val, 0)

        if mask.any():
            integral[mask] = (norm * reference * np.log(energy_max / energy_min))[mask]

        return integral

    @staticmethod
    def evaluate_energy_flux(energy_min, energy_max, tilt, norm, reference):
        """Evaluate the energy flux (static function)"""
        val = -1 * tilt + 2

        prefactor = norm * reference**2 / val
        upper = (energy_max / reference) ** val
        lower = (energy_min / reference) ** val
        energy_flux = prefactor * (upper - lower)

        mask = np.isclose(val, 0)

        if mask.any():
            # see https://www.wolframalpha.com/input/?i=a+*+x+*+(x%2Fb)+%5E+(-2)
            # for reference
            energy_flux[mask] = (
                norm * reference**2 * np.log(energy_max / energy_min)[mask]
            )

        return energy_flux

    def inverse(self, value, *args):
        """Return energy for a given function value of the spectral model.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Function value of the spectral model.
        """
        base = value / self.norm.quantity
        return self.reference.quantity * np.power(base, -1.0 / self.tilt.value)

    @property
    def pivot_energy(self):
        r"""The decorrelation energy is defined as:

        .. math::

            E_D = E_0 * \exp{cov(\phi_0, \Gamma) / (\phi_0 \Delta \Gamma^2)}

        Formula (1) in https://arxiv.org/pdf/0910.4881.pdf
        """
        tilt_err = self.tilt.error
        reference = self.reference.quantity
        norm = self.norm.quantity
        cov_tilt_norm = self.covariance.data[0, 1] * norm.unit
        return reference * np.exp(cov_tilt_norm / (norm * tilt_err**2))


class PowerLaw2SpectralModel(SpectralModel):
    r"""Spectral power-law model with integral as amplitude parameter.

    For more information see :ref:`powerlaw2-spectral-model`.

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

    See Also
    --------
    PowerLawSpectralModel, PowerLawNormSpectralModel
    """
    tag = ["PowerLaw2SpectralModel", "pl-2"]

    amplitude = Parameter(
        name="amplitude",
        value="1e-12 cm-2 s-1",
        scale_method="scale10",
        interp="log",
        is_norm=True,
    )
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

    @staticmethod
    def evaluate_integral(energy_min, energy_max, amplitude, index, emin, emax):
        r"""Integrate power law analytically.

        .. math::
            F(E_{min}, E_{max}) = F_0 \cdot \frac{E_{max}^{\Gamma + 1} \
                                - E_{min}^{\Gamma + 1}}{E_{0, max}^{\Gamma + 1} \
                                - E_{0, min}^{\Gamma + 1}}

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        """
        temp1 = np.power(energy_max, -index.value + 1)
        temp2 = np.power(energy_min, -index.value + 1)
        top = temp1 - temp2

        temp1 = np.power(emax, -index.value + 1)
        temp2 = np.power(emin, -index.value + 1)
        bottom = temp1 - temp2

        return amplitude * top / bottom

    def inverse(self, value, *args):
        """Return energy for a given function value of the spectral model.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Function value of the spectral model.
        """
        amplitude = self.amplitude.quantity
        index = self.index.value
        energy_min = self.emin.quantity
        energy_max = self.emax.quantity

        # to get the energies dimensionless we use a modified formula
        top = -index + 1
        bottom = energy_max - energy_min * (energy_min / energy_max) ** (-index)
        term = (bottom / top) * (value / amplitude)
        return np.power(term.to_value(""), -1.0 / index) * energy_max


class BrokenPowerLawSpectralModel(SpectralModel):
    r"""Spectral broken power-law model.

    For more information see :ref:`broken-powerlaw-spectral-model`.

    Parameters
    ----------
    index1 : `~astropy.units.Quantity`
        :math:`\Gamma1`
    index2 : `~astropy.units.Quantity`
        :math:`\Gamma2`
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    ebreak : `~astropy.units.Quantity`
        :math:`E_{break}`

    See Also
    --------
    SmoothBrokenPowerLawSpectralModel
    """

    tag = ["BrokenPowerLawSpectralModel", "bpl"]
    index1 = Parameter("index1", 2.0)
    index2 = Parameter("index2", 2.0)
    amplitude = Parameter(
        name="amplitude",
        value="1e-12 cm-2 s-1 TeV-1",
        scale_method="scale10",
        interp="log",
        is_norm=True,
    )
    ebreak = Parameter("ebreak", "1 TeV")

    @staticmethod
    def evaluate(energy, index1, index2, amplitude, ebreak):
        """Evaluate the model (static function)."""
        energy = np.atleast_1d(energy)
        cond = energy < ebreak
        bpwl = amplitude * np.ones(energy.shape)
        bpwl[cond] *= (energy[cond] / ebreak) ** (-index1)
        bpwl[~cond] *= (energy[~cond] / ebreak) ** (-index2)
        return bpwl


class SmoothBrokenPowerLawSpectralModel(SpectralModel):
    r"""Spectral smooth broken power-law model.

    For more information see :ref:`smooth-broken-powerlaw-spectral-model`.

    Parameters
    ----------
    index1 : `~astropy.units.Quantity`
        :math:`\Gamma1`
    index2 : `~astropy.units.Quantity`
        :math:`\Gamma2`
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    ebreak : `~astropy.units.Quantity`
        :math:`E_{break}`
    beta : `~astropy.units.Quantity`
        :math:`\beta`

    See Also
    --------
    BrokenPowerLawSpectralModel
    """

    tag = ["SmoothBrokenPowerLawSpectralModel", "sbpl"]
    index1 = Parameter("index1", 2.0)
    index2 = Parameter("index2", 2.0)
    amplitude = Parameter(
        name="amplitude",
        value="1e-12 cm-2 s-1 TeV-1",
        scale_method="scale10",
        interp="log",
        is_norm=True,
    )
    ebreak = Parameter("ebreak", "1 TeV")
    reference = Parameter("reference", "1 TeV", frozen=True)
    beta = Parameter("beta", 1, frozen=True)

    @staticmethod
    def evaluate(energy, index1, index2, amplitude, ebreak, reference, beta):
        """Evaluate the model (static function)."""
        beta *= np.sign(index2 - index1)
        pwl = amplitude * (energy / reference) ** (-index1)
        brk = (1 + (energy / ebreak) ** ((index2 - index1) / beta)) ** (-beta)
        return pwl * brk


class PiecewiseNormSpectralModel(SpectralModel):
    """Piecewise spectral correction
       with a free normalization at each fixed energy nodes.

       For more information see :ref:`piecewise-norm-spectral`.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Array of energies at which the model values are given (nodes).
    norms : `~numpy.ndarray` or list of `Parameter`
        Array with the initial norms of the model at energies ``energy``.
        A normalisation parameters is created for each value.
        Default is one at each node.
    interp : str
        Interpolation scaling in {"log", "lin"}. Default is "log"
    """

    tag = ["PiecewiseNormSpectralModel", "piecewise-norm"]

    def __init__(self, energy, norms=None, interp="log"):
        self._energy = energy
        self._interp = interp

        if norms is None:
            norms = np.ones(len(energy))

        if len(norms) != len(energy):
            raise ValueError("dimension mismatch")

        if len(norms) < 2:
            raise ValueError("Input arrays must contain at least 2 elements")

        if not isinstance(norms[0], Parameter):
            parameters = Parameters(
                [Parameter(f"norm_{k}", norm) for k, norm in enumerate(norms)]
            )
        else:
            parameters = Parameters(norms)

        self.default_parameters = parameters
        super().__init__()

    @property
    def energy(self):
        """Energy nodes"""
        return self._energy

    @property
    def norms(self):
        """Norm values"""
        return u.Quantity(self.parameters.value)

    def evaluate(self, energy, **norms):
        scale = interpolation_scale(scale=self._interp)
        e_eval = scale(np.atleast_1d(energy.value))
        e_nodes = scale(self.energy.to(energy.unit).value)
        v_nodes = scale(self.norms)
        log_interp = scale.inverse(np.interp(e_eval, e_nodes, v_nodes))
        return log_interp

    def to_dict(self, full_output=False):
        data = super().to_dict(full_output=full_output)
        data["spectral"]["energy"] = {
            "data": self.energy.data.tolist(),
            "unit": str(self.energy.unit),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Create model from dict"""
        data = data["spectral"]
        energy = u.Quantity(data["energy"]["data"], data["energy"]["unit"])
        parameters = Parameters.from_dict(data["parameters"])
        return cls.from_parameters(parameters, energy=energy)

    @classmethod
    def from_parameters(cls, parameters, **kwargs):
        """Create model from parameters"""
        return cls(norms=parameters, **kwargs)


class ExpCutoffPowerLawSpectralModel(SpectralModel):
    r"""Spectral exponential cutoff power-law model.

    For more information see :ref:`exp-cutoff-powerlaw-spectral-model`.

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
    alpha : `~astropy.units.Quantity`
        :math:`\alpha`

    See Also
    --------
    ExpCutoffPowerLawNormSpectralModel
    """

    tag = ["ExpCutoffPowerLawSpectralModel", "ecpl"]

    index = Parameter("index", 1.5)
    amplitude = Parameter(
        name="amplitude",
        value="1e-12 cm-2 s-1 TeV-1",
        scale_method="scale10",
        interp="log",
        is_norm=True,
    )
    reference = Parameter("reference", "1 TeV", frozen=True)
    lambda_ = Parameter("lambda_", "0.1 TeV-1")
    alpha = Parameter("alpha", "1.0", frozen=True)

    @staticmethod
    def evaluate(energy, index, amplitude, reference, lambda_, alpha):
        """Evaluate the model (static function)."""
        pwl = amplitude * (energy / reference) ** (-index)
        cutoff = np.exp(-np.power(energy * lambda_, alpha))

        return pwl * cutoff

    @property
    def e_peak(self):
        r"""Spectral energy distribution peak energy (`~astropy.units.Quantity`).

        This is the peak in E^2 x dN/dE and is given by:

        .. math::
            E_{Peak} =  \left(\frac{2 - \Gamma}{\alpha}\right)^{1/\alpha} / \lambda
        """
        reference = self.reference.quantity
        index = self.index.quantity
        lambda_ = self.lambda_.quantity
        alpha = self.alpha.quantity

        if index >= 2 or lambda_ == 0.0 or alpha == 0.0:
            return np.nan * reference.unit
        else:
            return np.power((2 - index) / alpha, 1 / alpha) / lambda_


class ExpCutoffPowerLawNormSpectralModel(SpectralModel):
    r"""Norm spectral exponential cutoff power-law model.

    Parameters
    ----------
    index : `~astropy.units.Quantity`
        :math:`\Gamma`
    norm : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    lambda_ : `~astropy.units.Quantity`
        :math:`\lambda`
    alpha : `~astropy.units.Quantity`
        :math:`\alpha`

    See Also
    --------
    ExpCutoffPowerLawSpectralModel
    """
    tag = ["ExpCutoffPowerLawNormSpectralModel", "ecpl-norm"]

    index = Parameter("index", 1.5)
    norm = Parameter("norm", 1, unit="", interp="log", is_norm=True)
    reference = Parameter("reference", "1 TeV", frozen=True)
    lambda_ = Parameter("lambda_", "0.1 TeV-1")
    alpha = Parameter("alpha", "1.0", frozen=True)

    @staticmethod
    def evaluate(energy, index, norm, reference, lambda_, alpha):
        """Evaluate the model (static function)."""
        pwl = norm * (energy / reference) ** (-index)
        cutoff = np.exp(-np.power(energy * lambda_, alpha))

        return pwl * cutoff


class ExpCutoffPowerLaw3FGLSpectralModel(SpectralModel):
    r"""Spectral exponential cutoff power-law model used for 3FGL.

    For more information see :ref:`exp-cutoff-powerlaw-3fgl-spectral-model`.

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

    tag = ["ExpCutoffPowerLaw3FGLSpectralModel", "ecpl-3fgl"]
    index = Parameter("index", 1.5)
    amplitude = Parameter(
        "amplitude",
        "1e-12 cm-2 s-1 TeV-1",
        scale_method="scale10",
        interp="log",
        is_norm=True,
    )
    reference = Parameter("reference", "1 TeV", frozen=True)
    ecut = Parameter("ecut", "10 TeV")

    @staticmethod
    def evaluate(energy, index, amplitude, reference, ecut):
        """Evaluate the model (static function)."""
        pwl = amplitude * (energy / reference) ** (-index)
        cutoff = np.exp((reference - energy) / ecut)
        return pwl * cutoff


class SuperExpCutoffPowerLaw3FGLSpectralModel(SpectralModel):
    r"""Spectral super exponential cutoff power-law model used for 3FGL.

    For more information see :ref:`super-exp-cutoff-powerlaw-3fgl-spectral-model`.

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

    tag = ["SuperExpCutoffPowerLaw3FGLSpectralModel", "secpl-3fgl"]
    amplitude = Parameter(
        "amplitude",
        "1e-12 cm-2 s-1 TeV-1",
        scale_method="scale10",
        interp="log",
        is_norm=True,
    )
    reference = Parameter("reference", "1 TeV", frozen=True)
    ecut = Parameter("ecut", "10 TeV")
    index_1 = Parameter("index_1", 1.5)
    index_2 = Parameter("index_2", 2)

    @staticmethod
    def evaluate(energy, amplitude, reference, ecut, index_1, index_2):
        """Evaluate the model (static function)."""
        pwl = amplitude * (energy / reference) ** (-index_1)
        cutoff = np.exp((reference / ecut) ** index_2 - (energy / ecut) ** index_2)
        return pwl * cutoff


class SuperExpCutoffPowerLaw4FGLSpectralModel(SpectralModel):
    r"""Spectral super exponential cutoff power-law model used for 4FGL-DR1 (and DR2).
    See equation (4) of https://arxiv.org/pdf/1902.10045.pdf
    For more information see :ref:`super-exp-cutoff-powerlaw-4fgl-spectral-model`.

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
        internally assumes unit of :math: `[E_0]` power :math:`-\Gamma_2`
    """

    tag = ["SuperExpCutoffPowerLaw4FGLSpectralModel", "secpl-4fgl"]
    amplitude = Parameter(
        "amplitude",
        "1e-12 cm-2 s-1 TeV-1",
        scale_method="scale10",
        interp="log",
        is_norm=True,
    )
    reference = Parameter("reference", "1 TeV", frozen=True)
    expfactor = Parameter("expfactor", "1e-2")
    index_1 = Parameter("index_1", 1.5)
    index_2 = Parameter("index_2", 2)

    @staticmethod
    def evaluate(energy, amplitude, reference, expfactor, index_1, index_2):
        """Evaluate the model (static function)."""
        pwl = amplitude * (energy / reference) ** (-index_1)
        cutoff = np.exp(
            expfactor
            / reference.unit**index_2
            * (reference**index_2 - energy**index_2)
        )
        return pwl * cutoff


class SuperExpCutoffPowerLaw4FGLDR3SpectralModel(SpectralModel):
    r"""Spectral super exponential cutoff power-law model used for 4FGL-DR3.
    See equations (2) and (3) of https://arxiv.org/pdf/2201.11184.pdf
    For more information see :ref:`super-exp-cutoff-powerlaw-4fgl-dr3-spectral-model`.

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
        :math:`a`, given as dimensionless value
    """

    tag = ["SuperExpCutoffPowerLaw4FGLDR3SpectralModel", "secpl-4fgl-dr3"]
    amplitude = Parameter(
        name="amplitude",
        value="1e-12 cm-2 s-1 TeV-1",
        scale_method="scale10",
        interp="log",
        is_norm=True,
    )
    reference = Parameter("reference", "1 TeV", frozen=True)
    expfactor = Parameter("expfactor", "1e-2")
    index_1 = Parameter("index_1", 1.5)
    index_2 = Parameter("index_2", 2)

    @staticmethod
    def evaluate(energy, amplitude, reference, expfactor, index_1, index_2):
        """Evaluate the model (static function)."""
        # https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html#PLSuperExpCutoff4
        pwl = amplitude * (energy / reference) ** (-index_1)
        cutoff = (energy / reference) ** (expfactor / index_2) * np.exp(
            expfactor / index_2**2 * (1 - (energy / reference) ** index_2)
        )

        mask = np.abs(index_2 * np.log(energy / reference)) < 1e-2
        ln_ = np.log(energy[mask] / reference)
        power = expfactor * (
            ln_ / 2.0 + index_2 / 6.0 * ln_**2.0 + index_2**2.0 / 24.0 * ln_**3
        )
        cutoff[mask] = (energy[mask] / reference) ** power
        return pwl * cutoff


class LogParabolaSpectralModel(SpectralModel):
    r"""Spectral log parabola model.

    For more information see :ref:`logparabola-spectral-model`.

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

    See Also
    --------
    LogParabolaNormSpectralModel

    """
    tag = ["LogParabolaSpectralModel", "lp"]
    amplitude = Parameter(
        "amplitude",
        "1e-12 cm-2 s-1 TeV-1",
        scale_method="scale10",
        interp="log",
        is_norm=True,
    )
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
        xx = energy / reference
        exponent = -alpha - beta * np.log(xx)
        return amplitude * np.power(xx, exponent)

    @property
    def e_peak(self):
        r"""Spectral energy distribution peak energy (`~astropy.units.Quantity`).

        This is the peak in E^2 x dN/dE and is given by:

        .. math::
            E_{Peak} = E_{0} \exp{ (2 - \alpha) / (2 * \beta)}
        """
        reference = self.reference.quantity
        alpha = self.alpha.quantity
        beta = self.beta.quantity
        return reference * np.exp((2 - alpha) / (2 * beta))


class LogParabolaNormSpectralModel(SpectralModel):
    r"""Norm spectral log parabola model.

    Parameters
    ----------
    norm : `~astropy.units.Quantity`
        :math:`\phi_0`
    reference : `~astropy.units.Quantity`
        :math:`E_0`
    alpha : `~astropy.units.Quantity`
        :math:`\alpha`
    beta : `~astropy.units.Quantity`
        :math:`\beta`

    See Also
    --------
    LogParabolaSpectralModel
    """
    tag = ["LogParabolaNormSpectralModel", "lp-norm"]
    norm = Parameter("norm", 1, unit="", interp="log", is_norm=True)
    reference = Parameter("reference", "10 TeV", frozen=True)
    alpha = Parameter("alpha", 2)
    beta = Parameter("beta", 1)

    @classmethod
    def from_log10(cls, norm, reference, alpha, beta):
        """Construct from :math:`log_{10}` parametrization."""
        beta_ = beta / np.log(10)
        return cls(norm=norm, reference=reference, alpha=alpha, beta=beta_)

    @staticmethod
    def evaluate(energy, norm, reference, alpha, beta):
        """Evaluate the model (static function)."""
        xx = energy / reference
        exponent = -alpha - beta * np.log(xx)
        return norm * np.power(xx, exponent)


class TemplateSpectralModel(SpectralModel):
    """A model generated from a table of energy and value arrays.

    For more information see :ref:`template-spectral-model`.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Array of energies at which the model values are given
    values : array
        Array with the values of the model at energies ``energy``.
    interp_kwargs : dict
        Interpolation keyword arguments passed to `scipy.interpolate.RegularGridInterpolator`.
        By default all values outside the interpolation range are set to zero.
        If you want to apply linear extrapolation you can pass `interp_kwargs={'fill_value':
        'extrapolate', 'kind': 'linear'}`. If you want to choose the interpolation
        scaling applied to values, you can use `interp_kwargs={"values_scale": "log"}`.
    meta : dict, optional
        Meta information, meta['filename'] will be used for serialization
    """

    norm = Parameter("norm", 1, unit="", interp="log", is_norm=True, frozen=True)
    tag = ["TemplateSpectralModel", "template"]

    def __init__(
        self,
        energy,
        values,
        norm=1.0,
        interp_kwargs=None,
        meta=None,
    ):
        self.energy = energy
        self.values = u.Quantity(values, copy=False)
        self.meta = {} if meta is None else meta
        interp_kwargs = interp_kwargs or {}
        interp_kwargs.setdefault("values_scale", "log")
        interp_kwargs.setdefault("points_scale", ("log",))

        if len(energy) == 1:
            interp_kwargs["method"] = "nearest"

        self._evaluate = ScaledRegularGridInterpolator(
            points=(energy,), values=values, **interp_kwargs
        )

        super().__init__(norm=norm)

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

    def evaluate(self, energy, norm):
        """Evaluate the model (static function)."""
        return norm * self._evaluate((energy,), clip=True)

    def to_dict(self, full_output=False):
        data = super().to_dict(full_output)
        data["spectral"]["energy"] = {
            "data": self.energy.data.tolist(),
            "unit": str(self.energy.unit),
        }
        data["spectral"]["values"] = {
            "data": self.values.data.tolist(),
            "unit": str(self.values.unit),
        }

        return data

    @classmethod
    def from_dict(cls, data):
        data = data["spectral"]
        energy = u.Quantity(data["energy"]["data"], data["energy"]["unit"])
        values = u.Quantity(data["values"]["data"], data["values"]["unit"])
        norm = [p["value"] for p in data["parameters"] if p["name"] == "norm"][0]
        return cls(energy=energy, values=values, norm=norm)

    @classmethod
    def from_region_map(cls, map, **kwargs):
        """Create model from region map"""
        energy = map.geom.axes["energy_true"].center
        values = map.quantity[:, 0, 0]
        return cls(energy=energy, values=values, **kwargs)


class TemplateNDSpectralModel(SpectralModel):
    """A model generated from a ND map where extra dimensions define the parameter space.

    Parameters
    ----------
    map : `~gammapy.maps.RegionNDMap`
        Map template.
    meta : dict, optional
        Meta information, meta['filename'] will be used for serialization
    interp_kwargs : dict
        Interpolation keyword arguments passed to `gammapy.maps.Map.interp_by_pix`.
        Default arguments are {'method': 'linear', 'fill_value': 0}.
    """

    tag = ["TemplateNDSpectralModel", "templateND"]

    def __init__(self, map, interp_kwargs=None, meta=None, filename=None):
        self._map = map.copy()
        self.meta = dict() if meta is None else meta
        if filename is not None:
            filename = str(make_path(filename))
        self.filename = filename

        points_scale = [
            "lin",
            "lin",
            "log",
        ]
        parameters = []
        has_energy = False
        for axis in map.geom.axes:
            if axis.name not in ["energy_true", "energy"]:
                center = (axis.bounds[1] + axis.bounds[0]) / 2
                parameter = Parameter(
                    name=axis.name,
                    value=center,
                    unit=axis.unit,
                    scale_method="scale10",
                    min=axis.bounds[0],
                    max=axis.bounds[-1],
                    interp=axis.interp,
                )
                points_scale.append(axis.interp)
                parameters.append(parameter)
            else:
                has_energy |= True
        if not has_energy:
            raise ValueError("Invalid map, no energy axis found")

        self.default_parameters = Parameters(parameters)

        interp_kwargs = interp_kwargs or {}
        interp_kwargs.setdefault("values_scale", "log")
        interp_kwargs.setdefault("points_scale", points_scale)
        self._interp_kwargs = interp_kwargs
        super().__init__()

    @property
    def map(self):
        """Template map  (`~gammapy.maps.RegionNDMap`)"""
        return self._map

    def evaluate(self, energy, **kwargs):
        coord = {"energy_true": energy}
        coord.update(kwargs)

        pixels = [0, 0] + [
            self.map.geom.axes[key].coord_to_pix(value) for key, value in coord.items()
        ]

        val = self.map.interp_by_pix(pixels, **self._interp_kwargs)
        return u.Quantity(val, self.map.unit, copy=False)

    def write(self, overwrite=False):
        if self.filename is None:
            raise IOError("Missing filename")
        elif os.path.isfile(self.filename) and not overwrite:
            log.warning("Template file already exits, and overwrite is False")
        else:
            self.map.write(self.filename)

    @classmethod
    def from_dict(cls, data):
        data = data["spectral"]
        filename = data["filename"]
        m = RegionNDMap.read(filename)
        model = cls(m, filename=filename)
        for idx, p in enumerate(model.parameters):
            par = p.to_dict()
            par.update(data["parameters"][idx])
            setattr(model, p.name, Parameter(**par))
        return model

    def to_dict(self, full_output=False):
        """Create dict for YAML serilisation"""
        data = super().to_dict(full_output)
        data["spectral"]["filename"] = self.filename
        data["spectral"]["unit"] = str(self.map.unit)
        return data


class ScaleSpectralModel(SpectralModel):
    """Wrapper to scale another spectral model by a norm factor.

    Parameters
    ----------
    model : `SpectralModel`
        Spectral model to wrap.
    norm : float
        Multiplicative norm factor for the model value.
    """

    tag = ["ScaleSpectralModel", "scale"]
    norm = Parameter("norm", 1, unit="", interp="log", is_norm=True)

    def __init__(self, model, norm=norm.quantity):
        self.model = model
        self._covariance = None
        super().__init__(norm=norm)

    def evaluate(self, energy, norm):
        return norm * self.model(energy)

    def integral(self, energy_min, energy_max, **kwargs):
        return self.norm.value * self.model.integral(energy_min, energy_max, **kwargs)


class EBLAbsorptionNormSpectralModel(SpectralModel):
    r"""Gamma-ray absorption models.

    For more information see :ref:`absorption-spectral-model`.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy node values
    param : `~astropy.units.Quantity`
        Parameter node values
    data : `~astropy.units.Quantity`
        Model value
    redshift : float
        Redshift of the absorption model
    alpha_norm: float
        Norm of the EBL model
    interp_kwargs : dict
        Interpolation option passed to `ScaledRegularGridInterpolator`.
        By default the models are extrapolated outside the range. To prevent
        this and raise an error instead use interp_kwargs = {"extrapolate": False}
    """

    tag = ["EBLAbsorptionNormSpectralModel", "ebl-norm"]
    alpha_norm = Parameter("alpha_norm", 1.0, frozen=True)
    redshift = Parameter("redshift", 0.1, frozen=True)

    def __init__(self, energy, param, data, redshift, alpha_norm, interp_kwargs=None):
        self.filename = None
        # set values log centers
        self.param = param
        self.energy = energy
        self.data = u.Quantity(data, copy=False)

        interp_kwargs = interp_kwargs or {}
        interp_kwargs.setdefault("points_scale", ("lin", "log"))
        interp_kwargs.setdefault("values_scale", "log")
        interp_kwargs.setdefault("extrapolate", True)

        self._evaluate_table_model = ScaledRegularGridInterpolator(
            points=(self.param, self.energy), values=self.data, **interp_kwargs
        )
        super().__init__(redshift=redshift, alpha_norm=alpha_norm)

    def to_dict(self, full_output=False):
        data = super().to_dict(full_output=full_output)
        if self.filename is None:
            data["spectral"]["energy"] = {
                "data": self.energy.data.tolist(),
                "unit": str(self.energy.unit),
            }
            data["spectral"]["param"] = {
                "data": self.param.data.tolist(),
                "unit": str(self.param.unit),
            }
            data["spectral"]["values"] = {
                "data": self.data.data.tolist(),
                "unit": str(self.data.unit),
            }
        else:
            data["spectral"]["filename"] = str(self.filename)
        return data

    @classmethod
    def from_dict(cls, data):
        data = data["spectral"]
        redshift = [p["value"] for p in data["parameters"] if p["name"] == "redshift"][
            0
        ]
        alpha_norm = [
            p["value"] for p in data["parameters"] if p["name"] == "alpha_norm"
        ][0]
        if "filename" in data:
            return cls.read(data["filename"], redshift=redshift, alpha_norm=alpha_norm)
        else:
            energy = u.Quantity(data["energy"]["data"], data["energy"]["unit"])
            param = u.Quantity(data["param"]["data"], data["param"]["unit"])
            values = u.Quantity(data["values"]["data"], data["values"]["unit"])
            return cls(
                energy=energy,
                param=param,
                data=values,
                redshift=redshift,
                alpha_norm=alpha_norm,
            )

    @classmethod
    def read(cls, filename, redshift=0.1, alpha_norm=1, interp_kwargs=None):
        """Build object from an XSPEC model.

        Parameters
        ----------

        filename : str
            File containing the model.
        redshift : float
            Redshift of the absorption model
        alpha_norm: float
            Norm of the EBL model
        interp_kwargs : dict
            Interpolation option passed to `ScaledRegularGridInterpolator`.

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
        model = cls(
            energy=energy,
            param=param,
            data=data,
            redshift=redshift,
            alpha_norm=alpha_norm,
            interp_kwargs=interp_kwargs,
        )
        model.filename = filename
        return model

    @classmethod
    def read_builtin(
        cls, reference="dominguez", redshift=0.1, alpha_norm=1, interp_kwargs=None
    ):
        """Read  from one of the built-in absorption models.

        Parameters
        ----------
        reference : {'franceschini', 'dominguez', 'finke'}
            name of one of the available model in gammapy-data
        redshift : float
            Redshift of the absorption model
        alpha_norm: float
            Norm of the EBL model

        References
        ----------
        .. [1] Franceschini et al., "Extragalactic optical-infrared background radiation, its time evolution and the cosmic photon-photon opacity",  # noqa: E501
            `Link <https://ui.adsabs.harvard.edu/abs/2008A%26A...487..837F>`__
        .. [2] Dominguez et al., " Extragalactic background light inferred from AEGIS galaxy-SED-type fractions"  # noqa: E501
            `Link <https://ui.adsabs.harvard.edu/abs/2011MNRAS.410.2556D>`__
        .. [3] Finke et al., "Modeling the Extragalactic Background Light from Stars and Dust"
            `Link <https://ui.adsabs.harvard.edu/abs/2010ApJ...712..238F>`__

        """
        models = {}
        models["franceschini"] = "$GAMMAPY_DATA/ebl/ebl_franceschini.fits.gz"
        models["dominguez"] = "$GAMMAPY_DATA/ebl/ebl_dominguez11.fits.gz"
        models["finke"] = "$GAMMAPY_DATA/ebl/frd_abs.fits.gz"

        return cls.read(
            models[reference], redshift, alpha_norm, interp_kwargs=interp_kwargs
        )

    def evaluate(self, energy, redshift, alpha_norm):
        """Evaluate model for energy and parameter value."""
        absorption = np.clip(self._evaluate_table_model((redshift, energy)), 0, 1)
        return np.power(absorption, alpha_norm)


class NaimaSpectralModel(SpectralModel):
    r"""A wrapper for Naima models.

    For more information see :ref:`naima-spectral-model`.

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
    nested_models : dict
        Additional parameters for nested models not supplied by the radiative model,
        for now this is used  only for synchrotron self-compton model
    """

    tag = ["NaimaSpectralModel", "naima"]

    def __init__(
        self,
        radiative_model,
        distance=1.0 * u.kpc,
        seed=None,
        nested_models=None,
        use_cache=False,
    ):
        import naima

        self.radiative_model = radiative_model
        self.radiative_model._memoize = use_cache
        self.distance = u.Quantity(distance)
        self.seed = seed

        if nested_models is None:
            nested_models = {}

        self.nested_models = nested_models

        if isinstance(self.particle_distribution, naima.models.TableModel):
            param_names = ["amplitude"]
        else:
            param_names = self.particle_distribution.param_names

        parameters = []

        for name in param_names:
            value = getattr(self.particle_distribution, name)
            is_norm = name == "amplitude"
            parameter = Parameter(name, value, is_norm=is_norm)
            parameters.append(parameter)

        # In case of a synchrotron radiative model, append B to the fittable parameters
        if "B" in self.radiative_model.param_names:
            value = getattr(self.radiative_model, "B")
            parameter = Parameter("B", value)
            parameters.append(parameter)

        # In case of a synchrotron self compton model, append B and Rpwn to the fittable parameters
        if self.include_ssc:
            B = self.nested_models["SSC"]["B"]
            radius = self.nested_models["SSC"]["radius"]
            parameters.append(Parameter("B", B))
            parameters.append(Parameter("radius", radius, frozen=True))

        self.default_parameters = Parameters(parameters)
        self.ssc_energy = np.logspace(-7, 9, 100) * u.eV
        super().__init__()

    @property
    def include_ssc(self):
        """Whether the model includes an SSC component"""
        import naima

        is_ic_model = isinstance(self.radiative_model, naima.models.InverseCompton)
        return is_ic_model and "SSC" in self.nested_models

    @property
    def ssc_model(self):
        """Synchrotron model"""
        import naima

        if self.include_ssc:
            return naima.models.Synchrotron(
                self.particle_distribution,
                B=self.B.quantity,
                Eemax=self.radiative_model.Eemax,
                Eemin=self.radiative_model.Eemin,
            )

    @property
    def particle_distribution(self):
        """Particle distribution"""
        return self.radiative_model.particle_distribution

    def _evaluate_ssc(
        self,
        energy,
    ):
        """
        Compute photon density spectrum from synchrotron emission for synchrotron self-compton
        model, assuming uniform synchrotron emissivity inside a sphere of radius R (see Section
        4.1 of Atoyan & Aharonian 1996)

        Based on :

        https://naima.readthedocs.io/en/latest/examples.html#crab-nebula-ssc-model

        """
        Lsy = self.ssc_model.flux(
            self.ssc_energy, distance=0 * u.cm
        )  # use distance 0 to get luminosity
        phn_sy = Lsy / (4 * np.pi * self.radius.quantity**2 * const.c) * 2.24
        # The factor 2.24 comes from the assumption on uniform synchrotron
        # emissivity inside a sphere

        if "SSC" not in self.radiative_model.seed_photon_fields:
            self.radiative_model.seed_photon_fields["SSC"] = {
                "isotropic": True,
                "type": "array",
                "energy": self.ssc_energy,
                "photon_density": phn_sy,
            }
        else:
            self.radiative_model.seed_photon_fields["SSC"]["photon_density"] = phn_sy

        dnde = self.radiative_model.flux(
            energy, seed=self.seed, distance=self.distance
        ) + self.ssc_model.flux(energy, distance=self.distance)
        return dnde

    def _update_naima_parameters(self, **kwargs):
        """Update Naima model parameters"""
        for name, value in kwargs.items():
            setattr(self.particle_distribution, name, value)

        if "B" in self.radiative_model.param_names:
            self.radiative_model.B = self.B.quantity

    def evaluate(self, energy, **kwargs):
        """Evaluate the model.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy to evaluate the model at.

        Returns
        -------
        dnde : `~astropy.units.Quantity`
            Differential flux at given energy.
        """
        self._update_naima_parameters(**kwargs)

        if self.include_ssc:
            dnde = self._evaluate_ssc(energy.flatten())
        elif self.seed is not None:
            dnde = self.radiative_model.flux(
                energy.flatten(), seed=self.seed, distance=self.distance
            )
        else:
            dnde = self.radiative_model.flux(energy.flatten(), distance=self.distance)

        dnde = dnde.reshape(energy.shape)
        unit = 1 / (energy.unit * u.cm**2 * u.s)
        return dnde.to(unit)

    def to_dict(self, full_output=True):
        # for full_output to True otherwise broken
        return super().to_dict(full_output=True)

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError(
            "Currently the NaimaSpectralModel cannot be read from YAML"
        )

    @classmethod
    def from_parameters(cls, parameters, **kwargs):
        raise NotImplementedError(
            "Currently the NaimaSpectralModel cannot be built from a list of parameters."
        )


class GaussianSpectralModel(SpectralModel):
    r"""Gaussian spectral model.

    For more information see :ref:`gaussian-spectral-model`.

    Parameters
    ----------
    amplitude : `~astropy.units.Quantity`
        :math:`N_0`
    mean : `~astropy.units.Quantity`
        :math:`\bar{E}`
    sigma : `~astropy.units.Quantity`
        :math:`\sigma`
    """

    tag = ["GaussianSpectralModel", "gauss"]
    amplitude = Parameter(
        "amplitude", 1e-12 * u.Unit("cm-2 s-1"), interp="log", is_norm=True
    )
    mean = Parameter("mean", 1 * u.TeV)
    sigma = Parameter("sigma", 2 * u.TeV)

    @staticmethod
    def evaluate(energy, amplitude, mean, sigma):
        return (
            amplitude
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-((energy - mean) ** 2) / (2 * sigma**2))
        )

    def integral(self, energy_min, energy_max, **kwargs):
        r"""Integrate Gaussian analytically.

        .. math::
            F(E_{min}, E_{max}) = \frac{N_0}{2} \left[ erf(\frac{E - \bar{E}}{\sqrt{2} \sigma})\right]_{E_{min}}^{E_{max}}  # noqa: E501

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Lower and upper bound of integration range
        """
        # kwargs are passed to this function but not used
        # this is to get a consistent API with SpectralModel.integral()
        u_min = (
            (energy_min - self.mean.quantity) / (np.sqrt(2) * self.sigma.quantity)
        ).to_value("")
        u_max = (
            (energy_max - self.mean.quantity) / (np.sqrt(2) * self.sigma.quantity)
        ).to_value("")

        return (
            self.amplitude.quantity
            / 2
            * (scipy.special.erf(u_max) - scipy.special.erf(u_min))
        )

    def energy_flux(self, energy_min, energy_max):
        r"""Compute energy flux in given energy range analytically.

        .. math::
            G(E_{min}, E_{max}) =  \frac{N_0 \sigma}{\sqrt{2*\pi}}* \left[ - \exp(\frac{E_{min}-\bar{E}}{\sqrt{2} \sigma})   # noqa: E501
            \right]_{E_{min}}^{E_{max}} + \frac{N_0 * \bar{E}}{2} \left[ erf(\frac{E - \bar{E}}{\sqrt{2} \sigma})   # noqa: E501
             \right]_{E_{min}}^{E_{max}}


        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        """
        u_min = (
            (energy_min - self.mean.quantity) / (np.sqrt(2) * self.sigma.quantity)
        ).to_value("")
        u_max = (
            (energy_max - self.mean.quantity) / (np.sqrt(2) * self.sigma.quantity)
        ).to_value("")
        a = self.amplitude.quantity * self.sigma.quantity / np.sqrt(2 * np.pi)
        b = self.amplitude.quantity * self.mean.quantity / 2
        return a * (np.exp(-(u_min**2)) - np.exp(-(u_max**2))) + b * (
            scipy.special.erf(u_max) - scipy.special.erf(u_min)
        )
