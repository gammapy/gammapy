# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy."""
import operator
import numpy as np
import scipy.optimize
import scipy.special
import astropy.units as u
from astropy import constants as const
from astropy.table import Table
from gammapy.maps import MapAxis
from gammapy.maps.utils import edges_from_lo_hi
from gammapy.modeling import Covariance, Parameter, Parameters
from gammapy.utils.integrate import evaluate_integral_pwl, trapz_loglog
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from gammapy.utils.scripts import make_path
from .core import Model


def integrate_spectrum(func, emin, emax, ndecade=100, intervals=False):
    """Integrate 1d function using the log-log trapezoidal rule.

    If scalar values for xmin and xmax are passed an oversampled grid is generated using the
    ``ndecade`` keyword argument. If xmin and xmax arrays are passed, no
    oversampling is performed and the integral is computed in the provided
    grid.

    Parameters
    ----------
    func : callable
        Function to integrate.
    emin : `~astropy.units.Quantity`
        Integration range minimum
    emax : `~astropy.units.Quantity`
        Integration range minimum
    ndecade : int, optional
        Number of grid points per decade used for the integration.
        Default : 100.
    intervals : bool, optional
        Return integrals in the grid not the sum, default: False
    """
    if emin.isscalar and emax.isscalar:
        energies = MapAxis.from_energy_bounds(
            emin=emin, emax=emax, nbin=ndecade, per_decade=True
        ).edges
    else:
        energies = edges_from_lo_hi(emin, emax)

    values = func(energies)

    integral = trapz_loglog(values, energies)

    if intervals:
        return integral

    return integral.sum()


class SpectralModel(Model):
    """Spectral model base class."""

    def __call__(self, energy):
        kwargs = {par.name: par.quantity for par in self.parameters}
        kwargs = self._convert_evaluate_unit(kwargs, energy)
        return self.evaluate(energy, **kwargs)

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

    def __radd__(self, model):
        return self.__add__(model)

    def __sub__(self, model):
        if not isinstance(model, SpectralModel):
            model = ConstantSpectralModel(const=model)
        return CompoundSpectralModel(self, model, operator.sub)

    def __rsub__(self, model):
        return self.__sub__(model)

    def _evaluate_gradient(self, energy, eps):
        n = len(self.parameters)
        f = self(energy)
        shape = (n, len(np.atleast_1d(energy)))
        df_dp = np.zeros(shape)

        for idx, parameter in enumerate(self.parameters):
            if parameter.frozen or eps[idx] == 0:
                continue

            parameter.value += eps[idx]
            df = self(energy) - f
            df_dp[idx] = df.value / eps[idx]

            # Reset model to original parameter
            parameter.value -= eps[idx]

        return df_dp

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
        p_cov = self.covariance
        eps = np.sqrt(np.diag(p_cov)) * epsilon

        df_dp = self._evaluate_gradient(energy, eps)
        f_cov = df_dp.T @ p_cov @ df_dp
        f_err = np.sqrt(np.diagonal(f_cov))

        q = self(energy)
        return u.Quantity([q.value, f_err], unit=q.unit)

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
        energy = MapAxis.from_energy_bounds(emin, emax, n_points, energy_unit).edges

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
        energy = MapAxis.from_energy_bounds(emin, emax, n_points, energy_unit).edges

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

    For more information see :ref:`constant-spectral-model`.

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

    For more information see :ref:`compound-spectral-model`.
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

    For more information see :ref:`powerlaw-spectral-model`.

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
    evaluate_integral = staticmethod(evaluate_integral_pwl)

    @staticmethod
    def evaluate(energy, index, amplitude, reference):
        """Evaluate the model (static function)."""
        return amplitude * np.power((energy / reference), -index)

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
        kwargs = {par.name: par.quantity for par in self.parameters}
        kwargs = self._convert_evaluate_unit(kwargs, emin)
        return self.evaluate_integral(emin=emin, emax=emax, **kwargs)

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
        kwargs = {par.name: par.quantity for par in self.parameters}
        kwargs = self._convert_evaluate_unit(kwargs, emin)
        return self.evaluate_energy_flux(emin=emin, emax=emax, **kwargs)

    def inverse(self, value):
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
        return reference * np.exp(cov_index_ampl / (amplitude * index_err ** 2))


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
        temp1 = np.power(emax, -self.index.value + 1)
        temp2 = np.power(emin, -self.index.value + 1)
        top = temp1 - temp2

        temp1 = np.power(self.emax.quantity, -self.index.value + 1)
        temp2 = np.power(self.emin.quantity, -self.index.value + 1)
        bottom = temp1 - temp2

        return self.amplitude.quantity * top / bottom

    def inverse(self, value):
        """Return energy for a given function value of the spectral model.

        Parameters
        ----------
        value : `~astropy.units.Quantity`
            Function value of the spectral model.
        """
        amplitude = self.amplitude.quantity
        index = self.index.value
        emin = self.emin.quantity
        emax = self.emax.quantity

        # to get the energies dimensionless we use a modified formula
        top = -index + 1
        bottom = emax - emin * (emin / emax) ** (-index)
        term = (bottom / top) * (value / amplitude)
        return np.power(term.to_value(""), -1.0 / index) * emax


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
    """

    tag = "BrokenPowerLawSpectralModel"
    index1 = Parameter("index1", 2.0)
    index2 = Parameter("index2", 2.0)
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1")
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
    """

    tag = "SmoothBrokenPowerLawSpectralModel"
    index1 = Parameter("index1", 2.0)
    index2 = Parameter("index2", 2.0)
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1")
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


class PiecewiseBrokenPowerLawSpectralModel(SpectralModel):
    """Piecewise broken power-law at fixed energy nodes.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Array of energies at which the model values are given (nodes).
    values : array
        Array with the initial values of the model at energies ``energy``.
        A normalisation parameters is created for each value.
    """

    tag = "PiecewiseBrokenPowerLawSpectralModel"

    def __init__(self, energy, values, parameters=None):
        self._energy = energy
        self.init_values = values
        if len(values) != len(energy):
            raise ValueError("dimension mismatch")
        if len(values) < 2:
            raise ValueError("Input arrays must contians at least 2 elements")
        if parameters is None:
            parameters = Parameters(
                [Parameter(f"norm{k}", 1.0) for k, _ in enumerate(values)]
            )
        for parameter in parameters:
            setattr(self, parameter.name, parameter)
        self.default_parameters = parameters

    @classmethod
    def from_parameters(cls, parameters, **kwargs):
        return cls(kwargs["energy"], kwargs["values"], parameters=parameters)

    @classmethod
    def from_template(cls, model, energy=None):
        """Create from TemplateSpectralModel.

        Parameters
        ----------
        model : `~gammapy.modeling.models.TemplateSpectralModel`
            Template evaluated to determine values at given `energy`
        energy : `~astropy.units.Quantity`
            Array of energies at which the model values are given (nodes).
            By default energy are set as model.energy.
        """

        if not isinstance(model, TemplateSpectralModel):
            raise TypeError("model must be a TemplateSpectralModel")
        if energy is None:
            energy = model.energy
        return cls(energy, model(energy))

    @property
    def values(self):
        return np.array([p.value for p in self.parameters]) * self.init_values

    @property
    def energy(self):
        return self._energy

    def __call__(self, energy):
        return self.evaluate(energy)

    def evaluate(self, energy):
        logedata = np.log10(np.atleast_1d(energy.value))
        loge = np.log10(self.energy.to(energy.unit).value)
        logv = np.log10(self.values.value)
        ne = len(loge)
        conds = (
            [(logedata < loge[1])]
            + [
                (logedata >= loge[k]) & (logedata < loge[k + 1])
                for k in range(1, ne - 2)
            ]
            + [(logedata >= loge[-2])]
        )
        a = (logv[1:] - logv[:-1]) / (loge[1:] - loge[:-1])
        b = logv[1:] - a * loge[1:]

        output = np.zeros(logedata.shape)
        for k in range(ne - 1):
            output[conds[k]] = 10 ** (a[k] * logedata[conds[k]] + b[k])
        return output * self.values.unit

    def to_dict(self):
        return {
            "type": self.tag,
            "parameters": self.parameters.to_dict(),
            "energy": {
                "data": self.energy.data.tolist(),
                "unit": str(self.energy.unit),
            },
            "values": {
                "data": self.init_values.data.tolist(),
                "unit": str(self.values.unit),
            },
        }

    @classmethod
    def from_dict(cls, data):
        energy = u.Quantity(data["energy"]["data"], data["energy"]["unit"])
        values = u.Quantity(data["values"]["data"], data["values"]["unit"])
        if "parameters" in data:
            parameters = Parameters.from_dict(data["parameters"])
            return cls.from_parameters(parameters, energy=energy, values=values)
        else:
            return cls(energy=energy, values=values)


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
    """

    tag = "ExpCutoffPowerLawSpectralModel"

    index = Parameter("index", 1.5)
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1")
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

    tag = "ExpCutoffPowerLaw3FGLSpectralModel"
    index = Parameter("index", 1.5)
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1")
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

    tag = "SuperExpCutoffPowerLaw3FGLSpectralModel"
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1")
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
    r"""Spectral super exponential cutoff power-law model used for 4FGL.

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
        internally assumes unit of :math:`[E_0]` power :math:`-\Gamma_2`
    """

    tag = "SuperExpCutoffPowerLaw4FGLSpectralModel"
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1")
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
            / reference.unit ** index_2
            * (reference ** index_2 - energy ** index_2)
        )
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


class TemplateSpectralModel(SpectralModel):
    """A model generated from a table of energy and value arrays.

    For more information see :ref:`template-spectral-model`.

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
        self.values = u.Quantity(values, copy=False)
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
            "parameters": self.parameters.to_dict(),
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
        parameters = Parameters.from_dict(data["parameters"])
        energy = u.Quantity(data["energy"]["data"], data["energy"]["unit"])
        values = u.Quantity(data["values"]["data"], data["values"]["unit"])
        return cls.from_parameters(parameters, energy=energy, values=values)


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
        self._covariance = None
        super().__init__(norm=norm)

    def evaluate(self, energy, norm):
        return norm * self.model(energy)

    def integral(self, emin, emax, **kwargs):
        return self.norm.value * self.model.integral(emin, emax, **kwargs)


class Absorption:
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
    filename : str
        Filename of the absorption model used for serialisation.
    interp_kwargs : dict
        Interpolation option passed to `ScaledRegularGridInterpolator`.
        By default the models are extrapolated outside the range. To prevent
        this and raise an error instead use interp_kwargs = {"extrapolate": False}
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
        interp_kwargs.setdefault("extrapolate", True)

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
    def read(cls, filename, interp_kwargs=None):
        """Build object from an XSPEC model.

        Todo: Format of XSPEC binary files should be referenced at https://gamma-astro-data-formats.readthedocs.io/en/latest/

        Parameters
        ----------
        filename : str
            File containing the model.
        interp_kwargs : dict
            Interpolation option passed to `ScaledRegularGridInterpolator`.

        Returns
        -------
        absorption : `Absorption`
            Absorption model.
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
        return cls(
            energy=energy,
            param=param,
            data=data,
            filename=filename,
            interp_kwargs=interp_kwargs,
        )

    @classmethod
    def read_builtin(cls, name, interp_kwargs=None):
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

        Returns
        -------
        absorption : `Absorption`
            Absorption model.

        """
        models = dict()
        models["franceschini"] = "$GAMMAPY_DATA/ebl/ebl_franceschini.fits.gz"
        models["dominguez"] = "$GAMMAPY_DATA/ebl/ebl_dominguez11.fits.gz"
        models["finke"] = "$GAMMAPY_DATA/ebl/frd_abs.fits.gz"

        return cls.read(models[name], interp_kwargs=interp_kwargs)

    def table_model(self, parameter):
        """Table model for a given parameter value.

        Parameters
        ----------
        parameter : float
            Parameter value.

        Returns
        -------
        template_model : `TemplateSpectralModel`
            Template spectral model.
        """
        energy = self.energy
        values = self.evaluate(energy=energy, parameter=parameter)
        return TemplateSpectralModel(
            energy=energy, values=values, interp_kwargs={"values_scale": "log"}
        )

    def evaluate(self, energy, parameter):
        """Evaluate model for energy and parameter value."""
        return np.clip(self._evaluate((parameter, energy)), 0, 1)


class AbsorbedSpectralModel(SpectralModel):
    r"""Spectral model with EBL absorption.

    For more information see :ref:`absorbed-spectral-model`.

    Parameters
    ----------
    spectral_model : `SpectralModel`
        Spectral model.
    absorption : `Absorption`
        Absorption model.
    redshift : float
        Redshift of the absorption model
    alpha_norm: float
        Norm of the EBL model
    """

    tag = "AbsorbedSpectralModel"
    alpha_norm = Parameter("alpha_norm", 1.0, frozen=True)
    redshift = Parameter("redshift", 0.1, frozen=True)

    def __init__(
        self, spectral_model, absorption, redshift, alpha_norm=alpha_norm.quantity,
    ):
        self.spectral_model = spectral_model
        self.absorption = absorption

        min_ = self.absorption.param.min()
        max_ = self.absorption.param.max()

        redshift = Parameter("redshift", redshift, frozen=True, min=min_, max=max_)
        super().__init__(redshift=redshift, alpha_norm=alpha_norm)

    def _check_covariance(self):
        if not self.parameters == self._covariance.parameters:
            self._covariance = Covariance(self.parameters)

    @property
    def covariance(self):
        self._check_covariance()
        self._covariance.set_subcovariance(self.spectral_model.covariance)
        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        self._check_covariance()
        self._covariance.data = covariance

        subcovar = self._covariance.get_subcovariance(
            self.spectral_model.covariance.parameters
        )
        self.spectral_model.covariance = subcovar

    @property
    def parameters(self):
        return (
            Parameters([self.redshift, self.alpha_norm])
            + self.spectral_model.parameters
        )

    def evaluate(self, energy, **kwargs):
        """Evaluate the model at a given energy."""
        # assign redshift value and remove it from dictionary
        # since it does not belong to the spectral model
        parameter = kwargs.pop("redshift")
        alpha_norm = kwargs.pop("alpha_norm")

        dnde = self.spectral_model.evaluate(energy=energy, **kwargs)
        absorption = self.absorption.evaluate(energy=energy, parameter=parameter)
        # Power rule: (e ^ a) ^ b = e ^ (a * b)
        absorption = np.power(absorption, alpha_norm)
        return dnde * absorption

    def to_dict(self):
        return {
            "type": self.tag,
            "base_model": self.spectral_model.to_dict(),
            "absorption": self.absorption.to_dict(),
            "absorption_parameter": {"name": "redshift", "value": self.redshift.value,},
            "parameters": Parameters([self.redshift, self.alpha_norm]).to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY

        model_class = SPECTRAL_MODEL_REGISTRY.get_cls(data["base_model"]["type"])

        model = cls(
            spectral_model=model_class.from_dict(data["base_model"]),
            absorption=Absorption.from_dict(data["absorption"]),
            redshift=data["absorption_parameter"]["value"],
        )
        return model


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
        Additionnal parameters for nested models not supplied by the radiative model,
        for now this is used  only for synchrotron self-compton model
    """

    tag = "NaimaSpectralModel"

    def __init__(
        self, radiative_model, distance=1.0 * u.kpc, seed=None, nested_models=None
    ):
        import naima

        self.radiative_model = radiative_model
        self._particle_distribution = self.radiative_model.particle_distribution
        self.distance = u.Quantity(distance)
        self.seed = seed

        if nested_models is None:
            nested_models = {}

        self.nested_models = nested_models

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

        # In case of a synchrotron self compton model, append B and Rpwn to the fittable parameters
        if (
            isinstance(self.radiative_model, naima.models.InverseCompton)
            and "SSC" in self.nested_models
        ):
            B = self.nested_models["SSC"]["B"]
            radius = self.nested_models["SSC"]["radius"]
            parameters.append(Parameter("B", B))
            parameters.append(Parameter("radius", radius, frozen=True))

        self.default_parameters = Parameters(parameters)
        super().__init__()

    def _evaluate_ssc(
        self, energy,
    ):
        """
        Compute photon density spectrum from synchrotron emission for synchrotron self-compton model,
        assuming uniform synchrotron emissivity inside a sphere of radius R
        (see Section 4.1 of Atoyan & Aharonian 1996)

        based on :
        "https://naima.readthedocs.io/en/latest/examples.html#crab-nebula-ssc-model"

        """
        import naima

        SYN = naima.models.Synchrotron(
            self._particle_distribution,
            B=self.B.quantity,
            Eemax=self.radiative_model.Eemax,
            Eemin=self.radiative_model.Eemin,
        )

        Esy = np.logspace(-7, 9, 100) * u.eV
        Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
        phn_sy = Lsy / (4 * np.pi * self.radius.quantity ** 2 * const.c) * 2.24
        # The factor 2.24 comes from the assumption on uniform synchrotron
        # emissivity inside a sphere

        if "SSC" not in self.radiative_model.seed_photon_fields:
            self.radiative_model.seed_photon_fields["SSC"] = {
                "isotropic": True,
                "type": "array",
                "energy": Esy,
                "photon_density": phn_sy,
            }
        else:
            self.radiative_model.seed_photon_fields["SSC"]["photon_density"] = phn_sy

        dnde = self.radiative_model.flux(
            energy, seed=self.seed, distance=self.distance
        ) + SYN.flux(energy, distance=self.distance)
        return dnde

    def evaluate(self, energy, **kwargs):
        """Evaluate the model."""
        import naima

        for name, value in kwargs.items():
            setattr(self._particle_distribution, name, value)

        if "B" in self.radiative_model.param_names:
            self.radiative_model.B = self.B.quantity

        if (
            isinstance(self.radiative_model, naima.models.InverseCompton)
            and "SSC" in self.nested_models
        ):
            dnde = self._evaluate_ssc(energy.flatten())
        elif self.seed is not None:
            dnde = self.radiative_model.flux(
                energy.flatten(), seed=self.seed, distance=self.distance
            )
        else:
            dnde = self.radiative_model.flux(energy.flatten(), distance=self.distance)

        dnde = dnde.reshape(energy.shape)
        unit = 1 / (energy.unit * u.cm ** 2 * u.s)
        return dnde.to(unit)

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
            * np.exp(-((energy - mean) ** 2) / (2 * sigma ** 2))
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
        u_min = (
            (emin - self.mean.quantity) / (np.sqrt(2) * self.sigma.quantity)
        ).to_value("")
        u_max = (
            (emax - self.mean.quantity) / (np.sqrt(2) * self.sigma.quantity)
        ).to_value("")

        return (
            self.norm.quantity
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
        u_min = (
            (emin - self.mean.quantity) / (np.sqrt(2) * self.sigma.quantity)
        ).to_value("")
        u_max = (
            (emax - self.mean.quantity) / (np.sqrt(2) * self.sigma.quantity)
        ).to_value("")
        a = self.norm.quantity * self.sigma.quantity / np.sqrt(2 * np.pi)
        b = self.norm.quantity * self.mean.quantity / 2
        return a * (np.exp(-(u_min ** 2)) - np.exp(-(u_max ** 2))) + b * (
            scipy.special.erf(u_max) - scipy.special.erf(u_min)
        )
