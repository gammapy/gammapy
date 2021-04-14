# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import inspect
from copy import deepcopy
import numpy as np
from astropy.table import Table
from astropy import units as u
from gammapy.modeling.models import Model
from gammapy.maps import MapAxis

__all__ = ["Estimator", "FluxEstimate"]

SED_TYPES = ["dnde", "e2dnde", "flux", "eflux"]

OPTIONAL_QUANTITIES = [
    "err", "errn", "errp", "ul", "scan"
]

OPTIONAL_QUANTITIES_COMMON = [
    "ts",
    "sqrt_ts",
    "npred",
    "npred_excess",
    "npred_null",
    "stat",
    "stat_null",
    "niter"
]


DEFAULT_UNIT = {
    "dnde": u.Unit("cm-2 s-1 TeV-1"),
    "e2dnde": u.Unit("erg cm-2 s-1"),
    "flux": u.Unit("cm-2 s-1"),
    "eflux": u.Unit("erg cm-2 s-1"),
}


class Estimator(abc.ABC):
    """Abstract estimator base class."""

    _available_selection_optional = {}

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @abc.abstractmethod
    def run(self, datasets):
        pass

    @property
    def selection_optional(self):
        """"""
        return self._selection_optional

    @selection_optional.setter
    def selection_optional(self, selection):
        """Set optional selection"""
        available = self._available_selection_optional

        if "all" in selection:
            self._selection_optional = available
        elif selection is None:
            self._selection_optional = []
        else:
            if set(selection).issubset(set(available)):
                self._selection_optional = selection
            else:
                difference = set(selection).difference(set(available))
                raise ValueError(f"{difference} is not a valid method.")

    @staticmethod
    def get_sqrt_ts(ts, norm):
        r"""Compute sqrt(TS) value.

        Compute sqrt(TS) as defined by:

        .. math::
            \sqrt{TS} = \left \{
            \begin{array}{ll}
              -\sqrt{TS} & : \text{if} \ norm < 0 \\
              \sqrt{TS} & : \text{else}
            \end{array}
            \right.

        Parameters
        ----------
        ts : `~numpy.ndarray`
            TS value.
        norm : `~numpy.ndarray`
            norm value
        Returns
        -------
        sqrt_ts : `~numpy.ndarray`
            Sqrt(TS) value.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(norm > 0, np.sqrt(ts), -np.sqrt(ts))

    def copy(self):
        """Copy estimator"""
        return deepcopy(self)

    @property
    def config_parameters(self):
        """Config parameters"""
        pars = {}
        names = self.__init__.__code__.co_varnames
        for name in names:
            if name == "self":
                continue

            pars[name] = getattr(self, name)
        return pars

    def __str__(self):
        s = f"{self.__class__.__name__}\n"
        s += "-" * (len(s) - 1) + "\n\n"

        pars = self.config_parameters
        max_len = np.max([len(_) for _ in pars]) + 1

        for name, value in sorted(pars.items()):
            if isinstance(value, Model):
                s += f"\t{name:{max_len}s}: {value.__class__.__name__}\n"
            elif inspect.isclass(value):
                s += f"\t{name:{max_len}s}: {value.__name__}\n"
            elif isinstance(value, np.ndarray):
                s += f"\t{name:{max_len}s}: {value}\n"
            else:
                s += f"\t{name:{max_len}s}: {value}\n"

        return s.expandtabs(tabsize=2)


class FluxEstimate:
    """A flux estimate produced by an Estimator.

    Follows the likelihood SED type description and allows norm values
    to be converted to dnde, flux, eflux and e2dnde

    The flux is converted according to the input spectral model. The latter must be the one used to
    obtain the 'norm' values of the input data.

    The energy axis is obtained from the input data:
    - directly from the energy `MapAxis if the input data is a `dict` of `Map``
    - from the 'e_min' and 'e_max' columns in the input data is an `~astropy.table.Table`

    Parameters
    ----------
    data : dict of `Map` or `Table`
        Mappable containing the sed data with at least a 'norm' entry.
        If data is a Table, it should contain 'e_min' and 'e_max' columns.
    reference_spectral_model : `SpectralModel`
        Reference spectral model used to produce the input data.
    """

    def __init__(self, data, reference_spectral_model):
        # TODO: Check data
        self._data = data

        if hasattr(self._data["norm"], "geom"):
            self.energy_axis = self.data["norm"].geom.axes["energy"]
            self._expand_slice = (slice(None), np.newaxis, np.newaxis)
        else:
            # Here we assume there is only one row per energy
            e_edges = self._data["e_min"].quantity
            e_edges = e_edges.insert(len(self._data), self._data["e_max"].quantity[-1])
            self.energy_axis = MapAxis.from_energy_edges(e_edges)
            self._expand_slice = slice(None)

        # Note that here we could use the specification from dnde_ref to build piecewise PL
        # But does it work beyond min and max centers?

        self._reference_spectral_model = reference_spectral_model

    @property
    def reference_spectral_model(self):
        """Reference spectral model (`SpectralModel`)"""
        return self._reference_spectral_model

    @property
    def data(self):
        return self._data

    @property
    def available_quantities(self):
        """Available quantities"""
        try:
            keys = self.data.keys()
        except AttributeError:
            keys = self.data.columns

        available_quantities = []

        for quantity in OPTIONAL_QUANTITIES:
            norm_quantity = f"norm_{quantity}"
            if norm_quantity in keys:
                available_quantities.append(quantity)

        for quantity in OPTIONAL_QUANTITIES_COMMON:
            if quantity in keys:
                available_quantities.append(quantity)

        return available_quantities

    # TODO: add support for scan
    def _check_norm_quantity(self, quantity):
        if quantity not in self.available_quantities:
            raise KeyError(
                f"Cannot compute required flux quantity. {quantity} "
                "is not defined on current flux estimate."
            )

    @property
    def energy_ref(self):
        """Reference energy"""
        return self.energy_axis.center

    @property
    def energy_min(self):
        """Energy min"""
        return self.energy_axis.edges[:-1]

    @property
    def energy_max(self):
        """Energy max"""
        return self.energy_axis.edges[1:]

    # TODO: keep or remove?
    @property
    def niter(self):
        """Number of iterations of fit"""
        self._check_norm_quantity("niter")
        return self.data["niter"]

    @property
    def npred(self):
        """Predicted counts"""
        self._check_norm_quantity("npred")
        return self.data["npred"]

    @property
    def npred_null(self):
        """Predicted counts null hypothesis"""
        self._check_norm_quantity("npred_null")
        return self.data["npred_null"]

    @property
    def npred_excess(self):
        """Predicted excess counts"""
        self._check_norm_quantity("npred")
        self._check_norm_quantity("npred_null")
        return self.data["npred"] - self.data["npred_null"]

    @property
    def stat(self):
        """Fit statistic value"""
        self._check_norm_quantity("stat")
        return self.data["stat"]

    @property
    def stat_null(self):
        """Fit statistic value for thenull hypothesis"""
        self._check_norm_quantity("stat_null")
        return self.data["stat_null"]

    @property
    def ts(self):
        """ts map (`Map`)"""
        self._check_norm_quantity("ts")
        return self.data["ts"]

    # TODO: just always derive from ts?
    @property
    def sqrt_ts(self):
        """sqrt(TS) as defined by:

        .. math::

            \sqrt{TS} = \left \{
            \begin{array}{ll}
              -\sqrt{TS} & : \text{if} \ norm < 0 \\
              \sqrt{TS} & : \text{else}
            \end{array}
            \right.

        """
        return self.data["sqrt_ts"]

    @property
    def norm(self):
        """Norm values"""
        return self.data["norm"]

    @property
    def norm_err(self):
        """Norm error"""
        self._check_norm_quantity("err")
        return self.data["norm_err"]

    @property
    def norm_errn(self):
        """Negative norm error"""
        self._check_norm_quantity("errn")
        return self.data["norm_errn"]

    @property
    def norm_errp(self):
        """Positive norm error"""
        self._check_norm_quantity("errp")
        return self.data["norm_errp"]

    @property
    def norm_ul(self):
        """Norm upper limit"""
        self._check_norm_quantity("ul")
        return self.data["norm_ul"]

    @property
    def dnde_ref(self):
        """Reference differential flux"""
        result = self.reference_spectral_model(self.energy_axis.center)
        return result[self._expand_slice]

    @property
    def e2dnde_ref(self):
        """Reference differential flux * energy ** 2"""
        energy = self.energy_axis.center
        result = (
            self.reference_spectral_model(energy) * energy ** 2
        )
        return result[self._expand_slice]

    @property
    def flux_ref(self):
        """Reference integral flux"""
        energy_min = self.energy_axis.edges[:-1]
        energy_max = self.energy_axis.edges[1:]
        result = self.reference_spectral_model.integral(energy_min, energy_max)
        return result[self._expand_slice]

    @property
    def eflux_ref(self):
        """Reference energy flux"""
        energy_min = self.energy_axis.edges[:-1]
        energy_max = self.energy_axis.edges[1:]
        result = self.reference_spectral_model.energy_flux(energy_min, energy_max)
        return result[self._expand_slice]

    @property
    def dnde(self):
        """Return differential flux (dnde) SED values."""
        return self.norm * self.dnde_ref

    @property
    def dnde_err(self):
        """Return differential flux (dnde) SED errors."""
        return self.norm_err * self.dnde_ref

    @property
    def dnde_errn(self):
        """Return differential flux (dnde) SED negative errors."""
        return self.norm_errn * self.dnde_ref

    @property
    def dnde_errp(self):
        """Return differential flux (dnde) SED positive errors."""
        return self.norm_errp * self.dnde_ref

    @property
    def dnde_ul(self):
        """Return differential flux (dnde) SED upper limit."""
        return self.norm_ul * self.dnde_ref

    @property
    def e2dnde(self):
        """Return differential energy flux (e2dnde) SED values."""
        return self.norm * self.e2dnde_ref

    @property
    def e2dnde_err(self):
        """Return differential energy flux (e2dnde) SED errors."""
        return self.norm_err * self.e2dnde_ref

    @property
    def e2dnde_errn(self):
        """Return differential energy flux (e2dnde) SED negative errors."""
        return self.norm_errn * self.e2dnde_ref

    @property
    def e2dnde_errp(self):
        """Return differential energy flux (e2dnde) SED positive errors."""
        return self.norm_errp * self.e2dnde_ref

    @property
    def e2dnde_ul(self):
        """Return differential energy flux (e2dnde) SED upper limit."""
        return self.norm_ul * self.e2dnde_ref

    @property
    def flux(self):
        """Return integral flux (flux) SED values."""
        return self.norm * self.flux_ref

    @property
    def flux_err(self):
        """Return integral flux (flux) SED values."""
        return self.norm_err * self.flux_ref

    @property
    def flux_errn(self):
        """Return integral flux (flux) SED negative errors."""
        return self.norm_errn * self.flux_ref

    @property
    def flux_errp(self):
        """Return integral flux (flux) SED positive errors."""
        return self.norm_errp * self.flux_ref

    @property
    def flux_ul(self):
        """Return integral flux (flux) SED upper limits."""
        return self.norm_ul * self.flux_ref

    @property
    def eflux(self):
        """Return energy flux (eflux) SED values."""
        return self.norm * self.eflux_ref

    @property
    def eflux_err(self):
        """Return energy flux (eflux) SED errors."""
        return self.norm_err * self.eflux_ref

    @property
    def eflux_errn(self):
        """Return energy flux (eflux) SED negative errors."""
        return self.norm_errn * self.eflux_ref

    @property
    def eflux_errp(self):
        """Return energy flux (eflux) SED positive errors."""
        return self.norm_errp * self.eflux_ref

    @property
    def eflux_ul(self):
        """Return energy flux (eflux) SED upper limits."""
        return self.norm_ul * self.eflux_ref
