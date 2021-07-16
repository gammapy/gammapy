# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import inspect
from copy import deepcopy
import numpy as np
from astropy.table import Table
from astropy import units as u
from gammapy.modeling.models import Model
from gammapy.maps import MapAxis, Map

__all__ = ["Estimator", "FluxEstimate"]


DEFAULT_UNIT = {
    "dnde": u.Unit("cm-2 s-1 TeV-1"),
    "e2dnde": u.Unit("erg cm-2 s-1"),
    "flux": u.Unit("cm-2 s-1"),
    "eflux": u.Unit("erg cm-2 s-1"),
}

REQUIRED_MAPS = {
    "dnde": ["dnde"],
    "e2dnde": ["e2dnde"],
    "flux": ["flux"],
    "eflux": ["eflux"],
    "likelihood": ["norm"],
}

REQUIRED_COLUMNS = {
    "dnde": ["e_ref", "dnde"],
    "e2dnde": ["e_ref", "e2dnde"],
    "flux": ["e_min", "e_max", "flux"],
    "eflux": ["e_min", "e_max", "eflux"],
    # TODO: extend required columns
    "likelihood": ["e_min", "e_max", "e_ref", "ref_dnde", "ref_flux", "ref_eflux", "norm"],
}


REQUIRED_QUANTITIES_SCAN = ["stat_scan", "stat"]

OPTIONAL_QUANTITIES = {
    "dnde": ["dnde_err", "dnde_errp", "dnde_errn", "dnde_ul"],
    "e2dnde": ["e2dnde_err", "e2dnde_errp", "e2dnde_errn", "e2dnde_ul"],
    "flux": ["flux_err", "flux_errp", "flux_errn", "flux_ul"],
    "eflux": ["eflux_err", "eflux_errp", "eflux_errn", "eflux_ul"],
    "likelihood": ["norm_err", "norm_errn", "norm_errp", "norm_ul"],
}

VALID_QUANTITIES = [
    "norm",
    "norm_err",
    "norm_errn",
    "norm_errp",
    "norm_ul",
    "ts",
    "sqrt_ts",
    "npred",
    "npred_excess",
    "npred_null",
    "stat",
    "stat_scan",
    "stat_null",
    "niter",
    "is_ul",
    "counts"
]


OPTIONAL_QUANTITIES_COMMON = [
    "ts",
    "sqrt_ts",
    "npred",
    "npred_excess",
    "npred_null",
    "stat",
    "stat_null",
    "niter",
#    "is_ul",
    "counts"
]


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

        if selection is None:
            self._selection_optional = []
        elif "all" in selection:
            self._selection_optional = available
        else:
            if set(selection).issubset(set(available)):
                self._selection_optional = selection
            else:
                difference = set(selection).difference(set(available))
                raise ValueError(f"{difference} is not a valid method.")

    def _get_energy_axis(self, dataset):
        """Energy axis"""
        if self.energy_edges is None:
            energy_axis = dataset.counts.geom.axes["energy"].squash()
        else:
            energy_axis = MapAxis.from_energy_edges(self.energy_edges)

        return energy_axis

    def copy(self):
        """Copy estimator"""
        return deepcopy(self)

    @property
    def config_parameters(self):
        """Config parameters"""
        pars = self.__dict__.copy()
        pars = {key.strip("_"): value for key, value in pars.items()}
        return pars

    def __str__(self):
        s = f"{self.__class__.__name__}\n"
        s += "-" * (len(s) - 1) + "\n\n"

        pars = self.config_parameters
        max_len = np.max([len(_) for _ in pars]) + 1

        for name, value in sorted(pars.items()):
            if isinstance(value, Model):
                s += f"\t{name:{max_len}s}: {value.tag[0]}\n"
            elif inspect.isclass(value):
                s += f"\t{name:{max_len}s}: {value.__name__}\n"
            elif isinstance(value, u.Quantity):
                s += f"\t{name:{max_len}s}: {value}\n"
            elif isinstance(value, Estimator):
                pass
            elif isinstance(value, np.ndarray):
                value = np.array_str(value, precision=2, suppress_small=True)
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
    data : dict of `Map`
        Mappable containing the sed data with at least a 'norm' entry.
        If data is a Table, it should contain 'e_min' and 'e_max' columns.
    reference_spectral_model : `SpectralModel`
        Reference spectral model used to produce the input data.
    meta : dict
        Flux maps meta data.
    """
    _expand_slice = (slice(None), np.newaxis, np.newaxis)

    def __init__(self, data, reference_spectral_model, meta=None):
        self._data = data
        self._reference_spectral_model = reference_spectral_model

        if meta is None:
            meta = {}

        self.meta = meta

    @property
    def available_quantities(self):
        """Available quantities"""
        keys = self._data.keys()

        available_quantities = []

        for quantity in VALID_QUANTITIES:
            if quantity in keys:
                available_quantities.append(quantity)

        return available_quantities

    @staticmethod
    def _validate_data(data, sed_type, check_scan=False):
        """Check that map input is valid and correspond to one of the SED type."""
        try:
            keys = data.keys()
            required = set(REQUIRED_MAPS[sed_type])
        except KeyError:
            raise ValueError(f"Unknown SED type: '{sed_type}'")

        if check_scan:
            required = required.union(REQUIRED_QUANTITIES_SCAN)

        if not required.issubset(keys):
            missing = required.difference(keys)
            raise ValueError(
                "Missing data / column for sed type '{}':" " {}".format(sed_type, missing)
            )

    # TODO: add support for scan
    def _check_quantity(self, quantity):
        if quantity not in self.available_quantities:
            raise AttributeError(
                f"Quantity '{quantity}' is not defined on current flux estimate."
            )

    @property
    def n_sigma(self):
        """n sigma UL"""
        return self.meta.get("n_sigma", 1)

    @property
    def n_sigma_ul(self):
        """n sigma UL"""
        return self.meta.get("n_sigma_ul")

    @property
    def ts_threshold_ul(self):
        """TS threshold for upper limits"""
        return self.meta.get("ts_threshold_ul", 4)

    @property
    def sed_type_init(self):
        """Initial sed type"""
        return self.meta.get("sed_type_init")

    @property
    def geom(self):
        """Reference map geometry (`Geom`)"""
        return self.norm.geom

    @property
    def energy_axis(self):
        """Energy axis (`MapAxis`)"""
        return self.geom.axes["energy"]

    @property
    def reference_spectral_model(self):
        """Reference spectral model (`SpectralModel`)"""
        return self._reference_spectral_model

    @property
    def energy_ref(self):
        """Reference energy.

        Defined by `energy_ref` column in `FluxPoints.table` or computed as log
        center, if `energy_min` and `energy_max` columns are present in `FluxEstimate.data`.

        Returns
        -------
        energy_ref : `~astropy.units.Quantity`
            Reference energy.
        """
        return self.energy_axis.center

    @property
    def energy_min(self):
        """Energy min

        Returns
        -------
        energy_min : `~astropy.units.Quantity`
            Lower bound of energy bin.
        """
        return self.energy_axis.edges[:-1]

    @property
    def energy_max(self):
        """Energy max

        Returns
        -------
        energy_max : `~astropy.units.Quantity`
            Upper bound of energy bin.
        """
        return self.energy_axis.edges[1:]

    # TODO: keep or remove?
    @property
    def niter(self):
        """Number of iterations of fit"""
        self._check_quantity("niter")
        return self._data["niter"]

    @property
    def is_ul(self):
        """Whether data is an upper limit"""
        # TODO: make this a well defined behaviour
        is_ul = self.norm.copy()

        if "ts" in self._data and "norm_ul" in self._data:
            is_ul.data = self.ts.data < self.ts_threshold_ul
        elif "norm_ul" in self._data:
            is_ul.data = np.isfinite(self.norm_ul)
        else:
            is_ul.data = np.isnan(self.norm)

        return is_ul

    @property
    def counts(self):
        """Predicted counts null hypothesis"""
        self._check_quantity("counts")
        return self._data["counts"]

    @property
    def npred(self):
        """Predicted counts"""
        self._check_quantity("npred")
        return self._data["npred"]

    @property
    def npred_null(self):
        """Predicted counts null hypothesis"""
        self._check_quantity("npred_null")
        return self._data["npred_null"]

    @property
    def npred_excess(self):
        """Predicted excess counts"""
        self._check_quantity("npred")
        self._check_quantity("npred_null")
        return self._data["npred"] - self._data["npred_null"]

    @property
    def stat_scan(self):
        """Fit statistic value"""
        self._check_quantity("stat_scan")
        return self._data["stat_scan"]

    @property
    def stat(self):
        """Fit statistic value"""
        self._check_quantity("stat")
        return self._data["stat"]

    @property
    def stat_null(self):
        """Fit statistic value for the null hypothesis"""
        self._check_quantity("stat_null")
        return self._data["stat_null"]

    @property
    def ts(self):
        """ts map (`Map`)"""
        self._check_quantity("ts")
        return self._data["ts"]

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

        Returns
        -------
        sqrt_ts : `Map`
            sqrt(TS) map
        """
        if "sqrt_ts" in self._data:
            return self._data["sqrt_ts"]
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                data = np.where(self.norm > 0, np.sqrt(self.ts), -np.sqrt(self.ts))
                return Map.from_geom(geom=self.geom, data=data)

    @property
    def norm(self):
        """Norm values"""
        return self._data["norm"]

    @property
    def norm_err(self):
        """Norm error"""
        self._check_quantity("norm_err")
        return self._data["norm_err"]

    @property
    def norm_errn(self):
        """Negative norm error"""
        self._check_quantity("norm_errn")
        return self._data["norm_errn"]

    @property
    def norm_errp(self):
        """Positive norm error"""
        self._check_quantity("norm_errp")
        return self._data["norm_errp"]

    @property
    def norm_ul(self):
        """Norm upper limit"""
        self._check_quantity("norm_ul")
        return self._data["norm_ul"]

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

    def to_dict(self, sed_type="likelihood"):
        """Return maps in a given SED type in the form of a dictionary.

        Parameters
        ----------
        sed_type : str
            sed type to convert to. Default is `Likelihood`

        Returns
        -------
        map_dict : dict
            Dictionary containing the requested maps.
        """
        data = {}
        all_maps = REQUIRED_MAPS[sed_type] + OPTIONAL_QUANTITIES[sed_type] + OPTIONAL_QUANTITIES_COMMON

        for quantity in all_maps:
            try:
                data[quantity] = getattr(self, quantity)
            except AttributeError:
                pass

        return data

    @classmethod
    def from_dict(cls, maps, sed_type="likelihood", reference_model=None, gti=None):
        """Create FluxMaps from a dictionary of maps.

        Parameters
        ----------
        maps : dict
            Dictionary containing the input maps.
        sed_type : str
            SED type of the input maps. Default is `Likelihood`
        reference_model : `~gammapy.modeling.models.SkyModel`, optional
            Reference model to use for conversions. Default in None.
            If None, a model consisting of a point source with a power law spectrum of index 2 is assumed.
        gti : `~gammapy.data.GTI`
            Maps GTI information. Default is None.

        Returns
        -------
        flux_maps : `~gammapy.estimators.FluxMaps`
            Flux maps object.
        """
        cls._validate_data(data=maps, sed_type=sed_type)

        if sed_type == "likelihood":
            return cls(data=maps, reference_model=reference_model)

        if reference_model is None:
            log.warning(
                "No reference model set for FluxMaps. Assuming point source with E^-2 spectrum."
            )
            reference_model = cls.default_model

        map_ref = maps[sed_type]

        energy_axis = map_ref.geom.axes["energy"]

        with np.errstate(invalid="ignore", divide="ignore"):
            fluxes = reference_model.spectral_model.reference_fluxes(energy_axis=energy_axis)

        # TODO: handle reshaping in MapAxis
        factor = fluxes[f"ref_{sed_type}"].to(map_ref.unit)[:, np.newaxis, np.newaxis]

        data = dict()
        data["norm"] = map_ref / factor

        for key in OPTIONAL_QUANTITIES[sed_type]:
            if key in maps:
                norm_type = key.replace(sed_type, "norm")
                data[norm_type] = maps[key] / factor

        # We add the remaining maps
        for key in OPTIONAL_QUANTITIES_COMMON:
            if key in maps:
                data[key] = maps[key]

        return cls(data=data, reference_model=reference_model, gti=gti)

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__)
        str_ += "\n\n"
        str_ += "\t" + f"geom            : {self.geom.__class__.__name__}\n"
        str_ += "\t" + f"axes            : {self.geom.axes_names}\n"
        str_ += "\t" + f"shape           : {self.geom.data_shape[::-1]}\n"
        str_ += "\t" + f"quantities      : {list(self.available_quantities)}\n"
        str_ += "\t" + f"ref. model      : {self.reference_spectral_model.tag[-1]}\n"
        str_ += "\t" + f"n_sigma         : {self.n_sigma}\n"
        str_ += "\t" + f"n_sigma_ul      : {self.n_sigma_ul}\n"
        str_ += "\t" + f"ts_threshold_ul : {self.ts_threshold_ul}\n"
        str_ += "\t" + f"sed type init   : {self.sed_type_init}\n"
        return str_.expandtabs(tabsize=2)
