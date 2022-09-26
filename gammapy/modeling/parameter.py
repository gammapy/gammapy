# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model parameter classes."""
import collections.abc
import copy
import itertools
import logging
import numpy as np
from astropy import units as u
from gammapy.utils.interpolation import interpolation_scale
from gammapy.utils.table import table_from_row_data

__all__ = ["Parameter", "Parameters"]

log = logging.getLogger(__name__)


def _get_parameters_str(parameters):
    str_ = ""

    for par in parameters:
        if par.name == "amplitude":
            value_format, error_format = "{:10.2e}", "{:7.1e}"
        else:
            value_format, error_format = "{:10.3f}", "{:7.2f}"

        line = "\t{:21} {:8}: " + value_format + "\t {} {:<12s}\n"

        if par._link_label_io is not None:
            name = par._link_label_io
        else:
            name = par.name

        if par.frozen:
            frozen, error = "(frozen)", "\t\t"
        else:
            frozen = ""
            try:
                error = "+/- " + error_format.format(par.error)
            except AttributeError:
                error = ""
        str_ += line.format(name, frozen, par.value, error, par.unit)
    return str_.expandtabs(tabsize=2)


class Parameter:
    """A model parameter.

    Note that the parameter value has been split into
    a factor and scale like this::

        value = factor x scale

    Users should interact with the ``value``, ``quantity``
    or ``min`` and ``max`` properties and consider the fact
    that there is a ``factor``` and ``scale`` an implementation detail.

    That was introduced for numerical stability in parameter and error
    estimation methods, only in the Gammapy optimiser interface do we
    interact with the ``factor``, ``factor_min`` and ``factor_max`` properties,
    i.e. the optimiser "sees" the well-scaled problem.

    Parameters
    ----------
    name : str
        Name
    value : float or `~astropy.units.Quantity`
        Value
    scale : float, optional
        Scale (sometimes used in fitting)
    unit : `~astropy.units.Unit` or str, optional
        Unit
    min : float, optional
        Minimum (sometimes used in fitting)
    max : float, optional
        Maximum (sometimes used in fitting)
    frozen : bool, optional
        Frozen? (used in fitting)
    error : float
        Parameter error
    scan_min : float
        Minimum value for the parameter scan. Overwrites scan_n_sigma.
    scan_max : float
        Minimum value for the parameter scan. Overwrites scan_n_sigma.
    scan_n_values: int
        Number of values to be used for the parameter scan.
    scan_n_sigma : int
        Number of sigmas to scan.
    scan_values: `numpy.array`
        Scan values. Overwrites all of the scan keywords before.
    scale_method : {'scale10', 'factor1', None}
        Method used to set ``factor`` and ``scale``
    interp : {"lin", "sqrt", "log"}
        Parameter scaling to use for the scan.
    is_norm : bool
        Whether the parameter represents the flux norm of the model.
    """

    def __init__(
        self,
        name,
        value,
        unit="",
        scale=1,
        min=np.nan,
        max=np.nan,
        frozen=False,
        error=0,
        scan_min=None,
        scan_max=None,
        scan_n_values=11,
        scan_n_sigma=2,
        scan_values=None,
        scale_method="scale10",
        interp="lin",
        is_norm=False,
    ):
        if not isinstance(name, str):
            raise TypeError(f"Name must be string, got '{type(name)}' instead")

        self._name = name
        self._link_label_io = None
        self.scale = scale
        self.min = min
        self.max = max
        self.frozen = frozen
        self._error = error
        self._is_norm = is_norm
        self._type = None

        # TODO: move this to a setter method that can be called from `__set__` also!
        # Having it here is bad: behaviour not clear if Quantity and `unit` is passed.
        if isinstance(value, u.Quantity) or isinstance(value, str):
            val = u.Quantity(value)
            self.value = val.value
            self.unit = val.unit
        else:
            self.factor = value
            self.unit = unit

        self.scan_min = scan_min
        self.scan_max = scan_max
        self.scan_values = scan_values
        self.scan_n_values = scan_n_values
        self.scan_n_sigma = scan_n_sigma
        self.interp = interp
        self.scale_method = scale_method

    def __get__(self, instance, owner):
        if instance is None:
            return self

        par = instance.__dict__[self.name]
        par._type = getattr(instance, "type", None)
        return par

    def __set__(self, instance, value):
        if isinstance(value, Parameter):
            instance.__dict__[self.name] = value
        else:
            par = instance.__dict__[self.name]
            raise TypeError(f"Cannot assign {value!r} to parameter {par!r}")

    def __set_name__(self, owner, name):
        if not self._name == name:
            raise ValueError(f"Expected parameter name '{name}', got {self._name}")

    @property
    def is_norm(self):
        """Whether the parameter represents the norm of the model"""
        return self._is_norm

    @property
    def type(self):
        return self._type

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, value):
        self._error = float(u.Quantity(value, unit=self.unit).value)

    @property
    def name(self):
        """Name (str)."""
        return self._name

    @property
    def factor(self):
        """Factor (float)."""
        return self._factor

    @factor.setter
    def factor(self, val):
        self._factor = float(val)

    @property
    def scale(self):
        """Scale (float)."""
        return self._scale

    @scale.setter
    def scale(self, val):
        self._scale = float(val)

    @property
    def unit(self):
        """Unit (`~astropy.units.Unit`)."""
        return self._unit

    @unit.setter
    def unit(self, val):
        self._unit = u.Unit(val)

    @property
    def min(self):
        """Minimum (float)."""
        return self._min

    @min.setter
    def min(self, val):
        "Astropy Table has masked values for NaN. Replacing with np.nan."
        if isinstance(val, np.ma.core.MaskedConstant):
            self._min = np.nan
        else:
            self._min = float(val)

    @property
    def factor_min(self):
        """Factor min (float).

        This ``factor_min = min / scale`` is for the optimizer interface.
        """
        return self.min / self.scale

    @property
    def max(self):
        """Maximum (float)."""
        return self._max

    @max.setter
    def max(self, val):
        "Astropy Table has masked values for NaN. Replacing with np.nan."
        if isinstance(val, np.ma.core.MaskedConstant):
            self._max = np.nan
        else:
            self._max = float(val)

    @property
    def factor_max(self):
        """Factor max (float).

        This ``factor_max = max / scale`` is for the optimizer interface.
        """
        return self.max / self.scale

    @property
    def scale_method(self):
        """Method used to set ``factor`` and ``scale``"""
        return self._scale_method

    @scale_method.setter
    def scale_method(self, val):
        if val not in ["scale10", "factor1"] and val is not None:
            raise ValueError(f"Invalid method: {val}")
        self._scale_method = val

    @property
    def frozen(self):
        """Frozen? (used in fitting) (bool)."""
        return self._frozen

    @frozen.setter
    def frozen(self, val):
        if val in ["True", "False"]:
            val = bool(val)
        if not isinstance(val, bool) and not isinstance(val, np.bool_):
            raise TypeError(f"Invalid type: {val}, {type(val)}")
        self._frozen = val

    @property
    def value(self):
        """Value = factor x scale (float)."""
        return self._factor * self._scale

    @value.setter
    def value(self, val):
        self.factor = float(val) / self._scale

    @property
    def quantity(self):
        """Value times unit (`~astropy.units.Quantity`)."""
        return self.value * self.unit

    @quantity.setter
    def quantity(self, val):
        val = u.Quantity(val)

        if not val.unit.is_equivalent(self.unit):
            raise u.UnitConversionError(
                f"Unit must be equivalent to {self.unit} for parameter {self.name}"
            )

        self.value = val.value
        self.unit = val.unit

    # TODO: possibly allow to set this independently
    @property
    def conf_min(self):
        """Confidence min value (`float`)

        Returns parameter minimum if defined else the scan_min
        """
        if not np.isnan(self.min):
            return self.min
        else:
            return self.scan_min

    # TODO: possibly allow to set this independently
    @property
    def conf_max(self):
        """Confidence max value (`float`)

        Returns parameter maximum if defined else the scan_max
        """
        if not np.isnan(self.max):
            return self.max
        else:
            return self.scan_max

    @property
    def scan_min(self):
        """Stat scan min"""
        if self._scan_min is None:
            return self.value - self.error * self.scan_n_sigma

        return self._scan_min

    @property
    def scan_max(self):
        """Stat scan max"""
        if self._scan_max is None:
            return self.value + self.error * self.scan_n_sigma

        return self._scan_max

    @scan_min.setter
    def scan_min(self, value):
        """Stat scan min setter"""
        self._scan_min = value

    @scan_max.setter
    def scan_max(self, value):
        """Stat scan max setter"""
        self._scan_max = value

    @property
    def scan_n_sigma(self):
        """Stat scan n sigma"""
        return self._scan_n_sigma

    @scan_n_sigma.setter
    def scan_n_sigma(self, n_sigma):
        """Stat scan n sigma"""
        self._scan_n_sigma = int(n_sigma)

    @property
    def scan_values(self):
        """Stat scan values (`~numpy.ndarray`)"""
        if self._scan_values is None:
            scale = interpolation_scale(self.interp)
            parmin, parmax = scale([self.scan_min, self.scan_max])
            values = np.linspace(parmin, parmax, self.scan_n_values)
            return scale.inverse(values)

        return self._scan_values

    @scan_values.setter
    def scan_values(self, values):
        """Set scan values"""
        self._scan_values = values

    def check_limits(self):
        """Emit a warning or error if value is outside the min/max range"""
        if not self.frozen:
            if (~np.isnan(self.min) and (self.value <= self.min)) or (
                ~np.isnan(self.max) and (self.value >= self.max)
            ):
                log.warning(
                    f"Value {self.value} is outside bounds [{self.min}, {self.max}]"
                    f" for parameter '{self.name}'"
                )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name!r}, value={self.value!r}, "
            f"factor={self.factor!r}, scale={self.scale!r}, unit={self.unit!r}, "
            f"min={self.min!r}, max={self.max!r}, frozen={self.frozen!r}, id={hex(id(self))})"
        )

    def copy(self):
        """A deep copy"""
        return copy.deepcopy(self)

    def update_from_dict(self, data):
        """Update parameters from a dict.
        Protection against changing parameter model, type, name."""
        keys = ["value", "unit", "min", "max", "frozen"]
        for k in keys:
            setattr(self, k, data[k])

    def to_dict(self):
        """Convert to dict."""
        output = {
            "name": self.name,
            "value": self.value,
            "unit": self.unit.to_string("fits"),
            "error": self.error,
            "min": self.min,
            "max": self.max,
            "frozen": self.frozen,
            "interp": self.interp,
            "scale_method": self.scale_method,
            "is_norm": self.is_norm,
        }

        if self._link_label_io is not None:
            output["link"] = self._link_label_io

        return output

    def autoscale(self):
        """Autoscale the parameters.

        Set ``factor`` and ``scale`` according to ``scale_method`` attribute

        Available ``scale_method``

        * ``scale10`` sets ``scale`` to power of 10,
          so that abs(factor) is in the range 1 to 10
        * ``factor1`` sets ``factor, scale = 1, value``

        In both cases the sign of value is stored in ``factor``,
        i.e. the ``scale`` is always positive.
        If ``scale_method`` is None the scaling is ignored.

        """
        if self.scale_method == "scale10":
            value = self.value
            if value != 0:
                exponent = np.floor(np.log10(np.abs(value)))
                scale = np.power(10.0, exponent)
                self.factor = value / scale
                self.scale = scale

        elif self.scale_method == "factor1":
            self.factor, self.scale = 1, self.value


class Parameters(collections.abc.Sequence):
    """Parameters container.

    - List of `Parameter` objects.
    - Covariance matrix.

    Parameters
    ----------
    parameters : list of `Parameter`
        List of parameters
    """

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = []
        else:
            parameters = list(parameters)

        self._parameters = parameters

    def check_limits(self):
        """Check parameter limits and emit a warning"""
        for par in self:
            par.check_limits()

    @property
    def types(self):
        """Parameter types"""
        return [par.type for par in self]

    @property
    def min(self):
        """Parameter mins (`numpy.ndarray`)."""
        return np.array([_.min for _ in self._parameters], dtype=np.float64)

    @min.setter
    def min(self, min_array):
        """Parameter minima (`numpy.ndarray`)."""
        if not len(self) == len(min_array):
            raise ValueError("Minima must have same length as parameter list")

        for min_, par in zip(min_array, self):
            par.min = min_

    @property
    def max(self):
        """Parameter maxima (`numpy.ndarray`)."""
        return np.array([_.max for _ in self._parameters], dtype=np.float64)

    @max.setter
    def max(self, max_array):
        """Parameter maxima (`numpy.ndarray`)."""
        if not len(self) == len(max_array):
            raise ValueError("Maxima must have same length as parameter list")

        for max_, par in zip(max_array, self):
            par.max = max_

    @property
    def value(self):
        """Parameter values (`numpy.ndarray`)."""
        return np.array([_.value for _ in self._parameters], dtype=np.float64)

    @value.setter
    def value(self, values):
        """Parameter values (`numpy.ndarray`)."""
        if not len(self) == len(values):
            raise ValueError("Values must have same length as parameter list")

        for value, par in zip(values, self):
            par.value = value

    @classmethod
    def from_stack(cls, parameters_list):
        """Create `Parameters` by stacking a list of other `Parameters` objects.

        Parameters
        ----------
        parameters_list : list of `Parameters`
            List of `Parameters` objects
        """
        pars = itertools.chain(*parameters_list)
        return cls(pars)

    def copy(self):
        """A deep copy"""
        return copy.deepcopy(self)

    @property
    def free_parameters(self):
        """List of free parameters"""
        return self.__class__([par for par in self._parameters if not par.frozen])

    @property
    def unique_parameters(self):
        """Unique parameters (`Parameters`)."""
        return self.__class__(dict.fromkeys(self._parameters))

    @property
    def names(self):
        """List of parameter names"""
        return [par.name for par in self._parameters]

    def index(self, val):
        """Get position index for a given parameter.

        The input can be a parameter object, parameter name (str)
        or if a parameter index (int) is passed in, it is simply returned.
        """
        if isinstance(val, int):
            return val
        elif isinstance(val, Parameter):
            return self._parameters.index(val)
        elif isinstance(val, str):
            for idx, par in enumerate(self._parameters):
                if val == par.name:
                    return idx
            raise IndexError(f"No parameter: {val!r}")
        else:
            raise TypeError(f"Invalid type: {type(val)!r}")

    def __getitem__(self, key):
        """Access parameter by name, index or boolean mask"""
        if isinstance(key, np.ndarray) and key.dtype == bool:
            return self.__class__(list(np.array(self._parameters)[key]))
        else:
            idx = self.index(key)
            return self._parameters[idx]

    def __len__(self):
        return len(self._parameters)

    def __add__(self, other):
        if isinstance(other, Parameters):
            return Parameters.from_stack([self, other])
        else:
            raise TypeError(f"Invalid type: {other!r}")

    def to_dict(self):
        data = []

        for par in self._parameters:
            data.append(par.to_dict())

        return data

    def to_table(self):
        """Convert parameter attributes to `~astropy.table.Table`."""
        rows = []
        for p in self._parameters:
            d = p.to_dict()
            if "link" not in d:
                d["link"] = ""
            for key in ["scale_method", "interp"]:
                if key in d:
                    del d[key]
            rows.append({**dict(type=p.type), **d})
        table = table_from_row_data(rows)

        table["value"].format = ".4e"
        for name in ["error", "min", "max"]:
            table[name].format = ".3e"

        return table

    def __eq__(self, other):
        all_equal = np.all([p is p_new for p, p_new in zip(self, other)])
        return all_equal and len(self) == len(other)

    @classmethod
    def from_dict(cls, data):
        parameters = []

        for par in data:
            link_label = par.pop("link", None)
            parameter = Parameter(**par)
            parameter._link_label_io = link_label
            parameters.append(parameter)

        return cls(parameters=parameters)

    def set_parameter_factors(self, factors):
        """Set factor of all parameters.

        Used in the optimizer interface.
        """
        idx = 0
        for parameter in self._parameters:
            if not parameter.frozen:
                parameter.factor = factors[idx]
                idx += 1

    def autoscale(self):
        """Autoscale all parameters.

        See :func:`~gammapy.modeling.Parameter.autoscale`

        """
        for par in self._parameters:
            par.autoscale()

    def select(
        self,
        name=None,
        type=None,
        frozen=None,
    ):
        """Create a mask of models, true if all conditions are verified

        Parameters
        ----------
        name : str or list
            Name of the parameter
        type : {None, spatial, spectral, temporal}
           type of models
        frozen : bool
            Select frozen parameters if True, exclude them if False.

        Returns
        -------
        parameters : `Parameters`
           Selected parameters
        """
        selection = np.ones(len(self), dtype=bool)

        if name and not isinstance(name, list):
            name = [name]

        for idx, par in enumerate(self):
            if name:
                selection[idx] &= np.any([_ == par.name for _ in name])

            if type:
                selection[idx] &= type == par.type

            if frozen is not None:
                if frozen:
                    selection[idx] &= par.frozen
                else:
                    selection[idx] &= ~par.frozen

        return self[selection]

    def freeze_all(self):
        """Freeze all parameters"""
        for par in self._parameters:
            par.frozen = True

    def unfreeze_all(self):
        """Unfreeze all parameters (even those frozen by default)"""
        for par in self._parameters:
            par.frozen = False

    def restore_status(self, restore_values=True):
        """Context manager to restore status.

        A copy of the values is made on enter,
        and those values are restored on exit.

        Parameters
        ----------
        restore_values : bool
            Restore values if True, otherwise restore only frozen status.

        Examples
        --------
        ::

            from gammapy.modeling.models import PowerLawSpectralModel
            pwl = PowerLawSpectralModel(index=2)
            with pwl.parameters.restore_status():
                pwl.parameters["index"].value = 3
            print(pwl.parameters["index"].value)
        """
        return restore_parameters_status(self, restore_values)


class restore_parameters_status:
    def __init__(self, parameters, restore_values=True):
        self.restore_values = restore_values
        self._parameters = parameters
        self.values = [_.value for _ in parameters]
        self.frozen = [_.frozen for _ in parameters]

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        for value, par, frozen in zip(self.values, self._parameters, self.frozen):
            if self.restore_values:
                par.value = value
            par.frozen = frozen
