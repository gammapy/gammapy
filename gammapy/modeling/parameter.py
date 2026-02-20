# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model parameter classes."""

import collections.abc
import copy
import html
import itertools
import logging
import numpy as np
from astropy import units as u
from astropy.table import Table
from gammapy.utils.interpolation import interpolation_scale
from gammapy.utils.scripts import make_name


__all__ = ["Parameter", "Parameters", "PriorParameter", "PriorParameters"]

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
    that there is a ``factor`` and ``scale`` an implementation detail.

    That was introduced for numerical stability in parameter and error
    estimation methods, only in the Gammapy optimiser interface do we
    interact with the ``factor``, ``factor_min`` and ``factor_max`` properties,
    i.e. the optimiser "sees" the well-scaled problem.

    Parameters
    ----------
    name : str
        Name.
    value : float or `~astropy.units.Quantity`
        Value.
    scale : float, optional
        Scale (sometimes used in fitting).
    unit : `~astropy.units.Unit` or str, optional
        Unit. Default is "".
    min : float, str or `~astropy.units.quantity`, optional
        Minimum (sometimes used in fitting). If `None`, set to `numpy.nan`. Default is None.
    max : float, str or `~astropy.units.quantity`, optional
        Maximum (sometimes used in fitting). Default is `numpy.nan`.
    frozen : bool, optional
        Frozen (used in fitting).  Default is False.
    error : float, optional
        Parameter error. Default is 0.
    scan_min : float, optional
        Minimum value for the parameter scan. Overwrites scan_n_sigma.
        Default is None.
    scan_max : float, optional
        Maximum value for the parameter scan. Overwrites scan_n_sigma.
        Default is None.
    scan_n_values: int, optional
        Number of values to be used for the parameter scan. Default is 11.
    scan_n_sigma : int, optional
        Number of sigmas to scan. Default is 2.
    scan_values: `numpy.array`, optional
        Scan values. Overwrites all the scan keywords before.
        Default is None.
    scale_method : {'scale10', 'factor1', None}, optional
        Method used to set ``factor`` and ``scale``. Default is "scale10".
    interp : {"lin", "sqrt", "log"}, optional
        Parameter scaling to use for the scan. Default is "lin".
    prior : `~gammapy.modeling.models.Prior`, optional
        Prior set on the parameter. Default is None.
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
        prior=None,
        scale_transform="lin",
    ):
        if not isinstance(name, str):
            raise TypeError(f"Name must be string, got '{type(name)}' instead")

        self._name = name
        self._link_label_io = None
        self._scale_method = scale_method
        self._scale_transform = scale_transform
        self.interp = interp
        self._scale = float(scale)
        self.frozen = frozen
        self._error = error
        self._type = None

        # TODO: move this to a setter method that can be called from `__set__` also!
        # Having it here is bad: behaviour not clear if Quantity and `unit` is passed.
        if isinstance(value, u.Quantity) or isinstance(value, str):
            val = u.Quantity(value)
            self.value = val.value
            self.unit = val.unit
        else:
            self.value = float(value)
            self.unit = unit

        self.min = min
        self.max = max
        self.scan_min = scan_min
        self.scan_max = scan_max
        self.scan_values = scan_values
        self.scan_n_values = scan_n_values
        self.scan_n_sigma = scan_n_sigma
        self.prior = prior

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
    def prior(self):
        """Prior applied to the parameter  as a `~gammapy.modeling.models.Prior`."""
        return self._prior

    @prior.setter
    def prior(self, value):
        if value is not None:
            from .models import Prior

            if isinstance(value, dict):
                from .models import Model

                self._prior = Model.from_dict({"prior": value})
            elif isinstance(value, Prior):
                self._prior = value
            else:
                raise TypeError(f"Invalid type: {value!r}")
        else:
            self._prior = value

    def prior_stat_sum(self):
        if self.prior is not None:
            return self.prior(self)

    @property
    def type(self):
        return self._type

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, value):
        self._error = self._set_quantity_str_float(value)

    @property
    def name(self):
        """Name as a string."""
        return self._name

    @property
    def factor(self):
        """Factor as a float."""
        return self._factor

    @factor.setter
    def factor(self, val):
        self._factor = float(val)
        self._value = float(self.inverse_transform(self._factor))

    @property
    def scale(self):
        """Scale as a float."""
        return self._scale

    @property
    def unit(self):
        """Unit as a `~astropy.units.Unit` object."""
        return self._unit

    @unit.setter
    def unit(self, val):
        self._unit = u.Unit(val)

    @property
    def min(self):
        """Minimum as a float."""
        from .models import UniformPrior, LogUniformPrior

        if isinstance(self.prior, (UniformPrior, LogUniformPrior)):
            return self.prior.min.value
        return self._min

    @min.setter
    def min(self, val):
        """`~astropy.table.Table` has masked values for NaN. Replacing with NaN."""
        if isinstance(val, np.ma.core.MaskedConstant) or (val is None):
            self._min = np.nan
        else:
            self._min = self._set_quantity_str_float(val)

    @property
    def factor_min(self):
        """Factor minimum as a float (used by the optimizer).

        By default, when no transform is applied, ``factor_min = min / scale``,
        otherwise ``factor_min = transform(min)``.
        """
        return self.transform(self.min)

    @property
    def max(self):
        """Maximum as a float."""
        from .models import UniformPrior, LogUniformPrior

        if isinstance(self.prior, (UniformPrior, LogUniformPrior)):
            return self.prior.max.value
        return self._max

    @max.setter
    def max(self, val):
        """`~astropy.table.Table` has masked values for NaN. Replacing with NaN."""
        if isinstance(val, np.ma.core.MaskedConstant) or (val is None):
            self._max = np.nan
        else:
            self._max = self._set_quantity_str_float(val)

    @property
    def factor_max(self):
        """Factor maximum as a float (used by the optimizer).

        By default, when no transform is applied, ``factor_max = max / scale``,
        otherwise ``factor_max = transform(max)``.
        """
        return self.transform(self.max)

    def _set_quantity_str_float(self, value):
        """Logics for min and max setter."""
        if isinstance(value, u.Quantity) or isinstance(value, str):
            value = u.Quantity(value)
            return float(value.to(self.unit).value)
        else:
            return float(value)

    def set_lim(self, min=None, max=None):
        """
        Set the min and/or max value for the parameter.

        Parameters
        ----------
        min, max: float, `~astropy.units.Quantity` or str, optional
            Minimum and Maximum value to assign to the parameter `min` and `max`.
            Default is None, which set `min` and `max` to `~numpy.nan`.
        """
        if min is not None:
            self.min = min
        if max is not None:
            self.max = max

    @property
    def scale_method(self):
        """Method used to set ``factor`` and ``scale``."""
        return self._scale_method

    @scale_method.setter
    def scale_method(self, val):
        if val not in ["scale10", "factor1"] and val is not None:
            raise ValueError(f"Invalid method: {val}")
        self.reset_autoscale()
        self._scale_method = val

    @property
    def scale_transform(self):
        """Scale interp : {"lin", "sqrt", "log"}."""
        return self._scale_transform

    @scale_transform.setter
    def scale_transform(self, val):
        if val not in ["lin", "log", "sqrt"]:
            raise ValueError(f"Invalid transform: {val}")
        self.reset_autoscale()
        self._scale_transform = val

    @property
    def frozen(self):
        """Frozen (used in fitting) (bool)."""
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
        return self._value

    @value.setter
    def value(self, val):
        self._value = float(val)
        self._factor = self.transform(val)

    @property
    def quantity(self):
        """Value times unit as a `~astropy.units.Quantity`."""
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

    @property
    def _step(self):
        return self.error if self.error > 0.0 else np.abs(self.value)

    # TODO: possibly allow to set this independently
    @property
    def conf_min(self):
        """Confidence minimum value as a `float`.

        Return parameter minimum if defined, otherwise  a default is estimated from value and error.
        """
        if not np.isnan(self.min):
            return self.min
        else:
            min_ = self.value - self._step * self.scan_n_sigma
            large_step = np.maximum(self._step, np.abs(self.value))
            min_ = np.minimum(min_, -large_step * 1e5)
            return min_

    # TODO: possibly allow to set this independently
    @property
    def conf_max(self):
        """Confidence maximum value as a `float`.

        Return parameter maximum if defined, otherwise a default is estimated from value and error.
        """
        if not np.isnan(self.max):
            return self.max
        else:
            max_ = self.value + self._step * self.scan_n_sigma
            large_step = np.maximum(self._step, np.abs(self.value))
            max_ = np.maximum(max_, large_step * 1e5)
            return max_

    @property
    def scan_min(self):
        """Stat scan minimum."""
        if self._scan_min is None:
            return self.value - self._step * self.scan_n_sigma

        return self._scan_min

    @property
    def scan_max(self):
        """Stat scan maximum."""
        if self._scan_max is None:
            return self.value + self._step * self.scan_n_sigma

        return self._scan_max

    @scan_min.setter
    def scan_min(self, value):
        """Stat scan minimum setter."""
        self._scan_min = value

    @scan_max.setter
    def scan_max(self, value):
        """Stat scan maximum setter."""
        self._scan_max = value

    @property
    def scan_n_sigma(self):
        """Stat scan n sigma."""
        return self._scan_n_sigma

    @scan_n_sigma.setter
    def scan_n_sigma(self, n_sigma):
        """Stat scan n sigma."""
        self._scan_n_sigma = int(n_sigma)

    @property
    def scan_values(self):
        """Stat scan values as a `numpy.ndarray`."""
        if self._scan_values is None:
            scale = interpolation_scale(self.interp)
            parmin, parmax = scale([self.scan_min, self.scan_max])
            values = np.linspace(parmin, parmax, self.scan_n_values)
            return scale.inverse(values)

        return self._scan_values

    @scan_values.setter
    def scan_values(self, values):
        """Set scan values."""
        self._scan_values = values

    def check_limits(self):
        """Emit a warning or error if value is outside the minimum/maximum range."""
        if not self.frozen:
            if (~np.isnan(self.min) and (self.value < self.min)) or (
                ~np.isnan(self.max) and (self.value > self.max)
            ):
                log.warning(
                    f"Value {self.value} is outside bounds [{self.min}, {self.max}]"
                    f" for parameter '{self.name}'"
                )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name!r}, value={self.value!r}, "
            f"factor={self.factor!r}, scale={self.scale!r}, unit={self.unit!r}, "
            f"min={self.min!r}, max={self.max!r}, frozen={self.frozen!r}, prior={self.prior!r}, id={hex(id(self))})"
        )

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def copy(self):
        """Deep copy."""
        return copy.deepcopy(self)

    def update_from_dict(self, data):
        """Update parameters from a dictionary."""
        keys = ["value", "unit", "min", "max", "frozen", "prior"]
        for k in keys:
            if k == "prior" and data[k] == "":
                data[k] = None
            setattr(self, k, data[k])

    def to_dict(self):
        """Convert to dictionary."""
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
            "scale_transform": self.scale_transform,
        }

        if self._link_label_io is not None:
            output["link"] = self._link_label_io
        if self.prior is not None:
            output["prior"] = self.prior.to_dict()["prior"]
        return output

    def update_scale(self, value):
        """Update the parameter scale.

        Set ``factor`` and ``scale`` according to ``scale_method`` attribute.

        Available ``scale_method``.

        * ``scale10`` sets ``scale`` to power of 10,
          so that abs(factor) is in the range 1 to 10
        * ``factor1`` sets ``factor, scale = 1, value``

        In both cases the sign of value is stored in ``factor``,
        i.e. the ``scale`` is always positive.
        If ``scale_method`` is None the scaling is ignored.

        """
        if self.scale_method == "scale10":
            if value != 0:
                exponent = np.floor(np.log10(np.abs(value)))
                self._scale = np.power(10.0, exponent)

        elif self.scale_method == "factor1":
            self._scale = value

    def transform(self, value, update_scale=False):
        """Transform from value to factor (used by the optimizer).

        Parameters
        ----------
        value : float
            Parameter value.
        update_scale : bool, optional
            Update the scaling (used by the autoscale). Default is False.
        """
        interp_scale = interpolation_scale(self.scale_transform)
        transformed_value = interp_scale(value)
        if update_scale:
            self.update_scale(transformed_value)
        return transformed_value / self.scale

    def inverse_transform(self, factor):
        """Inverse transform from factor (used by the optimizer) to value.

        Parameters
        ----------
        factor : float
            Parameter factor.
        """
        interp_scale = interpolation_scale(self.scale_transform)
        value = interp_scale.inverse(self.scale * factor)
        return value

    def _inverse_transform_derivative(self, factor):
        """Inverse transform from factor (used by the optimizer) to value.

        Parameters
        ----------
        factor : float
            Parameter factor.
        """
        interp_scale = interpolation_scale(self.scale_transform)
        return interp_scale._inverse_deriv(self.scale * factor) * self.scale

    def autoscale(self):
        """Apply `~gammapy.utils.interpolation.interpolation_scale` and `scale_method` to the parameter."""
        self.factor = self.transform(self.value, update_scale=True)

    def reset_autoscale(self):
        """Reset scaling such as factor=value, scale=1."""
        self._factor = self._value
        self._scale = 1.0


class Parameters(collections.abc.Sequence):
    """Parameters container.

    - List of `Parameter` objects.
    - Covariance matrix.

    Parameters
    ----------
    parameters : list of `Parameter`
        List of parameters.
    """

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = []
        else:
            parameters = list(parameters)

        self._parameters = parameters

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def check_limits(self):
        """Check parameter limits and emit a warning."""
        for par in self:
            par.check_limits()

    @property
    def prior(self):
        return [par.prior for par in self]

    def prior_stat_sum(self):
        parameters_stat_sum = 0
        for par in self:
            if par.prior is not None:
                parameters_stat_sum += par.prior_stat_sum()
        return parameters_stat_sum

    @property
    def types(self):
        """Parameter types."""
        return [par.type for par in self]

    @property
    def min(self):
        """Parameter minima as a `numpy.ndarray`."""
        return np.array([_.min for _ in self._parameters], dtype=np.float64)

    @min.setter
    def min(self, min_array):
        """Parameter minima as a `numpy.ndarray`."""
        if not len(self) == len(min_array):
            raise ValueError("Minima must have same length as parameter list")

        for min_, par in zip(min_array, self):
            par.min = min_

    @property
    def max(self):
        """Parameter maxima as a `numpy.ndarray`."""
        return np.array([_.max for _ in self._parameters], dtype=np.float64)

    @max.setter
    def max(self, max_array):
        """Parameter maxima as a `numpy.ndarray`."""
        if not len(self) == len(max_array):
            raise ValueError("Maxima must have same length as parameter list")

        for max_, par in zip(max_array, self):
            par.max = max_

    @property
    def value(self):
        """Parameter values as a `numpy.ndarray`."""
        return np.array([_.value for _ in self._parameters], dtype=np.float64)

    @value.setter
    def value(self, values):
        """Parameter values as a `numpy.ndarray`."""
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
            List of `Parameters` objects.
        """
        pars = itertools.chain(*parameters_list)
        return cls(pars)

    def copy(self):
        """Deep copy."""
        return copy.deepcopy(self)

    @property
    def free_parameters(self):
        """List of free parameters."""
        return self.__class__([par for par in self._parameters if not par.frozen])

    @property
    def unique_parameters(self):
        """Unique parameters as a `Parameters` object."""
        return self.__class__(dict.fromkeys(self._parameters))

    @property
    def free_unique_parameters(self):
        """List of free and unique parameters."""
        return self.__class__([par for par in self.unique_parameters if not par.frozen])

    @property
    def names(self):
        """List of parameter names."""
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
        """Access parameter by name, index or boolean mask."""
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

    @staticmethod
    def _create_default_table():
        name_to_type = {
            "type": "str",
            "name": "str",
            "value": "float",
            "unit": "str",
            "error": "float",
            "min": "float",
            "max": "float",
            "frozen": "bool",
            "link": "str",
            "prior": "str",
        }
        return Table(names=name_to_type.keys(), dtype=name_to_type.values())

    def update_link_label(self):
        """Update linked parameters labels used for serialisation and print."""
        params_list = []
        params_shared = []
        for param in self:
            if param not in params_list:
                params_list.append(param)
                params_list.append(param)
            elif param not in params_shared:
                params_shared.append(param)
        for param in params_shared:
            param._link_label_io = param.name + "@" + make_name()

    def to_table(self):
        """Convert parameter attributes to `~astropy.table.Table`."""
        self.update_link_label()
        table = self._create_default_table()

        for p in self._parameters:
            d = {k: v for k, v in p.to_dict().items() if k in table.colnames}
            if "prior" in d:
                d["prior"] = d["prior"]["type"]
            table.add_row(d)

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
            par.pop("is_norm", None)
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

        See :func:`~gammapy.modeling.Parameter.autoscale`.

        """
        for par in self._parameters:
            par.autoscale()

    def select(
        self,
        name=None,
        type=None,
        frozen=None,
    ):
        """Create a mask of models, true if all conditions are verified.

        Parameters
        ----------
        name : str or list, optional
            Name of the parameter. Default is None.
        type : {None, "spatial", "spectral", "temporal"}
            Type of models. Default is None.
        frozen : bool, optional
            Select frozen parameters if True, exclude them if False. Default is None.

        Returns
        -------
        parameters : `Parameters`
           Selected parameters.
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
                    selection[idx] &= not par.frozen

        return self[selection]

    def freeze_all(self):
        """Freeze all parameters."""
        for par in self._parameters:
            par.frozen = True

    def unfreeze_all(self):
        """Unfreeze all parameters (even those frozen by default)."""
        for par in self._parameters:
            par.frozen = False

    def restore_status(self, restore_values=True):
        """Context manager to restore status.

        A copy of the values is made on enter,
        and those values are restored on exit.

        Parameters
        ----------
        restore_values : bool, optional
            Restore values if True, otherwise restore only frozen status. Default is None.

        Examples
        --------
        >>> from gammapy.modeling.models import PowerLawSpectralModel
        >>> pwl = PowerLawSpectralModel(index=2)
        >>> with pwl.parameters.restore_status():
        ...     pwl.parameters["index"].value = 3
        >>> print(pwl.parameters["index"].value) # doctest: +SKIP
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

    def __exit__(self, exception_type, exception_value, exception_traceback):
        for value, par, frozen in zip(self.values, self._parameters, self.frozen):
            if self.restore_values:
                par.value = value
            par.frozen = frozen


class PriorParameter(Parameter):
    """Parameter of a `~gammapy.modeling.models.Prior`.

    A prior is a probability density function of a model parameter and can take different forms, including but not limited to Gaussian
    distributions and uniform distributions. The prior includes information or knowledge about the dataset or the
    parameters of the fit.

    Parameters
    ----------
    name : str
        Name.
    value : float or `~astropy.units.Quantity`
        Value.
    unit : `~astropy.units.Unit` or str, optional
        Unit. Default is "".

    Examples
    --------
    For a usage example see :doc:`/tutorials/details/priors` tutorial.
    """

    def __init__(
        self,
        name,
        value,
        unit="",
        scale=1,
        min=np.nan,
        max=np.nan,
        error=0,
        scale_method="scale10",
        scale_transform="lin",
    ):
        if not isinstance(name, str):
            raise TypeError(f"Name must be string, got '{type(name)}' instead")

        self._name = name
        self._scale_method = scale_method
        self._scale_transform = scale_transform
        self._scale = float(scale)
        self.min = min
        self.max = max
        self._error = error
        self._prior = None
        if isinstance(value, u.Quantity) or isinstance(value, str):
            val = u.Quantity(value)
            self.value = val.value
            self.unit = val.unit
        else:
            self.factor = value
            self.unit = unit

        self._type = "prior"

    def to_dict(self):
        """Convert to dictionary."""
        output = {
            "name": self.name,
            "value": self.value,
            "unit": self.unit.to_string("fits"),
            "error": self.error,
            "min": self.min,
            "max": self.max,
        }
        return output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name!r}, value={self.value!r}, "
            f"factor={self.factor!r}, scale={self.scale!r}, unit={self.unit!r}, "
            f"min={self.min!r}, max={self.max!r})"
        )


class PriorParameters(Parameters):
    """Container of parameter priors.

    - List of `PriorParameter` objects.

    Parameters
    ----------
    parameters : list of `PriorParameter`
        List of parameters.
    """

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = []
        else:
            parameters = list(parameters)

        self._parameters = parameters

    def to_table(self):
        """Convert parameter attributes to `~astropy.table.Table`."""
        rows = []
        for p in self._parameters:
            d = p.to_dict()
            rows.append({**dict(type=p.type), **d})
        table = Table(rows)

        table["value"].format = ".4e"
        for name in ["error", "min", "max"]:
            table[name].format = ".3e"

        return table

    @classmethod
    def from_dict(cls, data):
        parameters = []

        for par in data:
            parameter = PriorParameter(**par)
            parameters.append(parameter)

        return cls(parameters=parameters)
