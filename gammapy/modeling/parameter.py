# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model parameter classes."""
import collections.abc
import copy
import itertools
import logging
import numpy as np
from astropy import units as u
from gammapy.utils.table import table_from_row_data

__all__ = ["Parameter", "Parameters"]

log = logging.getLogger(__name__)


def _get_parameters_str(parameters):
    str_ = ""

    for par in parameters:
        if par.name == "amplitude":
            line = "\t{:12} {:11}: {:10.2e} {} {:<12s}\n"
        else:
            line = "\t{:12} {:11}: {:7.3f} {} {:<12s}\n"

        frozen = "(frozen)" if par.frozen else ""
        try:
            error = "+/- {:7.2f}".format(parameters.get_error(par))
        except AttributeError:
            error = ""

        str_ += line.format(par.name, frozen, par.value, error, par.unit)
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
    ):
        self.name = name
        self._link_label_io = None
        self.scale = scale
        self.min = min
        self.max = max
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
            self.factor = value
            self.unit = unit

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

    @name.setter
    def name(self, val):
        if not isinstance(val, str):
            raise TypeError(f"Invalid type: {val}, {type(val)}")
        self._name = val

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
        self._max = float(val)

    @property
    def factor_max(self):
        """Factor max (float).

        This ``factor_max = max / scale`` is for the optimizer interface.
        """
        return self.max / self.scale

    @property
    def frozen(self):
        """Frozen? (used in fitting) (bool)."""
        return self._frozen

    @frozen.setter
    def frozen(self, val):
        if not isinstance(val, bool):
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
        val = u.Quantity(val, unit=self.unit)
        self.value = val.value
        self.unit = val.unit

    def check_limits(self):
        """Emit a warning or error if value is outside the min/max range"""
        if not self.frozen:
            if (~np.isnan(self.min) and (self.value <= self.min)) or (
                ~np.isnan(self.max) and (self.value >= self.max)
            ):
                log.warning(
                    f"Value {self.value} is outside bounds [{self.min}, {self.max}] for parameter '{self.name}'"
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

    def to_dict(self):
        """Convert to dict."""
        output = {
            "name": self.name,
            "value": self.value,
            "unit": self.unit.to_string("fits"),
            "min": self.min,
            "max": self.max,
            "frozen": self.frozen,
            "error": self.error,
        }

        if self._link_label_io is not None:
            output["link"] = self._link_label_io
        return output

    def autoscale(self, method="scale10"):
        """Autoscale the parameters.

        Set ``factor`` and ``scale`` according to ``method``

        Available methods:

        * ``scale10`` sets ``scale`` to power of 10,
          so that abs(factor) is in the range 1 to 10
        * ``factor1`` sets ``factor, scale = 1, value``

        In both cases the sign of value is stored in ``factor``,
        i.e. the ``scale`` is always positive.

        Parameters
        ----------
        method : {'factor1', 'scale10'}
            Method to apply
        """
        if method == "scale10":
            value = self.value
            if value != 0:
                exponent = np.floor(np.log10(np.abs(value)))
                scale = np.power(10.0, exponent)
                self.factor = value / scale
                self.scale = scale
        elif method == "factor1":
            self.factor, self.scale = 1, self.value
        else:
            raise ValueError(f"Invalid method: {method}")


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
    def values(self):
        """Parameter values (`numpy.ndarray`)."""
        return np.array([_.value for _ in self._parameters], dtype=np.float64)

    @values.setter
    def values(self, values):
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

    def __getitem__(self, name):
        """Access parameter by name or index"""
        idx = self.index(name)
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
        rows = [p.to_dict() for p in self._parameters]
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

    def autoscale(self, method="scale10"):
        """Autoscale all parameters.

        See :func:`~gammapy.modeling.Parameter.autoscale`

        Parameters
        ----------
        method : {'factor1', 'scale10'}
            Method to apply
        """
        for par in self._parameters:
            par.autoscale(method)

    @property
    def restore_values(self):
        """Context manager to restore values.

        A copy of the values is made on enter,
        and those values are restored on exit.

        Examples
        --------
        ::

            from gammapy.modeling.models import PowerLawSpectralModel
            pwl = PowerLawSpectralModel(index=2)
            with pwl.parameters.restore_values:
                pwl.parameters["index"].value = 3
            print(pwl.parameters["index"].value)
        """
        return restore_parameters_values(self)

    def freeze_all(self):
        """Freeze all parameters"""
        for par in self._parameters:
            par.frozen = True


class restore_parameters_values:
    def __init__(self, parameters):
        self._parameters = parameters
        self.values = [_.value for _ in parameters]
        self.frozen = [_.frozen for _ in parameters]

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        for value, par, frozen in zip(self.values, self._parameters, self.frozen):
            par.value = value
            par.frozen = frozen
