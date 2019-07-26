# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model parameter classes."""
import copy
import numpy as np
from astropy import units as u
from astropy.units.core import UnitConversionError
from astropy.table import Table
from ..array import check_type

__all__ = ["Parameter", "Parameters"]


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
    factor : float or `~astropy.units.Quantity`
        Factor
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

    __slots__ = ["_name", "_factor", "_scale", "_unit", "_min", "_max", "_frozen"]

    def __init__(
        self, name, factor, unit="", scale=1, min=np.nan, max=np.nan, frozen=False
    ):
        self.name = name
        self.scale = scale

        if isinstance(factor, u.Quantity) or isinstance(factor, str):
            val = u.Quantity(factor)
            self.value = val.value
            self.unit = val.unit
        else:
            self.factor = factor
            self.unit = unit

        self.min = min
        self.max = max
        self.frozen = frozen

    @property
    def name(self):
        """Name (str)."""
        return self._name

    @name.setter
    def name(self, val):
        self._name = check_type(val, "str")

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
        self._frozen = check_type(val, "bool")

    @property
    def value(self):
        """Value = factor x scale (float)."""
        return self._factor * self._scale

    @value.setter
    def value(self, val):
        self._factor = float(val) / self._scale

    @property
    def quantity(self):
        """Value times unit (`~astropy.units.Quantity`)."""
        return self.value * self.unit

    @quantity.setter
    def quantity(self, val):
        val = u.Quantity(val)
        try:
            val.to(self.unit)
            self.value = val.value
            self.unit = val.unit
        except UnitConversionError:
            raise UnitConversionError(
                "{0} parameter must have units homogeneous with {1}".format(
                    self.name, self.unit
                )
            )

    def __repr__(self):
        return (
            "Parameter(name={name!r}, value={value!r}, "
            "factor={factor!r}, scale={scale!r}, unit={unit!r}, "
            "min={min!r}, max={max!r}, frozen={frozen!r})"
        ).format(**self.to_dict())

    def to_dict(self, selection="all"):
        """Convert to dict.

        Parameters
        -----------
        selection : {"all", "simple"}
            Selection of information to include
        """
        if selection == "simple":
            return dict(
                name=self.name,
                value=self.value,
                unit=self.unit.to_string("fits"),
                frozen=self.frozen,
            )
        elif selection == "all":
            return dict(
                name=self.name,
                value=self.value,
                factor=self.factor,
                scale=self.scale,
                unit=self.unit.to_string("fits"),
                min=self.min,
                max=self.max,
                frozen=self.frozen,
            )
        else:
            raise ValueError("Invalid selection: {!r}".format(selection))

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
            raise ValueError("Invalid method: {}".format(method))


class Parameters:
    """List of `Parameter`.

    Holds covariance matrix.

    Parameters
    ----------
    parameters : list of `Parameter`
        List of parameters
    covariance : `~numpy.ndarray`, optional
        Parameters covariance matrix.
        Order of values as specified by `parameters`.
    apply_autoscale : bool, optional
        Flag for optimizers, if True parameters are autoscaled before the fit,
        see `~gammapy.utils.modeling.Parameter.autoscale`
    """

    def __init__(self, parameters=None, covariance=None, apply_autoscale=True):
        if parameters is None:
            parameters = []

        self._parameters = self._filter_unique_parameters(parameters)
        self.covariance = covariance
        self.apply_autoscale = apply_autoscale

    @staticmethod
    def _filter_unique_parameters(parameters):
        """Filter unique parameters from a list of parameters"""
        unique_parameters = []

        for par in parameters:
            if par not in unique_parameters:
                unique_parameters.append(par)

        return unique_parameters

    def _init_covariance(self):
        if self.covariance is None:
            shape = (len(self.parameters), len(self.parameters))
            self.covariance = np.zeros(shape)

    def copy(self):
        """A deep copy"""
        return copy.deepcopy(self)

    @property
    def parameters(self):
        """List of `Parameter`."""
        return self._parameters

    @property
    def free_parameters(self):
        """List of free parameters"""
        return [par for par in self.parameters if not par.frozen]

    # TODO: replace this with a better API to update parameters
    @parameters.setter
    def parameters(self, vals):
        self._parameters = vals

    @property
    def names(self):
        """List of parameter names"""
        return [par.name for par in self.parameters]

    def __str__(self):
        str_ = self.__class__.__name__ + "\n\n"

        for par in self.parameters:
            if par.name == "amplitude":
                line = "\t{:12} {:11}: {:.2e} {} {}\n"
            else:
                line = "\t{:12} {:11}: {:.3f} {} {}\n"

            frozen = "(frozen)" if par.frozen else ""
            try:
                error = "+/- {:.2f}".format(self.get_error(par))
            except AttributeError:
                error = ""

            str_ += line.format(par.name, frozen, par.value, error, par.unit)

        return str_

    def _get_idx(self, val):
        """Get position index for a given parameter.

        The input can be a parameter object, parameter name (str)
        or if a parameter index (int) is passed in, it is simply returned.
        """
        if isinstance(val, int):
            return val
        elif isinstance(val, Parameter):
            return self.parameters.index(val)
        elif isinstance(val, str):
            for idx, par in enumerate(self.parameters):
                if val == par.name:
                    return idx
            raise IndexError("No parameter: {!r}".format(val))
        else:
            raise TypeError("Invalid type: {!r}".format(type(val)))

    def __getitem__(self, name):
        """Access parameter by name or index"""
        idx = self._get_idx(name)
        return self.parameters[idx]

    def to_dict(self, selection="all"):
        data = dict(parameters=[], covariance=None)
        for par in self.parameters:
            data["parameters"].append(par.to_dict(selection))
        if self.covariance is not None:
            data["covariance"] = self.covariance.tolist()

        return data

    def to_table(self):
        """Convert parameter attributes to `~astropy.table.Table`."""
        t = Table()
        t["name"] = [p.name for p in self.parameters]
        t["value"] = [p.value for p in self.parameters]
        if self.covariance is None:
            t["error"] = np.nan
        else:
            t["error"] = [self.error(idx) for idx in range(len(self.parameters))]

        t["unit"] = [p.unit.to_string("fits") for p in self.parameters]
        t["min"] = [p.min for p in self.parameters]
        t["max"] = [p.max for p in self.parameters]
        t["frozen"] = [p.frozen for p in self.parameters]

        for name in ["value", "error", "min", "max"]:
            t[name].format = ".3e"

        return t

    @classmethod
    def from_dict(cls, data):
        parameters = []
        for par in data["parameters"]:
            parameter = Parameter(
                name=par["name"],
                factor=float(par["value"]),
                unit=par.get("unit", ""),
                min=float(par.get("min", np.nan)),
                max=float(par.get("max", np.nan)),
                frozen=par.get("frozen", False),
            )
            parameters.append(parameter)

        try:
            covariance = np.array(data["covariance"])
        except KeyError:
            covariance = None

        return cls(parameters=parameters, covariance=covariance)

    def covariance_to_table(self):
        """Convert covariance matrix to `~astropy.table.Table`."""
        if self.covariance is None:
            raise ValueError("No covariance available")

        table = Table()
        table["name"] = self.names
        for idx, par in enumerate(self.parameters):
            vals = self.covariance[idx]
            table[par.name] = vals
            table[par.name].format = ".3e"
        return table

    @property
    def _ufloats(self):
        """Return dict of ufloats with covariance."""
        from uncertainties import correlated_values

        values = [_.value for _ in self.parameters]

        try:
            # convert existing parameters to ufloats
            uarray = correlated_values(values, self.covariance)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix not set.")

        upars = {}
        for par, upar in zip(self.parameters, uarray):
            upars[par.name] = upar

        return upars

    # TODO: deprecate or remove this?
    def set_parameter_errors(self, errors):
        """Set uncorrelated parameters errors.

        Parameters
        ----------
        errors : dict of `~astropy.units.Quantity`
            Dict of parameter errors.
        """
        diag = []
        for par in self.parameters:
            error = errors.get(par.name, 0)
            error = u.Quantity(error, par.unit).value
            diag.append(error)
        self.covariance = np.diag(np.power(diag, 2))

    # TODO: this is a temporary solution until we have a better way
    # to handle covariance matrices via a class
    def error(self, parname):
        """Get parameter error.

        Parameters
        ----------
        parname : str, int
            Parameter name or index
        """
        if self.covariance is None:
            raise ValueError("Covariance matrix not set.")

        idx = self._get_idx(parname)
        return np.sqrt(self.covariance[idx, idx])

    # TODO: this is a temporary solution until we have a better way
    # to handle covariance matrices via a class
    def set_error(self, parname, err):
        """Set parameter error.

        Parameters
        ----------
        parname : str, int
            Parameter name or index
        err : float or Quantity
            Parameter error
        """
        self._init_covariance()

        idx = self._get_idx(parname)
        err = u.Quantity(err, self[idx].unit).value
        self.covariance[idx, idx] = err ** 2

    @property
    def correlation(self):
        r"""Correlation matrix (`numpy.ndarray`).

        Correlation :math:`C` is related to covariance :math:`\Sigma` via:

        .. math::
            C_{ij} = \frac{ \Sigma_{ij} }{ \sqrt{\Sigma_{ii} \Sigma_{jj}} }
        """
        err = np.sqrt(np.diag(self.covariance))
        return self.covariance / np.outer(err, err)

    def set_parameter_factors(self, factors):
        """Set factor of all parameters.

        Used in the optimizer interface.
        """
        idx = 0
        for parameter in self.parameters:
            if not parameter.frozen:
                parameter.factor = factors[idx]
                idx += 1

    @property
    def _scale_matrix(self):
        scales = [par.scale for par in self.parameters]
        return np.outer(scales, scales)

    def _expand_factor_matrix(self, matrix):
        """Expand covariance matrix with zeros for frozen parameters"""
        shape = (len(self.parameters), len(self.parameters))
        matrix_expanded = np.zeros(shape)
        mask = np.array([par.frozen for par in self.parameters])
        free_parameters = ~(mask | mask[:, np.newaxis])
        matrix_expanded[free_parameters] = matrix.ravel()
        return matrix_expanded

    def set_covariance_factors(self, matrix):
        """Set covariance from factor covariance matrix.

        Used in the optimizer interface.
        """
        if not np.sqrt(matrix.size) == len(self.parameters):
            matrix = self._expand_factor_matrix(matrix)

        self.covariance = self._scale_matrix * matrix

    def autoscale(self, method="scale10"):
        """Autoscale all parameters.

        See :func:`~gammapy.utils.modelling.Parameter.autoscale`

        Parameters
        ----------
        method : {'factor1', 'scale10'}
            Method to apply
        """
        for par in self.parameters:
            par.autoscale(method)

    @property
    def restore_values(self):
        """Context manager to restore values.

        A copy of the values is made on enter,
        and those values are restored on exit.

        Examples
        --------
        ::

            from gammapy.spectrum.models import PowerLaw
            pwl = PowerLaw(index=2)
            with pwl.parameters.restore_values:
                pwl.parameters["index"].value = 3
            print(pwl.parameters["index"].value)
        """
        return restore_parameters_values(self)

    def freeze_all(self):
        """Freeze all parameters"""
        for par in self.parameters:
            par.frozen = True


class restore_parameters_values:
    def __init__(self, parameters):
        self.parameters = parameters
        self.values = [_.value for _ in parameters]
        self.frozen = [_.frozen for _ in parameters]

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        for value, par, frozen in zip(self.values, self.parameters, self.frozen):
            par.value = value
            par.frozen = frozen
