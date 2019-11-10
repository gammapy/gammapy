# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model parameter classes."""
import copy
import itertools
import numpy as np
import scipy.linalg
import scipy.stats
from astropy import units as u
from astropy.table import Table

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

    def __init__(
        self, name, factor, unit="", scale=1, min=np.nan, max=np.nan, frozen=False
    ):
        self.name = name
        self.scale = scale

        # TODO: move this to a setter method that can be called from `__set__` also!
        # Having it here is bad: behaviour not clear if Quantity and `unit` is passed.
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

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if isinstance(value, Parameter):
            instance.__dict__[self.name] = value
            # TODO: create the link in the parameters list
            # par = instance.__dict__[self.name]
            # instance.__dict__["_parameters"].link(par, value)
        else:
            par = instance.__dict__[self.name]
            raise TypeError(f"Cannot assign {value!r} to parameter {par!r}")

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
        self._factor = float(val) / self._scale

    @property
    def quantity(self):
        """Value times unit (`~astropy.units.Quantity`)."""
        return self.value * self.unit

    @quantity.setter
    def quantity(self, val):
        val = u.Quantity(val, unit=self.unit)
        self.value = val.value
        self.unit = val.unit

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
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit.to_string("fits"),
            "min": self.min,
            "max": self.max,
            "frozen": self.frozen,
        }

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


class Parameters:
    """Parameters container.

    - List of `Parameter` objects.
    - Covariance matrix.

    Parameters
    ----------
    parameters : list of `Parameter`
        List of parameters
    covariance : `~numpy.ndarray`, optional
        Parameters covariance matrix.
        Order of values as specified by `parameters`.
    """

    def __init__(self, parameters=None, covariance=None):
        if parameters is None:
            parameters = []
        else:
            parameters = list(parameters)

        self._parameters = parameters
        self._covariance = covariance

    @property
    def covariance(self):
        """Covariance matrix (`numpy.ndarray`)."""
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        value = np.asanyarray(value)

        shape = len(self), len(self)
        if value.shape != shape:
            raise ValueError(f"Invalid shape: {value.shape}, expected {shape}")

        self._covariance = value

    @classmethod
    def from_values(cls, values=None, covariance=None):
        """Create `Parameters` from values.

        TODO: document.
        """
        parameters = [
            Parameter(f"par_{idx}", value) for idx, value in enumerate(values)
        ]
        return cls(parameters, covariance)

    @property
    def values(self):
        """Parameter values (`numpy.ndarray`)."""
        return np.array([_.value for _ in self._parameters], dtype=np.float64)

    # TODO: add `values` setter, using array interface. Adapt callers to this!

    # TODO: use this, as in https://github.com/cdeil/multinorm/blob/master/multinorm.py
    @property
    def scipy_mvn(self):
        return scipy.stats.multivariate_normal(
            self.values, self.covariance, allow_singular=True
        )

    @classmethod
    def from_stack(cls, parameters_list):
        """Create `Parameters` by stacking a list of other `Parameters` objects.

        Parameters
        ----------
        parameters_list : list of `Parameters`
            List of `Parameters` objects
        """
        pars = itertools.chain(*parameters_list)
        parameters = cls(pars)

        if np.any([pars.covariance is not None for pars in parameters_list]):
            npars = len(parameters)
            parameters.covariance = np.zeros((npars, npars))

            for pars in parameters_list:
                if pars.covariance is not None:
                    parameters.set_subcovariance(pars)

        return parameters

    @property
    def _empty_covariance(self):
        return np.zeros((len(self), len(self)))

    @property
    def _any_covariance(self):
        return self._empty_covariance if self.covariance is None else self.covariance

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

    def _get_idx(self, val):
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
        idx = self._get_idx(name)
        return self._parameters[idx]

    # TODO: think about a better API for this, add docs.
    def link(self, par, other_par):
        """Create link to other parameter"""
        idx = self._get_idx(par)
        self._parameters[idx] = other_par

    def __len__(self):
        return len(self._parameters)

    def __add__(self, other):
        if isinstance(other, Parameters):
            return Parameters.from_stack([self, other])
        else:
            raise TypeError(f"Invalid type: {other!r}")

    def to_dict(self):
        data = dict(parameters=[], covariance=None)
        for par in self._parameters:
            data["parameters"].append(par.to_dict())
        if self.covariance is not None:
            data["covariance"] = self.covariance.tolist()

        return data

    def to_table(self):
        """Convert parameter attributes to `~astropy.table.Table`."""
        t = Table()
        t["name"] = [p.name for p in self._parameters]
        t["value"] = [p.value for p in self._parameters]
        if self.covariance is None:
            t["error"] = np.nan
        else:
            t["error"] = [self.error(idx) for idx in range(len(self))]

        t["unit"] = [p.unit.to_string("fits") for p in self._parameters]
        t["min"] = [p.min for p in self._parameters]
        t["max"] = [p.max for p in self._parameters]
        t["frozen"] = [p.frozen for p in self._parameters]

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

    @property
    def _ufloats(self):
        """Return dict of ufloats with covariance."""
        from uncertainties import correlated_values

        values = [_.value for _ in self._parameters]

        try:
            # convert existing parameters to ufloats
            uarray = correlated_values(values, self.covariance)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix not set.")

        upars = {}
        for par, upar in zip(self._parameters, uarray):
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
        for par in self._parameters:
            error = errors.get(par.name, 0)
            error = u.Quantity(error, par.unit).value
            diag.append(error)
        self.covariance = np.diag(np.power(diag, 2))
        # TODO: this assume no correlation between errors
        # from catalog values we could compute cov_ij = corr_ij*err_i*err_j
        # with corr_ij = np.corrcoef(err_i[:Nsources]), err_j[:Nsources])[0, 1] ?

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
        if self.covariance is None:
            self.covariance = self._empty_covariance

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
        for parameter in self._parameters:
            if not parameter.frozen:
                parameter.factor = factors[idx]
                idx += 1

    @property
    def _scale_matrix(self):
        scales = [par.scale for par in self._parameters]
        return np.outer(scales, scales)

    def _expand_factor_matrix(self, matrix):
        """Expand covariance matrix with zeros for frozen parameters"""
        matrix_expanded = self._empty_covariance
        mask = np.array([par.frozen for par in self._parameters])
        free_parameters = ~(mask | mask[:, np.newaxis])
        matrix_expanded[free_parameters] = matrix.ravel()
        return matrix_expanded

    def set_covariance_factors(self, matrix):
        """Set covariance from factor covariance matrix.

        Used in the optimizer interface.
        """
        # FIXME: this is weird to do sqrt(size). Simplify
        if not np.sqrt(matrix.size) == len(self):
            matrix = self._expand_factor_matrix(matrix)

        self.covariance = self._scale_matrix * matrix

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

    def get_subcovariance(self, parameters):
        """Get sub-covariance matrix

        Parameters
        ----------
        parameters : `Parameters`
            Sub list of parameters.

        Returns
        -------
        covariance : `~numpy.ndarray`
            Sub-covariance.
        """
        idx = [self._get_idx(par) for par in parameters]
        return self.covariance[np.ix_(idx, idx)]

    def set_subcovariance(self, parameters):
        """Set sub-covariance matrix

        Parameters
        ----------
        parameters : `Parameters`
            Sub list of parameters.

        """
        idx = [self._get_idx(par) for par in parameters]
        self.covariance[np.ix_(idx, idx)] = parameters.covariance


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
