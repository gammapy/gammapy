# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model parameter classes."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import copy
from ...extern import six
from astropy import units as u
from astropy.table import Table
from ..array import check_type

__all__ = ["Parameter", "Parameters"]


class Parameter(object):
    """
    Class representing model parameters.

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

        if isinstance(factor, u.Quantity) or isinstance(factor, six.string_types):
            self.quantity = factor
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
        self._factor = check_type(val, "number")

    @property
    def scale(self):
        """Scale (float)."""
        return self._scale

    @scale.setter
    def scale(self, val):
        self._scale = check_type(val, "number")

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
    def max(self):
        """Maximum (float)."""
        return self._max

    @max.setter
    def max(self, val):
        self._max = float(val)

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
        return self.value * u.Unit(self.unit)

    @quantity.setter
    def quantity(self, val):
        val = u.Quantity(val)
        self.value = val.value
        self.unit = str(val.unit)

    def __repr__(self):
        return (
            "Parameter(name={name!r}, value={value!r}, "
            "factor={factor!r}, scale={scale!r}, unit={unit!r}, "
            "min={min!r}, max={max!r}, frozen={frozen!r})"
        ).format(**self.to_dict())

    def to_dict(self):
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

    def autoscale(self, method="scale10"):
        """Autoscale the parameters.

        Set ``factor`` and ``scale`` according to ``method``

        Available methods:

        * ``scale10`` sets ``scale`` to power of 10,
          so that factor is in the range 1 to 10
        * ``factor1`` sets ``factor, scale = 1, value``

        Parameters
        ----------
        method : {'factor1', 'scale10'}
            Method to apply
        """
        if method == "scale10":
            value = self.value
            if value != 0:
                power = int(np.log10(np.absolute(value)))
                scale = 10 ** power
                self.factor = value / scale
                self.scale = scale
        elif method == "factor1":
            self.factor, self.scale = 1, self.value
        else:
            raise ValueError("Invalid method: {}".format(method))


class Parameters(object):
    """List of `Parameter`.

    Holds covariance matrix

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

    def __init__(self, parameters, covariance=None, apply_autoscale=True):
        self._parameters = parameters
        self.covariance = covariance
        self.apply_autoscale = apply_autoscale

    def _init_covar(self):
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

    # TODO: replace this with a better API to update parameters
    @parameters.setter
    def parameters(self, vals):
        self._parameters = vals

    @property
    def names(self):
        """List of parameter names"""
        return [par.name for par in self.parameters]

    def __str__(self):
        ss = self.__class__.__name__
        for par in self.parameters:
            ss += "\n{}".format(par)
        ss += "\n\nCovariance: \n{}".format(self.covariance)
        return ss

    def _get_idx(self, val):
        """Convert parameter name or index to index"""
        if isinstance(val, six.string_types):
            for idx, par in enumerate(self.parameters):
                if val == par.name:
                    return idx
            raise IndexError("Parameter {} not found for : {}".format(val, self))

        else:
            return val

    def __getitem__(self, name):
        """Access parameter by name or index"""
        idx = self._get_idx(name)
        return self.parameters[idx]

    def to_dict(self):
        retval = dict(parameters=[], covariance=None)
        for par in self.parameters:
            retval["parameters"].append(par.to_dict())
        if self.covariance is not None:
            retval["covariance"] = self.covariance.tolist()
        return retval

    def to_table(self):
        """Convert parameter attributes to `~astropy.table.Table`."""
        t = Table()
        t["name"] = [p.name for p in self.parameters]
        t["value"] = [p.value for p in self.parameters]
        if self.covariance is None:
            t["error"] = np.nan
        else:
            t["error"] = [self.error(idx) for idx in range(len(self.parameters))]

        t["unit"] = [p.unit for p in self.parameters]
        t["min"] = [p.min for p in self.parameters]
        t["max"] = [p.max for p in self.parameters]

        for name in ["value", "error", "min", "max"]:
            t[name].format = ".3e"

        return t

    @classmethod
    def from_dict(cls, val):
        pars = []
        for par in val["parameters"]:
            pars.append(
                Parameter(
                    name=par["name"],
                    factor=float(par["value"]),
                    unit=par["unit"],
                    min=float(par["min"]),
                    max=float(par["max"]),
                    frozen=par["frozen"],
                )
            )
        try:
            covariance = np.array(val["covariance"])
        except KeyError:
            covariance = None

        return cls(parameters=pars, covariance=covariance)

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
        """
        Return dict of ufloats with covariance
        """
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
        """
        Set uncorrelated parameters errors.

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
        self.covariance = np.diag(diag) ** 2

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
        self._init_covar()

        idx = self._get_idx(parname)
        err = u.Quantity(err, self[idx].unit).value
        self.covariance[idx, idx] = err ** 2

    def set_parameter_factors(self, factors):
        """Set factor of all parameters.

        Used in the optimiser interface.
        """
        for factor, parameter in zip(factors, self.parameters):
            parameter.factor = factor

    def set_covariance_factors(self, matrix):
        """Set covariance from factor covariance matrix.

        Used in the optimiser interface.
        """
        scales = np.array([par.scale for par in self.parameters])
        scale_matrix = scales[:, np.newaxis] * scales
        self.covariance = scale_matrix * matrix

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
