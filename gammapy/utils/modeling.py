# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Model parameter handling
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import copy
from ..extern import six
from astropy import units as u
from astropy.table import Table, Column, vstack

__all__ = [
    'Parameter',
    'ParameterList',
]


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
        Scale
    unit : str, optional
        Unit
    min : float, optional
        Minimum (sometimes used in fitting)
    max : float, optional
        Maximum (sometimes used in fitting)
    frozen : bool, optional
        Frozen? (used in fitting)
    """
    __slots__ = ['_name', '_factor', '_scale', '_unit', '_min', '_max', '_frozen']

    def __init__(self, name, factor, unit='', scale=1, min=np.nan, max=np.nan,
                 frozen=False):
        self.name = name

        if isinstance(factor, u.Quantity) or isinstance(factor, six.string_types):
            self.quantity = value
        else:
            self.factor = factor
            self.unit = unit
            self.scale = scale

        self.min = min
        self.max = max
        self.frozen = frozen

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = str(val)

    @property
    def factor(self):
        return self._factor

    @factor.setter
    def factor(self, val):
        self._factor = float(val)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        self._scale = float(val)

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, val):
        self._unit = str(val)

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, val):
        self._min = float(val)

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, val):
        self._max = float(val)

    @property
    def frozen(self):
        return self._frozen

    @frozen.setter
    def frozen(self, val):
        self._frozen = bool(val)

    @property
    def value(self):
        return self._factor * self._scale

    @value.setter
    def value(self, val):
        self._factor = float(val) / self._scale

    @property
    def quantity(self):
        return self.value * u.Unit(self.unit)

    @quantity.setter
    def quantity(self, par):
        par = u.Quantity(par)
        self.value = par.value
        self.unit = str(par.unit)

    def __repr__(self):
        ss = 'Parameter(name={name!r}, value={value!r}, unit={unit!r}, '
        ss += 'min={min!r}, max={max!r}, frozen={frozen!r})'
        return ss.format(**self.to_dict())

    def to_dict(self):
        return dict(
            name=self.name,
            value=self.value,
            unit=self.unit,
            min=self.min,
            max=self.max,
            frozen=self.frozen,
        )


class ParameterList(object):
    """List of `~gammapy.spectrum.models.Parameter`

    Holds covariance matrix

    Parameters
    ----------
    parameters : list of `Parameter`
        List of parameters
    covariance : `~numpy.ndarray`
        Parameters covariance matrix.
        Order of values as specified by `parameters`.
    """

    def __init__(self, parameters, covariance=None):
        self.parameters = parameters
        self.covariance = covariance

    def __str__(self):
        ss = self.__class__.__name__
        for par in self.parameters:
            ss += '\n{}'.format(par)
        ss += '\n\nCovariance: \n{}'.format(self.covariance)
        return ss

    def _get_idx(self, val):
        """Convert parameter name or index to index"""
        if isinstance(val, six.string_types):
            for idx, par in enumerate(self.parameters):
                if val == par.name:
                    return idx
            raise IndexError('Parameter {} not found for : {}'.format(val, self))

        else:
            return val

    def __getitem__(self, name):
        """Access parameter by name or index"""
        idx = self._get_idx(name)
        return self.parameters[idx]

    def to_dict(self):
        retval = dict(parameters=list(), covariance=None)
        for par in self.parameters:
            retval['parameters'].append(par.to_dict())
        if self.covariance is not None:
            retval['covariance'] = self.covariance.tolist()
        return retval

    def to_list_of_dict(self):
        result = []
        for parameter in self.parameters:
            vals = parameter.to_dict()
            if self.covariance is None:
                vals['error'] = np.nan
            else:
                vals['error'] = self.error(parameter.name)
            result.append(vals)
        return result

    def to_table(self):
        """
        Serialize parameter list into `~astropy.table.Table`
        """
        names = ['name', 'value', 'error', 'unit', 'min', 'max', 'frozen']
        formats = {'value': '.3e',
                   'error': '.3e',
                   'min': '.3e',
                   'max': '.3e'}
        table = Table(self.to_list_of_dict(), names=names)

        for name in formats:
            table[name].format = formats[name]
        return table

    @classmethod
    def from_dict(cls, val):
        pars = list()
        for par in val['parameters']:
            pars.append(Parameter(name=par['name'],
                                  value=float(par['value']),
                                  unit=par['unit'],
                                  min=float(par['min']),
                                  max=float(par['max']),
                                  frozen=par['frozen']))
        try:
            covariance = np.array(val['covariance'])
        except KeyError:
            covariance = None

        return cls(parameters=pars, covariance=covariance)

    # TODO: this is a temporary solution until we have a better way
    # to handle covariance matrices via a class
    def covariance_to_table(self):
        """
        Serialize parameter covariance into `~astropy.table.Table`
        """
        t = Table(self.covariance, names=self.names)[self.free]
        for name in t.colnames:
            t[name].format = '.3'

        col = Column(name='name/name', data=self.names)
        t.add_column(col, index=0)

        rows = [row for row in t if row['name/name'] in self.free]
        return vstack(rows)

    @property
    def names(self):
        """List of parameter names"""
        return [par.name for par in self.parameters]

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
            raise ValueError('Covariance matrix not set.')

        upars = {}
        for par, upar in zip(self.parameters, uarray):
            upars[par.name] = upar
        return upars

    @property
    def free(self):
        """
        Return list of free parameters names.
        """
        free_pars = [par.name for par in self.parameters if not par.frozen]
        return free_pars

    @property
    def frozen(self):
        """
        Return list of frozen parameters names.
        """
        frozen_pars = [par.name for par in self.parameters if par.frozen]
        return frozen_pars

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
            raise ValueError('Covariance matrix not set.')

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

    def _init_covar(self):
        if self.covariance is None:
            shape = (len(self.parameters), len(self.parameters))
            self.covariance = np.zeros(shape)

    def copy(self):
        """A deep copy"""
        return copy.deepcopy(self)

    def update_values_from_tuple(self, values):
        """Update parameter values from a tuple of values."""
        for value, parameter in zip(values, self.parameters):
            parameter.value = value

    def scale_parameter(self, parname, scale):
        """Scale parameter

        This method updates the ``scale`` of a given parameter and makes sure
        that the covariance matrix (if set) is scaled accordingly

        Parameters
        ----------
        parname : str, int
            Parameter name or index
        scale : float 
            Parameter scale
        """
        idx = self._get_idx(parname)
        par = self[idx]
        ratio = scale / par.scale
        par.scale = scale
        par.factor = par.factor / ratio

        if self.covariance is not None:
            self.covariance[idx,] /= scale
            self.covariance[:,idx] /= scale

    def scale_parameters_to_unity(self):
        """Scale all parameters

        This methods scales all parameters such that the factor is of order
        unity
        """
        for par in self.parameters:
            temp = int(np.log10(par.value))
            scale = 10 ** temp
            print(par.name, scale)
            self.scale_parameter(par.name, scale)
