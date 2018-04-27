# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Model parameter handling
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import numpy as np
import copy
from ..extern import six
from astropy import units as u
from astropy.table import Table, Column, vstack
from ..extern import xmltodict
from .scripts import make_path

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
        Name of the parameter
    value : float or `~astropy.units.Quantity`
        Value of the parameter
    unit : str, optional
        Unit of the parameter (if value is given as float)
    parmin : float, optional
        Parameter value minimum. Used as minimum boundary value
        in a model fit
    parmax : float, optional
        Parameter value maximum. Used as minimum boundary value
        in a model fit
    frozen : bool, optional
        Whether the parameter is free to be varied in a model fit
    """

    def __init__(self, name, value, unit='', parmin=None, parmax=None, frozen=False):
        self.name = name

        if isinstance(value, u.Quantity) or isinstance(value, six.string_types):
            self.quantity = value
        else:
            self.value = value
            self.unit = unit

        self.parmin = parmin or np.nan
        self.parmax = parmax or np.nan
        self.frozen = frozen

    @property
    def quantity(self):
        retval = self.value * u.Unit(self.unit)
        return retval

    @quantity.setter
    def quantity(self, par):
        par = u.Quantity(par)
        self.value = par.value
        self.unit = str(par.unit)

    def __str__(self):
        ss = 'Parameter(name={name!r}, value={value!r}, unit={unit!r}, '
        ss += 'min={parmin!r}, max={parmax!r}, frozen={frozen!r})'

        return ss.format(**self.__dict__)

    def to_dict(self):
        return dict(name=self.name,
                    value=float(self.value),
                    unit=str(self.unit),
                    frozen=self.frozen,
                    min=float(self.parmin),
                    max=float(self.parmax))

    def to_sherpa(self, modelname='Default'):
        """Convert to sherpa parameter"""
        from sherpa.models import Parameter

        parmin = np.finfo(np.float32).min if np.isnan(self.parmin) else self.parmin
        parmax = np.finfo(np.float32).max if np.isnan(self.parmax) else self.parmax

        par = Parameter(modelname=modelname, name=self.name,
                        val=self.value, units=self.unit,
                        min=parmin, max=parmax,
                        frozen=self.frozen)
        return par


class ParameterList(object):
    """List of `~gammapy.spectrum.models.Parameter`

    Holds covariance matrix

    Parameters
    ----------
    parameters : list of `Parameter`
        List of parameters
    covariance : `~numpy.ndarray`
        Parameters covariance matrix. Order of values as specified by
        `parameters`.
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

    def __getitem__(self, name):
        """Access parameter by name"""
        for par in self.parameters:
            if name == par.name:
                return par

        raise IndexError('Parameter {} not found for : {}'.format(name, self))

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
                                  parmin=float(par['min']),
                                  parmax=float(par['max']),
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

    # TODO: this is a temporary solution until we have a better way
    # to handle covariance matrices via a class
    def set_parameter_errors(self, errors):
        """
        Set uncorrelated parameters errors.

        Parameters
        ----------
        errors : dict of `~astropy.units.Quantity`
            Dict of parameter errors.
        """
        values = []
        for par in self.parameters:
            quantity = errors.get(par.name, 0 * u.Unit(par.unit))
            values.append(u.Quantity(quantity, par.unit).value)
        self.covariance = np.diag(values) ** 2

    # TODO: this is a temporary solution until we have a better way
    # to handle covariance matrices via a class
    def set_parameter_covariance(self, covariance, covar_axis):
        """
        Set full correlated parameters errors.

        Parameters
        ----------
        covariance : array-like
            Covariance matrix
        covar_axis : list
            List of strings defining the parameter order in covariance
        """
        shape = (len(self.parameters), len(self.parameters))
        covariance_new = np.zeros(shape)
        idx_lookup = dict([(par.name, idx) for idx, par in enumerate(self.parameters)])

        # TODO: make use of covariance matrix symmetry
        for i, par in enumerate(covar_axis):
            i_new = idx_lookup[par]
            for j, par_other in enumerate(covar_axis):
                j_new = idx_lookup[par_other]
                covariance_new[i_new, j_new] = covariance[i, j]

        self.covariance = covariance_new

    # TODO: this is a temporary solution until we have a better way
    # to handle covariance matrices via a class
    def error(self, parname):
        """
        Return error on a given parameter

        Parameters
        ----------
        parname : str
            Parameter
        """
        if self.covariance is None:
            raise ValueError('Covariance matrix not set.')

        for i, parameter in enumerate(self.parameters):
            if parameter.name == parname:
                return np.sqrt(self.covariance[i, i])
        raise ValueError('Could not find parameter {}'.format(parname))

    def copy(self):
        """A deep copy"""
        return copy.deepcopy(self)
