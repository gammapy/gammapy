# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Model classes to generate XML.

This is prototype code.

The goal was to be able to save gamma-cat in XML format
for the CTA data challenge GPS sky model.

TODO (needs a bit of experimentation / discussion / thought and a few days of coding):

* add repr to all classes
* integrate this the existing Gammapy model classes to make analysis possible.
* don't couple this to gamma-cat. Gamma-cat should change to create model classes that support XML I/O.
* sub-class Astropy Parameter and ParameterSet classes instead of starting from scratch?
* implement spatial and spectral mode registries instead of `if-elif` set on type to make SourceLibrary extensible.
* write test and docs
* Once modeling setup OK, ask new people to add missing models
  (see Gammalib, Fermi ST, naima, Sherpa, HESS)
  (it's one of the simplest and nicest things to get started with)

For XML model format definitions, see here:

* http://cta.irap.omp.eu/ctools/user_manual/getting_started/models.html#spectral-model-components
* http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import numpy as np
from ..extern import six
from astropy import units as u
from astropy.table import Table, Column, vstack
from ..extern import xmltodict
from .scripts import make_path

__all__ = [
    'Parameter',
    'ParameterList',
    'SourceLibrary',
    'SourceModel',

    'SpectralModel',
    'SpectralModelPowerLaw',
    'SpectralModelPowerLaw2',
    'SpectralModelExpCutoff',

    'SpatialModel',
    'SpatialModelPoint',
    'SpatialModelGauss',
    'SpatialModelShell',

    'UnknownModelError',
]


class UnknownModelError(ValueError):
    """
    Error when encountering unknown model types.
    """


class Parameter(object):
    """
    Class representing model parameters.

    Parameters
    ----------
    name : str
        Name of the parameter.
    value : float
        Value of the parameter.
    unit : str
        Unit of the parameter.
    parmin : float
        Parameter value minimum. Used as minimum boundary value
        in a model fit.
    parmax : float
        Parameter value maximum. Used as minimum boundary value
        in a model fit.
    frozen : bool
        Whether the parameter is free to be varied in a model fit.

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

    # TODO: I think this method is not very useful, because the same can be just
    # achieved with `Parameter(**data)`. Why duplicate?
    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data['name'],
            value=data['val'],
            unit=data['unit'],
        )

    @classmethod
    def from_dict_gammacat(cls, data):
        return cls(
            name=data['name'],
            value=float(data['value']),
            unit=data['unit'],
        )

    @classmethod
    def from_dict_xml(cls, data):
        unit = data.get('@unit', '')

        return cls(
            name=data['@name'],
            value=float(data['@value']),
            unit=unit,
        )

    def to_xml(self):
        return '        <parameter name="{name}" value="{value}" unit="{unit}"/>'.format(**self.__dict__)

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
    """List of `~gammapy.spectrum.models.Parameters`

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

    @classmethod
    def from_list_of_dict_gammacat(cls, data):
        return cls([Parameter.from_dict_gammacat(_) for _ in data])

    @classmethod
    def from_list_of_dict_xml(cls, data):
        return cls([Parameter.from_dict_xml(_) for _ in data])

    def to_xml(self):
        xml = [_.to_xml() for _ in self.parameters]
        return '\n'.join(xml)

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

    @classmethod
    def from_dict(cls, val):
        pars = list()
        for par in val['parameters']:
            pars.append(Parameter(name=par['name'], value=float(par['value']),
                                  unit=par['unit'], parmin=float(par['min']),
                                  parmax=float(par['max']),
                                  frozen=par['frozen']))
        try:
            covariance = np.array(val['covariance'])
        except KeyError:
            covariance = None

        return cls(parameters=pars, covariance=covariance)

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
            values.append(quantity.to(par.unit).value)
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


class SourceLibrary(object):
    def __init__(self, source_list):
        self.source_list = source_list

    @classmethod
    def read(cls, filename):
        path = make_path(filename)
        xml = path.read_text()
        return cls.from_xml(xml)

    @classmethod
    def from_xml(cls, xml):
        full_dict = xmltodict.parse(xml)
        sources_dict = full_dict['source_library']['source']
        return cls.from_list_of_dict(sources_dict)

    @classmethod
    def from_list_of_dict(cls, data):
        source_list = []
        for source_data in data:
            source_model = SourceModel.from_xml_dict(source_data)
            source_list.append(source_model)

        return cls(source_list=source_list)

    def to_xml(self, title='sources', header='\n'):
        sources_xml = []
        for source in self.source_list:
            sources_xml.append(source.to_xml())
        sources_xml = '\n'.join(sources_xml)

        fmt = '<?xml version="1.0" ?>\n'
        fmt += '{header}\n'
        fmt += '<source_library title="{title}">\n'
        fmt += '\n'
        fmt += '{sources_xml}\n'
        fmt += '</source_library>\n'

        return fmt.format(
            title=title,
            header=header,
            sources_xml=sources_xml,
        )


class SourceModel(object):
    """

    TODO: having "source_type" separate, but often inferred from the
    spatial model is weird. -> how to improve this?
    """

    def __init__(self, source_name, source_type,
                 spatial_model, spectral_model):
        self.source_name = source_name
        self.source_type = source_type
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model

    @classmethod
    def from_gammacat(cls, source):
        source_name = source.data['common_name']
        spectral_model = SpectralModel.from_gammacat(source)
        spatial_model = SpatialModel.from_gammacat(source)
        source_type = spatial_model.xml_source_type
        return cls(
            source_name=source_name,
            source_type=source_type,
            spatial_model=spatial_model,
            spectral_model=spectral_model,
        )

    @classmethod
    def from_xml_dict(cls, data):
        source_name = data['@name']
        source_type = data['@type']

        spectral_model = SpectralModel.from_xml_dict(data['spectrum'])

        if 'radialModel' in data:
            spatial_model = SpatialModel.from_xml_dict(data['radialModel'])
        elif 'spatialModel' in data:
            spatial_model = SpatialModel.from_xml_dict(data['spatialModel'])
        else:
            raise UnknownModelError('No spatial / radial model found for: {}'.format(data))

        return cls(
            source_name=source_name,
            source_type=source_type,
            spatial_model=spatial_model,
            spectral_model=spectral_model,
        )

    def to_xml(self):
        fmt = '<source name="{source_name}" type="{source_type}">\n'
        fmt += '    {spectrum_xml}\n'
        fmt += '    {spatial_xml}\n'
        fmt += '</source>\n'
        data = dict(
            source_name=self.source_name,
            source_type=self.source_type,
            spectrum_xml=self.spectral_model.to_xml(),
            spatial_xml=self.spatial_model.to_xml(),
        )
        return fmt.format(**data)


@six.add_metaclass(abc.ABCMeta)
class BaseModel(object):
    """
    Abstract base class to avoid code duplication between
    `SpectralModel` and `SpatialModel`.
    """

    @abc.abstractproperty
    def xml_type(self):
        """The XML type string"""
        pass

    @abc.abstractproperty
    def xml_types(self):
        """List of XML type strings (to support older versions of the format)"""
        pass

    def __init__(self, parameters):
        if isinstance(parameters, ParameterList):
            pass
        elif isinstance(parameters, list):
            parameters = ParameterList(parameters)
        else:
            raise ValueError('Need list of Parameter or ParameterList. '
                             'Invalid input: {}'.format(parameters))

        self.parameters = parameters


@six.add_metaclass(abc.ABCMeta)
class SpectralModel(BaseModel):
    """
    Spectral model abstract base class.
    """

    @classmethod
    def from_xml_dict(cls, data):
        model_type = data['@type']
        parameters = ParameterList.from_list_of_dict_xml(data['parameter'])

        if model_type == SpectralModelPowerLaw.xml_type:
            model = SpectralModelPowerLaw(parameters=parameters)
        elif model_type == SpectralModelPowerLaw2.xml_type:
            model = SpectralModelPowerLaw2(parameters=parameters)
        elif model_type == SpectralModelExpCutoff.xml_type:
            model = SpectralModelExpCutoff(parameters=parameters)
        else:
            raise UnknownModelError('Unknown spatial model: {}'.format(model_type))

        return model

    @classmethod
    def from_gammacat(cls, source):
        try:
            data = source.spectral_model.to_dict()
        except ValueError:
            from ..catalog.gammacat import NoDataAvailableError
            raise NoDataAvailableError(source)

        plist = ParameterList.from_list_of_dict_gammacat(data['parameters'])
        if data['name'] == 'PowerLaw':
            model = SpectralModelPowerLaw.from_plist(plist)
        elif data['name'] == 'PowerLaw2':
            model = SpectralModelPowerLaw2.from_plist(plist)
        elif data['name'] == 'ExponentialCutoffPowerLaw':
            model = SpectralModelExpCutoff.from_plist(plist)
        else:
            raise UnknownModelError('Unknown spectral model: {}'.format(data))

        return model

    def to_xml(self):
        return '<spectrum type="{}">\n{}\n    </spectrum>'.format(self.xml_type, self.parameters.to_xml())


class SpectralModelPowerLaw(SpectralModel):
    xml_type = 'PowerLaw'
    xml_types = [xml_type]

    @classmethod
    def from_plist(cls, plist):
        par = plist['amplitude']
        prefactor = Parameter(name='Prefactor', value=par.value, unit=par.unit)
        par = plist['index']
        index = Parameter(name='Index', value=-par.value, unit=par.unit)
        par = plist['reference']
        scale = Parameter(name='Scale', value=par.value, unit=par.unit)

        parameters = [prefactor, index, scale]
        return cls(parameters=parameters)


class SpectralModelPowerLaw2(SpectralModel):
    xml_type = 'PowerLaw2'
    xml_types = [xml_type]

    @classmethod
    def from_plist(cls, plist):
        par = plist['amplitude']
        integral = Parameter(name='Integral', value=par.value, unit=par.unit)
        par = plist['index']
        index = Parameter(name='Index', value=-par.value, unit=par.unit)
        par = plist['emin']
        lower_limit = Parameter(name='LowerLimit', value=par.value, unit=par.unit)
        par = plist['emax']
        upper_limit = Parameter(name='UpperLimit', value=par.value, unit=par.unit)

        parameters = [integral, index, lower_limit, upper_limit]
        return cls(parameters=parameters)


class SpectralModelExpCutoff(SpectralModel):
    xml_type = 'ExpCutoff'
    xml_types = [xml_type]

    @classmethod
    def from_plist(cls, plist):
        par = plist['amplitude']
        prefactor = Parameter(name='Prefactor', value=par.value, unit=par.unit)
        par = plist['index']
        index = Parameter(name='Index', value=par.value, unit=par.unit)
        par = plist['lambda_']
        cutoff = Parameter(name='Cutoff', value=1 / par.value, unit=par.unit)
        par = plist['reference']
        scale = Parameter(name='Scale', value=par.value, unit=par.unit)

        parameters = [prefactor, index, cutoff, scale]
        return cls(parameters=parameters)


@six.add_metaclass(abc.ABCMeta)
class SpatialModel(BaseModel):
    """
    Spatial model abstract base class
    """

    @classmethod
    def from_xml_dict(cls, data):
        model_type = data['@type']
        if model_type in SpatialModelPoint.xml_types:
            model = SpatialModelPoint(parameters=[
            ])
        elif model_type in SpatialModelGauss.xml_types:
            model = SpatialModelGauss(parameters=[
            ])
        elif model_type in SpatialModelShell.xml_types:
            model = SpatialModelShell(parameters=[
            ])
        else:
            raise UnknownModelError('Unknown spatial model: {}'.format(model_type))

        return model

    @classmethod
    def from_gammacat(cls, source):
        data = source.data

        if data['morph_type'] == 'point':
            model = SpatialModelPoint.from_gammacat(source)
        elif data['morph_type'] == 'gauss':
            model = SpatialModelGauss.from_gammacat(source)
        elif data['morph_type'] == 'shell':
            model = SpatialModelShell.from_gammacat(source)
        else:
            raise UnknownModelError('Unknown spatial model: {}'.format(source))

        return model

    def to_xml(self):
        return '<spatialModel type="{}">\n{}\n    </spatialModel>'.format(self.xml_type, self.parameters.to_xml())


class SpatialModelPoint(SpatialModel):
    xml_source_type = 'PointSource'
    xml_type = 'Point'
    xml_types = [xml_type]

    @classmethod
    def from_gammacat(cls, source):
        d = source.data
        glon = Parameter(name='GLON', value=d['glon'].value, unit=d['glon'].unit)
        glat = Parameter(name='GLAT', value=d['glat'].value, unit=d['glat'].unit)

        parameters = [glon, glat]
        return cls(parameters=parameters)


class SpatialModelGauss(SpatialModel):
    xml_source_type = 'ExtendedSource'
    xml_type = 'Gauss'
    xml_types = [xml_type, 'Gaussian']

    @classmethod
    def from_gammacat(cls, source):
        d = source.data
        glon = Parameter(name='GLON', value=d['glon'].value, unit=d['glon'].unit)
        glat = Parameter(name='GLAT', value=d['glat'].value, unit=d['glat'].unit)
        sigma = Parameter(name='Sigma', value=d['morph_sigma'].value, unit=d['morph_sigma'].unit)

        # TODO: fill `morph_sigma2` and `morph_pa` info

        parameters = [glon, glat, sigma]
        return cls(parameters=parameters)


class SpatialModelShell(SpatialModel):
    xml_source_type = 'ExtendedSource'
    xml_type = 'Shell'
    xml_types = [xml_type, 'RadialShell']

    @classmethod
    def from_gammacat(cls, source):
        d = source.data
        glon = Parameter(name='GLON', value=d['glon'].value, unit=d['glon'].unit)
        glat = Parameter(name='GLAT', value=d['glat'].value, unit=d['glat'].unit)
        radius = Parameter(name='Radius', value=d['morph_sigma'].value, unit=d['morph_sigma'].unit)
        width = Parameter(name='Width', value=0, unit='deg')

        parameters = [glon, glat, radius, width]
        return cls(parameters=parameters)
