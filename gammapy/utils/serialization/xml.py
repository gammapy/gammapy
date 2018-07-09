# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Model classes to generate XML.

For XML model format definitions, see here:

* http://cta.irap.omp.eu/ctools/user_manual/getting_started/models.html#spectral-model-components
* http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ...extern import xmltodict
from ...cube.models import SourceLibrary, SkyModel
from ..modeling import Parameter, ParameterList
from ...maps import Map
import numpy as np
import logging
import astropy.units as u
import gammapy.image.models as spatial
import gammapy.spectrum.models as spectral


log = logging.getLogger(__name__) 


__all__ = [
    'UnknownModelError',
    'UnknownParameterError',
    'xml_to_source_library',
    'xml_to_skymodel',
    'xml_to_model',
    'xml_to_parameter_list',
    'source_library_to_xml',
]


# TODO: Move to a separate file
model_registry = {
    'spectral':{
        'PowerLaw':{
            'model':spectral.PowerLaw,
            'parameters':{
                'Prefactor': ['amplitude', 'cm-2 s-1 MeV-1'],
                'Index': ['index', ''],
                'Scale': ['reference', 'MeV'],
            }
        }
    },
    'spatial':{
        'SkyDirFunction':{
            'model':spatial.SkyPointSource,
            'parameters':{
                'RA': ['lon_0', 'deg'],
                'DEC': ['lat_0', 'deg']
            }
        }
    }
}



class UnknownModelError(ValueError):
    """
    Error when encountering unknown model types.
    """


class UnknownParameterError(ValueError):
    """
    Error when encountering unknown model types.
    """


def xml_to_source_library(xml):
    """
    Convert XML to `~gammapy.cube.models.SkyModelList`
    """
    full_dict = xmltodict.parse(xml)
    skymodels = list()
    source_list = np.atleast_1d(full_dict['source_library']['source'])
    for xml_skymodel in source_list:
        skymodel = xml_to_skymodel(xml_skymodel)
        if skymodel is not None:
            skymodels.append(skymodel)
    return SourceLibrary(skymodels)


def xml_to_skymodel(xml):
    """
    Convert XML to `~gammapy.cube.models.SkyModel`
    """
    type_ = xml['@type']
    # TODO: Support ctools radial acceptance
    if type_ == 'RadialAcceptance':
        log.warn("Radial acceptance models are not supported")
        return None

    name = xml['@name']
    spatial = xml_to_model(xml['spatialModel'], 'spatial')
    spectral = xml_to_model(xml['spectrum'], 'spectral')
    return SkyModel(spatial_model=spatial, spectral_model=spectral, name=name)


def xml_to_model(xml, which):
    """
    Convert XML to `~gammapy.image.models.SkySpatialModel` or
    `~gammapy.spectrum.models.SpectralModel`
    """
    type_ = xml['@type']

    try:
        model = model_registry[which][type_]['model']
    except KeyError:
        msg = "{} model '{}' not registered"
        raise UnknownModelError(msg.format(which, type_))
    
    parameters = xml_to_parameter_list(xml['parameter'], which, type_)

    if type_ == 'MapCubeFunction':
        filename = xml['@file']
        map_ = Map.read(filename)
        model = model(map=map_, norm=-1, meta=dict(filename=filename))
        model.parameters = parameters
    elif type_ == 'FileFunction':
        filename = xml['@file']
        model = model.read_fermi_isotropic_model(filename,
                                                 meta=dict(filename=filename))
    else:
        # TODO: The new model API should support this, see issue #1398
        # >>> return model(parameters)
        # The following is a dirty workaround
        kwargs = dict()
        for par in parameters.parameters:
            kwargs[par.name] = -1 * u.Unit(par.unit)
        model = model(**kwargs)
        model.parameters = parameters
    return model


def xml_to_parameter_list(xml, which, type_):
    """
    Convert XML to `~gammapy.utils.modeling.ParameterList`

    TODO: Introduce scale argument to `~gammapy.utils.modeling.Parameter`.
    """
    parameters = list()
    for par in np.atleast_1d(xml):
        try:
            name, unit = model_registry[which][type_]['parameters'][par['@name']]
        except KeyError:
            msg = "Parameter '{}' not registered for {} model {}"
            raise UnknownParameterError(msg.format(par['@name'], which, type_))
        
        parameters.append(Parameter(
            name=name,
            value=float(par['@value']) * float(par['@scale']),
            unit=unit,
            parmin=float(par['@min']),
            parmax=float(par['@max']),
            frozen=bool(1 - int(par['@free']))
        ))
    return ParameterList(parameters)


def source_library_to_xml(sourcelib):
    """
    Convert `~gammapy.cube.models.SourceLibrary` to XML
    """
    xml = '<?xml version="1.0" encoding="utf-8"?>\n'
    xml += '<source_library title="source library">\n'
    for skymodel in sourcelib.skymodels:
        xml += skymodel_to_xml(skymodel)
    xml += '</source_library>'

    return xml


def skymodel_to_xml(skymodel):
    """
    Convert `~gammapy.cube.models.SkyModel` to XML
    """
    if 'Diffuse' in str(skymodel):
        type_ = 'DiffuseSource'
    else:
        type_ = 'PointSource'

    indent = 4 * ' '
    xml = indent + '<source name="{}" type="{}">\n'.format(skymodel.name, type_)
    xml += model_to_xml(skymodel.spectral_model, 'spectral')
    xml += model_to_xml(skymodel.spatial_model, 'spatial')
    xml += indent + '</source>\n'

    return xml


def model_to_xml(model, which):
    """
    Convert `~gammapy.image.models.SkySpatialModel` or
    `~gammapy.spectrum.models.SpectralModel` to XML
    """
    tag = 'spatialModel' if which == 'spatial' else 'spectrum'

    model_found = False
    for xml_type, type_ in model_registry[which].items():
        if isinstance(model, type_):
            model_found = True
            break

    if not model_found:
        msg = "{} model {} not in registry".format(which, model)
        raise UnknownModelError(msg)
    
    indent = 8 * ' '
    xml = indent + '<{} '.format(tag)
    if xml_type in ['MapCubeFunction', 'FileFunction']:
        xml += 'file="{}" '.format(model.meta['filename'])
    xml += 'type="{}">\n'.format(xml_type)
    xml += parameter_list_to_xml(model.parameters, which)
    xml += indent + '</{}>\n'.format(tag)
    return xml


def parameter_list_to_xml(parameters, which):
    """
    Convert `~gammapy.utils.modeling.ParameterList` to XML
    """
    indent = 12 * ' '
    xml = ''
    val = '<parameter free="{}" max="{}" min="{}" name="{}" scale="1.0" value="{}">'
    val += '</parameter>'
    for par in parameters.parameters:
        par_found = False
        for xml_par, (name, unit) in parname_registry[which].items():
            if par.name == name:
                par_found = True
                break

        if not par_found:
            msg = "{} parameter {} not in registry".format(which, par.name)
            raise UnknownParameterError(msg)

        xml += indent
        xml += val.format(int(not par.frozen),
                          par.parmax,
                          par.parmin,
                          xml_par,
                          par.quantity.to(unit).value)
        xml += '\n'

    return xml
