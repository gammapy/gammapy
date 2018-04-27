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
import astropy.units as u
import gammapy.image.models as spatial
import gammapy.spectrum.models as spectral

__all__ = [
    'UnknownModelError',
    'source_library_to_xml',
    'xml_to_source_library',
]


class UnknownModelError(ValueError):
    """
    Error when encountering unknown model types.
    """

class UnknownParameterError(ValueError):
    """
    Error when encountering unknown model types.
    """

def source_library_to_xml(skymodels):
    """
    Convert `~gammapy.cube.models.SkyModelList` to XML
    """

def xml_to_source_library(xml):
    """
    Convert XML to `~gammapy.cube.models.SkyModelList`
    """
    full_dict = xmltodict.parse(xml)
    skymodels = list()
    for xml_skymodel in full_dict['source_library']['source']:
        skymodels.append(xml_to_skymodel(xml_skymodel))
    return SourceLibrary(skymodels)


def xml_to_skymodel(xml):
    """
    Convert XML to `~gammapy.cube.models.SkyModel`
    """
    # TODO: type_ is not used anywhere
    type_ = xml['@type']

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
    parameters = xml_to_parameter_list(xml['parameter'], which)

    try:
        model = model_registry[which][type_]
    except KeyError:
        msg = "{} model '{}' not registered"
        raise UnknownModelError(msg.format(which, type_))

    if type_ == 'MapCubeFunction':
        filename = xml['@file']
        map_ = Map.read(filename)
        model = model(map=map_, norm=parameters['norm'].value)
    elif type_ == 'FileFunction':
        filename = xml['@file']
        model = model.read_fermi_isotropic_model(filename)
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


# TODO: MapCubeFunction does not have a good equivalent yet
model_registry = dict(spatial=dict(), spectral=dict())
model_registry['spatial']['SkyDirFunction'] = spatial.SkyPointSource
model_registry['spatial']['MapCubeFunction'] = spatial.SkyDiffuseMap
model_registry['spatial']['ConstantValue'] = spatial.SkyDiffuseConstant
model_registry['spectral']['PowerLaw'] = spectral.PowerLaw
model_registry['spectral']['FileFunction'] = spectral.TableModel


def xml_to_parameter_list(xml, which):
    """
    Convert XML to `~gammapy.utils.modeling.ParameterList`

    TODO: Introduce scale argument to `~gammapy.utils.modeling.Parameter`.
    """
    parameters = list()
    for par in np.atleast_1d(xml):
        try:
            name, unit = parname_registry[which][par['@name']]
        except KeyError:
            msg = "{} parameter '{}' not registered"
            raise UnknownParameterError(msg.format(which, par['@name']))

        parameters.append(Parameter(
            name = name,
            value = float(par['@value']) * float(par['@scale']),
            unit = unit,
            parmin = float(par['@min']),
            parmax = float(par['@max']),
            frozen = bool(1 - int(par['@free']))
        ))
    return ParameterList(parameters)


parname_registry = dict(spatial=dict(), spectral=dict())
parname_registry['spatial']['RA'] = 'lon_0', 'deg'
parname_registry['spatial']['DEC'] = 'lat_0', 'deg'
parname_registry['spatial']['Normalization'] = 'norm', ''
parname_registry['spatial']['Value'] = 'value', 'MeV cm-2 s-1'
parname_registry['spectral']['Prefactor'] = 'amplitude', 'MeV cm-2 s-1'
parname_registry['spectral']['Index'] = 'index', ''
parname_registry['spectral']['Scale'] = 'reference', 'MeV'
parname_registry['spectral']['Normalization'] = 'scale', ''
