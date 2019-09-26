# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model classes to generate XML.

For XML model format definitions, see here:

* http://cta.irap.omp.eu/ctools/user_manual/getting_started/models.html#spectral-model-components
* http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html
"""
import logging
import numpy as np
import astropy.units as u
from gammapy.extern import xmltodict
from gammapy.maps import Map
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import SkyModel, SkyModels, spatial, spectral

log = logging.getLogger(__name__)

__all__ = [
    "UnknownModelError",
    "UnknownParameterError",
    "xml_to_sky_models",
    "xml_to_skymodel",
    "xml_to_model",
    "xml_to_parameters",
    "sky_models_to_xml",
]

# TODO: Move to a separate file ?
model_registry = {
    "spectral": {
        "PowerLawSpectralModel": {
            "model": spectral.PowerLawSpectralModel,
            "parameters": {
                "Prefactor": ["amplitude", "cm-2 s-1 MeV-1"],
                "Index": ["index", ""],
                "Scale": ["reference", "MeV"],
                "PivotEnergy": ["reference", "MeV"],
            },
        },
        "ExpCutoffPowerLawSpectralModel": {
            "model": spectral.ExpCutoffPowerLawSpectralModel,
            "parameters": {
                "Prefactor": ["amplitude", "cm-2 s-1 MeV-1"],
                "Index": ["index", ""],
                "Scale": ["reference", "MeV"],
                "PivotEnergy": ["reference", "MeV"],
                "CutoffEnergy": [
                    "lambda_",
                    "MeV",
                ],  # parameter lambda_=1/Ecut will be inverted on model creation
            },
        },
        "ConstantValue": {
            "model": spectral.ConstantSpectralModel,
            "parameters": {
                "Value": ["const", "cm-2 s-1 MeV-1"],
                "Normalization": ["const", "cm-2 s-1 MeV-1"],
            },
        },
        # TODO: FileFunction is not working
        "FileFunction": {
            "model": spectral.TableModel,
            "parameters": {
                "Value": ["const", "cm-2 s-1 MeV-1"],
                "Normalization": ["const", "cm-2 s-1 MeV-1"],
            },
        },
    },
    "spatial": {
        "PointSource": {
            "model": spatial.PointSpatialModel,
            "parameters": {
                "RA": ["lon_0", "deg"],
                "DEC": ["lat_0", "deg"],
                "GLON": ["lon_0", "deg"],
                "GLAT": ["lat_0", "deg"],
            },
        },
        "RadialGaussian": {
            "model": spatial.GaussianSpatialModel,
            "parameters": {
                "RA": ["lon_0", "deg"],
                "DEC": ["lat_0", "deg"],
                "GLON": ["lon_0", "deg"],
                "GLAT": ["lat_0", "deg"],
                "Sigma": ["sigma", "deg"],
            },
        },
        "RadialDisk": {
            "model": spatial.DiskSpatialModel,
            "parameters": {
                "RA": ["lon_0", "deg"],
                "DEC": ["lat_0", "deg"],
                "GLON": ["lon_0", "deg"],
                "GLAT": ["lat_0", "deg"],
                "Radius": ["r_0", "deg"],
            },
        },
        "RadialShell": {
            "model": spatial.ShellSpatialModel,
            "parameters": {
                "RA": ["lon_0", "deg"],
                "DEC": ["lat_0", "deg"],
                "GLON": ["lon_0", "deg"],
                "GLAT": ["lat_0", "deg"],
                "Radius": ["radius", "deg"],
                "Width": ["width", "deg"],
            },
        },
        "DiffuseMap": {
            "model": spatial.TemplateSpatialModel,
            "parameters": {
                "Prefactor": ["norm", ""],
                "Normalization": ["norm", ""],
                "Value": ["norm", ""],
            },
        },
        "DiffuseIsotropic": {
            "model": spatial.ConstantSpatialModel,
            "parameters": {
                "Prefactor": ["value", ""],
                "Normalization": ["value", ""],
                "Value": ["value", ""],
            },
        },
    },
}
# For compatibility with the Fermi/LAT ScienceTools the model type PointSource can be replaced by SkyDirFunction.
model_registry["spatial"]["SkyDirFunction"] = model_registry["spatial"]["PointSource"]
model_registry["spatial"]["SpatialMap"] = model_registry["spatial"]["DiffuseMap"]
model_registry["spatial"]["DiffuseMapCube"] = model_registry["spatial"]["DiffuseMap"]
model_registry["spatial"]["MapCubeFunction"] = model_registry["spatial"]["DiffuseMap"]
model_registry["spatial"]["ConstantValue"] = model_registry["spatial"][
    "DiffuseIsotropic"
]
model_registry["spectral"]["Constant"] = model_registry["spectral"]["ConstantValue"]


class UnknownModelError(ValueError):
    """Error when encountering unknown models."""


class UnknownParameterError(ValueError):
    """Error when encountering unknown parameters."""


def xml_to_sky_models(xml):
    """
    Convert XML to `~gammapy.modeling.models.SkyModelList`
    """
    full_dict = xmltodict.parse(xml)
    skymodels = list()
    source_list = np.atleast_1d(full_dict["source_library"]["source"])
    for xml_skymodel in source_list:
        skymodel = xml_to_skymodel(xml_skymodel)
        if skymodel is not None:
            skymodels.append(skymodel)
    return SkyModels(skymodels)


def xml_to_skymodel(xml):
    """
    Convert XML to `~gammapy.modeling.models.SkyModel`
    """
    type_ = xml["@type"]
    # TODO: Support ctools radial acceptance
    if type_ == "RadialAcceptance":
        log.warning("Radial acceptance models are not supported")
        return None

    name = xml["@name"]
    spatial = xml_to_model(xml["spatialModel"], "spatial")
    spectral = xml_to_model(xml["spectrum"], "spectral")
    return SkyModel(spatial_model=spatial, spectral_model=spectral, name=name)


def xml_to_model(xml, which):
    """
    Convert XML to `~gammapy.modeling.models.SpatialModel` or
    `~gammapy.modeling.models.SpectralModel`
    """
    type_ = xml["@type"]

    try:
        model = model_registry[which][type_]["model"]
    except KeyError:
        raise UnknownModelError(f"{which} model {type_!r} not registered")

    parameters = xml_to_parameters(xml["parameter"], which, type_)

    if type_ in ["MapCubeFunction", "DiffuseMapCube", "DiffuseMap", "SpatialMap"]:
        filename = xml["@file"]
        map_ = Map.read(filename)
        model = model(map=map_, norm=-1, meta=dict(filename=filename))
        model.parameters = parameters
    elif type_ == "FileFunction":
        filename = xml["@file"]
        model = model.read_fermi_isotropic_model(filename, meta=dict(filename=filename))
    else:
        # TODO: The new model API should support this, see issue #1398
        # >>> return model(parameters)
        # The following is a dirty workaround
        kwargs = dict()
        for par in parameters.parameters:
            kwargs[par.name] = -1 * u.Unit(par.unit)
        model = model(**kwargs)
        model.parameters = parameters

        # Special case models for which the XML definition does not map one to
        # one to the gammapy model definition
        if type_ == "PowerLawSpectralModel":
            model.parameters["index"].value *= -1
            model.parameters["index"].min = np.nan
            model.parameters["index"].max = np.nan
        if type_ == "ExpCutoffPowerLawSpectralModel":
            model.parameters["lambda_"].value = 1 / model.parameters["lambda_"].value
            model.parameters["lambda_"].unit = (
                model.parameters["lambda_"].unit.to_string("fits") + "-1"
            )
            model.parameters["lambda_"].min = np.nan
            model.parameters["lambda_"].max = np.nan
            model.parameters["index"].value *= -1
            model.parameters["index"].min = np.nan
            model.parameters["index"].max = np.nan

    return model


def xml_to_parameters(xml, which, type_):
    """Convert XML to `~gammapy.utils.modeling.Parameters`."""
    parameters = []
    for par in np.atleast_1d(xml):
        try:
            name, unit = model_registry[which][type_]["parameters"][par["@name"]]
        except KeyError:
            raise UnknownParameterError(
                f"Parameter {par['@name']} not registered for {which} model {type_}"
            )

        factor = float(par["@value"])
        scale = float(par["@scale"])
        min_ = float(par.get("@min", "nan"))
        max_ = float(par.get("@max", "nan"))
        frozen = bool(1 - int(par["@free"]))

        parameters.append(
            Parameter(
                name=name,
                factor=factor,
                scale=scale,
                unit=unit,
                min=min_,
                max=max_,
                frozen=frozen,
            )
        )

    return Parameters(parameters)


def sky_models_to_xml(sourcelib):
    """
    Convert `~gammapy.modeling.models.SkyModels` to XML
    """
    xml = '<?xml version="1.0" encoding="utf-8"?>\n'
    xml += '<source_library title="source library">\n'
    for skymodel in sourcelib.skymodels:
        xml += skymodel_to_xml(skymodel)
    xml += "</source_library>"

    return xml


def skymodel_to_xml(skymodel):
    """
    Convert `~gammapy.modeling.models.SkyModel` to XML
    """
    if "Diffuse" in str(skymodel):
        type_ = "DiffuseSource"
    else:
        type_ = "PointSource"

    indent = 4 * " "
    xml = indent + f'<source name="{skymodel.name}" type="{type_}">\n'
    xml += model_to_xml(skymodel.spectral_model, "spectral")
    xml += model_to_xml(skymodel.spatial_model, "spatial")
    xml += indent + "</source>\n"

    return xml


def model_to_xml(model, which):
    """
    Convert `~gammapy.modeling.models.SpatialModel` or
    `~gammapy.modeling.models.SpectralModel` to XML
    """
    tag = "spatialModel" if which == "spatial" else "spectrum"

    model_found = False
    for xml_type, type_ in model_registry[which].items():
        if isinstance(model, type_):
            model_found = True
            break

    if not model_found:
        msg = f"{which} model {model} not in registry"
        raise UnknownModelError(msg)

    indent = 8 * " "
    xml = indent + f"<{tag} "
    if xml_type in ["MapCubeFunction", "FileFunction"]:
        xml += 'file="{}" '.format(model.meta["filename"])
    xml += f'type="{xml_type}">\n'
    xml += parameters_to_xml(model.parameters, which)
    xml += indent + f"</{tag}>\n"
    return xml


def parameters_to_xml(parameters, which):
    """Convert `~gammapy.utils.modeling.Parameters` to XML."""
    indent = 12 * " "
    xml = ""
    val = '<parameter free="{}" max="{}" min="{}" name="{}" scale="1.0" value="{}">'
    val += "</parameter>"
    for par in parameters.parameters:
        par_found = False
        for xml_par, (name, unit) in parname_registry[which].items():
            if par.name == name:
                par_found = True
                break

        if not par_found:
            msg = f"{which} parameter {par.name} not in registry"
            raise UnknownParameterError(msg)

        xml += indent
        free = int(not par.frozen)
        value = par.quantity.to_value(unit)
        xml += val.format(free, par.max, par.min, xml_par, value)
        xml += "\n"

    return xml
