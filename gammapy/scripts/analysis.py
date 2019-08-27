# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high-level interface API"""
import copy
import jsonschema
import logging
from ..utils.scripts import read_yaml
from astropy import units as u
from pathlib import Path

__all__ = ["Analysis"]

log = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent / "config"
SCHEMA_FILE = CONFIG_PATH / "schema.yaml"


class Analysis:
    """The Analysis class drives an analysis working session using the high-level interface API.
    It is initialized by default with a set of configuration parameters and values declared in
    an internal configuration schema YAML file, though you can also provide your own configuration
    file, as well as configuration parameters passed as a nested dictionary at the moment of
    instantiation. In that case these parameters will overwrite the values of those present in the
    configuration file.

    Parameters
    ----------
    configfile : string
        The name of a user defined configuration file.
    config : dict
        A nested dictionary with configuration parameters and values.

    Examples
    --------
    Here are different examples on how to create an `Analysis` session class:

    >>> from gammapy.scripts import Analysis
    >>> cfg = {"general": {"out_folder": "myfolder"}}
    >>> analysis = Analysis(configfile="myfile.yaml", config=cfg)
    >>> analysis = Analysis(config=cfg)
    >>> analysis = Analysis()
    """

    def __init__(self, configfile="", config=dict()):

        self.configuration = Config(configfile, config)
        self._set_logging()

    def _set_logging(self):
        """Set logging parameters for API."""
        logging.basicConfig(**self.configuration.settings["general"]["logging"])
        log.info(
            "Setting logging parameters ({}).".format(
                self.configuration.settings["general"]["logging"]["level"]
            )
        )


class Config:
    def __init__(self, configfile, config):

        self._default_settings = dict()
        self._file_settings = dict()
        self._command_settings = dict()
        self.settings = dict()
        self.validate()
        self._default_settings = copy.deepcopy(self.settings)

        if configfile:
            self._file_settings = read_yaml(configfile)
            self._update_settings(self._file_settings, self.settings)
        if len(config):
            self._command_settings = config
            self._update_settings(self._command_settings, self.settings)
        self.validate()

    def validate(self):
        """Validate config parameters against schema."""
        schema = read_yaml(SCHEMA_FILE)
        jsonschema.validate(self.settings, schema, _gp_validator)

    def _update_settings(self, source, target):
        for key, val in source.items():
            if key not in target:
                target[key] = {}
            if not isinstance(val, dict) or val == {}:
                target[key] = val
            else:
                self._update_settings(val, target[key])


def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]
    reserved = [
        "default",
        "const",
        "readOnly",
        "items",
        "uniqueItems",
        "definitions",
        "properties",
        "patternProperties",
    ]

    def set_defaults(validator, properties, instance, schema):
        for prop, sub_schema in properties.items():
            if prop not in reserved and "default" in sub_schema:
                instance.setdefault(prop, sub_schema["default"])
        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return jsonschema.validators.extend(validator_class, {"properties": set_defaults})


def _astropy_quantity(_, instance):
    """Check a number may also be an astropy quantity."""
    valid = jsonschema.Draft7Validator.TYPE_CHECKER.is_type(instance, "number")
    quantity = str(instance).split()
    if not valid and len(quantity) >= 2:
        value = str(instance).split()[0]
        unit = "".join(str(instance).split()[1:])
        try:
            valid = u.Quantity(float(value), unit).unit.physical_type != "dimensionless"
        except ValueError:
            log.error("{} is not a valid astropy quantity.".format(str(instance)))
    return valid


_type_checker = jsonschema.Draft7Validator.TYPE_CHECKER.redefine(
    "number", _astropy_quantity
)
_gp_units_validator = jsonschema.validators.extend(
    jsonschema.Draft7Validator, type_checker=_type_checker
)
_gp_validator = extend_with_default(_gp_units_validator)
