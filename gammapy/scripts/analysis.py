# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high-level interface API"""
import jsonschema
import logging
from ..utils.scripts import read_yaml
from astropy import units as u
from pathlib import Path

__all__ = ["Analysis"]

log = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent / "config"
CONFIG_FILE = CONFIG_PATH / "default.yaml"
SCHEMA_FILE = CONFIG_PATH / "schema.yaml"


class Analysis():
    """High-level interface analysis session class.

    The Analysis class drives an analysis working session using the high-level interface
    API. It is initialized by default with a set of configuration parameters and values
    declared in an internal configuration YAML file.

    Parameters
    ----------
    configfile : string
        The name of a user defined configuration file.
    """

    def __init__(self, configfile=CONFIG_FILE, **kwargs):
        self.config = dict()
        if configfile:
            self.configfile = configfile
            self.config = read_yaml(self.configfile)
        if len(kwargs):
            self.config.update(**kwargs)
        self._set_logging()

    def validate_config(self):
        """Validate config parameters against schema."""
        schema = read_yaml(SCHEMA_FILE)
        try:
            jsonschema.validate(self.config, schema, _gp_validator)
        except jsonschema.exceptions.ValidationError as ex:
            log.error("Error when validating configuration parameters against schema.")
            log.error("Parameter: {}".format(ex.schema_path[-2]))
            log.error(ex.message)
            raise ex

    def _set_logging(self):
        """Set logging parameters for API."""
        if "global" in self.config and "logging" in self.config["global"]:
            logging.basicConfig(**self.config["global"]["logging"])
            log.info("Setting logging parameters ({}).".format(self.config["global"]["logging"]["level"]))


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
_gp_validator = jsonschema.validators.extend(
    jsonschema.Draft7Validator, type_checker=_type_checker
)
