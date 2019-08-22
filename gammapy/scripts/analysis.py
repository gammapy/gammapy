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
SCHEMA_FILE = CONFIG_PATH / "schema.yaml"


class Singleton(type):
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__call__(*args, **kwargs)
            return cls.__instance
        else:
            return cls.__instance


class Analysis(metaclass = Singleton):
    """High-level interface analysis session class."""

    __slots__ = ["configfile", "params"]

    def __init__(self, configfile=CONFIG_PATH/"default.yaml"):
        self.configfile = configfile
        self.params = read_yaml(self.configfile)

        self.validate_schema()
        self.set_log()

    def set_log(self):
        """Set logging parameters for API."""

        cfg = self.params['global']
        if "logging" in cfg:
            log_params = dict()
            for par, val in cfg['logging'].items():
                log_params[par] = val
            log_params['level'] = cfg['logging']['level'].upper()
            logging.basicConfig(**log_params)
            log.info("Setting logging parameters.")

    def validate_schema(self):
        """Validate config parameters against schema."""
        schema = read_yaml(SCHEMA_FILE)

        try:
            jsonschema.validate(self.params, schema, _gp_validator)

        except jsonschema.exceptions.ValidationError as ex:
            log.error('Error when validating configuration parameters against schema.')
            log.error("Parameter: {}".format(ex.schema_path[-2]))
            log.error(ex.message)
            raise ex


def _astropy_quantity(_, instance):
    """Check a number may also be an astropy quantity."""
    valid = jsonschema.Draft7Validator.TYPE_CHECKER.is_type(instance, "number")
    quantity = str(instance).split()

    if not valid and len(quantity) >= 2:
        value = str(instance).split()[0]
        unit = "".join(str(instance).split()[1:])
        try:
            valid = u.Quantity(float(value), unit).unit.physical_type != "dimensionless"
        except ValueError as ex:
            log.error('Error when validating configuration parameters against schema.')
            log.error("{} is not a valid astropy quantity.".format(str(instance)))
            raise ex
    return valid


_type_checker = jsonschema.Draft7Validator.TYPE_CHECKER.redefine("number", _astropy_quantity)
_gp_validator = jsonschema.validators.extend(jsonschema.Draft7Validator, type_checker=_type_checker)
