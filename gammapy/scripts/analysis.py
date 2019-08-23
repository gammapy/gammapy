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


class Singleton(type):
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__call__(*args, **kwargs)
        return cls.__instance


class Analysis(metaclass=Singleton):
    """High-level interface analysis session class.

    The Analysis class drives an analysis working session with Gammapy using the
    high-level interface API. It is initialized by default with a set of configuration
    parameters and values declared in an internal configuration YAML file, though
    it also accepts a the name of user defined configuration file when instantiated.

    Parameters
    ----------
    configfile : string
        The name of a user defined configuration file.
    """

    __slots__ = ["configfile", "_params"]

    def __init__(self, configfile=CONFIG_PATH / "default.yaml"):
        self.configfile = configfile
        self._params = read_yaml(self.configfile)
        self._validate_schema()
        self._set_log()

    def reset(self):
        """Reset configuration parameters to initial state."""
        self.__init__(self.configfile)

    def list_config(self):
        """Returns list of configuration parameters."""
        return self._params

    def set_config(self, par, val):
        """Sets and removes configuration parameters.
        Examples
        --------
        >>> from gammapy.scripts import Analysis
        >>> analysis = Analysis()
        >>> analysis.set_config("global.logging.level", "info")
        >>> analysis.set_config("global.test", "test")
        >>> analysis.set_config("global.test", None)
        """
        branch = par.split(".")
        par_str = "self._params"
        if isinstance(val, str):
            val = "'{}'".format(val)
        for leaf in branch[:-1]:
            par_str += "['{}']".format(leaf)
        if val is None:
            par_str = "{}.pop('{}', None)".format(par_str, branch[-1])
        else:
            par_str += ".update({}={})".format(branch[-1], val)
        backup_params = copy.deepcopy(self._params)

        try:
            eval(par_str)
            self._validate_schema()
        except Exception as ex:
            log.error("Error when setting config param.")
            log.error(par_str)
            self._params = backup_params
            raise ex

    def _set_log(self):
        """Set logging parameters for API."""
        cfg = self._params["global"]
        if "logging" in cfg:
            log_params = dict()
            for par, val in cfg["logging"].items():
                log_params[par] = val
            log_params["level"] = cfg["logging"]["level"].upper()
            logging.basicConfig(**log_params)
            log.info("Setting logging parameters ({}).".format(log_params["level"]))

    def _validate_schema(self):
        """Validate config parameters against schema."""
        schema = read_yaml(SCHEMA_FILE)
        try:
            jsonschema.validate(self._params, schema, _gp_validator)
        except jsonschema.exceptions.ValidationError as ex:
            log.error("Error when validating configuration parameters against schema.")
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
            log.error("{} is not a valid astropy quantity.".format(str(instance)))
    return valid


_type_checker = jsonschema.Draft7Validator.TYPE_CHECKER.redefine(
    "number", _astropy_quantity
)
_gp_validator = jsonschema.validators.extend(
    jsonschema.Draft7Validator, type_checker=_type_checker
)
