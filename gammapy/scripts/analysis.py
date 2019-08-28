# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high-level interface API"""
import copy
import logging
from pathlib import Path
from astropy.coordinates import Angle
from astropy import units as u
import jsonschema
from gammapy.data import DataStore, Observations, ObservationTable
from gammapy.utils.scripts import make_path, read_yaml


__all__ = ["Analysis", "Config"]

log = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent / "config"
SCHEMA_FILE = CONFIG_PATH / "schema.yaml"


class Analysis:
    """Config-driven high-level analysis interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal configuration schema YAML file, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    Parameters
    ----------
    config : dict
        A nested dictionary with configuration parameters and values.

    Examples
    --------
    Here are different examples on how to create an `Analysis` session class:

    >>> from gammapy.scripts import Analysis
    >>> settings = {"general": {"outdir": "myfolder"}}
    >>> analysis = Analysis(settings)
    >>> analysis = Analysis()
    """

    def __init__(self, config=None):
        self._config = Config(config)
        self._set_logging()
        self.datastore = DataStore()
        self.observations = Observations()

    @property
    def config(self):
        """Analysis configuration (`Config`)"""
        return self._config

    @property
    def settings(self):
        """Configuration settings for the analysis session."""
        return self.config.settings

    def get_observations(self):
        """Fetch observations from the data store according to criteria defined in the configuration."""
        config_ds = make_path(self.settings["observations"]["data_store"])
        if config_ds.is_file():
            self.datastore = DataStore().from_file(config_ds)
        elif config_ds.is_dir():
            self.datastore = DataStore().from_dir(config_ds)
        else:
            log.error("Datastore {} not found.".format(config_ds))
            return False

        ids = set()
        selection = dict()
        for criteria in self.settings["observations"]["filter"]:
            selected_obs = ObservationTable()

            selection["type"] = criteria["filter_type"]
            if "inverted" in criteria and criteria["inverted"]:
                selection["inverted"] = True
            if selection["type"] == "sky_circle":
                selection["frame"] = criteria["frame"]
                selection["lon"] = Angle(criteria["lon"])
                selection["lat"] = Angle(criteria["lat"])
                selection["radius"] = Angle(criteria["radius"])
                selection["border"] = Angle(criteria["border"])
            if selection["type"] == "par_box":
                selection["variable"] = criteria["variable"]
                selection["value_range"] = criteria["value_range"]
            if selection["type"] == "angle_box":
                selection["type"] = "par_box"
                selection["variable"] = criteria["variable"]
                selection["value_range"] = Angle(criteria["value_range"])

            if selection["type"] != "ids" and selection["type"] != "all":
                selected_obs = self.datastore.obs_table.select_observations(selection)
            if selection["type"] == "ids":
                obs_list = self.datastore.get_observations(criteria["obs_ids"])
                selected_obs["OBS_ID"] = [obs.obs_id for obs in obs_list.list]
            if selection["type"] == "all":
                obs_list = self.datastore.get_observations()
                selected_obs["OBS_ID"] = [obs.obs_id for obs in obs_list.list]
            if len(selected_obs):
                if "exclude" in criteria and criteria["exclude"]:
                    ids.difference_update(selected_obs["OBS_ID"].tolist())
                else:
                    ids.update(selected_obs["OBS_ID"].tolist())
        if len(ids):
            self.observations = self.datastore.get_observations(ids)
        else:
            self.observations = Observations()

    def _set_logging(self):
        """Set logging parameters for API."""
        logging.basicConfig(**self.settings["general"]["logging"])
        log.info(
            "Setting logging parameters ({}).".format(
                self.settings["general"]["logging"]["level"]
            )
        )


class Config:
    """Analysis configuration.

    Parameters
    ----------
    config : dict
        Configuration parameters
    """

    def __init__(self, config=None):
        self._default_settings = {}
        self._command_settings = {}
        self.settings = {}

        # fill settings with default values
        self.validate()
        self._default_settings = copy.deepcopy(self.settings)

        # overwrite with config provided by the user
        if config is None:
            config = {}
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
