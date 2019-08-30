# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high-level interface API"""
import copy
import logging
import jsonschema
from pathlib import Path
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from gammapy.background import ReflectedRegionsBackgroundEstimator
from gammapy.data import DataStore, Observations, ObservationTable
from gammapy.maps import WcsGeom, WcsNDMap
from gammapy.spectrum import SpectrumExtraction
from gammapy.utils.scripts import make_path, read_yaml
from regions import CircleSkyRegion


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

        self.observations = None
        self.geom = None
        self.background_estimator = None
        self.extraction = None

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
        self.config.validate()

        datastore_path = make_path(self.settings["observations"]["datastore"])
        if datastore_path.is_file():
            datastore = DataStore().from_file(datastore_path)
        elif datastore_path.is_dir():
            datastore = DataStore().from_dir(datastore_path)
        else:
            raise FileNotFoundError("Datastore {} not found.".format(datastore_path))

        ids = set()
        selection = dict()
        for criteria in self.settings["observations"]["filters"]:
            selected_obs = ObservationTable()

            # -- TODO Handled by datastore.obs_table.select_observations
            # -
            selection["type"] = criteria["filter_type"]
            for key, val in criteria.items():
                if key in ["lon", "lat", "radius", "border"]:
                    val = Angle(val)
                selection[key] = val
            if selection["type"] == "angle_box":
                selection["type"] = "par_box"
                selection["value_range"] = Angle(criteria["value_range"])
            if selection["type"] == "sky_circle" or selection["type"].endswith("_box"):
                selected_obs = datastore.obs_table.select_observations(selection)
            if selection["type"] == "par_value":
                mask = (
                    datastore.obs_table[criteria["variable"]] == criteria["par_value"]
                )
                selected_obs = datastore.obs_table[mask]
            if selection["type"] == "ids":
                obs_list = datastore.get_observations(criteria["obs_ids"])
                selected_obs["OBS_ID"] = [obs.obs_id for obs in obs_list.list]
            if selection["type"] == "all":
                obs_list = datastore.get_observations()
                selected_obs["OBS_ID"] = [obs.obs_id for obs in obs_list.list]
            # --
            # -

            if len(selected_obs):
                if "exclude" in criteria and criteria["exclude"]:
                    ids.difference_update(selected_obs["OBS_ID"].tolist())
                else:
                    ids.update(selected_obs["OBS_ID"].tolist())
        self.observations = datastore.get_observations(ids)

    def reduce_data(self):
        """Produce reduced data sets."""
        self.config.validate()

        # create geometry
        self.geom = WcsGeom.create(
            skydir=tuple(self.settings["geometry"]["skydir"]),
            binsz=self.settings["geometry"]["binsz"],
            width=tuple(self.settings["geometry"]["width"]),
            coordsys=self.settings["geometry"]["coordsys"],
            proj=self.settings["geometry"]["proj"],
        )
        # axes=[energy_axis]

        if self.settings["reduction"]["data_reducer"] == "1d":
            self._spectrum_extraction()

    def _spectrum_extraction(self):
        """Run all steps for the spectrum extraction."""

        on = self.settings["reduction"]["background"]["on_region"]
        on_lon = Angle(on["center"][0])
        on_lat = Angle(on["center"][1])
        on_center = SkyCoord(on_lon, on_lat, frame=on["frame"])
        on_region = CircleSkyRegion(on_center, Angle(on["radius"]))
        background_pars = {"on_region": on_region}

        if "exclusion_region" in self.settings["reduction"]["background"]:
            exclusion = self.settings["reduction"]["background"]["exclusion_region"]
            exclusion_lon = Angle(exclusion["center"][0])
            exclusion_lat = Angle(exclusion["center"][1])
            exclusion_center = SkyCoord(
                exclusion_lon, exclusion_lat, frame=exclusion["frame"]
            )
            exclusion_region = CircleSkyRegion(
                exclusion_center, Angle(exclusion["radius"])
            )
            mask = self.geom.region_mask([exclusion_region], inside=True)
            exclusion_mask = WcsNDMap(geom=self.geom, data=mask)
            background_pars.update({"exclusion_mask": exclusion_mask})

        if self.settings["reduction"]["background"]["background_estimator"] == "reflected":
            self.background_estimator = ReflectedRegionsBackgroundEstimator(
                observations=self.observations, **background_pars
            )
            self.background_estimator.run()

        # e_reco
        # e_true
        # containment_correction=False,
        # extraction_pars = {"e_reco": e_reco, "e_true": e_true, 2containment_correction": containment_correction}

        self.extraction = SpectrumExtraction(
            observations=self.observations,
            bkg_estimate=self.background_estimator.result,
            # **extraction_pars
        )
        self.extraction.run()

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

    def view_settings(self):
        """Display settings in pretty YAML format."""
        print(yaml.dump(self.settings))
        
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
