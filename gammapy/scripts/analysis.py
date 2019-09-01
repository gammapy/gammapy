# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high-level interface API"""
import copy
import logging
import jsonschema
import yaml
from pathlib import Path
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from gammapy.spectrum import ReflectedRegionsBackgroundEstimator
from gammapy.data import DataStore, ObservationTable
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.spectrum import (
    FluxPointsDataset,
    FluxPointsEstimator,
    SpectrumExtraction,
    SpectrumDatasetOnOffStacker,
)
from gammapy.spectrum.models import SpectralModel
from gammapy.utils.fitting import Fit
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
        self.model = None
        self.fit_result = None
        self.flux_points_dataset = None

    @property
    def config(self):
        """Analysis configuration (`Config`)"""
        return self._config

    @property
    def settings(self):
        """Configuration settings for the analysis session."""
        return self.config.settings

    def fit(self, optimize_opts=None):
        """Fitting reduced data sets to model."""
        if self.settings["reduction"]["data_reducer"] == "1d":
            if self._validate_fitting_settings():
                self._read_model()
                self._fit_reduced_data(optimize_opts=optimize_opts)
        else:
            # TODO
            log.info("Fitting available only for 1D spectrum.")

    def get_flux_points(self):
        """Calculate flux points."""
        if self.settings["reduction"]["data_reducer"] == "1d":
            if self._validate_fp_settings():
                log.info("Calculating flux points.")
                obs_stacker = SpectrumDatasetOnOffStacker(
                    self.extraction.spectrum_observations
                )
                obs_stacker.run()
                stacked = obs_stacker.stacked_obs
                flux_model = self.model.copy()
                flux_model.parameters.covariance = self.fit_result.parameters.covariance
                stacked.model = flux_model

                # TODO
                # fp_binning handled in jsonschema validation class
                if "fp_binning" in self.settings["flux"]:
                    ax_pars = self.settings["flux"]["fp_binning"]
                    e_edges = MapAxis.from_bounds(**ax_pars).edges
                    flux_point_estimator = FluxPointsEstimator(
                        e_edges=e_edges, datasets=stacked
                    )
                    fp = flux_point_estimator.run()
                    fp.table["is_ul"] = fp.table["ts"] < 4
                    self.flux_points_dataset = FluxPointsDataset(
                        data=fp, model=self.model
                    )
                    cols = ["e_ref", "ref_flux", "dnde", "dnde_ul", "dnde_err", "is_ul"]
                    log.info("\n{}".format(self.flux_points_dataset.data.table[cols]))
                else:
                    log.error("Parameter flux.fp_binning is missing.")
        else:
            log.info("Flux point estimation available only for 1D spectrum.")

    def get_observations(self):
        """Fetch observations from the data store according to criteria defined in the configuration."""
        self.config.validate()
        log.info("Fetching observations.")
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

            # TODO
            # Reduce significantly the code.
            # This block would be handled by datastore.obs_table.select_observations
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
                    datastore.obs_table[criteria["variable"]] == criteria["value_param"]
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
        self.observations = datastore.get_observations(ids, skip_missing=True)
        for obs in self.observations.list:
            log.info(obs)

    def reduce(self):
        """Produce reduced data sets."""
        if self.settings["reduction"]["data_reducer"] == "1d":
            if self._validate_reduction_settings():
                log.info("Reducing data sets.")
                self._spectrum_extraction()
        else:
            # TODO
            log.info("Data reduction available only for 1D spectrum.")

    # TODO
    # add energy axes and other eventual params and types
    # validated and properly transformed in the jsonschema validation class
    def _create_geometry(self):
        """Create the geometry."""
        geom_params = self.settings["geometry"]
        self.geom = WcsGeom.create(**geom_params)

    def _fit_reduced_data(self, optimize_opts=None):
        """Fit data to models."""
        if self.settings["reduction"]["data_reducer"] == "1d":
            for obs in self.extraction.spectrum_observations:
                # TODO
                # fit_range handled in jsonschema validation class
                if "fit_range" in self.settings["fit"]:
                    e_min = u.Quantity(self.settings["fit"]["fit_range"]["min"])
                    e_max = u.Quantity(self.settings["fit"]["fit_range"]["e_max"])
                    obs.mask_fit = obs.counts.energy_mask(e_min, e_max)
                obs.model = self.model
            log.info("Fitting data sets to model.")
            fit = Fit(self.extraction.spectrum_observations)
            self.fit_result = fit.run(optimize_opts=optimize_opts)
            log.info(self.fit_result)
        else:
            # TODO
            log.info("Fitting available only for joint likelihood with 1D spectrum.")
            return False

    def _read_model(self):
        """Read the model from settings."""
        # TODO
        # make reading for generic spatial and spectral models with multiple components
        if self.settings["reduction"]["data_reducer"] == "1d":
            model_pars = self.settings["model"]["components"][0]["spectral"]
        else:
            log.info(
                "Model reading available only for single component spectral model."
            )
            return False
        log.info("Reading model.")
        self.model = SpectralModel.from_dict(model_pars)
        log.info(self.model)

    def _set_logging(self):
        """Set logging parameters for API."""
        logging.basicConfig(**self.settings["general"]["logging"])
        log.info(
            "Setting logging parameters ({}).".format(
                self.settings["general"]["logging"]["level"]
            )
        )

    def _spectrum_extraction(self):
        """Run all steps for the spectrum extraction."""
        background_params = self.settings["reduction"]["background"]

        on = background_params["on_region"]
        on_lon = Angle(on["center"][0])
        on_lat = Angle(on["center"][1])
        on_center = SkyCoord(on_lon, on_lat, frame=on["frame"])
        on_region = CircleSkyRegion(on_center, Angle(on["radius"]))
        background_pars = {"on_region": on_region}

        if "exclusion_mask" in background_params:
            map_hdu = {}
            filename = background_params["exclusion_mask"]["filename"]
            if "hdu" in background_params["exclusion_mask"]:
                map_hdu = {"hdu": background_params["exclusion_mask"]["hdu"]}
            exclusion_region = Map.read(filename, **map_hdu)
            background_pars.update({"exclusion_mask": exclusion_region})

        if background_params["background_estimator"] == "reflected":
            self.background_estimator = ReflectedRegionsBackgroundEstimator(
                observations=self.observations, **background_pars
            )
            self.background_estimator.run()
        else:
            # TODO
            log.info(
                "Background estimation available only for reflected regions method."
            )
            return False
        extraction_pars = {}
        if "containment_correction" in self.settings["reduction"]:
            extraction_pars.update(
                {
                    "containment_correction": self.settings["reduction"][
                        "containment_correction"
                    ]
                }
            )

        # TODO
        # e_reco/e_true handled in jsonschema validation class
        if "e_reco" in self.settings["geometry"]["axes"]:
            ax_pars = self.settings["geometry"]["axes"]["e_reco"]
            e_reco = MapAxis.from_bounds(**ax_pars).center
            extraction_pars.update({"e_reco": e_reco})
        if "e_true" in self.settings["geometry"]["axes"]:
            ax_pars = self.settings["geometry"]["axes"]["e_true"]
            e_true = MapAxis.from_bounds(**ax_pars).center
            extraction_pars.update({"e_true": e_true})
        self.extraction = SpectrumExtraction(
            observations=self.observations,
            bkg_estimate=self.background_estimator.result,
            **extraction_pars
        )
        self.extraction.run()

    def _validate_fitting_settings(self):
        """Validate settings before proceeding to fit."""
        if (
            self.settings["reduction"]["background"]["background_estimator"]
            == "reflected"
        ):
            if self.extraction and len(self.extraction.spectrum_observations):
                self.config.validate()
                return True
            else:
                log.info("No spectrum observations extracted.")
                log.info("Fit cannot be done.")
                return False
        else:
            # TODO
            log.info(
                "Background estimation available only for reflected regions method."
            )
            return False

    def _validate_fp_settings(self):
        """Validate settings before proceeding to flux points estimation."""
        if self.fit_result:
            self.config.validate()
            return True
        else:
            log.info("No observations selected.")
            log.info("Data reduction cannot be done.")
            return False

    def _validate_reduction_settings(self):
        """Validate settings before proceeding to data reduction."""
        if self.observations and len(self.observations):
            self.config.validate()
            return True
        else:
            log.info("No observations selected.")
            log.info("Data reduction cannot be done.")
            return False


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

    def __str__(self):
        """Display settings in pretty YAML format."""
        return yaml.dump(self.settings)

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
    quantity = str(instance).split()
    if len(quantity) >= 2:
        value = str(instance).split()[0]
        unit = "".join(str(instance).split()[1:])
        try:
            return u.Quantity(float(value), unit).unit.physical_type != "dimensionless"
        except ValueError:
            log.error("{} is not a valid astropy quantity.".format(str(instance)))
            raise ValueError("Not a valid astropy quantity.")
    else:
        try:
            number = float(instance)
        except ValueError:
            number = instance
        return jsonschema.Draft7Validator.TYPE_CHECKER.is_type(number, "number")


_type_checker = jsonschema.Draft7Validator.TYPE_CHECKER.redefine(
    "number", _astropy_quantity
)
_gp_units_validator = jsonschema.validators.extend(
    jsonschema.Draft7Validator, type_checker=_type_checker
)
_gp_validator = extend_with_default(_gp_units_validator)
