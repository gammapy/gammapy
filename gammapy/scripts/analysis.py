# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high-level interface API"""
import copy
import logging
from collections import defaultdict
from pathlib import Path
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
import jsonschema
import yaml
from gammapy.cube import MapDataset, MapMaker, PSFKernel
from gammapy.data import DataStore, ObservationTable
from gammapy.irf import make_mean_psf
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Datasets, Fit
from gammapy.modeling.models import SkyModel, BackgroundModel
from gammapy.modeling.serialize import dict_to_models
from gammapy.spectrum import (
    FluxPointsDataset,
    FluxPointsEstimator,
    ReflectedRegionsBackgroundEstimator,
    SpectrumExtraction,
)
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
        Configuration options following `Config` schema
    template : str
        Template for configuration settings

    Examples
    --------
    Example how to create an Analysis object:

    >>> from gammapy.scripts import Analysis
    >>> analysis = Analysis(template="1d")

    TODO: show a working example of running an analysis.
    Probably not here, but in high-level docs, linked to from class docstring.
    """

    def __init__(self, config=None, template="basic"):
        self._config = Config(config, template)
        self._set_logging()

        self.observations = None
        self.geom = None
        self.background_estimator = None
        self.datasets = None
        self.extraction = None
        self.images = None
        self.maps = None
        self.psf_kernel = None
        self.model = None
        self.fit = None
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

    def run_fit(self, optimize_opts=None):
        """Fitting reduced data sets to model."""
        if self._validate_fitting_settings():
            for obs in self.datasets.datasets:
                # TODO: fit_range handled in jsonschema validation class
                if "fit_range" in self.settings["fit"]:
                    e_min = u.Quantity(self.settings["fit"]["fit_range"]["min"])
                    e_max = u.Quantity(self.settings["fit"]["fit_range"]["max"])
                    obs.mask_fit = obs.counts.energy_mask(e_min, e_max)
                obs.model = self.model
            log.info("Fitting with joint likelihood.")
            self.fit = Fit(self.datasets)
            self.fit_result = self.fit.run(optimize_opts=optimize_opts)
            log.info(self.fit_result)

    @classmethod
    def from_file(cls, filename, template="basic"):
        """Instantiation of analysis from settings in config file.

        Parameters
        ----------
        filename : str, Path
            Configuration settings filename
        template: str
            Consider also values in configuration template
            Default value basic
        """
        filename = make_path(filename)
        config = read_yaml(filename)
        return cls(config=config, template=template)

    def get_flux_points(self):
        """Calculate flux points."""
        if self.settings["reduction"]["data_reducer"] == "1d":
            if self._validate_fp_settings():
                log.info("Calculating flux points.")
                stacked = self.datasets.stack_reduce()
                flux_model = self.model.copy()
                flux_model.parameters.covariance = self.fit_result.parameters.covariance
                stacked.model = flux_model
                axis_params = self.settings["flux"]["fp_binning"]
                e_edges = MapAxis.from_bounds(**axis_params).edges
                flux_point_estimator = FluxPointsEstimator(
                    e_edges=e_edges, datasets=stacked
                )
                fp = flux_point_estimator.run()
                fp.table["is_ul"] = fp.table["ts"] < 4
                self.flux_points_dataset = FluxPointsDataset(data=fp, model=self.model)
                cols = ["e_ref", "ref_flux", "dnde", "dnde_ul", "dnde_err", "is_ul"]
                log.info("\n{}".format(self.flux_points_dataset.data.table[cols]))
        else:
            # TODO raise error?
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
            raise FileNotFoundError(f"Datastore {datastore_path} not found.")
        ids = set()
        selection = dict()
        for criteria in self.settings["observations"]["filters"]:
            selected_obs = ObservationTable()

            # TODO: Reduce significantly the code.
            # This block would be handled by datastore.obs_table.select_observations
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

            if len(selected_obs):
                if "exclude" in criteria and criteria["exclude"]:
                    ids.difference_update(selected_obs["OBS_ID"].tolist())
                else:
                    ids.update(selected_obs["OBS_ID"].tolist())
        self.observations = datastore.get_observations(ids, skip_missing=True)
        for obs in self.observations.list:
            log.info(obs)

    def get_datasets(self):
        """Produce reduced data sets."""
        if not self._validate_reduction_settings():
            return False
        if self.settings["reduction"]["data_reducer"] == "1d":
            self._spectrum_extraction()
        elif self.settings["reduction"]["data_reducer"] == "3d":
            self._create_geometry()
            self._map_making()
        else:
            # TODO raise error?
            log.info("Data reduction method not available.")

    def _create_geometry(self):
        """Create the geometry."""
        # TODO: handled in jsonschema validation class
        geom_params = copy.deepcopy(self.settings["geometry"])
        e_reco, e_true = self._energy_axes()
        geom_params["axes"] = []
        geom_params["axes"].append(e_reco)
        geom_params["skydir"] = tuple(geom_params["skydir"])
        if "offset_max" in geom_params:
            del geom_params["offset_max"]
        if "psf_max_radius" in geom_params:
            del geom_params["psf_max_radius"]
        self.geom = WcsGeom.create(**geom_params)

    def _energy_axes(self):
        """Builds energy axes from settings in geometry."""
        # TODO: e_reco/e_true handled in jsonschema validation class
        e_reco = e_true = None
        if "e_reco" in self.settings["geometry"]["axes"]:
            axis_params = self.settings["geometry"]["axes"]["e_reco"]
            e_reco = MapAxis.from_bounds(**axis_params)
        if "e_true" in self.settings["geometry"]["axes"]:
            axis_params = self.settings["geometry"]["axes"]["e_true"]
            e_true = MapAxis.from_bounds(**axis_params)
        return e_reco, e_true

    def _map_making(self):
        """Make maps and data sets for 3d analysis."""
        maker_params = {}
        if "offset_max" in self.settings["geometry"]:
            maker_params = {"offset_max": self.settings["geometry"]["offset_max"]}
        maker = MapMaker(self.geom, **maker_params)
        self.maps = maker.run(self.observations)
        self.images = maker.run_images()
        table_psf = make_mean_psf(self.observations, self.geom.center_skydir)
        psf_params = {}
        if "psf_max_radius" in self.settings["geometry"]:
            psf_params = {"max_radius": self.settings["geometry"]["psf_max_radius"]}
        self.psf_kernel = PSFKernel.from_table_psf(table_psf, self.geom, **psf_params)
        self._read_model()
        # TODO: background model may come from YAML parameters
        background_model = BackgroundModel(self.maps["background"], norm=1.0)
        background_model.parameters["tilt"].frozen = False
        self.datasets = Datasets(
            MapDataset(
                model=self.model,
                counts=self.maps["counts"],
                exposure=self.maps["exposure"],
                background_model=background_model,
                psf=self.psf_kernel,
            )
        )

    def _read_model(self):
        """Read the model from settings."""
        # TODO: Deal with multiple components
        log.info("Reading model.")
        model_yaml = self.settings["model"]
        if self.settings["reduction"]["data_reducer"] == "1d":
            self.model = dict_to_models(model_yaml)[0].spectral_model
        else:
            spatial_model = dict_to_models(model_yaml)[0].spatial_model
            spectral_model = dict_to_models(model_yaml)[0].spectral_model
            self.model = SkyModel(
                spatial_model=spatial_model, spectral_model=spectral_model
            )
        log.info(self.model)

    def _set_logging(self):
        """Set logging parameters for API."""
        logging.basicConfig(**self.settings["general"]["logging"])
        log.info(
            "Setting logging config: {!r}".format(self.settings["general"]["logging"])
        )

    def _spectrum_extraction(self):
        """Run all steps for the spectrum extraction."""
        background = self.settings["reduction"]["background"]
        log.info("Reducing spectrum data sets.")
        on = background["on_region"]
        on_lon = Angle(on["center"][0])
        on_lat = Angle(on["center"][1])
        on_center = SkyCoord(on_lon, on_lat, frame=on["frame"])
        on_region = CircleSkyRegion(on_center, Angle(on["radius"]))
        background_params = {"on_region": on_region}

        if "exclusion_mask" in background_params:
            map_hdu = {}
            filename = background["exclusion_mask"]["filename"]
            if "hdu" in background["exclusion_mask"]:
                map_hdu = {"hdu": background["exclusion_mask"]["hdu"]}
            exclusion_region = Map.read(filename, **map_hdu)
            background_params["exclusion_mask"] = exclusion_region

        if background["background_estimator"] == "reflected":
            self.background_estimator = ReflectedRegionsBackgroundEstimator(
                observations=self.observations, **background_params
            )
            self.background_estimator.run()
        else:
            # TODO: raise error?
            log.info("Background estimation only for reflected regions method.")
            return False

        extraction_params = {}
        if "containment_correction" in self.settings["reduction"]:
            extraction_params["containment_correction"] = self.settings["reduction"][
                "containment_correction"
            ]
        e_reco, e_true = self._energy_axes()
        if e_reco:
            extraction_params["e_reco"] = e_reco.center
        if e_true:
            extraction_params["e_true"] = e_true.center
        self.extraction = SpectrumExtraction(
            observations=self.observations,
            bkg_estimate=self.background_estimator.result,
            **extraction_params,
        )
        self.extraction.run()
        self._read_model()
        self.datasets = Datasets(self.extraction.spectrum_observations)

    def _validate_fitting_settings(self):
        """Validate settings before proceeding to fit 1D."""
        if self.datasets.datasets:
            if (self.extraction and
                self.settings["reduction"]["background"]["background_estimator"]
                != "reflected"
            ):
                # TODO raise error?
                log.info("Background estimation only for reflected regions method.")
                return False
            self.config.validate()
            return True
        else:
            log.info("No datasets reduced.")
            log.info("Fit cannot be done.")
            return False

    def _validate_fp_settings(self):
        """Validate settings before proceeding to flux points estimation."""
        if self.fit:
            self.config.validate()
            return True
        else:
            log.info("No results available from fit.")
            log.info("Flux points calculation cannot be done.")
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

    def __init__(self, config=None, template="basic"):
        self._user_settings = {}
        self._template = template
        if template not in _implemented_templates:
            log.warning(f"Template {template} not implemented.")
            log.warning("Fetching basic template settings.")
            self._template = "basic"
        # fill with default values
        self.settings = {}
        self.validate()
        self._default_settings = copy.deepcopy(self.settings)
        # add user settings
        self.update_settings(config)

    def __str__(self):
        """Display settings in pretty YAML format."""
        return yaml.dump(self.settings)

    def dump(self, filename="config.yaml"):
        """Serialize config into a yaml formatted file.

        Parameters
        ----------
        filename : str, Path
            Configuration settings filename
            Default config.yaml
        """

        settings_str = ""
        doc_dic = self._get_doc_sections()
        for section in doc_dic.keys():
            if section in self.settings:
                settings_str += doc_dic[section] + "\n"
                settings_str += yaml.dump({section: self.settings[section]}) + "\n"
        filename = make_path(filename)
        path_file = Path(self.settings["general"]["outdir"]) / filename
        path_file.write_text(settings_str)

    def print_help(self, section=""):
        """Print template configuration settings."""
        doc = self._get_doc_sections()
        for keyword in doc.keys():
            if section == "" or section == keyword:
                print(doc[keyword])

    def update_settings(self, config=None, configfile=""):
        """Update settings with config dictionary or values in configfile"""
        if configfile:
            filename = make_path(configfile)
            config = read_yaml(filename)
        if config is None:
            config = {}
        if len(config):
            self._user_settings = config
            self._update_settings(self._user_settings, self.settings)
            self.validate()

    def validate(self):
        """Validate and/or fill initial config parameters against schema."""
        validator = _gp_defaults[self._template]
        jsonschema.validate(self.settings, read_yaml(SCHEMA_FILE), validator)

    @staticmethod
    def _get_doc_sections():
        """Returns dict with commented docs from schema"""
        doc = defaultdict(str)
        with open(SCHEMA_FILE) as f:
            for line in filter(lambda line: line.startswith("#"), f):
                line = line.strip("\n")
                if line.startswith("# Block: "):
                    keyword = line.replace("# Block: ", "")
                doc[keyword] += line + "\n"
        return doc

    def _update_settings(self, source, target):
        for key, val in source.items():
            if key not in target:
                target[key] = {}
            if not isinstance(val, dict) or val == {}:
                target[key] = val
            else:
                self._update_settings(val, target[key])


def extend_with_default(validator_class, template):
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
    default_field = f"default_{template}"

    def set_defaults(validator, properties, instance, schema):
        for prop, sub_schema in properties.items():
            if prop not in reserved:
                if default_field in sub_schema:
                    instance.setdefault(prop, sub_schema[default_field])
        yield from validate_properties(validator, properties, instance, schema)

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
_gp_defaults = {
    "1d": extend_with_default(_gp_units_validator, template="1d"),
    "3d": extend_with_default(_gp_units_validator, template="3d"),
    "all": extend_with_default(_gp_units_validator, template="all"),
    "basic": extend_with_default(_gp_units_validator, template="basic"),
}
_implemented_templates = ["1d", "3d", "all", "basic"]
