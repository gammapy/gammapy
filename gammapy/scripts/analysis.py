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
from gammapy.cube import MapDataset, MapMakerObs
from gammapy.data import DataStore, ObservationTable
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Datasets, Fit
from gammapy.modeling.models import SkyModels
from gammapy.spectrum import (
    FluxPointsDataset,
    FluxPointsEstimator,
    ReflectedRegionsBackgroundEstimator,
    SpectrumExtraction,
)
from gammapy.utils.scripts import make_path, read_yaml

__all__ = ["Analysis", "AnalysisConfig"]

log = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent / "config"
SCHEMA_FILE = CONFIG_PATH / "schema.yaml"

ANALYSIS_TEMPLATES = {
    "basic": "template-basic.yaml",
    "1d": "template-1d.yaml",
    "3d": "template-3d.yaml"
}


class Analysis:
    """Config-driven high-level analysis interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal configuration schema YAML file, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    Parameters
    ----------
    config : dict or `AnalysisConfig`
        Configuration options following `AnalysisConfig` schema

    Examples
    --------
    Example how to create an Analysis object:

    >>> from gammapy.scripts import Analysis
    >>> analysis = Analysis.from_template(template="1d")

    TODO: show a working example of running an analysis.
    Probably not here, but in high-level docs, linked to from class docstring.
    """

    def __init__(self, config=None):
        if config is None:
            filename = CONFIG_PATH / ANALYSIS_TEMPLATES["basic"]
            self._config = AnalysisConfig.from_yaml(filename)
        elif isinstance(config, dict):
            self._config = AnalysisConfig(config)
        elif isinstance(config, AnalysisConfig):
            self._config = config
        else:
            raise ValueError("Dict or `AnalysiConfig` object required.")

        self._set_logging()
        self.observations = None
        self.background_estimator = None
        self.datasets = None
        self.extraction = None
        self.model = None
        self.fit = None
        self.fit_result = None
        self.flux_points_dataset = None

    @property
    def config(self):
        """Analysis configuration (`AnalysisConfig`)"""
        return self._config

    @property
    def settings(self):
        """Configuration settings for the analysis session."""
        return self.config.settings

    def run_fit(self, optimize_opts=None):
        """Fitting reduced data sets to model."""
        if self._validate_fitting_settings():
            for ds in self.datasets.datasets:
                # TODO: fit_range handled in jsonschema validation class
                if "fit_range" in self.settings["fit"]:
                    e_min = u.Quantity(self.settings["fit"]["fit_range"]["min"])
                    e_max = u.Quantity(self.settings["fit"]["fit_range"]["max"])
                    if isinstance(ds, MapDataset):
                        ds.mask_fit = ds.counts.geom.energy_mask(e_min, e_max)
                    else:
                        ds.mask_fit = ds.counts.energy_mask(e_min, e_max)

            log.info("Fitting reduced data sets.")
            self.fit = Fit(self.datasets)
            self.fit_result = self.fit.run(optimize_opts=optimize_opts)
            log.info(self.fit_result)

    @classmethod
    def from_yaml(cls, filename):
        """Create analysis from settings in config file.

        Parameters
        ----------
        filename : str, Path
            Configuration settings filename

        Returns
        -------
        analysis : `Analysis`
            Analysis class
        """
        config = AnalysisConfig.from_yaml(filename)
        return cls(config=config)

    @classmethod
    def from_template(cls, template="basic"):
        """Create Analysis from existing templates.

        Parameters
        ----------
        template : {"basic", "1d", "3d"}
            Build in templates.

        Returns
        -------
        analysis : `Analysis`
            Analysis class
        """
        filename = CONFIG_PATH / ANALYSIS_TEMPLATES[template]
        return cls.from_yaml(filename)

    def get_datasets(self):
        """Produce reduced data sets."""
        if not self._validate_reduction_settings():
            return False
        if self.settings["reduction"]["dataset-type"] == "SpectrumDatasetOnOff":
            self._spectrum_extraction()
        elif self.settings["reduction"]["dataset-type"] == "MapDataset":
            self._map_making()
        else:
            # TODO raise error?
            log.info("Data reduction method not available.")

    def get_flux_points(self):
        """Calculate flux points."""
        if self._validate_fp_settings():
            # TODO: add "source" to config
            source = "source"
            log.info("Calculating flux points.")

            axis_params = self.settings["flux"]["fp_binning"]
            e_edges = MapAxis.from_bounds(**axis_params).edges

            flux_point_estimator = FluxPointsEstimator(
                e_edges=e_edges, datasets=self.datasets, source=source,
            )
            fp = flux_point_estimator.run()
            fp.table["is_ul"] = fp.table["ts"] < 4

            model = self.model[source].spectral_model.copy()
            self.flux_points_dataset = FluxPointsDataset(data=fp, model=model)
            cols = ["e_ref", "ref_flux", "dnde", "dnde_ul", "dnde_err", "is_ul"]
            log.info("\n{}".format(self.flux_points_dataset.data.table[cols]))

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

    @staticmethod
    def _create_geometry(params):
        """Create the geometry."""
        # TODO: handled in jsonschema validation class
        geom_params = copy.deepcopy(params)

        axes = []
        for axis_params in geom_params.get("axes", []):
            ax = MapAxis.from_bounds(**axis_params)
            axes.append(ax)

        geom_params["axes"] = axes
        geom_params["skydir"] = tuple(geom_params["skydir"])
        return WcsGeom.create(**geom_params)

    def _map_making(self):
        """Make maps and data sets for 3d analysis."""
        geom = self._create_geometry(self.settings["reduction"]["geom"])
        geom_irf = self._create_geometry(self.settings["reduction"]["geom-irf"])
        offset_max = Angle(self.settings["reduction"]["offset-max"])
        stack_datasets = self.settings["reduction"]["stack-datasets"]

        if stack_datasets:
            stacked = MapDataset.create(geom=geom, geom_irf=geom_irf, name="stacked")

            for obs in self.observations:
                dataset = self._get_dataset(obs, geom, geom_irf, offset_max)
                stacked.stack(dataset)

            self._extract_irf_kernels(stacked)
            datasets = [stacked]
        else:
            datasets = []
            for obs in self.observations:
                dataset = self._get_dataset(obs, geom, geom_irf, offset_max)
                self._extract_irf_kernels(dataset)
                datasets.append(dataset)

        self.datasets = Datasets(datasets)

    @staticmethod
    def _get_dataset(obs, geom, geom_irf, offset_max):
        position, width = obs.pointing_radec, 2 * offset_max
        geom_cutout = geom.cutout(position=position, width=width)
        geom_irf_cutout = geom_irf.cutout(position=position, width=width)

        maker = MapMakerObs(
            observation=obs,
            geom=geom_cutout,
            geom_true=geom_irf_cutout,
            offset_max=offset_max
        )

        return maker.run()

    def _extract_irf_kernels(self, dataset):
        # TODO: remove hard-coded default value
        max_radius = self.settings["reduction"].get("psf-kernel-radius", "0.5 deg")

        # TODO: handle IRF maps in fit
        geom = dataset.counts.geom
        geom_irf = dataset.exposure.geom

        position = geom.center_skydir
        geom_psf = geom.to_image().to_cube(geom_irf.axes)
        dataset.psf = dataset.psf.get_psf_kernel(position=position, geom=geom_psf, max_radius=max_radius)

        e_reco = geom.get_axis_by_name("energy").edges
        dataset.edisp = dataset.edisp.get_energy_dispersion(position=position, e_reco=e_reco)

    def get_model(self):
        """Read the model from settings."""
        # TODO: Deal with multiple components
        log.info("Reading model.")
        model_yaml = Path(self.settings["model"])
        base_path = self.config.filename.parent

        self.model = SkyModels.from_yaml(base_path / model_yaml)

        for dataset in self.datasets.datasets:
            if isinstance(dataset, MapDataset):
                dataset.model = self.model
            else:
                if len(self.model.skymodels) > 1:
                    raise ValueError("Can only fit a single spectral model at one time.")
                dataset.model = self.model.skymodels[0].spectral_model

        log.info(self.model)

    def _set_logging(self):
        """Set logging parameters for API."""
        logging.basicConfig(**self.settings["general"]["logging"])
        log.info(
            "Setting logging config: {!r}".format(self.settings["general"]["logging"])
        )

    def _spectrum_extraction(self):
        """Run all steps for the spectrum extraction."""
        region = self.settings["reduction"]["geom"]["region"]
        log.info("Reducing spectrum data sets.")
        on_lon = Angle(region["center"][0])
        on_lat = Angle(region["center"][1])
        on_center = SkyCoord(on_lon, on_lat, frame=region["frame"])
        on_region = CircleSkyRegion(on_center, Angle(region["radius"]))
        background_params = {"on_region": on_region}

        background = self.settings["reduction"]["background"]

        if "exclusion_mask" in background:
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

        params = self.settings["reduction"]["geom"]["axes"][0]
        e_reco = MapAxis.from_bounds(**params).edges
        extraction_params["e_reco"] = e_reco
        extraction_params["e_true"] = None
        self.extraction = SpectrumExtraction(
            observations=self.observations,
            bkg_estimate=self.background_estimator.result,
            **extraction_params,
        )
        self.extraction.run()
        self.datasets = Datasets(self.extraction.spectrum_observations)

        if self.settings["reduction"]["stack-datasets"]:
            stacked = self.datasets.stack_reduce()
            stacked.name = "stacked"
            self.datasets = Datasets([stacked])

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


class AnalysisConfig:
    """Analysis configuration.

    Parameters
    ----------
    config : dict
        Configuration parameters
    """

    def __init__(self, config=None, filename="config.yaml"):
        self._user_settings = {}
        self.settings = {}
        # add user settings
        self.update_settings(config)
        self.filename = filename

    def __str__(self):
        """Display settings in pretty YAML format."""
        info = self.__class__.__name__ + "\n\n\t"

        data = yaml.dump(self.settings, sort_keys=False, indent=4)
        data = data.replace("\n", "\n\t")
        info += data
        return info.expandtabs(tabsize=4)

    def to_yaml(self, filename=None, overwrite=False):
        """Serialize config into a yaml formatted file.

        Parameters
        ----------
        filename : str, Path
            Configuration settings filename
            Default config.yaml
        overwrite : bool
            Whether to overwrite an existing file.
        """
        if filename is None:
            filename = self.filename

        filename = make_path(filename)
        path_file = Path(self.settings["general"]["outdir"]) / filename
        self.filename = path_file

        if path_file.exists() and not overwrite:
            raise IOError(f"File {filename} already exists.")

        path_file.write_text(yaml.dump(self.settings, sort_keys=False, indent=4))
        log.info(f"Configuration settings saved into {path_file}")

    @classmethod
    def from_yaml(cls, filename):
        """Read config from filename"""
        filename = make_path(filename)
        config = read_yaml(filename)
        return cls(config, filename=filename)

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
            self._user_settings.update(config)
            self._update_settings(config, self.settings)
        self.validate()

    def validate(self):
        """Validate and/or fill initial config parameters against schema."""
        validator = _gp_units_validator
        try:
            jsonschema.validate(self.settings, read_yaml(SCHEMA_FILE), validator)
        except jsonschema.exceptions.ValidationError as ex:
            log.error('Error when validating configuration parameters against schema.')
            log.error(ex.message)

    @staticmethod
    def _get_doc_sections():
        """Returns dict with commented docs from schema"""
        doc = defaultdict(str)
        with open(SCHEMA_FILE) as f:
            for line in filter(lambda line: line.startswith("# "), f):
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


def is_quantity(instance):
    try:
        _ = u.Quantity(instance)
        return True
    except ValueError:
        return False


def _astropy_quantity(_, instance):
    """Check a number may also be an astropy quantity."""
    is_number = jsonschema.Draft7Validator.TYPE_CHECKER.is_type(instance, "number")
    return is_number or is_quantity(instance)


_type_checker = jsonschema.Draft7Validator.TYPE_CHECKER.redefine(
    "number", _astropy_quantity
)
_gp_units_validator = jsonschema.validators.extend(
    jsonschema.Draft7Validator, type_checker=_type_checker
)
