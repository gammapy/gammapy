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
from gammapy.cube import MapDataset, MapDatasetMaker
from gammapy.cube.fit import BINSZ_IRF
from gammapy.data import DataStore, ObservationTable
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Datasets, Fit
from gammapy.modeling.models import SkyModels
from gammapy.modeling.serialize import dict_to_models
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
DOCS_FILE = CONFIG_PATH / "docs.yaml"

ANALYSIS_TEMPLATES = {
    "basic": "template-basic.yaml",
    "1d": "template-1d.yaml",
    "3d": "template-3d.yaml",
}


class Analysis:
    """Config-driven high-level analysis interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal configuration schema YAML file, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    For more info see  :ref:`HLI`.

    Parameters
    ----------
    config : dict or `AnalysisConfig`
        Configuration options following `AnalysisConfig` schema
    """

    def __init__(self, config=None):
        if isinstance(config, dict):
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
        self.flux_points = None

    @property
    def config(self):
        """Analysis configuration (`AnalysisConfig`)"""
        return self._config

    @property
    def settings(self):
        """Configuration settings for the analysis session."""
        return self.config.settings

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
        """Produce reduced datasets."""
        if not self._validate_reduction_settings():
            return False
        if self.settings["datasets"]["dataset-type"] == "SpectrumDatasetOnOff":
            self._spectrum_extraction()
        elif self.settings["datasets"]["dataset-type"] == "MapDataset":
            self._map_making()
        else:
            # TODO raise error?
            log.info("Data reduction method not available.")
            return False

    def set_model(self, model=None, filename=""):
        """Read the model from dict or filename and attach it to datasets.

        Parameters
        ----------
        model: dict or string
            Dictionary or string in YAML format with the serialized model.
        filename : string
            Name of the model YAML file describing the model.
        """
        if not self._validate_set_model():
            return False
        log.info(f"Reading model.")
        if isinstance(model, str):
            model = yaml.safe_load(model)
        if model:
            self.model = SkyModels(dict_to_models(model))
        elif filename:
            filepath = make_path(filename)
            self.model = SkyModels.from_yaml(filepath)
        else:
            return False
        # TODO: Deal with multiple components
        for dataset in self.datasets.datasets:
            if isinstance(dataset, MapDataset):
                dataset.model = self.model
            else:
                if len(self.model.skymodels) > 1:
                    raise ValueError(
                        "Can only fit a single spectral model at one time."
                    )
                dataset.model = self.model.skymodels[0].spectral_model
        log.info(self.model)

    def run_fit(self, optimize_opts=None):
        """Fitting reduced datasets to model."""
        if not self._validate_fitting_settings():
            return False

        for ds in self.datasets.datasets:
            # TODO: fit_range handled in jsonschema validation class
            if "fit_range" in self.settings["fit"]:
                e_min = u.Quantity(self.settings["fit"]["fit_range"]["min"])
                e_max = u.Quantity(self.settings["fit"]["fit_range"]["max"])
                if isinstance(ds, MapDataset):
                    ds.mask_fit = ds.counts.geom.energy_mask(e_min, e_max)
                else:
                    ds.mask_fit = ds.counts.energy_mask(e_min, e_max)
        log.info("Fitting reduced datasets.")
        self.fit = Fit(self.datasets)
        self.fit_result = self.fit.run(optimize_opts=optimize_opts)
        log.info(self.fit_result)

    def get_flux_points(self, source="source"):
        """Calculate flux points for a specific model component.

        Parameters
        ----------
        source : string
            Name of the model component where to calculate the flux points.
        """
        if not self._validate_fp_settings():
            return False

        # TODO: add "source" to config
        log.info("Calculating flux points.")
        axis_params = self.settings["flux-points"]["fp_binning"]
        e_edges = MapAxis.from_bounds(**axis_params).edges
        flux_point_estimator = FluxPointsEstimator(
            e_edges=e_edges, datasets=self.datasets, source=source
        )
        fp = flux_point_estimator.run()
        fp.table["is_ul"] = fp.table["ts"] < 4
        model = self.model[source].spectral_model.copy()
        self.flux_points = FluxPointsDataset(data=fp, model=model)
        cols = ["e_ref", "ref_flux", "dnde", "dnde_ul", "dnde_err", "is_ul"]
        log.info("\n{}".format(self.flux_points.data.table[cols]))

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
        """Make maps and datasets for 3d analysis."""
        log.info("Creating geometry.")
        geom = self._create_geometry(self.settings["datasets"]["geom"])

        if "geom-irf" in self.settings["datasets"]:
            geom_irf = self._create_geometry(self.settings["datasets"]["geom-irf"])
        else:
            geom_irf = geom.to_binsz(binsz=BINSZ_IRF)

        offset_max = Angle(self.settings["datasets"]["offset-max"])
        stack_datasets = self.settings["datasets"]["stack-datasets"]
        log.info("Creating datasets.")

        maker = MapDatasetMaker(geom=geom, geom_true=geom_irf, offset_max=offset_max)
        if stack_datasets:
            stacked = MapDataset.create(geom=geom, geom_irf=geom_irf, name="stacked")
            for obs in self.observations:
                dataset = maker.run(obs)
                stacked.stack(dataset)
            self._extract_irf_kernels(stacked)
            datasets = [stacked]
        else:
            datasets = []
            for obs in self.observations:
                dataset = maker.run(obs)
                self._extract_irf_kernels(dataset)
                datasets.append(dataset)

        self.datasets = Datasets(datasets)

    def _extract_irf_kernels(self, dataset):
        # TODO: remove hard-coded default value
        max_radius = self.settings["datasets"].get("psf-kernel-radius", "0.6 deg")
        # TODO: handle IRF maps in fit
        geom = dataset.counts.geom
        geom_irf = dataset.exposure.geom
        position = geom.center_skydir
        geom_psf = geom.to_image().to_cube(geom_irf.axes)
        dataset.psf = dataset.psf.get_psf_kernel(
            position=position, geom=geom_psf, max_radius=max_radius
        )
        e_reco = geom.get_axis_by_name("energy").edges
        dataset.edisp = dataset.edisp.get_energy_dispersion(
            position=position, e_reco=e_reco
        )

    def _set_logging(self):
        """Set logging parameters for API."""
        logging.basicConfig(**self.settings["general"]["logging"])
        log.info(
            "Setting logging config: {!r}".format(self.settings["general"]["logging"])
        )

    def _spectrum_extraction(self):
        """Run all steps for the spectrum extraction."""
        region = self.settings["datasets"]["geom"]["region"]
        log.info("Reducing spectrum datasets.")
        on_lon = Angle(region["center"][0])
        on_lat = Angle(region["center"][1])
        on_center = SkyCoord(on_lon, on_lat, frame=region["frame"])
        on_region = CircleSkyRegion(on_center, Angle(region["radius"]))
        background_params = {"on_region": on_region}
        background = self.settings["datasets"]["background"]
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

        extraction_params = {}
        if "containment_correction" in self.settings["datasets"]:
            extraction_params["containment_correction"] = self.settings["datasets"][
                "containment_correction"
            ]
        params = self.settings["datasets"]["geom"]["axes"][0]
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
        if self.settings["datasets"]["stack-datasets"]:
            stacked = self.datasets.stack_reduce()
            stacked.name = "stacked"
            self.datasets = Datasets([stacked])

    def _validate_reduction_settings(self):
        """Validate settings before proceeding to data reduction."""
        if self.observations and len(self.observations):
            self.config.validate()
            return True
        else:
            log.info("No observations selected.")
            log.info("Data reduction cannot be done.")
            return False

    def _validate_set_model(self):
        if self.datasets and self.datasets.datasets:
            self.config.validate()
            return True
        else:
            log.info("No datasets reduced.")
            return False

    def _validate_fitting_settings(self):
        """Validate settings before proceeding to fit 1D."""
        if not self.model:
            log.info("No model fetched for datasets.")
            log.info("Fit cannot be done.")
            return False
        else:
            return True

    def _validate_fp_settings(self):
        """Validate settings before proceeding to flux points estimation."""
        valid = True
        if self.fit:
            self.config.validate()
        else:
            log.info("No results available from fit.")
            valid = False
        if "flux-points" not in self.settings:
            log.info("No values declared for the energy bins.")
            valid = False
        elif "fp_binning" not in self.settings["flux-points"]:
            log.info("No values declared for the energy bins.")
            valid = False
        if not valid:
            log.info("Flux points calculation cannot be done.")
        return valid


class AnalysisConfig:
    """Analysis configuration.

    Parameters
    ----------
    config : dict
        Configuration parameters
    """

    def __init__(self, config=None, filename="config.yaml"):
        self.settings = {}
        self.template = ""
        if config is None:
            self.template = CONFIG_PATH / ANALYSIS_TEMPLATES["basic"]
        # add user settings
        self.update_settings(config, self.template)
        self.filename = Path(filename).name

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

        self.filename = Path(filename).name
        path_file = Path(self.settings["general"]["outdir"]) / self.filename

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

    @classmethod
    def from_template(cls, template="basic"):
        """Create AnalysisConfig from existing templates.

        Parameters
        ----------
        template : {"basic", "1d", "3d"}
            Build in templates.

        Returns
        -------
        analysis : `AnalysisConfig`
            AnalysisConfig class
        """
        filename = CONFIG_PATH / ANALYSIS_TEMPLATES[template]
        return cls.from_yaml(filename)

    def help(self, section=""):
        """Print template configuration settings."""
        doc = self._get_doc_sections()
        for keyword in doc.keys():
            if section == "" or section == keyword:
                print(doc[keyword])

    def update_settings(self, config=None, filename=""):
        """Update settings with config dictionary or values in configfile"""
        if filename:
            filepath = make_path(filename)
            config = read_yaml(filepath)
        if config is None:
            config = {}
        if isinstance(config, str):
            config = yaml.safe_load(config)
        if len(config):
            self._update_settings(config, self.settings)
        self.validate()

    def validate(self):
        """Validate and/or fill initial config parameters against schema."""
        validator = _gp_units_validator
        try:
            jsonschema.validate(self.settings, read_yaml(SCHEMA_FILE), validator)
        except jsonschema.exceptions.ValidationError as ex:
            log.error("Error when validating configuration parameters against schema.")
            log.error(ex.message)

    @staticmethod
    def _get_doc_sections():
        """Returns dict with commented docs from docs file"""
        doc = defaultdict(str)
        with open(DOCS_FILE) as f:
            for line in filter(lambda line: not line.startswith("---"), f):
                line = line.strip("\n")
                if line.startswith("# Section: "):
                    keyword = line.replace("# Section: ", "")
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
