# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high-level interface API"""
import logging
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
import yaml
from gammapy.analysis.config import AnalysisConfig
from gammapy.cube import MapDataset, MapDatasetMaker, SafeMaskMaker
from gammapy.data import DataStore, ObservationTable
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Datasets, Fit
from gammapy.modeling.models import SkyModels
from gammapy.modeling.serialize import dict_to_models
from gammapy.spectrum import (
    FluxPointsDataset,
    FluxPointsEstimator,
    ReflectedRegionsBackgroundMaker,
    SpectrumDatasetMaker,
)
from gammapy.utils.scripts import make_path, read_yaml

__all__ = ["Analysis"]

log = logging.getLogger(__name__)


class Analysis:
    """Config-driven high-level analysis interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal configuration schema YAML file, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    For more info see  :ref:`analysis`.

    Parameters
    ----------
    config : dict or `AnalysisConfig`
        Configuration options following `AnalysisConfig` schema
    """

    def __init__(self, config=None):
        if isinstance(config, dict):
            self._config = AnalysisConfig(**config)
        elif isinstance(config, AnalysisConfig):
            self._config = config
        else:
            raise ValueError("Dict or `AnalysiConfig` object required.")

        self._set_logging()
        self.datastore = None
        self.observations = None
        self.datasets = None
        self.model = None
        self.fit = None
        self.fit_result = None
        self.flux_points = None

    @property
    def config(self):
        """Analysis configuration (`AnalysisConfig`)"""
        return self._config

    def get_observations(self):
        """Fetch observations from the data store according to criteria defined in the configuration."""
        if not self._validate_observations_settings():
            return False

        log.info("Fetching observations.")
        datastore_path = make_path(self.settings["observations"]["datastore"])
        if datastore_path.is_file():
            self.datastore = DataStore().from_file(datastore_path)
        elif datastore_path.is_dir():
            self.datastore = DataStore().from_dir(datastore_path)
        else:
            raise FileNotFoundError(f"Datastore {datastore_path} not found.")
        ids = []
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
                selected_obs = self.datastore.obs_table.select_observations(selection)
            if selection["type"] == "par_value":
                mask = (
                    self.datastore.obs_table[criteria["variable"]]
                    == criteria["value_param"]
                )
                selected_obs = self.datastore.obs_table[mask]
            if selection["type"] == "ids":
                obs_list = self.datastore.get_observations(criteria["obs_ids"])
                selected_obs["OBS_ID"] = [obs.obs_id for obs in obs_list.list]
            if selection["type"] == "all":
                obs_list = self.datastore.get_observations()
                selected_obs["OBS_ID"] = [obs.obs_id for obs in obs_list.list]

            if len(selected_obs):
                if "exclude" in criteria and criteria["exclude"]:
                    exclude = selected_obs["OBS_ID"].tolist()
                    selection = np.isin(ids, exclude)
                    ids = list(np.array(ids)[~selection])
                else:
                    ids.extend(selected_obs["OBS_ID"].tolist())
        self.observations = self.datastore.get_observations(ids, skip_missing=True)
        log.info(f"{len(self.observations.list)} observations were selected.")
        for obs in self.observations.list:
            log.debug(obs)

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
        if not self._validate_set_model_settings():
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
        for dataset in self.datasets:
            dataset.model = self.model

        log.info(self.model)

    def run_fit(self, optimize_opts=None):
        """Fitting reduced datasets to model."""
        if not self._validate_fitting_settings():
            return False

        for ds in self.datasets:
            # TODO: fit_range handled in jsonschema validation class
            if "fit" in self.settings and "fit_range" in self.settings["fit"]:
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
        model = self.model[source].copy()
        self.flux_points = FluxPointsDataset(data=fp, model=model)
        cols = ["e_ref", "ref_flux", "dnde", "dnde_ul", "dnde_err", "is_ul"]
        log.info("\n{}".format(self.flux_points.data.table[cols]))

    def update_config(self, config=None, filename=""):
        """Updates config with provided settings.

        Parameters
        ----------
        config : string dict or `AnalysisConfig` object
            Configuration settings provided in dict() syntax.
        filename : string
            Filename in YAML format.
        """
        upd_cfg = None
        try:
            if filename:
                filepath = make_path(filename)
                config = dict(read_yaml(filepath))
            if isinstance(config, str):
                config = dict(yaml.safe_load(config))
            if isinstance(config, dict):
                upd_cfg = AnalysisConfig(**config)
            if isinstance(config, AnalysisConfig):
                upd_cfg = config
        except Exception as ex:
            log.error("Could not parse the config settings provided.")
            log.error(ex)
            return False
        if upd_cfg is None:
            self._config = AnalysisConfig()
        else:
            self._config = self.config._update(upd_cfg)

    @staticmethod
    def _create_geometry(params):
        """Create the geometry."""
        geom_params = copy.deepcopy(params)
        axes = []
        for axis_params in params.get("axes", []):
            ax = MapAxis.from_bounds(**axis_params)
            axes.append(ax)
        geom_params["axes"] = axes
        if "skydir" in geom_params:
            geom_params["skydir"] = tuple(geom_params["skydir"])
        return WcsGeom.create(**geom_params)

    def _map_making(self):
        """Make maps and datasets for 3d analysis."""
        log.info("Creating geometry.")

        geom = self._create_geometry(self.settings["datasets"]["geom"])

        geom_irf = dict(energy_axis_true=None, binsz_irf=None, margin_irf=None)
        if "energy-axis-true" in self.settings["datasets"]:
            axis_params = self.settings["datasets"]["energy-axis-true"]
            geom_irf["energy_axis_true"] = MapAxis.from_bounds(**axis_params)
        geom_irf["binsz_irf"] = self.settings["datasets"].get("binsz", None)
        geom_irf["margin_irf"] = self.settings["datasets"].get("margin", None)

        offset_max = Angle(self.settings["datasets"]["offset-max"])
        log.info("Creating datasets.")

        maker = MapDatasetMaker()
        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=offset_max)

        stacked = MapDataset.create(geom=geom, name="stacked", **geom_irf)

        if self.settings["datasets"]["stack-datasets"]:
            for obs in self.observations:
                log.info(f"Processing observation {obs.obs_id}")
                cutout = stacked.cutout(obs.pointing_radec, width=2 * offset_max)
                dataset = maker.run(cutout, obs)
                dataset = maker_safe_mask.run(dataset, obs)
                dataset.background_model.name = f"bkg_{dataset.name}"
                # TODO remove this once dataset and model have unique identifiers
                log.debug(dataset)
                stacked.stack(dataset)
            self._extract_irf_kernels(stacked)
            datasets = [stacked]
        else:
            datasets = []
            for obs in self.observations:
                log.info(f"Processing observation {obs.obs_id}")
                cutout = stacked.cutout(obs.pointing_radec, width=2 * offset_max)
                dataset = maker.run(cutout, obs)
                dataset = maker_safe_mask.run(dataset, obs)
                dataset.background_model.name = f"bkg_{dataset.name}"
                # TODO remove this once dataset and model have unique identifiers
                self._extract_irf_kernels(dataset)
                log.debug(dataset)
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

    def _set_logging(self):
        """Set logging parameters for API."""
        self.config.general.log.level = self.config.general.log.level.upper()
        logging.basicConfig(**self.config.general.log.dict())
        log.info("Setting logging config: {!r}".format(self.config.general.log.dict()))

    def _spectrum_extraction(self):
        """Run all steps for the spectrum extraction."""
        region = self.settings["datasets"]["geom"]["region"]
        log.info("Reducing spectrum datasets.")
        on_lon = Angle(region["center"][0])
        on_lat = Angle(region["center"][1])
        on_center = SkyCoord(on_lon, on_lat, frame=region["frame"])
        on_region = CircleSkyRegion(on_center, Angle(region["radius"]))

        maker_config = {}
        if "containment_correction" in self.settings["datasets"]:
            maker_config["containment_correction"] = self.settings["datasets"][
                "containment_correction"
            ]
        params = self.settings["datasets"]["geom"]["axes"][0]
        e_reco = MapAxis.from_bounds(**params).edges
        maker_config["e_reco"] = e_reco

        # TODO: remove hard-coded e_true and make it configurable
        maker_config["e_true"] = np.logspace(-2, 2.5, 109) * u.TeV
        maker_config["region"] = on_region

        dataset_maker = SpectrumDatasetMaker(**maker_config)
        bkg_maker_config = {}
        background = self.settings["datasets"]["background"]

        if "exclusion_mask" in background:
            map_hdu = {}
            filename = background["exclusion_mask"]["filename"]
            if "hdu" in background["exclusion_mask"]:
                map_hdu = {"hdu": background["exclusion_mask"]["hdu"]}
            exclusion_region = Map.read(filename, **map_hdu)
            bkg_maker_config["exclusion_mask"] = exclusion_region
        if background["background_estimator"] == "reflected":
            reflected_bkg_maker = ReflectedRegionsBackgroundMaker(**bkg_maker_config)
        else:
            # TODO: raise error?
            log.info("Background estimation only for reflected regions method.")

        safe_mask_maker = SafeMaskMaker(methods=["aeff-default", "aeff-max"])

        datasets = []
        for obs in self.observations:
            log.info(f"Processing observation {obs.obs_id}")
            selection = ["counts", "aeff", "edisp"]
            dataset = dataset_maker.run(obs, selection=selection)
            dataset = reflected_bkg_maker.run(dataset, obs)
            if dataset.counts_off is None:
                log.info(
                    f"No OFF region found for observation {obs.obs_id}. Discarding."
                )
                continue
            dataset = safe_mask_maker.run(dataset, obs)
            log.debug(dataset)
            datasets.append(dataset)

        self.datasets = Datasets(datasets)

        if self.settings["datasets"]["stack-datasets"]:
            stacked = self.datasets.stack_reduce()
            stacked.name = "stacked"
            self.datasets = Datasets([stacked])

    def _validate_observations_settings(self):
        """Validate settings before proceeding to observations selection."""
        if not self.config.data.datastore:
            log.info("No datastore defined")
            log.info("Observation selection cannot be done.")
            return False
        return True

    def _validate_reduction_settings(self):
        """Validate settings before proceeding to data reduction."""
        if not self.observations or len(self.observations) == 0:
            log.info("No observations selected.")
            log.info("Data reduction cannot be done.")
            return False
        return True

    def _validate_set_model_settings(self):
        """Validate settings before proceeding to model attach."""
        if not self.datasets or len(self.datasets) == 0:
            log.info("No datasets reduced.")
            return False
        return True

    def _validate_fitting_settings(self):
        """Validate settings before proceeding to fit."""
        if not self.model:
            log.info("No model attached to datasets.")
            log.info("Fit cannot be done.")
            return False
        return True

    def _validate_fp_settings(self):
        """Validate settings before proceeding to flux points estimation."""
        valid = True
        if not self.fit:
            log.info("No results available from fit.")
            valid = False
        if not self.config.flux_points.energy:
            log.info("No values declared for the energy bins.")
            valid = False
        if not valid:
            log.info("Flux points calculation cannot be done.")
        return valid
