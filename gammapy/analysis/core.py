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
from gammapy.utils.scripts import make_path

__all__ = ["Analysis"]

log = logging.getLogger(__name__)


class Analysis:
    """Config-driven high-level analysis interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal high-level interface model, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

    For more info see  :ref:`analysis`.

    Parameters
    ----------
    config : dict or `AnalysisConfig`
        Configuration options following `AnalysisConfig` schema
    """

    def __init__(self, config):
        self.config = config
        self.config.set_logging()
        self.datastore = None
        self.observations = None
        self.datasets = None
        self.models = None
        self.fit = None
        self.fit_result = None
        self.flux_points = None

    @property
    def config(self):
        """Analysis configuration (`AnalysisConfig`)"""
        return self._config

    @config.setter
    def config(self, value):
        if isinstance(value, dict):
            self._config = AnalysisConfig(**value)
        elif isinstance(value, AnalysisConfig):
            self._config = value
        else:
            raise TypeError("config must be dict or AnalysisConfig.")

    def get_observations(self):
        """Fetch observations from the data store according to criteria defined in the configuration."""
        path = make_path(self.config.observations.datastore)

        if path.is_file():
            self.datastore = DataStore.from_file(path)
        elif path.is_dir():
            self.datastore = DataStore.from_dir(path)
        else:
            raise FileNotFoundError(f"Datastore not found: {path}")

        log.info("Fetching observations.")
        data_settings = self.config.observations
        selected_obs = ObservationTable()
        obs_list = self.datastore.get_observations()
        if len(self.config.observations.obs_ids):
            obs_list = self.datastore.get_observations(data_settings.obs_ids)
        selected_obs["OBS_ID"] = [obs.obs_id for obs in obs_list]
        ids = selected_obs["OBS_ID"].tolist()
        if self.config.observations.obs_file:
            path = make_path(self.config.observations.obs_file)
            with open(path, 'r') as f:
                ids_file = [int(obs_id) for obs_id in f.readlines()]
            ids.extend(ids_file)
        if data_settings.obs_cone.lon is not None:
            # TODO remove border keyword
            cone = dict(
                type="sky_circle",
                frame=data_settings.obs_cone.frame,
                lon=data_settings.obs_cone.lon,
                lat=data_settings.obs_cone.lat,
                radius=data_settings.obs_cone.radius,
                border="1 deg",
            )
            selected_cone = self.datastore.obs_table.select_observations(cone)
            ids = list(set(ids) & set(selected_cone["OBS_ID"].tolist()))
        self.observations = self.datastore.get_observations(ids, skip_missing=True)
        if self.config.observations.obs_time.start is not None:
            start = self.config.observations.obs_time.start
            stop = self.config.observations.obs_time.stop
            self.observations = self.observations.select_time([(start, stop)])
        log.info(f"Number of selected observations: {len(self.observations)}")
        for obs in self.observations:
            log.debug(obs)

    def get_datasets(self):
        """Produce reduced datasets."""
        if not self.observations or len(self.observations) == 0:
            raise RuntimeError("No observations have been selected.")

        if self.config.datasets.type == "1d":
            self._spectrum_extraction()
        elif self.config.datasets.type == "3d":
            self._map_making()
        else:
            ValueError(f"Invalid dataset type: {self.config.datasets.type}")

    def set_models(self, models):
        """Set models on datasets.

        Parameters
        ----------
        models : `~gammapy.modeling.models.SkyModels` or str
            SkyModels object or YAML models string
        """
        if not self.datasets or len(self.datasets) == 0:
            raise RuntimeError("Missing datasets")

        log.info(f"Reading model.")
        if isinstance(models, str):
            # FIXME: SkyModels should offer a method to create from YAML str
            models = yaml.safe_load(models)
            self.models = SkyModels(dict_to_models(models))
        elif isinstance(models, SkyModels):
            self.models = models
        else:
            raise TypeError(f"Invalid type: {models!r}")

        for dataset in self.datasets:
            dataset.models = self.models

        log.info(self.models)

    def read_models(self, path):
        """Read models from YAML file."""
        path = make_path(path)
        models = SkyModels.from_yaml(path)
        self.set_models(models)

    def run_fit(self, optimize_opts=None):
        """Fitting reduced datasets to model."""
        if not self.models:
            raise RuntimeError("Missing models")

        fit_settings = self.config.fit
        for dataset in self.datasets:
            if fit_settings.fit_range:
                e_min = fit_settings.fit_range.min
                e_max = fit_settings.fit_range.max
                if isinstance(dataset, MapDataset):
                    dataset.mask_fit = dataset.counts.geom.energy_mask(e_min, e_max)
                else:
                    dataset.mask_fit = dataset.counts.energy_mask(e_min, e_max)

        log.info("Fitting datasets.")
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
        if not self.fit:
            raise RuntimeError("No results available from Fit.")

        fp_settings = self.config.flux_points
        # TODO: add "source" to config
        log.info("Calculating flux points.")
        e_edges = self._make_energy_axis(fp_settings.energy).edges
        flux_point_estimator = FluxPointsEstimator(
            e_edges=e_edges, datasets=self.datasets, source=source
        )
        fp = flux_point_estimator.run()
        fp.table["is_ul"] = fp.table["ts"] < 4
        self.flux_points = FluxPointsDataset(data=fp, models=self.models[source])
        cols = ["e_ref", "ref_flux", "dnde", "dnde_ul", "dnde_err", "is_ul"]
        log.info("\n{}".format(self.flux_points.data.table[cols]))

    def update_config(self, config):
        self.config = self.config.update(config=config)

    def _create_geometry(self):
        """Create the geometry."""
        geom_params = {}
        geom_settings = self.config.datasets.geom
        skydir_settings = geom_settings.wcs.skydir
        if skydir_settings.lon is not None:
            skydir = SkyCoord(
                skydir_settings.lon, skydir_settings.lat, frame=skydir_settings.frame
            )
            geom_params["skydir"] = skydir
        if skydir_settings.frame == "icrs":
            geom_params["coordsys"] = "CEL"
        if skydir_settings.frame == "galactic":
            geom_params["coordsys"] = "GAL"
        axes = [self._make_energy_axis(geom_settings.axes.energy)]
        geom_params["axes"] = axes
        geom_params["binsz"] = geom_settings.wcs.binsize
        width = geom_settings.wcs.fov.width.to("deg").value
        height = geom_settings.wcs.fov.height.to("deg").value
        geom_params["width"] = (width, height)
        return WcsGeom.create(**geom_params)

    def _map_making(self):
        """Make maps and datasets for 3d analysis."""
        log.info("Creating geometry.")

        geom = self._create_geometry()
        geom_settings = self.config.datasets.geom
        geom_irf = dict(energy_axis_true=None, binsz_irf=None, margin_irf=None)
        if geom_settings.axes.energy_true.min is not None:
            geom_irf["energy_axis_true"] = self._make_energy_axis(
                geom_settings.axes.energy_true
            )
        geom_irf["binsz_irf"] = geom_settings.wcs.binsize_irf.to("deg").value
        geom_irf["margin_irf"] = geom_settings.wcs.margin_irf.to("deg").value
        offset_max = geom_settings.selection.offset_max
        log.info("Creating datasets.")

        maker = MapDatasetMaker(selection=self.config.datasets.map_selection)
        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=offset_max)
        stacked = MapDataset.create(geom=geom, name="stacked", **geom_irf)

        if self.config.datasets.stack:
            for obs in self.observations:
                log.info(f"Processing observation {obs.obs_id}")
                cutout = stacked.cutout(obs.pointing_radec, width=2 * offset_max)
                dataset = maker.run(cutout, obs)
                dataset = maker_safe_mask.run(dataset, obs)
                if "background" in self.config.datasets.map_selection:
                    dataset.background_model.name = f"bkg_{dataset.name}"
                    # TODO remove this once dataset and model have unique identifiers
                log.debug(dataset)
                stacked.stack(dataset)
            datasets = [stacked]
        else:
            datasets = []
            for obs in self.observations:
                log.info(f"Processing observation {obs.obs_id}")
                cutout = stacked.cutout(obs.pointing_radec, width=2 * offset_max)
                dataset = maker.run(cutout, obs)
                dataset = maker_safe_mask.run(dataset, obs)
                if "background" in self.config.datasets.map_selection:
                    dataset.background_model.name = f"bkg_{dataset.name}"
                    # TODO remove this once dataset and model have unique identifiers
                log.debug(dataset)
                datasets.append(dataset)
        self.datasets = Datasets(datasets)

    def _spectrum_extraction(self):
        """Run all steps for the spectrum extraction."""
        log.info("Reducing spectrum datasets.")
        datasets_settings = self.config.datasets
        on_lon = datasets_settings.on_region.lon
        on_lat = datasets_settings.on_region.lat
        on_center = SkyCoord(on_lon, on_lat, frame=datasets_settings.on_region.frame)
        on_region = CircleSkyRegion(on_center, datasets_settings.on_region.radius)

        maker_config = {}
        if datasets_settings.containment_correction:
            maker_config[
                "containment_correction"
            ] = datasets_settings.containment_correction
        e_reco = self._make_energy_axis(datasets_settings.geom.axes.energy).edges
        maker_config["e_reco"] = e_reco
        # TODO: remove hard-coded e_true and make it configurable
        maker_config["e_true"] = np.logspace(-2, 2.5, 109) * u.TeV
        maker_config["region"] = on_region

        dataset_maker = SpectrumDatasetMaker(**maker_config)
        bkg_maker_config = {}
        if datasets_settings.background.exclusion:
            exclusion_region = Map.read(datasets_settings.background.exclusion)
            bkg_maker_config["exclusion_mask"] = exclusion_region
        bkg_maker = ReflectedRegionsBackgroundMaker(**bkg_maker_config)
        safe_mask_maker = SafeMaskMaker(methods=["aeff-default", "aeff-max"])

        datasets = []
        for obs in self.observations:
            log.info(f"Processing observation {obs.obs_id}")
            selection = ["counts", "aeff", "edisp"]
            dataset = dataset_maker.run(obs, selection=selection)
            dataset = bkg_maker.run(dataset, obs)
            if dataset.counts_off is None:
                log.info(
                    f"No OFF region found for observation {obs.obs_id}. Discarding."
                )
                continue
            dataset = safe_mask_maker.run(dataset, obs)
            log.debug(dataset)
            datasets.append(dataset)

        self.datasets = Datasets(datasets)

        if self.config.datasets.stack:
            stacked = self.datasets.stack_reduce()
            stacked.name = "stacked"
            self.datasets = Datasets([stacked])

    @staticmethod
    def _make_energy_axis(axis):
        return MapAxis.from_bounds(
            name="energy",
            lo_bnd=axis.min.value,
            hi_bnd=axis.max.to_value(axis.min.unit),
            nbin=axis.nbins,
            unit=axis.min.unit,
            interp="log",
            node_type="edges",
        )
