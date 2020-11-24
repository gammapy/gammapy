# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high-level interface API"""
import logging
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import CircleSkyRegion
from gammapy.analysis.config import AnalysisConfig
from gammapy.data import DataStore
from gammapy.datasets import Datasets, FluxPointsDataset, MapDataset, SpectrumDataset
from gammapy.estimators import FluxPointsEstimator
from gammapy.makers import (
    FoVBackgroundMaker,
    MapDatasetMaker,
    ReflectedRegionsBackgroundMaker,
    RingBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import FoVBackgroundModel, Models
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
        observations_settings = self.config.observations
        path = make_path(observations_settings.datastore)
        if path.is_file():
            self.datastore = DataStore.from_file(path)
        elif path.is_dir():
            self.datastore = DataStore.from_dir(path)
        else:
            raise FileNotFoundError(f"Datastore not found: {path}")

        log.info("Fetching observations.")
        if (
            len(observations_settings.obs_ids)
            and observations_settings.obs_file is not None
        ):
            raise ValueError(
                "Values for both parameters obs_ids and obs_file are not accepted."
            )
        elif (
            not len(observations_settings.obs_ids)
            and observations_settings.obs_file is None
        ):
            obs_list = self.datastore.get_observations()
            ids = [obs.obs_id for obs in obs_list]
        elif len(observations_settings.obs_ids):
            obs_list = self.datastore.get_observations(observations_settings.obs_ids)
            ids = [obs.obs_id for obs in obs_list]
        else:
            path = make_path(observations_settings.obs_file)
            ids = list(Table.read(path, format="ascii", data_start=0).columns[0])

        if observations_settings.obs_cone.lon is not None:
            cone = dict(
                type="sky_circle",
                frame=observations_settings.obs_cone.frame,
                lon=observations_settings.obs_cone.lon,
                lat=observations_settings.obs_cone.lat,
                radius=observations_settings.obs_cone.radius,
                border="0 deg",
            )
            selected_cone = self.datastore.obs_table.select_observations(cone)
            ids = list(set(ids) & set(selected_cone["OBS_ID"].tolist()))
        self.observations = self.datastore.get_observations(ids, skip_missing=True)
        if observations_settings.obs_time.start is not None:
            start = observations_settings.obs_time.start
            stop = observations_settings.obs_time.stop
            self.observations = self.observations.select_time([(start, stop)])
        log.info(f"Number of selected observations: {len(self.observations)}")
        for obs in self.observations:
            log.debug(obs)

    def get_datasets(self):
        """Produce reduced datasets."""
        datasets_settings = self.config.datasets
        if not self.observations or len(self.observations) == 0:
            raise RuntimeError("No observations have been selected.")

        if datasets_settings.type == "1d":
            self._spectrum_extraction()
        else:  # 3d
            self._map_making()

    def set_models(self, models):
        """Set models on datasets.

        Adds `FoVVackgroundModel` if not present already

        Parameters
        ----------
        models : `~gammapy.modeling.models.Models` or str
            Models object or YAML models string
        """
        if not self.datasets or len(self.datasets) == 0:
            raise RuntimeError("Missing datasets")

        log.info(f"Reading model.")
        if isinstance(models, str):
            self.models = Models.from_yaml(models)
        elif isinstance(models, Models):
            self.models = models
        else:
            raise TypeError(f"Invalid type: {models!r}")

        self.models.extend(self.datasets.models)

        for dataset in self.datasets:

            if dataset.background_model is None:
                bkg_model = FoVBackgroundModel(dataset_name=dataset.name)

            self.models.append(bkg_model)

        self.datasets.models = self.models

        log.info(self.models)

    def read_models(self, path):
        """Read models from YAML file."""
        path = make_path(path)
        models = Models.read(path)
        self.set_models(models)

    def run_fit(self, optimize_opts=None):
        """Fitting reduced datasets to model."""
        if not self.models:
            raise RuntimeError("Missing models")

        fit_settings = self.config.fit
        for dataset in self.datasets:
            if fit_settings.fit_range:
                energy_min = fit_settings.fit_range.min
                energy_max = fit_settings.fit_range.max
                geom = dataset.counts.geom
                data = geom.energy_mask(energy_min, energy_max)
                dataset.mask_fit = Map.from_geom(geom=geom, data=data)

        log.info("Fitting datasets.")
        self.fit = Fit(self.datasets)
        self.fit_result = self.fit.run(optimize_opts=optimize_opts)
        log.info(self.fit_result)

    def get_flux_points(self):
        """Calculate flux points for a specific model component."""
        if not self.fit:
            raise RuntimeError("No results available from Fit.")

        fp_settings = self.config.flux_points
        log.info("Calculating flux points.")
        energy_edges = self._make_energy_axis(fp_settings.energy).edges
        flux_point_estimator = FluxPointsEstimator(
            energy_edges=energy_edges,
            source=fp_settings.source,
            **fp_settings.parameters,
        )
        fp = flux_point_estimator.run(datasets=self.datasets)
        fp.table["is_ul"] = fp.table["ts"] < 4
        self.flux_points = FluxPointsDataset(
            data=fp, models=self.models[fp_settings.source]
        )
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
            geom_params["frame"] = "icrs"
        if skydir_settings.frame == "galactic":
            geom_params["frame"] = "galactic"
        axes = [self._make_energy_axis(geom_settings.axes.energy)]
        geom_params["axes"] = axes
        geom_params["binsz"] = geom_settings.wcs.binsize
        width = geom_settings.wcs.fov.width.to("deg").value
        height = geom_settings.wcs.fov.height.to("deg").value
        geom_params["width"] = (width, height)
        return WcsGeom.create(**geom_params)

    def _map_making(self):
        """Make maps and datasets for 3d analysis."""
        datasets_settings = self.config.datasets
        log.info("Creating geometry.")
        geom = self._create_geometry()
        geom_settings = datasets_settings.geom
        geom_irf = dict(energy_axis_true=None, binsz_irf=None)
        if geom_settings.axes.energy_true.min is not None:
            geom_irf["energy_axis_true"] = self._make_energy_axis(
                geom_settings.axes.energy_true, name="energy_true"
            )
        geom_irf["binsz_irf"] = geom_settings.wcs.binsize_irf.to("deg").value
        offset_max = geom_settings.selection.offset_max
        log.info("Creating datasets.")

        maker = MapDatasetMaker(selection=datasets_settings.map_selection)

        safe_mask_selection = datasets_settings.safe_mask.methods
        safe_mask_settings = datasets_settings.safe_mask.parameters
        maker_safe_mask = SafeMaskMaker(
            methods=safe_mask_selection, **safe_mask_settings
        )

        bkg_maker_config = {}
        if datasets_settings.background.exclusion:
            exclusion_region = Map.read(datasets_settings.background.exclusion)
            bkg_maker_config["exclusion_mask"] = exclusion_region
        bkg_maker_config.update(datasets_settings.background.parameters)

        bkg_method = datasets_settings.background.method
        if bkg_method == "fov_background":
            log.debug(f"Creating FoVBackgroundMaker with arguments {bkg_maker_config}")
            bkg_maker = FoVBackgroundMaker(**bkg_maker_config)
        elif bkg_method == "ring":
            bkg_maker = RingBackgroundMaker(**bkg_maker_config)
            log.debug(f"Creating RingBackgroundMaker with arguments {bkg_maker_config}")
            if datasets_settings.geom.axes.energy.nbins > 1:
                raise ValueError(
                    "You need to define a single-bin energy geometry for your dataset."
                )
        else:
            bkg_maker = None
            log.warning(
                f"No background maker set for 3d analysis. Check configuration."
            )

        stacked = MapDataset.create(geom=geom, name="stacked", **geom_irf)

        if datasets_settings.stack:
            for obs in self.observations:
                log.info(f"Processing observation {obs.obs_id}")
                cutout = stacked.cutout(obs.pointing_radec, width=2 * offset_max)
                dataset = maker.run(cutout, obs)
                dataset = maker_safe_mask.run(dataset, obs)
                if bkg_maker is not None:
                    dataset = bkg_maker.run(dataset)

                if bkg_method == "ring":
                    dataset = dataset.to_map_dataset()

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
                if bkg_maker is not None:
                    dataset = bkg_maker.run(dataset)
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
        e_reco = self._make_energy_axis(datasets_settings.geom.axes.energy)

        maker_config["selection"] = ["counts", "exposure", "edisp"]
        dataset_maker = SpectrumDatasetMaker(**maker_config)

        bkg_maker_config = {}
        if datasets_settings.background.exclusion:
            exclusion_region = Map.read(datasets_settings.background.exclusion)
            bkg_maker_config["exclusion_mask"] = exclusion_region
        bkg_maker_config.update(datasets_settings.background.parameters)
        bkg_method = datasets_settings.background.method
        if bkg_method == "reflected":
            bkg_maker = ReflectedRegionsBackgroundMaker(**bkg_maker_config)
            log.debug(
                f"Creating ReflectedRegionsBackgroundMaker with arguments {bkg_maker_config}"
            )
        else:
            bkg_maker = None
            log.warning(
                f"No background maker set for 1d analysis. Check configuration."
            )

        safe_mask_selection = datasets_settings.safe_mask.methods
        safe_mask_settings = datasets_settings.safe_mask.parameters
        safe_mask_maker = SafeMaskMaker(
            methods=safe_mask_selection, **safe_mask_settings
        )

        e_true = self._make_energy_axis(
            datasets_settings.geom.axes.energy_true, name="energy_true"
        )

        reference = SpectrumDataset.create(
            e_reco=e_reco, e_true=e_true, region=on_region
        )

        datasets = []
        for obs in self.observations:
            log.info(f"Processing observation {obs.obs_id}")
            dataset = dataset_maker.run(reference.copy(), obs)
            if bkg_maker is not None:
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

        if datasets_settings.stack:
            stacked = self.datasets.stack_reduce(name="stacked")
            self.datasets = Datasets([stacked])

    @staticmethod
    def _make_energy_axis(axis, name="energy"):
        return MapAxis.from_bounds(
            name=name,
            lo_bnd=axis.min.value,
            hi_bnd=axis.max.to_value(axis.min.unit),
            nbin=axis.nbins,
            unit=axis.min.unit,
            interp="log",
            node_type="edges",
        )
