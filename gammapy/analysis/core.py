# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high level interface API"""
import logging
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import CircleSkyRegion
from gammapy.analysis.config import AnalysisConfig
from gammapy.data import DataStore
from gammapy.datasets import Datasets, FluxPointsDataset, MapDataset, SpectrumDataset
from gammapy.estimators import (
    ExcessMapEstimator,
    FluxPointsEstimator,
    LightCurveEstimator,
)
from gammapy.makers import (
    DatasetsMaker,
    FoVBackgroundMaker,
    MapDatasetMaker,
    ReflectedRegionsBackgroundMaker,
    RingBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.maps import Map, MapAxis, RegionGeom, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import DatasetModels, FoVBackgroundModel, Models
from gammapy.utils.pbar import progress_bar
from gammapy.utils.scripts import make_path

__all__ = ["Analysis"]

log = logging.getLogger(__name__)


class Analysis:
    """Config-driven high level analysis interface.

    It is initialized by default with a set of configuration parameters and values declared in
    an internal high level interface model, though the user can also provide configuration
    parameters passed as a nested dictionary at the moment of instantiation. In that case these
    parameters will overwrite the default values of those present in the configuration file.

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
        self.fit = Fit()
        self.fit_result = None
        self.flux_points = None

    @property
    def models(self):
        if not self.datasets:
            raise RuntimeError("No datasets defined. Impossible to set models.")
        return self.datasets.models

    @models.setter
    def models(self, models):
        self.set_models(models, extend=False)

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

    def _set_data_store(self):
        """Set the datastore on the Analysis object."""
        path = make_path(self.config.observations.datastore)
        if path.is_file():
            log.debug(f"Setting datastore from file: {path}")
            self.datastore = DataStore.from_file(path)
        elif path.is_dir():
            log.debug(f"Setting datastore from directory: {path}")
            self.datastore = DataStore.from_dir(path)
        else:
            raise FileNotFoundError(f"Datastore not found: {path}")

    def _make_obs_table_selection(self):
        """Return list of obs_ids after filtering on datastore observation table."""
        obs_settings = self.config.observations

        # Reject configs with list of obs_ids and obs_file set at the same time
        if len(obs_settings.obs_ids) and obs_settings.obs_file is not None:
            raise ValueError(
                "Values for both parameters obs_ids and obs_file are not accepted."
            )

        # First select input list of observations from obs_table
        if len(obs_settings.obs_ids):
            selected_obs_table = self.datastore.obs_table.select_obs_id(
                obs_settings.obs_ids
            )
        elif obs_settings.obs_file is not None:
            path = make_path(obs_settings.obs_file)
            ids = list(Table.read(path, format="ascii", data_start=0).columns[0])
            selected_obs_table = self.datastore.obs_table.select_obs_id(ids)
        else:
            selected_obs_table = self.datastore.obs_table

        # Apply cone selection
        if obs_settings.obs_cone.lon is not None:
            cone = dict(
                type="sky_circle",
                frame=obs_settings.obs_cone.frame,
                lon=obs_settings.obs_cone.lon,
                lat=obs_settings.obs_cone.lat,
                radius=obs_settings.obs_cone.radius,
                border="0 deg",
            )
            selected_obs_table = selected_obs_table.select_observations(cone)

        return selected_obs_table["OBS_ID"].tolist()

    def get_observations(self):
        """Fetch observations from the data store according to criteria defined
        in the configuration."""
        observations_settings = self.config.observations
        self._set_data_store()

        log.info("Fetching observations.")
        ids = self._make_obs_table_selection()
        required_irf = [_.value for _ in observations_settings.required_irf]
        self.observations = self.datastore.get_observations(
            ids, skip_missing=True, required_irf=required_irf
        )

        if observations_settings.obs_time.start is not None:
            start = observations_settings.obs_time.start
            stop = observations_settings.obs_time.stop
            if len(start.shape) == 0:
                time_intervals = [(start, stop)]
            else:
                time_intervals = [(tstart, tstop) for tstart, tstop in zip(start, stop)]
            self.observations = self.observations.select_time(time_intervals)

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

    def set_models(self, models, extend=True):
        """Set models on datasets.
        Adds `FoVBackgroundModel` if not present already

        Parameters
        ----------
        models : `~gammapy.modeling.models.Models` or str
            Models object or YAML models string
        extend : bool
            Extend the exiting models on the datasets or replace them.
        """
        if not self.datasets or len(self.datasets) == 0:
            raise RuntimeError("Missing datasets")

        log.info("Reading model.")
        if isinstance(models, str):
            models = Models.from_yaml(models)
        elif isinstance(models, Models):
            pass
        elif isinstance(models, DatasetModels) or isinstance(models, list):
            models = Models(models)
        else:
            raise TypeError(f"Invalid type: {models!r}")

        if extend:
            models.extend(self.datasets.models)

        self.datasets.models = models

        bkg_models = []
        for dataset in self.datasets:
            if dataset.tag == "MapDataset" and dataset.background_model is None:
                bkg_models.append(FoVBackgroundModel(dataset_name=dataset.name))
        if bkg_models:
            models.extend(bkg_models)
            self.datasets.models = models

        log.info(models)

    def read_models(self, path, extend=True):
        """Read models from YAML file.

        Parameters
        ----------
        path : str
            path to the model file
        extend : bool
            Extend the exiting models on the datasets or replace them.
        """

        path = make_path(path)
        models = Models.read(path)
        self.set_models(models, extend=extend)
        log.info(f"Models loaded from {path}.")

    def write_models(self, overwrite=True, write_covariance=True):
        """Write models to YAML file.
        File name is taken from the configuration file.
        """

        filename_models = self.config.general.models_file
        if filename_models is not None:
            self.models.write(
                filename_models, overwrite=overwrite, write_covariance=write_covariance
            )
            log.info(f"Models loaded from {filename_models}.")
        else:
            raise RuntimeError("Missing models_file in config.general")

    def read_datasets(self):
        """Read datasets from YAML file.
        File names are taken from the configuration file.

        """

        filename = self.config.general.datasets_file
        filename_models = self.config.general.models_file
        if filename is not None:
            self.datasets = Datasets.read(filename)
            log.info(f"Datasets loaded from {filename}.")
            if filename_models is not None:
                self.read_models(filename_models, extend=False)
        else:
            raise RuntimeError("Missing datasets_file in config.general")

    def write_datasets(self, overwrite=True, write_covariance=True):
        """Write datasets to YAML file.
        File names are taken from the configuration file.

        Parameters
        ----------
        overwrite : bool
            overwrite datasets FITS files
        write_covariance : bool
            save covariance or not
        """

        filename = self.config.general.datasets_file
        filename_models = self.config.general.models_file
        if filename is not None:
            self.datasets.write(
                filename,
                filename_models,
                overwrite=overwrite,
                write_covariance=write_covariance,
            )
            log.info(f"Datasets stored to {filename}.")
            log.info(f"Datasets stored to {filename_models}.")
        else:
            raise RuntimeError("Missing datasets_file in config.general")

    def run_fit(self):
        """Fitting reduced datasets to model."""
        if not self.models:
            raise RuntimeError("Missing models")

        fit_settings = self.config.fit
        for dataset in self.datasets:
            if fit_settings.fit_range:
                energy_min = fit_settings.fit_range.min
                energy_max = fit_settings.fit_range.max
                geom = dataset.counts.geom
                dataset.mask_fit = geom.energy_mask(energy_min, energy_max)

        log.info("Fitting datasets.")
        result = self.fit.run(datasets=self.datasets)
        self.fit_result = result
        log.info(self.fit_result)

    def get_flux_points(self):
        """Calculate flux points for a specific model component."""
        if not self.datasets:
            raise RuntimeError(
                "No datasets defined. Impossible to compute flux points."
            )

        fp_settings = self.config.flux_points
        log.info("Calculating flux points.")
        energy_edges = self._make_energy_axis(fp_settings.energy).edges
        flux_point_estimator = FluxPointsEstimator(
            energy_edges=energy_edges,
            source=fp_settings.source,
            fit=self.fit,
            **fp_settings.parameters,
        )

        fp = flux_point_estimator.run(datasets=self.datasets)

        self.flux_points = FluxPointsDataset(
            data=fp, models=self.models[fp_settings.source]
        )
        cols = ["e_ref", "dnde", "dnde_ul", "dnde_err", "sqrt_ts"]
        table = self.flux_points.data.to_table(sed_type="dnde")
        log.info("\n{}".format(table[cols]))

    def get_excess_map(self):
        """Calculate excess map with respect to the current model."""
        excess_settings = self.config.excess_map
        log.info("Computing excess maps.")

        # TODO: Here we could possibly stack the datasets if needed
        # or allow to compute the excess map for each dataset
        if len(self.datasets) > 1:
            raise ValueError("Datasets must be stacked to compute the excess map")

        if self.datasets[0].tag not in ["MapDataset", "MapDatasetOnOff"]:
            raise ValueError("Cannot compute excess map for 1D dataset")

        energy_edges = self._make_energy_axis(excess_settings.energy_edges)
        if energy_edges is not None:
            energy_edges = energy_edges.edges

        excess_map_estimator = ExcessMapEstimator(
            correlation_radius=excess_settings.correlation_radius,
            energy_edges=energy_edges,
            **excess_settings.parameters,
        )
        self.excess_map = excess_map_estimator.run(self.datasets[0])

    def get_light_curve(self):
        """Calculate light curve for a specific model component."""
        lc_settings = self.config.light_curve
        log.info("Computing light curve.")
        energy_edges = self._make_energy_axis(lc_settings.energy_edges).edges

        if (
            lc_settings.time_intervals.start is None
            or lc_settings.time_intervals.stop is None
        ):
            log.info(
                "Time intervals not defined. Extract light curve on datasets GTIs."
            )
            time_intervals = None
        else:
            time_intervals = [
                (t1, t2)
                for t1, t2 in zip(
                    lc_settings.time_intervals.start, lc_settings.time_intervals.stop
                )
            ]

        light_curve_estimator = LightCurveEstimator(
            time_intervals=time_intervals,
            energy_edges=energy_edges,
            source=lc_settings.source,
            fit=self.fit,
            **lc_settings.parameters,
        )
        lc = light_curve_estimator.run(datasets=self.datasets)
        self.light_curve = lc
        log.info(
            "\n{}".format(
                self.light_curve.to_table(format="lightcurve", sed_type="flux")
            )
        )

    def update_config(self, config):
        self.config = self.config.update(config=config)

    @staticmethod
    def _create_wcs_geometry(wcs_geom_settings, axes):
        """Create the WCS geometry."""
        geom_params = {}
        skydir_settings = wcs_geom_settings.skydir
        if skydir_settings.lon is not None:
            skydir = SkyCoord(
                skydir_settings.lon, skydir_settings.lat, frame=skydir_settings.frame
            )
            geom_params["skydir"] = skydir

        if skydir_settings.frame in ["icrs", "galactic"]:
            geom_params["frame"] = skydir_settings.frame
        else:
            raise ValueError(
                f"Incorrect skydir frame: expect 'icrs' or 'galactic'. Got {skydir_settings.frame}"
            )

        geom_params["axes"] = axes
        geom_params["binsz"] = wcs_geom_settings.binsize
        width = wcs_geom_settings.width.width.to("deg").value
        height = wcs_geom_settings.width.height.to("deg").value
        geom_params["width"] = (width, height)

        return WcsGeom.create(**geom_params)

    @staticmethod
    def _create_region_geometry(on_region_settings, axes):
        """Create the region geometry."""
        on_lon = on_region_settings.lon
        on_lat = on_region_settings.lat
        on_center = SkyCoord(on_lon, on_lat, frame=on_region_settings.frame)
        on_region = CircleSkyRegion(on_center, on_region_settings.radius)

        return RegionGeom.create(region=on_region, axes=axes)

    def _create_geometry(self):
        """Create the geometry."""
        log.debug("Creating geometry.")
        datasets_settings = self.config.datasets
        geom_settings = datasets_settings.geom
        axes = [self._make_energy_axis(geom_settings.axes.energy)]
        if datasets_settings.type == "3d":
            geom = self._create_wcs_geometry(geom_settings.wcs, axes)
        elif datasets_settings.type == "1d":
            geom = self._create_region_geometry(datasets_settings.on_region, axes)
        else:
            raise ValueError(
                f"Incorrect dataset type. Expect '1d' or '3d'. Got {datasets_settings.type}."
            )
        return geom

    def _create_reference_dataset(self, name=None):
        """Create the reference dataset for the current analysis."""
        log.debug("Creating target Dataset.")
        geom = self._create_geometry()

        geom_settings = self.config.datasets.geom
        geom_irf = dict(energy_axis_true=None, binsz_irf=None)
        if geom_settings.axes.energy_true.min is not None:
            geom_irf["energy_axis_true"] = self._make_energy_axis(
                geom_settings.axes.energy_true, name="energy_true"
            )
        if geom_settings.wcs.binsize_irf is not None:
            geom_irf["binsz_irf"] = geom_settings.wcs.binsize_irf.to("deg").value

        if self.config.datasets.type == "1d":
            return SpectrumDataset.create(geom, name=name, **geom_irf)
        else:
            return MapDataset.create(geom, name=name, **geom_irf)

    def _create_dataset_maker(self):
        """Create the Dataset Maker."""
        log.debug("Creating the target Dataset Maker.")

        datasets_settings = self.config.datasets
        if datasets_settings.type == "3d":
            maker = MapDatasetMaker(selection=datasets_settings.map_selection)
        elif datasets_settings.type == "1d":
            maker_config = {}
            if datasets_settings.containment_correction:
                maker_config[
                    "containment_correction"
                ] = datasets_settings.containment_correction

            maker_config["selection"] = ["counts", "exposure", "edisp"]

            maker = SpectrumDatasetMaker(**maker_config)

        return maker

    def _create_safe_mask_maker(self):
        """Create the SafeMaskMaker."""
        log.debug("Creating the mask_safe Maker.")

        safe_mask_selection = self.config.datasets.safe_mask.methods
        safe_mask_settings = self.config.datasets.safe_mask.parameters
        return SafeMaskMaker(methods=safe_mask_selection, **safe_mask_settings)

    def _create_background_maker(self):
        """Create the Background maker."""
        log.info("Creating the background Maker.")

        datasets_settings = self.config.datasets
        bkg_maker_config = {}
        if datasets_settings.background.exclusion:
            path = make_path(datasets_settings.background.exclusion)
            exclusion_mask = Map.read(path)
            exclusion_mask.data = exclusion_mask.data.astype(bool)
            bkg_maker_config["exclusion_mask"] = exclusion_mask
        bkg_maker_config.update(datasets_settings.background.parameters)

        bkg_method = datasets_settings.background.method

        bkg_maker = None
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
        elif bkg_method == "reflected":
            bkg_maker = ReflectedRegionsBackgroundMaker(**bkg_maker_config)
            log.debug(
                f"Creating ReflectedRegionsBackgroundMaker with arguments {bkg_maker_config}"
            )
        else:
            log.warning("No background maker set. Check configuration.")
        return bkg_maker

    def _map_making(self):
        """Make maps and datasets for 3d analysis"""
        datasets_settings = self.config.datasets
        offset_max = datasets_settings.geom.selection.offset_max

        log.info("Creating reference dataset and makers.")
        stacked = self._create_reference_dataset(name="stacked")

        maker = self._create_dataset_maker()
        maker_safe_mask = self._create_safe_mask_maker()
        bkg_maker = self._create_background_maker()

        makers = [maker, maker_safe_mask, bkg_maker]
        makers = [maker for maker in makers if maker is not None]

        log.info("Start the data reduction loop.")

        datasets_maker = DatasetsMaker(
            makers,
            stack_datasets=datasets_settings.stack,
            n_jobs=self.config.general.n_jobs,
            cutout_mode="trim",
            cutout_width=2 * offset_max,
        )
        self.datasets = datasets_maker.run(stacked, self.observations)
        # TODO: move progress bar to DatasetsMaker but how with multiprocessing ?

    def _spectrum_extraction(self):
        """Run all steps for the spectrum extraction."""
        log.info("Reducing spectrum datasets.")
        datasets_settings = self.config.datasets
        dataset_maker = self._create_dataset_maker()
        safe_mask_maker = self._create_safe_mask_maker()
        bkg_maker = self._create_background_maker()

        reference = self._create_reference_dataset()

        datasets = []
        for obs in progress_bar(self.observations, desc="Observations"):
            log.debug(f"Processing observation {obs.obs_id}")
            dataset = dataset_maker.run(reference.copy(), obs)
            if bkg_maker is not None:
                dataset = bkg_maker.run(dataset, obs)
                if dataset.counts_off is None:
                    log.debug(
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
        if axis.min is None or axis.max is None:
            return None
        elif axis.nbins is None or axis.nbins < 1:
            return None
        else:
            return MapAxis.from_bounds(
                name=name,
                lo_bnd=axis.min.value,
                hi_bnd=axis.max.to_value(axis.min.unit),
                nbin=axis.nbins,
                unit=axis.min.unit,
                interp="log",
                node_type="edges",
            )
