# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Session class driving the high level interface API"""
import abc
import os
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
    FoVBackgroundMaker,
    MapDatasetMaker,
    ReflectedRegionsBackgroundMaker,
    RingBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
    DatasetsMaker
)
from gammapy.maps import Map, MapAxis, RegionGeom, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import FoVBackgroundModel, Models
from gammapy.utils.pbar import progress_bar
from gammapy.utils.scripts import make_path
from gammapy.utils.registry import Registry

__all__ = ["Analysis"]

log = logging.getLogger(__name__)


class AnalysisStepBase(abc.ABC):
    tag = "analysis-step"
    def __init__(self, analysis, overwrite=True):
        self.analysis = analysis
        self.overwrite = overwrite

    @abc.abstractmethod
    def run(self):
        pass
    
class AnalysisStep:
    "Create one of the analysis step class listed in the registry"
    @staticmethod
    def create(tag, analysis, **kwargs):
        cls = ANALYSIS_STEP_REGISTRY.get_cls(tag)
        return cls(analysis, **kwargs)

class DataReductionAnalysisStep(AnalysisStepBase):
    tag = "data-reduction"

    def run(self):
        filepath = self.analysis.config.general.datasets_file
        if filepath is not None and os.path.exists(filepath) and not self.overwrite:
            self.analysis.read_datasets()
        else:
            ObservationsAnalysisStep(self.analysis, self.overwrite).run()
            DatasetsAnalysisStep(self.analysis, self.overwrite).get_datasets()
            self.analysis.write_datasets()

class ObservationsAnalysisStep(AnalysisStepBase):
    tag = "observations"

    def run(self):
        """Fetch observations from the data store according to criteria defined in the configuration."""
        observations_settings = self.analysis.config.observations
        self._set_data_store()

        log.info("Fetching observations.")
        ids = self._make_obs_table_selection()

        self.analysis.observations = self.analysis.datastore.get_observations(
            ids, skip_missing=True, required_irf=observations_settings.required_irf
        )

        if observations_settings.obs_time.start is not None:
            start = observations_settings.obs_time.start
            stop = observations_settings.obs_time.stop
            if len(start.shape) == 0:
                time_intervals = [(start, stop)]
            else:
                time_intervals = [(tstart, tstop) for tstart, tstop in zip(start, stop)]
            self.analysis.observations = self.analysis.observations.select_time(time_intervals)

        for obs in self.analysis.observations:
            log.debug(obs)

    def _set_data_store(self):
        """Set the datastore on the Analysis object."""
        path = make_path(self.analysis.config.observations.datastore)
        if path.is_file():
            log.debug(f"Setting datastore from file: {path}")
            self.analysis.datastore = DataStore.from_file(path)
        elif path.is_dir():
            log.debug(f"Setting datastore from directory: {path}")
            self.analysis.datastore = DataStore.from_dir(path)
        else:
            raise FileNotFoundError(f"Datastore not found: {path}")

    def _make_obs_table_selection(self):
        """Return list of obs_ids after filtering on datastore observation table."""
        obs_settings = self.analysis.config.observations

        # Reject configs with list of obs_ids and obs_file set at the same time
        if len(obs_settings.obs_ids) and obs_settings.obs_file is not None:
            raise ValueError(
                "Values for both parameters obs_ids and obs_file are not accepted."
            )

        # First select input list of observations from obs_table
        if len(obs_settings.obs_ids):
            selected_obs_table = self.analysis.datastore.obs_table.select_obs_id(
                obs_settings.obs_ids
            )
        elif obs_settings.obs_file is not None:
            path = make_path(obs_settings.obs_file)
            ids = list(Table.read(path, format="ascii", data_start=0).columns[0])
            selected_obs_table = self.analysis.datastore.obs_table.select_obs_id(ids)
        else:
            selected_obs_table = self.analysis.datastore.obs_table

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


class DatasetsAnalysisStep(AnalysisStepBase):
    tag = "datasets"

    def run(self):
        filepath = self.analysis.config.general.datasets_file
        if filepath is not None and os.path.exists(filepath) and not self.overwrite:
            self.analysis.read_datasets()
        else:
            self.get_datasets()
            self.analysis.write_datasets()

    def get_datasets(self):
        """Produce reduced datasets."""
        datasets_settings = self.analysis.config.datasets
        if not self.analysis.observations or len(self.analysis.observations) == 0:
            raise RuntimeError("No observations have been selected.")

        if datasets_settings.type == "1d":
            self._spectrum_extraction()
        else:  # 3d
            self._map_making()
            
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
        datasets_settings = self.analysis.config.datasets
        geom_settings = datasets_settings.geom
        axes = [make_energy_axis(geom_settings.axes.energy)]
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

        geom_settings = self.analysis.config.datasets.geom
        geom_irf = dict(energy_axis_true=None, binsz_irf=None)
        if geom_settings.axes.energy_true.min is not None:
            geom_irf["energy_axis_true"] = make_energy_axis(
                geom_settings.axes.energy_true, name="energy_true"
            )
        if geom_settings.wcs.binsize_irf is not None:
            geom_irf["binsz_irf"] = geom_settings.wcs.binsize_irf.to("deg").value

        if self.analysis.config.datasets.type == "1d":
            return SpectrumDataset.create(geom, name=name, **geom_irf)
        else:
            return MapDataset.create(geom, name=name, **geom_irf)

    def _create_dataset_maker(self):
        """Create the Dataset Maker."""
        log.debug("Creating the target Dataset Maker.")

        datasets_settings = self.analysis.config.datasets
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

        safe_mask_selection = self.analysis.config.datasets.safe_mask.methods
        safe_mask_settings = self.analysis.config.datasets.safe_mask.parameters
        return SafeMaskMaker(methods=safe_mask_selection, **safe_mask_settings)

    def _create_background_maker(self):
        """Create the Background maker."""
        log.info("Creating the background Maker.")

        datasets_settings = self.analysis.config.datasets
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
        datasets_settings = self.analysis.config.datasets
        offset_max = datasets_settings.geom.selection.offset_max

        log.info("Creating reference dataset and makers.")
        stacked = self._create_reference_dataset(name="stacked")

        maker = self._create_dataset_maker()
        maker_safe_mask = self._create_safe_mask_maker()
        bkg_maker = self._create_background_maker()
        makers = [maker, maker_safe_mask, bkg_maker]
        makers = [maker for maker in makers if maker is not None]
        log.info("Start the data reduction loop.")
        
        datasets_maker = DatasetsMaker(makers,
                                      stack_datasets=datasets_settings.stack,
                                      n_jobs=self.analysis.config.general.n_jobs,
                                      cutout_mode='partial',
                                      cutout_width=2 * offset_max)
        self.analysis.datasets = datasets_maker.run(stacked, self.analysis.observations)
        #TODO: move progress bar to DatasetsMaker but how with multiprocessing ?

    def _spectrum_extraction(self):
        """Run all steps for the spectrum extraction."""
        log.info("Reducing spectrum datasets.")
        datasets_settings = self.analysis.config.datasets
        dataset_maker = self._create_dataset_maker()
        safe_mask_maker = self._create_safe_mask_maker()
        bkg_maker = self._create_background_maker()

        reference = self._create_reference_dataset()

        datasets = []
        for obs in progress_bar(self.analysis.observations, desc="Observations"):
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
        self.analysis.datasets = Datasets(datasets)

        if datasets_settings.stack:
            stacked = self.analysis.datasets.stack_reduce(name="stacked")
            self.analysis.datasets = Datasets([stacked])


def make_energy_axis(axis, name="energy"):
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

class ExcessMapAnalysisStep(AnalysisStepBase):
    tag = "excess-map"
    def run(self):
        """Calculate excess map with respect to the current model."""
        excess_settings = self.analysis.config.excess_map
        #TODO: allow a list of kernel sizes
        log.info("Computing excess maps.")

        if self.analysis.config.datasets.type == "1d":
            raise ValueError("Cannot compute excess map for 1D dataset")

        # Here we could possibly stack the datasets if needed.
        if len(self.analysis.datasets) > 1:
            raise ValueError("Datasets must be stacked to compute the excess map")

        energy_edges = make_energy_axis(excess_settings.energy_edges)
        if energy_edges is not None:
            energy_edges = energy_edges.edges

        excess_map_estimator = ExcessMapEstimator(
            correlation_radius=excess_settings.correlation_radius,
            energy_edges=energy_edges,
            **excess_settings.parameters,
        )
        self.analysis.excess_map = excess_map_estimator.run(self.analysis.datasets[0])

class FitAnalysisStep(AnalysisStepBase):
    tag = "fit"
    def run(self):
        """Fitting reduced datasets to model."""
        if not self.analysis.models:
            raise RuntimeError("Missing models")
    
        fit_settings = self.analysis.config.fit
        for dataset in self.analysis.datasets:
            if fit_settings.fit_range:
                energy_min = fit_settings.fit_range.min
                energy_max = fit_settings.fit_range.max
                #TODO : add fit range in lon/lat 
                geom = dataset.counts.geom
                dataset.mask_fit = geom.energy_mask(energy_min, energy_max)
    
        log.info("Fitting datasets.")
        result = self.analysis.fit.run(datasets=self.analysis.datasets)
        self.analysis.fit_result = result
        log.info(self.analysis.fit_result)


class FluxPointsAnalysisStep(AnalysisStepBase):
    tag="flux-points"
    def run(self):
        """Calculate flux points for a specific model component."""
        if not self.analysis.datasets:
            raise RuntimeError("No datasets set.")
    
        fp_settings = self.analysis.config.flux_points
        log.info("Calculating flux points.")
        energy_edges = make_energy_axis(fp_settings.energy).edges
        flux_point_estimator = FluxPointsEstimator(
            energy_edges=energy_edges,
            source=fp_settings.source,
            fit=self.analysis.fit,
            **fp_settings.parameters,
        )
    
        fp = flux_point_estimator.run(datasets=self.analysis.datasets)
    
        self.analysis.flux_points = FluxPointsDataset(
            data=fp, models=self.analysis.models[fp_settings.source]
        )
        cols = ["e_ref", "dnde", "dnde_ul", "dnde_err", "sqrt_ts"]
        table = self.analysis.flux_points.data.to_table(sed_type="dnde")
        log.info("\n{}".format(table[cols]))

class LightCurveAnalysisStep(AnalysisStepBase):
    tag= "light-curve"
    def run(self):
        """Calculate light curve for a specific model component."""
        lc_settings = self.analysis.config.light_curve
        log.info("Computing light curve.")
        energy_edges = make_energy_axis(lc_settings.energy_edges).edges

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
            fit=self.analysis.fit,
            **lc_settings.parameters,
        )
        lc = light_curve_estimator.run(datasets=self.analysis.datasets)
        self.analysis.light_curve = lc
        log.info(
            "\n{}".format(
                self.analysis.light_curve.to_table(format="lightcurve", sed_type="flux")
            )
        )

ANALYSIS_STEP_REGISTRY = Registry([
        DataReductionAnalysisStep,
        ObservationsAnalysisStep,
        DatasetsAnalysisStep,
        ExcessMapAnalysisStep,
        FitAnalysisStep,
        FluxPointsAnalysisStep,
        LightCurveAnalysisStep,
        ])

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
        self.read_datasets()
        self.read_models()
        self.fit = Fit()
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

    def update_config(self, config):
        self.config = self.config.update(config=config)

    def run(self, steps=None, overwrite=None, **kwargs):
        if steps is None:
            steps = self.config.general.steps
            overwrite = self.config.general.overwrite
        else:
            if overwrite is None:
                overwrite = [True]*len(steps)
        for k,step in enumerate(steps):
            if isinstance(overwrite, list):
                overwrite_step = overwrite[k]
            else :
                overwrite_step = overwrite
            anlysis_step = AnalysisStep.create(step,
                                               self,
                                               overwrite=overwrite_step,
                                               **kwargs)
            anlysis_step.run()

    def set_models(self, models, extend=True):
        """Set models on datasets.

        Adds `FoVBackgroundModel` if not present already

        Parameters
        ----------
        models : `~gammapy.modeling.models.Models` or str
            Models object or YAML models string
        extend : bool
            Extent the exiting models on the datasets or replace them.
        """
        if not self.datasets or len(self.datasets) == 0:
            raise RuntimeError("Missing datasets")

        log.info("Reading model.")
        if isinstance(models, str):
            self.models = Models.from_yaml(models)
        elif isinstance(models, Models):
            self.models = models
        else:
            raise TypeError(f"Invalid type: {models!r}")

        if extend:
            self.models.extend(self.datasets.models)

        if self.config.datasets.type == "3d":
            for dataset in self.datasets:
                if dataset.background_model is None:
                    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)

                self.models.append(bkg_model)

        self.datasets.models = self.models

        log.info(self.models)

    def read_models(self, filename=None, extend=True):
        """Read models from YAML file."""
        _, filename = self._update_datasets_files(filename_models=filename)
        if filename is not None :
            models = Models.read(filename)
            self.set_models(models, extend)
            log.info(f"Models loaded from {filename}.")
        else:
            self.models = None
 
    def read_datasets(self, filename=None, filename_models=None):
        """Read datasets from YAML file."""
        filename, filename_models = self._update_datasets_files(filename, filename_models)
        if filename is not None:
            self.datasets = Datasets.read(filename, filename_models)
            log.info(f"Datasets loaded from {filename}.")
            log.info(f"Models loaded from {filename_models}.")
        else:
            self.datasets = None


    def write_datasets(self, filename=None, filename_models=None):
        """Read datasets to YAML file."""
        filename, filename_models = self._update_datasets_files(filename, filename_models)
        if filename is not None:
            self.datasets.write(filename,
                                 filename_models,
                                 overwrite=True,
                                 write_covariance=True)
            log.info(f"Datasets stored to {filename}.")
            log.info(f"Datasets stored to {filename_models}.")


    def _update_datasets_files(self, filename=None, filename_models=None):
        config = self.config
        if filename is None:
            filename = self.config.general.datasets_file
        else: 
            config.general.datasets_file = filename
            self.update_config(config)
        if filename_models is None:
            filename_models = self.config.general.models_file
        else: 
            config.general.models_file = filename_models
            self.update_config(config)
        return filename, filename_models