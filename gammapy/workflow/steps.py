# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""classes containing the workflow steps supported by the high level interface"""

import abc
import logging
import os
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import CircleSkyRegion
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
from gammapy.utils.pbar import progress_bar
from gammapy.utils.scripts import make_name, make_path


class WorkflowStepBase(abc.ABC):
    tag = "workflow-step"

    def __init__(self, config, log=None, name=None, overwrite=True):
        self.config = config
        self.overwrite = overwrite
        self._name = make_name(name)

        if log is None:
            log = logging.getLogger(__name__)
        self.log = log

    @property
    def name(self):
        return self._name

    def run(self, data):
        # TODO: for now data is an workflow class
        # but it will be replaced by a new specific container
        # and the type and origin of data required will be defined by config
        self.data = data
        self.products = self.data
        # TODO: for now it is the workflow class
        # but in future each step will declare the products they create
        # and init a specific container
        self._run()

    @abc.abstractmethod
    def _run(self):
        pass


class WorkflowStep:
    """Create one of the workflow step class listed in the registry."""

    @staticmethod
    def create(tag, config, **kwargs):
        from . import WORKFLOW_STEP_REGISTRY

        cls = WORKFLOW_STEP_REGISTRY.get_cls(tag)
        return cls(config, **kwargs)


class DataSelectionWorkflowStep(WorkflowStepBase):
    tag = "data-selection"

    def _run(self):
        filepath = self.config.general.datasets_file
        if filepath is not None and os.path.exists(filepath) and not self.overwrite:
            self.data.read_datasets()
        else:
            obs_step = ObservationsWorkflowStep(
                self.config, log=self.log, overwrite=self.overwrite
            )
            datasets_step = DatasetsWorkflowStep(
                self.config, log=self.log, overwrite=self.overwrite
            )
            obs_step.run(self.data)
            datasets_step.run(self.data)
            self.data.write_datasets()


class ObservationsWorkflowStep(WorkflowStepBase):
    tag = "observations"

    def _run(self):
        """Fetch observations from the data store according to criteria defined in the configuration."""
        observations_settings = self.config.observations
        self.data.datastore = self._get_data_store()

        self.log.info("Fetching observations.")
        ids = self._make_obs_table_selection()
        required_irf = observations_settings.required_irf
        self.data.observations = self.data.datastore.get_observations(
            ids, skip_missing=True, required_irf=required_irf
        )

        if observations_settings.obs_time.start is not None:
            start = observations_settings.obs_time.start
            stop = observations_settings.obs_time.stop
            if len(start.shape) == 0:
                time_intervals = [(start, stop)]
            else:
                time_intervals = [(tstart, tstop) for tstart, tstop in zip(start, stop)]
            self.data.observations = self.data.observations.select_time(time_intervals)

        self.log.info(f"Number of selected observations: {len(self.data.observations)}")

        for obs in self.data.observations:
            self.log.debug(obs)

    def _get_data_store(self):
        """Set the datastore on the Workflow object."""
        path = make_path(self.config.observations.datastore)
        if path.is_file():
            self.log.debug(f"Setting datastore from file: {path}")
            return DataStore.from_file(path)
        elif path.is_dir():
            self.log.debug(f"Setting datastore from directory: {path}")
            return DataStore.from_dir(path)
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
            selected_obs_table = self.data.datastore.obs_table.select_obs_id(
                obs_settings.obs_ids
            )
        elif obs_settings.obs_file is not None:
            path = make_path(obs_settings.obs_file)
            ids = list(Table.read(path, format="ascii", data_start=0).columns[0])
            selected_obs_table = self.data.datastore.obs_table.select_obs_id(ids)
        else:
            selected_obs_table = self.data.datastore.obs_table

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


class DatasetsWorkflowStep(WorkflowStepBase):
    tag = "datasets"

    def _run(self):
        # TODO: check if exits and read else run and write
        self.get_datasets()

    def get_datasets(self):
        """Produce reduced datasets."""
        datasets_settings = self.config.datasets
        if not self.data.observations or len(self.data.observations) == 0:
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
        self.log.debug("Creating geometry.")
        datasets_settings = self.config.datasets
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
        """Create the reference dataset for the current workflow."""
        self.log.debug("Creating target Dataset.")
        geom = self._create_geometry()

        geom_settings = self.config.datasets.geom
        geom_irf = dict(energy_axis_true=None, binsz_irf=None)
        if geom_settings.axes.energy_true.min is not None:
            geom_irf["energy_axis_true"] = make_energy_axis(
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
        self.log.debug("Creating the target Dataset Maker.")

        datasets_settings = self.config.datasets
        if datasets_settings.type == "3d":
            maker = MapDatasetMaker(selection=datasets_settings.map_selection)
        elif datasets_settings.type == "1d":
            maker_config = {}
            if datasets_settings.containment_correction:
                maker_config["containment_correction"] = (
                    datasets_settings.containment_correction
                )

            maker_config["selection"] = ["counts", "exposure", "edisp"]

            maker = SpectrumDatasetMaker(**maker_config)

        return maker

    def _create_safe_mask_maker(self):
        """Create the SafeMaskMaker."""
        self.log.debug("Creating the mask_safe Maker.")

        safe_mask_selection = self.config.datasets.safe_mask.methods
        safe_mask_settings = self.config.datasets.safe_mask.parameters
        return SafeMaskMaker(methods=safe_mask_selection, **safe_mask_settings)

    def _create_background_maker(self):
        """Create the Background maker."""
        self.log.info("Creating the background Maker.")

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
            self.log.debug(
                f"Creating FoVBackgroundMaker with arguments {bkg_maker_config}"
            )
            bkg_maker = FoVBackgroundMaker(**bkg_maker_config)
        elif bkg_method == "ring":
            bkg_maker = RingBackgroundMaker(**bkg_maker_config)
            self.log.debug(
                f"Creating RingBackgroundMaker with arguments {bkg_maker_config}"
            )
            if datasets_settings.geom.axes.energy.nbins > 1:
                raise ValueError(
                    "You need to define a single-bin energy geometry for your dataset."
                )
        elif bkg_method == "reflected":
            bkg_maker = ReflectedRegionsBackgroundMaker(**bkg_maker_config)
            self.log.debug(
                f"Creating ReflectedRegionsBackgroundMaker with arguments {bkg_maker_config}"
            )
        else:
            self.log.warning("No background maker set. Check configuration.")
        return bkg_maker

    def _map_making(self):
        """Make maps and datasets for 3d workflow"""
        datasets_settings = self.config.datasets
        offset_max = datasets_settings.geom.selection.offset_max

        self.log.info("Creating reference dataset and makers.")
        stacked = self._create_reference_dataset(name="stacked")

        maker = self._create_dataset_maker()
        maker_safe_mask = self._create_safe_mask_maker()
        bkg_maker = self._create_background_maker()

        makers = [maker, maker_safe_mask, bkg_maker]
        makers = [maker for maker in makers if maker is not None]

        self.log.info("Start the data reduction loop.")

        datasets_maker = DatasetsMaker(
            makers,
            stack_datasets=datasets_settings.stack,
            n_jobs=self.config.general.n_jobs,
            cutout_mode="trim",
            cutout_width=2 * offset_max,
        )
        self.products.datasets = datasets_maker.run(stacked, self.data.observations)

    def _spectrum_extraction(self):
        """Run all steps for the spectrum extraction."""
        self.log.info("Reducing spectrum datasets.")
        datasets_settings = self.config.datasets
        dataset_maker = self._create_dataset_maker()
        safe_mask_maker = self._create_safe_mask_maker()
        bkg_maker = self._create_background_maker()

        reference = self._create_reference_dataset()

        datasets = []
        for obs in progress_bar(self.data.observations, desc="Observations"):
            self.log.debug(f"Processing observation {obs.obs_id}")
            dataset = dataset_maker.run(reference.copy(), obs)
            if bkg_maker is not None:
                dataset = bkg_maker.run(dataset, obs)
                if dataset.counts_off is None:
                    self.log.debug(
                        f"No OFF region found for observation {obs.obs_id}. Discarding."
                    )
                    continue
            dataset = safe_mask_maker.run(dataset, obs)
            self.log.debug(dataset)
            datasets.append(dataset)
        self.products.datasets = Datasets(datasets)

        if datasets_settings.stack:
            stacked = self.data.datasets.stack_reduce(name="stacked")
            self.products.datasets = Datasets([stacked])


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


class ExcessMapWorkflowStep(WorkflowStepBase):
    tag = "excess-map"

    def _run(self):
        """Calculate excess map with respect to the current model."""
        excess_settings = self.config.excess_map
        # TODO: allow a list of kernel sizes
        self.log.info("Computing excess maps.")

        if self.config.datasets.type == "1d":
            raise ValueError("Cannot compute excess map for 1D dataset")

        # Here we could possibly stack the datasets if needed.
        if len(self.data.datasets) > 1:
            raise ValueError("Datasets must be stacked to compute the excess map")

        energy_edges = make_energy_axis(excess_settings.energy_edges)
        if energy_edges is not None:
            energy_edges = energy_edges.edges

        excess_map_estimator = ExcessMapEstimator(
            correlation_radius=excess_settings.correlation_radius,
            energy_edges=energy_edges,
            **excess_settings.parameters,
        )
        self.products.excess_map = excess_map_estimator.run(self.data.datasets[0])


class FitWorkflowStep(WorkflowStepBase):
    tag = "fit"

    def _run(self):
        """Fitting reduced datasets to model."""
        if not self.data.models:
            raise RuntimeError("Missing models")

        fit_settings = self.config.fit
        for dataset in self.data.datasets:
            if fit_settings.fit_range:
                energy_min = fit_settings.fit_range.min
                energy_max = fit_settings.fit_range.max
                # TODO : add fit range in lon/lat
                geom = dataset.counts.geom
                dataset.mask_fit = geom.energy_mask(energy_min, energy_max)

        self.log.info("Fitting datasets.")
        result = self.data.fit.run(datasets=self.data.datasets)
        self.products.fit_result = result
        self.log.info(self.products.fit_result)


class FluxPointsWorkflowStep(WorkflowStepBase):
    tag = "flux-points"

    def _run(self):
        """Calculate flux points for a specific model component."""
        if not self.data.datasets:
            raise RuntimeError(
                "No datasets defined. Impossible to compute flux points."
            )

        fp_settings = self.config.flux_points
        self.log.info("Calculating flux points.")

        energy_edges = make_energy_axis(fp_settings.energy).edges
        flux_point_estimator = FluxPointsEstimator(
            energy_edges=energy_edges,
            source=fp_settings.source,
            fit=self.data.fit,
            **fp_settings.parameters,
        )

        fp = flux_point_estimator.run(datasets=self.data.datasets)

        self.products.flux_points = FluxPointsDataset(
            data=fp, models=self.data.models[fp_settings.source]
        )
        cols = ["e_ref", "dnde", "dnde_ul", "dnde_err", "sqrt_ts"]
        table = self.products.flux_points.data.to_table(sed_type="dnde")
        self.log.info("\n{}".format(table[cols]))


class LightCurveWorkflowStep(WorkflowStepBase):
    tag = "light-curve"

    def _run(self):
        """Calculate light curve for a specific model component."""
        lc_settings = self.config.light_curve
        self.log.info("Computing light curve.")
        energy_edges = make_energy_axis(lc_settings.energy_edges).edges

        if (
            lc_settings.time_intervals.start is None
            or lc_settings.time_intervals.stop is None
        ):
            self.log.info(
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
            fit=self.data.fit,
            **lc_settings.parameters,
        )
        lc = light_curve_estimator.run(datasets=self.data.datasets)
        self.products.light_curve = lc
        self.log.info(
            "\n{}".format(
                self.products.light_curve.to_table(format="lightcurve", sed_type="flux")
            )
        )
