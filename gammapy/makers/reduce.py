# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from astropy.coordinates import Angle
import gammapy.utils.parallel as parallel
from gammapy.datasets import Datasets, MapDataset, MapDatasetOnOff, SpectrumDataset
from .core import Maker
from .safe import SafeMaskMaker

log = logging.getLogger(__name__)


__all__ = [
    "DatasetsMaker",
]


class DatasetsMaker(Maker, parallel.ParallelMixin):
    """Run makers in a chain.

    Parameters
    ----------
    makers : list of `Maker` objects
        Makers.
    stack_datasets : bool, optional
        If True, stack into the reference dataset (see `run` method arguments).
        Default is True.
    n_jobs : int, optional
        Number of processes to run in parallel.
        Default is one, unless `~gammapy.utils.parallel.N_JOBS_DEFAULT` was modified.
    cutout_mode : {'trim', 'partial', 'strict'}
        Used only to cutout the reference `MapDataset` around each processed observation.
        Mode is an option for Cutout2D, for details see `~astropy.nddata.utils.Cutout2D`.
        Default is "trim".
    cutout_width : tuple of `~astropy.coordinates.Angle`, optional
        Angular sizes of the region in (lon, lat) in that specific order.
        If only one value is passed, a square region is extracted.
        If None it returns an error, except if the list of makers includes a `SafeMaskMaker`
        with the offset-max method defined. In that case it is set to two times `offset_max`.
        Default is None.
    parallel_backend : {'multiprocessing', 'ray'}, optional
        Which backend to use for multiprocessing.
        Default is None.
    """

    tag = "DatasetsMaker"

    def __init__(
        self,
        makers,
        stack_datasets=True,
        n_jobs=None,
        cutout_mode="trim",
        cutout_width=None,
        parallel_backend=None,
    ):
        self.log = logging.getLogger(__name__)
        self.makers = makers
        self.cutout_mode = cutout_mode

        if cutout_width is not None:
            cutout_width = Angle(cutout_width)

        self.cutout_width = cutout_width
        self._apply_cutout = True

        if self.cutout_width is None:
            if self.offset_max is None:
                self._apply_cutout = False
            else:
                self.cutout_width = 2 * self.offset_max

        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.stack_datasets = stack_datasets

        self._datasets = []
        self._error = False

    @property
    def offset_max(self):
        maker = self.safe_mask_maker
        if maker is not None and hasattr(maker, "offset_max"):
            return maker.offset_max

    @property
    def safe_mask_maker(self):
        for m in self.makers:
            if isinstance(m, SafeMaskMaker):
                return m

    def make_dataset(self, dataset, observation):
        """Make single dataset.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Reference dataset.
        observation : `Observation`
            Observation.
        """
        if self._apply_cutout:
            cutouts_kwargs = {
                "position": observation.get_pointing_icrs(observation.tmid).galactic,
                "width": self.cutout_width,
                "mode": self.cutout_mode,
            }
            dataset_obs = dataset.cutout(
                **cutouts_kwargs,
            )
        else:
            dataset_obs = dataset.copy()

        if dataset.models is not None:
            models = dataset.models.copy()
            models.reassign(dataset.name, dataset_obs.name)
            dataset_obs.models = models

        log.info(f"Computing dataset for observation {observation.obs_id}")

        for maker in self.makers:
            log.info(f"Running {maker.tag}")
            dataset_obs = maker.run(dataset=dataset_obs, observation=observation)

        return dataset_obs

    def callback(self, dataset):
        if self.stack_datasets:
            if type(self._dataset) is MapDataset and type(dataset) is MapDatasetOnOff:
                dataset = dataset.to_map_dataset(name=dataset.name)
            self._dataset.stack(dataset)
        else:
            self._datasets.append(dataset)

    def error_callback(self, dataset):
        # parallel run could cause a memory error with non-explicit message.
        self._error = True

    def run(self, dataset, observations, datasets=None):
        """Run data reduction.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Reference dataset (used only for stacking if datasets are provided).
        observations : `Observations`
            Observations.
        datasets : `~gammapy.datasets.Datasets`
            Base datasets, if provided its length must be the same as the observations.

        Returns
        -------
        datasets : `~gammapy.datasets.Datasets`
            Datasets.

        """
        if isinstance(dataset, MapDataset):
            # also valid for Spectrum as it inherits from MapDataset
            self._dataset = dataset
        else:
            raise TypeError("Invalid reference dataset.")

        if isinstance(dataset, SpectrumDataset):
            self._apply_cutout = False

        if datasets is not None:
            self._apply_cutout = False
        else:
            datasets = len(observations) * [dataset]

        n_jobs = min(self.n_jobs, len(observations))

        parallel.run_multiprocessing(
            self.make_dataset,
            zip(datasets, observations),
            backend=self.parallel_backend,
            pool_kwargs=dict(processes=n_jobs),
            method="apply_async",
            method_kwargs=dict(
                callback=self.callback,
                error_callback=self.error_callback,
            ),
            task_name="Data reduction",
        )

        if self._error:
            raise RuntimeError("Execution of a sub-process failed")

        if self.stack_datasets:
            return Datasets([self._dataset])

        lookup = {
            d.meta_table["OBS_ID"][0]: idx for idx, d in enumerate(self._datasets)
        }
        return Datasets([self._datasets[lookup[obs.obs_id]] for obs in observations])
