import logging
from multiprocessing import Pool
import numpy as np
from astropy.coordinates import Angle
from gammapy.datasets import Datasets, MapDataset, MapDatasetOnOff, SpectrumDataset
from .core import Maker
from .safe import SafeMaskMaker

log = logging.getLogger(__name__)


__all__ = [
    "DatasetsMaker",
]


class DatasetsMaker(Maker):
    """Run makers in a chain

    Parameters
    ----------
    makers : list of `Maker` objects
        Makers
    stack_datasets : bool
        If True stack into the reference dataset (see `run` method arguments).
    n_jobs : int
        Number of processes to run in parallel
    cutout_mode : {'trim', 'partial', 'strict'}
        Used only to cutout the reference `MapDataset` around each processed observation.
        Mode is an option for Cutout2D, for details see `~astropy.nddata.utils.Cutout2D`.
        Default is "trim".
    cutout_width : tuple of `~astropy.coordinates.Angle`
        Angular sizes of the region in (lon, lat) in that specific order.
        If only one value is passed, a square region is extracted.
        If None it returns an error, except if the list of makers includes a `SafeMaskMaker`
        with the offset-max method defined. In that case it is set to two times `offset_max`.
    """

    tag = "DatasetsMaker"

    def __init__(
        self,
        makers,
        stack_datasets=True,
        n_jobs=None,
        cutout_mode="trim",
        cutout_width=None,
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
            Reference dataset
        observation : `Observation`
            Observation
        """

        if self._apply_cutout:
            cutouts_kwargs = {
                "position": observation.pointing_radec.galactic,
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
            if isinstance(self._dataset, MapDataset) and isinstance(
                dataset, MapDatasetOnOff
            ):
                dataset = dataset.to_map_dataset(name=dataset.name)
            self._dataset.stack(dataset)
        else:
            self._datasets.append(dataset)

    def error_callback(self, dataset):
        # parallel run could cause a memory error with non-explicit message.
        self._error = True

    def run(self, dataset, observations, datasets=None):
        """Run data reduction

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset`
            Reference dataset (used only for stacking if datasets are provided)
        observations : `Observations`
            Observations
        datasets : `~gammapy.datasets.Datasets`
            Base datasets, if provided its length must be the same than the observations.

        Returns
        -------
        datasets : `~gammapy.datasets.Datasets`
            Datasets

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

        if self.n_jobs is not None and self.n_jobs > 1:
            n_jobs = min(self.n_jobs, len(observations))
            with Pool(processes=n_jobs) as pool:
                log.info("Using {} jobs.".format(n_jobs))
                results = []
                for base, obs in zip(datasets, observations):
                    result = pool.apply_async(
                        self.make_dataset,
                        (
                            base,
                            obs,
                        ),
                        callback=self.callback,
                        error_callback=self.error_callback,
                    )
                    results.append(result)
                # wait async run is done
                [result.wait() for result in results]
            if self._error:
                raise RuntimeError("Execution of a sub-process failed")
        else:
            for base, obs in zip(datasets, observations):
                dataset = self.make_dataset(base, obs)
                self.callback(dataset)

        if self.stack_datasets:
            return Datasets([self._dataset])
        else:
            # have to sort datasets because of async
            obs_ids = [d.meta_table["OBS_ID"][0] for d in self._datasets]
            ordered = []
            for obs in observations:
                ind = np.where(np.array(obs_ids) == obs.obs_id)[0][0]
                ordered.append(self._datasets[ind])
            self._datasets = ordered
            return Datasets(self._datasets)
