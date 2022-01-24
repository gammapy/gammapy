import contextlib
import logging
from multiprocessing import Pool
import numpy as np
from astropy.coordinates import Angle
from gammapy.datasets import Datasets, MapDataset, SpectrumDataset
from .core import Maker
from .safe import SafeMaskMaker

log = logging.getLogger(__name__)


class DatasetsMaker(Maker):
    """Run makers in a chain

    Parameters
    ----------
    makers : list of `Maker` objects
        Makers
    stack_datasets : bool
        Stack into reference dataset or not
    n_jobs : int
        Number of processes to run in parallel
        Default is None
    cutout_mode : str
        Cutout mode. Default is "partial"
    cutout_width : str or `~astropy.coordinates.Angle`,
        Cutout width. Default is None, If Default is determined
    """

    tag = "DatasetsMaker"

    def __init__(
        self,
        makers,
        stack_datasets=True,
        n_jobs=None,
        cutout_mode="partial",
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
            with contextlib.closing(Pool(processes=n_jobs)) as pool:
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
