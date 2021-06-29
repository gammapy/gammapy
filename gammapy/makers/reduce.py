import logging
import contextlib
from multiprocessing import Pool
import numpy as np

from astropy.coordinates import Angle
from .core import Maker
from .safe import SafeMaskMaker
from gammapy.datasets import Datasets, MapDataset, SpectrumDataset

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
        Defalut is None
    cutout_mode : str
        Cutout mode. Default is "partial"
    cutout_width : str or `~astropy.coordinates.Angle`,
        Cutout width. Default is None, If Default is determined 
    
    Returns
    -------
    datasets : `~gammapy.datasets.Datasets`
        Datasets
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
        self.n_jobs = n_jobs
        self.stack_datasets = stack_datasets

        self._datasets = []
        self._order = []

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

    def make_dataset(self, observation):
        """Make single dataset.
        Parameters
        ----------
        observation : `Observation`
            Observation
        """

        if isinstance(self._dataset, SpectrumDataset):
            dataset_obs = self._dataset.copy()
        elif isinstance(self._dataset, MapDataset):
            cutouts_kwargs = {
                "position": observation.pointing_radec.galactic,
                "width": self.cutout_width,
                "mode": self.cutout_mode,
            }
            dataset_obs = self._dataset.cutout(**cutouts_kwargs)

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

    def run(self, dataset, observations):
        """Run and write
        Parameters
        ----------
         dataset : `~gammapy.datasets.MapDataset`
            Reference dataset
         observations : `Observations`
            Observations
        """

        if isinstance(dataset, MapDataset):
            # also valid for Spectrum as it inherits from MapDataset
            self._dataset = dataset
        else:
            raise TypeError("Invalid reference dataset.")

        if self.cutout_width is None and not isinstance(dataset, SpectrumDataset):
            if self.offset_max is None:
                raise Exception(
                    ValueError("cutout_width must be defined if there is no offset_max")
                )
            else:
                self.cutout_width = 2 * self.offset_max

        if self.n_jobs is not None and self.n_jobs > 1:
            n_jobs = min(self.n_jobs, len(observations))
            with contextlib.closing(Pool(processes=n_jobs)) as pool:
                log.info("Using {} jobs.".format(n_jobs))
                results = []
                for observation in observations:
                    result = pool.apply_async(
                        self.make_dataset, (observation,), callback=self.callback
                    )
                    results.append(result)
                # wait async run is done
                [result.wait() for result in results]
        else:
            for obs in observations:
                dataset = self.make_dataset(obs)
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
