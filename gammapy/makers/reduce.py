import logging
import contextlib
from multiprocessing import Pool
from pathlib import Path

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
    dataset : `~gammapy.datasets.MapDataset`
        Reference dataset
    stacking : bool
        Stack into reference dataset or not
    n_jobs : int
        Number of processes to run in parallel
        Defalut is None
    cutout_mode : str
        Cutout mode. Default is "partial"
    cutout_width : str or `~astropy.coordinates.Angle`,
        Cutout width. Default is None, If Default is determined 
        Path
    write_all : bool
        Write a dataset per observation or not
    overwrite : bool
        Whether to overwrite
    
    Returns
    -------
    datasets : `~gammapy.datasets.Datasets`
        Datasets
    """

    tag = "DatasetsMaker"

    def __init__(
        self,
        makers,
        dataset,
        stacking=True,
        n_jobs=None,
        cutout_mode="partial",
        cutout_width=None,
        path=".",
        write_all=False,
        overwrite=True,
    ):
        self.log = logging.getLogger(__name__)
        self.makers = makers
        self.path = Path(path)
        self.overwrite = overwrite
        self.cutout_mode = cutout_mode
        if cutout_width is not None:
            cutout_width = Angle(cutout_width)
        self.cutout_width = cutout_width
        self.n_jobs = n_jobs
        self.write_all = write_all
        self.stacking = stacking

        if isinstance(dataset, MapDataset):
            # also valid for Spectrum as it inherits from MapDataset
            self.dataset = dataset
        else:
            raise Exception(TypeError("Invalid reference dataset."))

        self._datasets = []

        if self.cutout_width is None and not isinstance(dataset, SpectrumDataset):
            if self.offset_max is None:
                raise Exception(
                    ValueError("cutout_width must be defined if there is no offset_max")
                )
            else:
                self.cutout_width = 2 * self.offset_max

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

        if isinstance(self.dataset, SpectrumDataset):
            dataset_obs = self.dataset.copy(
                name=f"obs-{observation.obs_id}_{self.dataset.name}"
            )
        elif isinstance(self.dataset, MapDataset):
            cutouts_kwargs = {
                "position": observation.pointing_radec.galactic,
                "width": self.cutout_width,
                "mode": self.cutout_mode,
            }
            dataset_obs = self.dataset.cutout(
                **cutouts_kwargs, name=f"obs-{observation.obs_id}_{self.dataset.name}"
            )

        log.info(f"Computing dataset for observation {observation.obs_id}")
        for maker in self.makers:
            log.info(f"Running {maker.tag}")
            dataset_obs = maker.run(dataset=dataset_obs, observation=observation)
        if self.write_all:
            self._write(dataset_obs)
        return dataset_obs

    def _write(self, dataset):
        """Write individual dataset"""
        path = self.path / f"{dataset.name}"
        path.mkdir(exist_ok=True)

        filename = path / f"{dataset.name}.fits"
        log.info(f"Writing {filename}")
        dataset.write(filename, overwrite=self.overwrite)

    def callback(self, dataset):
        print(dataset.name)
        if self.stacking:
            self.dataset.stack(dataset)
        else:

            self._datasets.append(dataset)

    def run(self, observations):
        """Run and write
        Parameters
        ----------
        observations : `Observations`
            Observations
        """

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

        if self.stacking:
            return Datasets([self.dataset])
        else:
            return Datasets(self._datasets)
