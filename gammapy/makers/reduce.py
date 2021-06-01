import logging
import contextlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from .core import Maker
from .safe import SafeMaskMaker


log = logging.getLogger(__name__)


class Makers(Maker):
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
    """

    tag = "Makers"

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
        self.dataset = dataset

        if self.cutout_width is None and self.offset_max is None:
            raise Exception(
                ValueError("cutout_width must be defined if there is no offset_max")
            )

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

    def make_dataset(self, observations):
        """Make single dataset.
        Parameters
        ----------
        observations : `Observations`
            Observations
        """

        observations = np.atleast_1d(observations)

        cutouts_kwargs = self._setup_cutouts(observations)
        dataset = self.dataset.cutout(**cutouts_kwargs, name=self.dataset.name)

        for observation in observations:
            log.info(f"Computing dataset for observation {observation.obs_id}")
            dataset_obs = dataset.copy(name=f"obs-{observation.obs_id}_{dataset.name}")
            for maker in self.makers:
                log.info(f"Running {maker.tag}")
                dataset_obs = maker.run(dataset=dataset_obs, observation=observation)
            if self.write_all:
                self._write(dataset_obs)
            dataset.stack(dataset_obs)
        return dataset

    def _setup_cutouts(self, observations):
        """set cutout parameters"""
        if self.cutout_width is None:
            self.cutout_width = 2 * self.offset_max

        cutout_width = self.cutout_width
        obs_positions = [obs.pointing_radec.galactic for obs in observations]
        lon = np.mean([pos.l.value for pos in obs_positions])
        lat = np.mean([pos.b.value for pos in obs_positions])
        position = SkyCoord(lon * u.deg, lat * u.deg, frame="galactic")
        if self.stacking:
            sep = np.max(position.separation(SkyCoord(obs_positions)))
            cutout_width += 2 * (sep + 0.1 * u.deg)
        return {
            "position": position,
            "width": cutout_width,
            "mode": self.cutout_mode,
        }

    def _write(self, dataset):
        """Write individual dataset"""

        filename = self.path / f"dataset.fits"
        log.info(f"Writing {filename}")
        dataset.write(filename, overwrite=self.overwrite)

    def run(self, observations):
        """Run and write
        Parameters
        ----------
        observations : `Observations`
            Observations
        """

        if self.n_jobs is not None and self.n_jobs > 1:
            if self.stacking:
                observations = self._group_observations(self.n_jobs, observations)
            with contextlib.closing(Pool(processes=self.n_jobs)) as pool:
                log.info("Using {} jobs.".format(self.n_jobs))
                results = pool.map(self.make_dataset, observations)
            pool.join()
        else:
            results = [self.make_dataset(obs) for obs in observations]

        if self.stacking:
            for dataset in results:
                self.dataset.stack(dataset)
            self._write(self.dataset)
            return self.dataset

    @staticmethod
    def _group_observations(n_groups, observations):
        """Splint obsevration in multiple groups for parallel run """
        # TODO: maybe we could introduce observations.select and observations.group
        # for this kind of operations and move this later.

        observations = observations.copy()

        obs_groups = np.empty(n_groups, dtype=object)
        n_obs = len(observations)
        n_per_group = int(np.ceil(n_obs / n_groups))

        lons = np.array([obs.pointing_radec.galactic.l.value for obs in observations])
        if np.any(lons > 90) & np.any(lons < 90):
            # TODO: simplify ?
            lons[lons > 180] -= 360
        group_lons = np.linspace(np.min(lons), np.max(lons), n_groups)
        for k, lon in enumerate(group_lons):
            group_position = SkyCoord(lon * u.deg, 0 * u.deg, frame="galactic")
            obs_positions = [obs.pointing_radec for obs in observations]
            sep = group_position.separation(SkyCoord(obs_positions))
            ind = np.argsort(sep)[: min(n_per_group, len(observations))]
            ind = ind[ind < n_per_group]
            obs_groups[k] = []
            for idx in ind:
                obs = observations[idx]
                obs_groups[k].append(obs)
                observations.remove(obs)
        return obs_groups
