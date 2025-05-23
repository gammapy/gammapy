# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate observations"""

from itertools import repeat
from gammapy.utils import parallel as parallel
from gammapy.utils.scripts import make_path


class ObservationsEventsSampler(parallel.ParallelMixin):
    """Run event sampling for an emsemble of observations

    Parameters
    ----------
    sampler_kwargs : dict, optional
        Arguments passed to `~gammapy.datasets.MapDatasetEventSampler`.
    dataset_kwargs : dict, optional
        Arguments passed to `~gammapy.datasets.create_map_dataset_from_observation()`.
    outdir : str, Path
        path of the output files created. Default is "./simulated_data/".
        If None a list of `~gammapy.data.Observation` is returned.
    overwrite : bool
        Overwrite the output files or not
    n_jobs : int, optional
        Number of processes to run in parallel.
        Default is one, unless `~gammapy.utils.parallel.N_JOBS_DEFAULT` was modified.
    parallel_backend : {'multiprocessing', 'ray'}, optional
        Which backend to use for multiprocessing.
        Default is None.
    """

    def __init__(
        self,
        sampler_kwargs=None,
        dataset_kwargs=None,
        outdir="./simulated_data/",
        overwrite=True,
        n_jobs=None,
        parallel_backend=None,
    ):
        if outdir is not None:
            outdir = make_path(outdir)
            outdir.mkdir(exist_ok=True, parents=True)
        self.outdir = outdir
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.overwrite = overwrite

        if sampler_kwargs is None:
            sampler_kwargs = {}
        self.sampler_kwargs = sampler_kwargs
        self.dataset_kwargs = dataset_kwargs

    def simulate_observation(self, observation, models=None):
        """Simulate a  single observation.

        Parameters
        ----------
        observation : `~gammapy.data.Observation`
            Observation to be simulated.
        models : `~gammapy.modeling.Models`, optional
            Models to simulate.
            Can be None to only sample background events. Default is None.
        """
        from gammapy.datasets import ObservationEventSampler

        sampler = ObservationEventSampler(
            **self.sampler_kwargs, dataset_kwargs=self.dataset_kwargs
        )
        observation = sampler.run(observation, models=models)

        if self.outdir is not None:
            observation.write(
                self.outdir / f"obs_{observation.obs_id}.fits",
                overwrite=self.overwrite,
            )
        else:
            return observation

    def run(self, observations, models=None):
        """Run event sampling for an ensemble of onservations

        Parameters
        ----------
        observation : `~gammapy.data.Observation`
            Observation to be simulated.
        models : `~gammapy.modeling.Models`, optional
            Models to simulate.
            Can be None to only sample background events. Default is None.
        """

        n_jobs = min(self.n_jobs, len(observations))

        observations = parallel.run_multiprocessing(
            self.simulate_observation,
            zip(
                observations,
                repeat(models),
            ),
            backend=self.parallel_backend,
            pool_kwargs=dict(processes=n_jobs),
            task_name="Simulate observations",
        )
        if self.outdir is None:
            return observations
