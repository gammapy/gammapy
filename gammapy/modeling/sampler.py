# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sampler parameter classes."""

from .utils import _parse_datasets

__all__ = ["Sampler", "SamplerLikelihood", "SamplerResult"]

SAMPLER_BACKENDS = ["ultranest", "nautilus"]

DEFAULT_SAMPLER_OPTS = {
    "ultranest": {
        "live_points": 400,
        "frac_remain": 0.5,
        "log_dir": None,
        "resume": "subfolder",
        "step_sampler": False,
        "nsteps": 10,
    },
    "nautilus": {"n_live": 2000, "filepath": None, "resume": True},
}

DEFAULT_RUN_OPTS = {
    "ultranest": {},
    "nautilus": {"f_live": 0.01, "n_eff": 2000, "verbose": True},
}


class Sampler:
    """Sampler class.

    The sampler class provides a uniform interface to multiple sampler backends. Currently available: "UltraNest".

    Parameters
    ----------
    backend : {"ultranest", "nautilus"}
        Global backend used for sampler. Default is "ultranest".
        UltraNest: Most options can be found in the
        `UltraNest doc <https://johannesbuchner.github.io/UltraNest/>`__.
        Nautilus uses neural-network-guided nested sampling. See the
    `   `Nautilus documentation <https://nautilus-sampler.readthedocs.io/>`__.
    sampler_opts : dict, optional
        Sampler options passed to the sampler. See the full list of options on the
        `UltraNest documentation <https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler>`__.
        Noteworthy options:

        live_points : int
            Minimum number of live points used in the sampling. Increase this number to get more accurate results.
            For more samples in the posterior increase this number or the min_ess parameter.
            Default is 400 live points.
        frac_remain : float
            Integrate until this fraction of the integral is left in the remainder. Set to a low number (1e-2 … 1e-5)
            to make sure peaks are discovered. Set to a higher number (0.5) if you know the posterior is simple.
            Default is 0.5.
        min_ess : int
            Target number of effective posterior samples. Increase this number to get more accurate results.
            Default is live_points, but you may need to increase it to 1000 or more for complex posteriors.
        log_dir : str
            Where to store output files.
            Default is None and no results are not stored.
        resume : str
            ‘overwrite’, overwrite previous data. ‘subfolder’, create a fresh subdirectory in `log_dir`.
            ‘resume’ or True, continue previous run if available. Only works when dimensionality, transform or
            likelihood are consistent.
        step_sampler : bool, optional
            Use a step sampler. This can be more efficient for higher dimensions (>10 or 15 parameters), but much
            slower for lower dimensions.
            Default is False.
        nsteps : int
            Number of steps to take in each direction in the step sampler. Increase this number to get more
            accurate results at the cost of more computation time.
            Default is 10.
    run_opts : dict, optional
        Optional run options passed to the given backend when running the sampler.
        See the full list of run options on the
        `UltraNest documentation <https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler.run>`__.

    Examples
    --------
    For a usage example, see :doc:`/tutorials/details/nested_sampling_Crab` tutorial.

    Notes
    -----
    If you are using the "UltraNest" library, please follow its citation scheme:
    `Cite UltraNest <https://johannesbuchner.github.io/UltraNest/issues.html#how-should-i-cite-ultranest>`__.

    """

    # TODO: add "zeusmc", "emcee"

    def __init__(self, backend="ultranest", sampler_opts=None, run_opts=None):
        if backend not in SAMPLER_BACKENDS:
            raise ValueError(f"Sampler {backend} is not supported.")

        self._sampler = None
        self.backend = backend
        self.sampler_opts = {} if sampler_opts is None else sampler_opts
        self.run_opts = {} if run_opts is None else run_opts

        for key, value in DEFAULT_SAMPLER_OPTS[backend].items():
            self.sampler_opts.setdefault(key, value)
        for key, value in DEFAULT_RUN_OPTS[backend].items():
            self.run_opts.setdefault(key, value)

    @staticmethod
    def _update_models_from_posterior(models, result):
        """
        Update the models with the posterior distribution.

        But this raises the question on how to estimate the error given the median and maxLogL.
        Covariance matrix is not defined in this sample approach (could be approximated via the samples).

        Parameters
        ----------
        models : `~gammapy.modeling.models`
            The models to update
        result : dict
            The sampler results dictionary containing the posterior distribution infos.
        """
        # TODO : add option for median, distribution peak, and maxLogL once Param object has asym errors

        posterior = result["posterior"]
        for i, par in enumerate(models.parameters.free_unique_parameters):
            par.value = posterior["mean"][i]
            par.error = posterior["stdev"][i]
        models._covariance = None

    def sampler_ultranest(self, parameters, like):
        """
        Defines the Ultranest sampler and options.

        Returns the result in the SamplerResult that contains the updated models, samples, posterior distribution and
        other information.

        Parameters
        ----------
        parameters : `~gammapy.modeling.Parameters`
            The models parameters to sample.
        like : `~gammapy.modeling.sampler.SamplerLikelihood`
            The likelihood function.

        Returns
        -------
        result : `~gammapy.modeling.sampler.SamplerResult`
            The sampler results.
        """
        import ultranest

        self._sampler = ultranest.ReactiveNestedSampler(
            parameters.names,
            like.fcn,
            transform=parameters._prior_inverse_cdf,
            log_dir=self.sampler_opts["log_dir"],
            resume=self.sampler_opts["resume"],
        )

        if self.sampler_opts["step_sampler"]:
            from ultranest.stepsampler import (
                SliceSampler,
                generate_mixture_random_direction,
            )

            self._sampler.stepsampler = SliceSampler(
                nsteps=self.sampler_opts["nsteps"],
                generate_direction=generate_mixture_random_direction,
            )

        result = self._sampler.run(
            min_num_live_points=self.sampler_opts["live_points"],
            frac_remain=self.sampler_opts["frac_remain"],
            **self.run_opts,
        )

        return result

    def sampler_nautilus(self, parameters, like):
        import nautilus
        import numpy as np

        self._sampler = nautilus.Sampler(
            prior=parameters._prior_inverse_cdf,
            likelihood=like.fcn,
            n_dim=len(parameters),
            n_live=self.sampler_opts["n_live"],
            filepath=self.sampler_opts["filepath"],
            resume=self.sampler_opts["resume"],
        )

        success = self._sampler.run(**self.run_opts)

        points, log_w, log_l = self._sampler.posterior()
        weights = np.exp(log_w - log_w.max())
        weights /= weights.sum()
        mean = np.average(points, weights=weights, axis=0)
        stdev = np.sqrt(np.average((points - mean) ** 2, weights=weights, axis=0))

        result = {
            "ncall": self._sampler.n_like,
            "success": success,
            "logz": self._sampler.log_z,
            "posterior": {"mean": mean, "stdev": stdev},
            "samples": self._sampler.posterior(equal_weight=True)[0],
            "points": points,
            "log_w": log_w,
            "log_l": log_l,
        }
        return result

    def run(self, datasets):
        """
        Run the sampler on the provided datasets.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets to fit.

        Returns
        -------
        result : `~gammapy.modeling.sampler.SamplerResult`
            The sampler results. See the class description to get the exact content.
        """
        datasets, parameters = _parse_datasets(datasets=datasets)
        parameters = parameters.free_unique_parameters

        like = SamplerLikelihood(
            function=datasets._stat_sum_likelihood, parameters=parameters
        )
        if self.backend == "ultranest":
            result_dict = self.sampler_ultranest(parameters, like)
            self._sampler.print_results()
            result = SamplerResult.from_ultranest(result_dict)
        elif self.backend == "nautilus":
            result_dict = self.sampler_nautilus(parameters, like)
            self._sampler.print_status()
            result = SamplerResult.from_nautilus(result_dict)

        models_copy = datasets.models.copy()
        self._update_models_from_posterior(models_copy, result_dict)
        result.models = models_copy
        return result


class SamplerResult:
    """SamplerResult class.

    Parameters
    ----------
    nfev : int
        Number of likelihood calls/evaluations.
    success : bool
        Did the sampler succeed in finding a good fit? Definition of convergence depends on the sampler backend.
    models : `~gammapy.modeling.models`
        The models updated after the sampler run.
    samples : `~numpy.ndarray`
        Array of (weighted) samples that can be used for histograms or corner plots.
    sampler_results : dict
        Output of sampler.
        See the :doc:`/tutorials/details/nested_sampling_Crab` tutorial for a complete description.
    """

    # TODO:
    #  - Support parameter posteriors directly on Parameter
    #    - e.g. adding a errn and errp entry
    #    - or creating a posterior entry on Parameter
    #  - Or support with a specific entry on the SamplerResult
    #  - Add write method to be consistent with FitResult

    def __init__(
        self, nfev=0, success=False, models=None, samples=None, sampler_results=None
    ):
        self.nfev = nfev
        self.success = success
        self.models = models
        self.samples = samples
        self.sampler_results = sampler_results

    @classmethod
    def from_ultranest(cls, ultranest_result):
        kwargs = {}
        kwargs["nfev"] = ultranest_result["ncall"]
        kwargs["success"] = ultranest_result["insertion_order_MWW_test"]["converged"]
        kwargs["samples"] = ultranest_result["samples"]
        kwargs["sampler_results"] = ultranest_result
        return cls(**kwargs)

    @classmethod
    def from_nautilus(cls, nautilus_result):
        kwargs = {}
        kwargs["nfev"] = nautilus_result["ncall"]
        kwargs["success"] = nautilus_result["success"]
        kwargs["samples"] = nautilus_result["samples"]
        kwargs["sampler_results"] = nautilus_result
        return cls(**kwargs)


class SamplerLikelihood:
    """Wrapper of the likelihood function used by the sampler.

    This is needed to modify parameters and likelihood by *-0.5

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        Parameters with starting values.
    function : callable
        Likelihood function.
    """

    # TODO: Will be updated with the FitStatistic class when ready.

    def __init__(self, function, parameters):
        self.function = function
        self.parameters = parameters

    def fcn(self, value):
        self.parameters.value = value
        total_stat = -0.5 * self.function()
        return total_stat
