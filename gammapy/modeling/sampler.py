# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sampler parameter classes."""

from .utils import _parse_datasets

__all__ = ["Sampler", "SamplerLikelihood", "SamplerResult"]


class Sampler:
    """Sampler class.

    The sampler class provides a uniform interface to multiple sampler backends. Currently available: "UltraNest".

    Parameters
    ----------
    backend : {"ultranest"}
        Global backend used for sampler. Default is "ultranest".
        UltraNest: Most options can be found in the
        `UltraNest doc <https://johannesbuchner.github.io/UltraNest/>`__.
    sampler_opts : dict, optional
        Sampler options passed to the sampler. Noteworthy options:

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
        step_sampler : bool
            Use a step sampler. This can be more efficient for higher dimensions (>10 or 15 parameters), but much
            slower for lower dimensions.
            Default is False.
        nsteps : int
            Number of steps to take in each direction in the step sampler. Increase this number to get more
            accurate results at the cost of more computation time.
            Default is 10.
        See the full list of options on the
        `UltraNest documentation <https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler>`__.
    run_opts : dict, optional
        Optional run options passed to the given backend when running the sampler.
        See the full list of run options on the
        `UltraNest documentation <https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler.run>`__.

    Example
    -------
    For a usage example, see :doc:`/tutorials/api/nested_sampling_Crab` tutorial.

    """

    # TODO: add "zeusmc", "emcee"

    def __init__(self, backend="ultranest", sampler_opts=None, run_opts=None):
        self._sampler = None
        self.backend = backend
        self.sampler_opts = {} if sampler_opts is None else sampler_opts
        self.run_opts = {} if run_opts is None else run_opts

        if self.backend == "ultranest":
            self.sampler_opts.setdefault("live_points", 400)
            self.sampler_opts.setdefault("frac_remain", 0.5)
            self.sampler_opts.setdefault("log_dir", None)
            self.sampler_opts.setdefault("resume", "subfolder")
            self.sampler_opts.setdefault("step_sampler", False)
            self.sampler_opts.setdefault("nsteps", 10)

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
        for i, par in enumerate(models.parameters.free_parameters):
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

        def _prior_inverse_cdf(values):
            """Returns a list of model parameters for a given list of values (that are bound in [0,1])."""
            if None in parameters:
                raise ValueError(
                    "Some parameters have no prior set. You need priors on all parameters."
                )
            return [par.prior._inverse_cdf(val) for par, val in zip(parameters, values)]

        self._sampler = ultranest.ReactiveNestedSampler(
            parameters.names,
            like.fcn,
            transform=_prior_inverse_cdf,
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
        parameters = parameters.free_parameters

        if self.backend == "ultranest":
            like = SamplerLikelihood(
                function=datasets._stat_sum_likelihood, parameters=parameters
            )
            result_dict = self.sampler_ultranest(parameters, like)
            self._sampler.print_results()

            models_copy = datasets.models.copy()
            self._update_models_from_posterior(models_copy, result_dict)

            result = SamplerResult.from_ultranest(result_dict)
            result.models = models_copy

            return result
        else:
            raise ValueError(f"Sampler {self.backend} is not supported.")


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
        See the :doc:`/tutorials/api/nested_sampling_Crab` tutorial for a complete description.
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
