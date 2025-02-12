# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sampler parameter classes."""

from .utils import _parse_datasets

__all__ = ["Sampler", "SamplerLikelihood", "SamplerResult"]


class Sampler:
    """Sampler class.

    The sampler class provides a uniform interface to multiple sampler backends.
    Currently available: "UltraNest", ("zeusmc", "emcee"  in #TODO).

    Parameters
    ----------
    backend : {"ultranest"}
        Global backend used for sampler. Default is "ultranest".
        UltraNest: Most options can be found in the UltraNest doc
        https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler

    #TODO : describe all parameters
    """

    def __init__(self, backend="ultranest", sampler_opts=None, run_opts=None):
        self._sampler = None
        self.backend = backend
        self.sampler_opts = sampler_opts
        self.run_opts = run_opts

        if self.backend == "ultranest":
            default_opts = {
                "live_points": 200,
                "frac_remain": 0.5,
                "log_dir": None,
                "resume": "subfolder",
                "step_sampler": False,
                "nsteps": 10,
            }

        self.sampler_opts = default_opts
        if sampler_opts is not None:
            self.sampler_opts.update(sampler_opts)
        if run_opts is None:
            self.run_opts = {}

    @staticmethod
    def _update_models_from_posterior(models, result):
        # TODO : add option for median, maxLogL once Param object has asym errors
        posterior = result["posterior"]
        for i, par in enumerate(models.parameters.free_parameters):
            par.value = posterior["mean"][i]
            par.error = posterior["stdev"][i]
        models._covariance = None

    def sampler_ultranest(self, parameters, like):
        """
        Defines the Ultranest sampler and options
        Returns the result dictionary that contains the samples and other information.
        """
        import ultranest

        def _prior_inverse_cdf(values):
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
    This is a placeholder to store the results from the sampler

    TODO:
    - Support parameter posteriors directly on Parameter
        - e.g. adding a errn and errp entry
        - or creating a posterior entry on Parameter
    - Or support with a specific entry on the SamplerResult

    Parameters
    ----------
    nfev : int
        number of likelihood calls/evaluations
    success : bool
        Did the sampler succeed in finding a good fit? Definition of convergence depends on the sampler backend.
    models : `~gammapy.modeling.models`
        the models used by the sampler
    samples : `~numpy.ndarray`, optional
        array of (weighted) samples
    sampler_results : dict, optional
        output of sampler.
    """

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
    #TODO: can this be done in a simpler manner without a class
    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        Parameters with starting values.
    function : callable
        Likelihood function.
    """

    def __init__(self, function, parameters):
        self.function = function
        self.parameters = parameters

    def fcn(self, value):
        self.parameters.value = value
        total_stat = -0.5 * self.function()
        return total_stat
