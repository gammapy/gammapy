# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sampler parameter classes."""

import ultranest
from gammapy.modeling.utils import _parse_datasets

__all__ = ["Sampler", "SamplerLikelihood"]  # , "SamplerResult"


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

    def __init__(self, backend="ultranest", sampler_opts=None):
        self._sampler = None
        self.backend = backend
        self.sampler_opts = sampler_opts

        if self.sampler_opts is None and self.backend == "ultranest":
            self.sampler_opts = dict(
                live_points=100,
                frac_remain=0.5,
                log_dir=None,
                resume="subfolder",
                step_sampler=False,
                nsteps=20,
            )

    def sampler_ultranest(self, parameters, like):
        """
        Defines the Ultranest sampler and options
        Returns the result dictionary that contains the samples and other information.
        """

        def _prior_inverse_cdf(values):
            if None in parameters:
                raise ValueError(
                    "Some parameters have no prior set. You need priors on all parameters."
                )
            return [par.prior._inverse_cdf(val) for par, val in zip(parameters, values)]

        # create ultranest object
        self._sampler = ultranest.ReactiveNestedSampler(
            parameters.names,
            like.fcn,
            transform=_prior_inverse_cdf,
            log_dir=self.sampler_opts["log_dir"],
            resume=self.sampler_opts["resume"],
        )

        if self.sampler_opts["step_sampler"]:
            self._sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps=self.sampler_opts["step_sampler"],
                generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
                adaptive_nsteps=False,
            )

        result = self._sampler.run(
            min_num_live_points=self.sampler_opts["live_points"],
            frac_remain=self.sampler_opts["frac_remain"],
        )

        return result

    def run(self, datasets):
        datasets, parameters = _parse_datasets(datasets=datasets)
        parameters = parameters.free_parameters

        if self.backend == "ultranest":
            # create log likelihood function
            like = SamplerLikelihood(
                function=datasets._stat_sum_likelihood, parameters=parameters
            )
            result = self.sampler_ultranest(parameters, like)

        print(self._sampler.print_results())
        return result


# class SamplerResult:
#   """SamplerResult class.
#   This is a placeholder to store the results from the sampler
#   """


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
