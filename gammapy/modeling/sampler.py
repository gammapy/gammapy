# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sampler parameter classes."""

import logging
import ultranest

__all__ = ["Sampler", "SamplerResult","SamplerLikelihood"]




class Sampler:
    """Sampler class.

    The sampler class provides a uniform interface to multiple sampler backends.
    Currently available: "UltraNest", "emcee" in #TODO .

    Parameters
    ----------
    backend : {"ultranest"}
        Global backend used for sampler. Default is "ultranest".

    #TODO : describe all parameters
    """

    def __init__(self, backend="ultranest", sampler_opts=None):
        self._sampler = None
        self.backend = backend
        self.sampler_opts = sampler_opts

        if self.sampler_opts is None and self.backend=="ultranest" :
            self.sampler_opts = dict(live_points=100, frac_remain=0.5, log_dir=None, resume="subfolder", step_sampler=False, nsteps=20)

    @staticmethod
    def _parse_datasets(datasets):
        from gammapy.datasets import Dataset, Datasets

        if isinstance(datasets, (list, Dataset)):
            datasets = Datasets(datasets)
#        return datasets, datasets.parameters.free_parameters
        return datasets, datasets.parameters

    def run(self, datasets):
        datasets, parameters = self._parse_datasets(datasets=datasets)
        parameters = parameters.free_parameters

        if self.backend == "ultranest":
            # create log likelihood function
            # need to remove prior penalty
            like = SamplerLikelihood(function=datasets.stat_sum_no_prior, parameters=parameters)
            
            # create ultranest object
            self._sampler = ultranest.ReactiveNestedSampler(parameters.names, like.fcn, transform=parameters.prior_inverse_cdf,
                                                            log_dir=self.sampler_opts["log_dir"], resume=self.sampler_opts["resume"])

            if self.sampler_opts["step_sampler"]: 
                self._sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps = self.sampler_opts["step_sampler"],
                generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
                adaptive_nsteps=False,
                )
            
            result = self._sampler.run(min_num_live_points=self.sampler_opts["live_points"], frac_remain=self.sampler_opts["frac_remain"])
        
        return result
   

#class SamplerResult:
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
        total_stat = -0.5*self.function()
        return total_stat