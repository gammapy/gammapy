# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sampler parameter classes."""

import logging
import ultranest
#from .sampler import SamplerLikelihood

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

    def __init__(self, live_points=100, frac_remain=0.5, log_dir=None, resume=False, step_sampler=False):
        self._sampler = None
        self.min_num_live_points = live_points
        self.frac_remain = frac_remain
        self.log_dir = log_dir
        self.step_sampler = step_sampler
        self.resume = resume

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

        # create log likelihood function
        # need to remove prior penalty
        like = SamplerLikelihood(function=datasets.stat_sum_no_prior, parameters=parameters)
        
        # create ultranest object
        self._sampler = ultranest.ReactiveNestedSampler(parameters.names, like.fcn, transform=parameters.prior_inverse_cdf,
                                                        log_dir=self.log_dir, resume=self.resume)

        if self.step_sampler: 
            print(f'step_sampler={step_sampler} => Using a step sampler' )
            nsteps = 10
            #RegionBallSliceSampler
            self._sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=nsteps,
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
            adaptive_nsteps=False,
            # max_nsteps=400
            )
        
        result = self._sampler.run(min_num_live_points=self.min_num_live_points, frac_remain=self.frac_remain)
        
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