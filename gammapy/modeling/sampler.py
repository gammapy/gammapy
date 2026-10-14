# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sampler parameter classes."""

import abc
import warnings
from gammapy.utils.deprecation import GammapyDeprecationWarning
from .utils import _parse_datasets

__all__ = ["Sampler", "SamplerLikelihood", "SamplerResult"]


class _SamplerBackend(abc.ABC):
    """Base class for the sampler backends.

    The options are passed to the backend library using its own keyword names:
    ``sampler_opts`` are forwarded to the sampler constructor and ``run_opts``
    to its ``run`` method. Options listed in ``gammapy_opts`` are handled by
    Gammapy and never forwarded.

    Parameters
    ----------
    sampler_opts : dict, optional
        Options passed to the backend sampler constructor. Default is None.
    run_opts : dict, optional
        Options passed to the backend sampler run method. Default is None.
    """

    tag = None
    default_sampler_opts = {}
    default_run_opts = {}
    gammapy_opts = {}

    def __init__(self, sampler_opts=None, run_opts=None):
        sampler_opts = {} if sampler_opts is None else dict(sampler_opts)
        run_opts = {} if run_opts is None else dict(run_opts)
        sampler_opts, run_opts = self._handle_deprecated_opts(sampler_opts, run_opts)

        self._sampler = None
        self.sampler_opts = {
            **self.default_sampler_opts,
            **self.gammapy_opts,
            **sampler_opts,
        }
        self.run_opts = {**self.default_run_opts, **run_opts}

    def _handle_deprecated_opts(self, sampler_opts, run_opts):
        return sampler_opts, run_opts

    def _forwarded_sampler_opts(self):
        """Sampler options without the ones handled by Gammapy."""
        return {
            key: value
            for key, value in self.sampler_opts.items()
            if key not in self.gammapy_opts
        }

    @abc.abstractmethod
    def build(self, parameters, like):
        """Create the backend sampler and store it in ``self._sampler``."""

    @abc.abstractmethod
    def run(self):
        """Run the backend sampler and return the result dictionary.

        The dictionary must contain a "posterior" entry with "mean" and "stdev".
        """

    @abc.abstractmethod
    def to_result(self, result_dict):
        """Convert the result dictionary to a `SamplerResult`."""

    @abc.abstractmethod
    def print_summary(self):
        """Print the summary of the sampler run."""


class _UltranestBackend(_SamplerBackend):
    """UltraNest backend using `ultranest.ReactiveNestedSampler`."""

    tag = "ultranest"
    default_sampler_opts = {"log_dir": None, "resume": "subfolder"}
    default_run_opts = {"min_num_live_points": 400, "frac_remain": 0.5}
    gammapy_opts = {"step_sampler": False, "nsteps": 10}

    # old sampler_opts name: new run_opts name
    _deprecated_sampler_opts = {
        "live_points": "min_num_live_points",
        "frac_remain": "frac_remain",
    }

    def _handle_deprecated_opts(self, sampler_opts, run_opts):
        for old_name, new_name in self._deprecated_sampler_opts.items():
            if old_name not in sampler_opts:
                continue
            warnings.warn(
                f"The UltraNest sampler option '{old_name}' is deprecated, "
                f"use run_opts={{'{new_name}': ...}} instead.",
                GammapyDeprecationWarning,
            )
            value = sampler_opts.pop(old_name)
            run_opts.setdefault(new_name, value)
        return sampler_opts, run_opts

    def build(self, parameters, like):
        import ultranest

        self._sampler = ultranest.ReactiveNestedSampler(
            parameters.names,
            like.fcn,
            transform=parameters._prior_inverse_cdf,
            **self._forwarded_sampler_opts(),
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

    def run(self):
        return self._sampler.run(**self.run_opts)

    def to_result(self, result_dict):
        return SamplerResult.from_ultranest(result_dict)

    def print_summary(self):
        self._sampler.print_results()


class _NautilusBackend(_SamplerBackend):
    """Nautilus backend using `nautilus.Sampler`."""

    tag = "nautilus"
    default_sampler_opts = {"n_live": 2000, "filepath": None, "resume": True}
    default_run_opts = {"f_live": 0.01, "n_eff": 2000, "verbose": True}

    def build(self, parameters, like):
        import nautilus

        self._sampler = nautilus.Sampler(
            prior=parameters._prior_inverse_cdf,
            likelihood=like.fcn,
            n_dim=len(parameters),
            **self._forwarded_sampler_opts(),
        )

    def run(self):
        import numpy as np

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
            "weighted_samples": dict(points=points, weights=np.exp(log_w)),
            "log_l": log_l,
        }
        return result

    def to_result(self, result_dict):
        return SamplerResult.from_nautilus(result_dict)

    def print_summary(self):
        self._sampler.print_status()


SAMPLER_BACKENDS = {
    "ultranest": _UltranestBackend,
    "nautilus": _NautilusBackend,
}


class Sampler:
    """Sampler class.

    The sampler class provides a uniform interface to multiple sampler backends.
    Currently available: "ultranest" and "nautilus".

    The options are passed directly to the chosen backend, using the backend's own
    keyword names: ``sampler_opts`` are forwarded to the backend sampler constructor
    and ``run_opts`` to its ``run`` method. Options not given take the Gammapy
    defaults listed below, otherwise the backend defaults apply.

    Parameters
    ----------
    backend : {"ultranest", "nautilus"}
        Global backend used for sampler. Default is "ultranest".
        UltraNest: Most options can be found in the
        `UltraNest doc <https://johannesbuchner.github.io/UltraNest/>`__.
        Nautilus uses neural-network-guided nested sampling. See the
        `Nautilus documentation <https://nautilus-sampler.readthedocs.io/>`__.
    sampler_opts : dict, optional
        Options passed to the backend sampler constructor. See the full list of options for
        `UltraNest <https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler>`__
        and `Nautilus <https://nautilus-sampler.readthedocs.io/en/latest/api.html#nautilus.Sampler>`__.
        Default is None.

        Gammapy defaults for "ultranest":

        log_dir : str, optional
            Where to store output files.
            Default is None and no results are stored.
        resume : str, optional
            ‘overwrite’, overwrite previous data. ‘subfolder’, create a fresh subdirectory in `log_dir`.
            ‘resume’ or True, continue previous run if available. Only works when dimensionality, transform or
            likelihood are consistent.
            Default is ‘subfolder’.

        Additional options for "ultranest", handled by Gammapy and not forwarded:

        step_sampler : bool, optional
            Use a step sampler. This can be more efficient for higher dimensions (>10 or 15 parameters), but much
            slower for lower dimensions.
            Default is False.
        nsteps : int, optional
            Number of steps to take in each direction in the step sampler. Increase this number to get more
            accurate results at the cost of more computation time.
            Default is 10.

        Gammapy defaults for "nautilus":

        n_live : int, optional
            Number of live points. Default is 2000.
        filepath : str, optional
            Path to the file where the sampler state is stored.
            Default is None and no results are stored.
        resume : bool, optional
            Resume from a previous run stored in `filepath`. Default is True.
    run_opts : dict, optional
        Options passed to the backend sampler run method. See the full list of options for
        `UltraNest <https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler.run>`__
        and `Nautilus <https://nautilus-sampler.readthedocs.io/en/latest/api.html#nautilus.Sampler.run>`__.
        Default is None.

        Gammapy defaults for "ultranest":

        min_num_live_points : int, optional
            Minimum number of live points used in the sampling. Increase this number to get more accurate results.
            For more samples in the posterior increase this number or the ``min_ess`` parameter.
            Default is 400 live points.
        frac_remain : float, optional
            Integrate until this fraction of the integral is left in the remainder. Set to a low number (1e-2 … 1e-5)
            to make sure peaks are discovered. Set to a higher number (0.5) if you know the posterior is simple.
            Default is 0.5.

        Other noteworthy option for "ultranest":

        min_ess : int, optional
            Target number of effective posterior samples. Increase this number to get more accurate results.
            UltraNest default is 400, but you may need to increase it to 1000 or more for complex posteriors.

        Gammapy defaults for "nautilus":

        f_live : float, optional
            Maximum fraction of the evidence contained in the live set before the run stops.
            Default is 0.01.
        n_eff : int, optional
            Minimum effective sample size. Default is 2000.
        verbose : bool, optional
            Print the status of the run. Default is True.

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

        self.backend = backend
        self._backend = SAMPLER_BACKENDS[backend](
            sampler_opts=sampler_opts, run_opts=run_opts
        )

    @property
    def sampler_opts(self):
        """Options passed to the backend sampler constructor, including defaults."""
        return self._backend.sampler_opts

    @property
    def run_opts(self):
        """Options passed to the backend sampler run method, including defaults."""
        return self._backend.run_opts

    @property
    def _sampler(self):
        return self._backend._sampler

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
            function=datasets.stat_sum_likelihood, parameters=parameters
        )
        self._backend.build(parameters, like)
        result_dict = self._backend.run()
        self._backend.print_summary()
        result = self._backend.to_result(result_dict)

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
