# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sampler parameter classes."""

import abc
import inspect
import warnings
from gammapy.utils.deprecation import (
    GammapyDeprecationWarning,
    deprecated_renamed_argument,
)
from .utils import _parse_datasets

__all__ = ["Sampler", "SamplerLikelihood", "SamplerResult"]

# Filled automatically with each backend class, keyed by its tag.
SAMPLER_BACKENDS = {}


class _SamplerBackend(abc.ABC):
    """Base class for the sampler backends.

    Subclasses defining a ``tag`` are registered in ``SAMPLER_BACKENDS``.

    The options use the backend library's own names. Before the run, each of them
    is sent to the backend sampler constructor or to its ``run`` method, by
    inspecting their signatures.

    Parameters
    ----------
    options : dict, optional
        Options routed automatically to the constructor or the run method.
        Default is None.
    sampler_opts : dict, optional
        Deprecated. Options passed as is to the backend sampler constructor.
        Default is None.
    run_opts : dict, optional
        Deprecated. Options passed as is to the backend sampler run method.
        Default is None.
    """

    tag = None
    default_options = {}
    # backend constructor arguments that Gammapy fills from the datasets and models,
    # excluded from the automatic options routing so that users cannot override them
    gammapy_filled_args = ()
    # old option name: new option name
    deprecated_options = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.tag is not None:
            if cls.tag in SAMPLER_BACKENDS:
                raise ValueError(f"Sampler backend '{cls.tag}' is already registered.")
            SAMPLER_BACKENDS[cls.tag] = cls

    def __init__(self, options=None, sampler_opts=None, run_opts=None):
        options = {} if options is None else dict(options)
        sampler_opts = {} if sampler_opts is None else dict(sampler_opts)
        run_opts = {} if run_opts is None else dict(run_opts)

        options = self._rename_deprecated_options(options)
        sampler_opts, run_opts = self._handle_deprecated_opts(sampler_opts, run_opts)

        duplicates = set(options) & (set(sampler_opts) | set(run_opts))
        if duplicates:
            raise ValueError(
                f"Options {sorted(duplicates)} are given twice, "
                "in options and in sampler_opts or run_opts."
            )

        self._sampler = None
        self.options = {**self.default_options, **options}
        self._explicit_sampler_opts = sampler_opts
        self._explicit_run_opts = run_opts
        self.sampler_opts = None
        self.run_opts = None

    def _rename_deprecated_options(self, options):
        for old_name, new_name in self.deprecated_options.items():
            if old_name in options:
                warnings.warn(
                    f"The {self.tag} sampler option '{old_name}' is deprecated, "
                    f"use '{new_name}' instead.",
                    GammapyDeprecationWarning,
                    stacklevel=5,
                )
                options.setdefault(new_name, options.pop(old_name))
        return options

    def resolve_options(self):
        """Split the options between the sampler constructor and its run method.

        This imports the backend library. The result is stored in ``sampler_opts``
        and ``run_opts`` and returned.
        """
        explicit = set(self._explicit_sampler_opts) | set(self._explicit_run_opts)
        options = {
            key: value for key, value in self.options.items() if key not in explicit
        }
        sampler_opts, run_opts = self._route_options(options)
        self.sampler_opts = {**sampler_opts, **self._explicit_sampler_opts}
        self.run_opts = {**run_opts, **self._explicit_run_opts}
        return self.sampler_opts, self.run_opts

    @staticmethod
    @abc.abstractmethod
    def _import_sampler_class():
        """Import and return the backend sampler class."""

    def _handle_deprecated_opts(self, sampler_opts, run_opts):
        return sampler_opts, run_opts

    def _route_options(self, options):
        """Split options between the sampler constructor and its run method.

        The split uses the signatures of the backend sampler class, so it follows
        the installed version of the backend library.
        """
        sampler_class = self._import_sampler_class()
        for function in [sampler_class.__init__, sampler_class.run]:
            if self._has_var_keyword(function):
                raise ValueError(
                    f"The {self.tag} sampler accepts arbitrary keyword arguments, "
                    "options cannot be routed automatically."
                )

        init_names = self._parameter_names(sampler_class.__init__)
        init_names -= set(self.gammapy_filled_args)
        run_names = self._parameter_names(sampler_class.run)

        sampler_opts, run_opts = {}, {}
        for key, value in options.items():
            in_init, in_run = key in init_names, key in run_names
            if in_init and in_run:
                raise ValueError(
                    f"Option '{key}' is accepted by both the {self.tag} sampler and "
                    "its run method, it cannot be routed automatically."
                )
            elif in_init:
                sampler_opts[key] = value
            elif in_run:
                run_opts[key] = value
            else:
                raise ValueError(
                    f"Unknown {self.tag} sampler option '{key}'. "
                    f"Valid options are: {sorted(init_names | run_names)}."
                )
        return sampler_opts, run_opts

    @staticmethod
    def _has_var_keyword(function):
        parameters = inspect.signature(function).parameters.values()
        return any(par.kind == par.VAR_KEYWORD for par in parameters)

    @staticmethod
    def _parameter_names(function):
        """Names of the keyword arguments of a function, without ``self``."""
        parameters = inspect.signature(function).parameters
        return {
            name
            for name, par in parameters.items()
            if name != "self"
            and par.kind in (par.POSITIONAL_OR_KEYWORD, par.KEYWORD_ONLY)
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
    default_options = {
        "log_dir": None,
        "resume": "subfolder",
        "min_num_live_points": 400,
        "frac_remain": 0.5,
    }
    gammapy_filled_args = ("param_names", "loglike", "transform")
    deprecated_options = {"live_points": "min_num_live_points"}

    # old sampler_opts name: new run_opts name
    _deprecated_sampler_opts = {
        "live_points": "min_num_live_points",
        "frac_remain": "frac_remain",
    }

    @staticmethod
    def _import_sampler_class():
        import ultranest

        return ultranest.ReactiveNestedSampler

    def _handle_deprecated_opts(self, sampler_opts, run_opts):
        for old_name, new_name in self._deprecated_sampler_opts.items():
            if old_name not in sampler_opts:
                continue
            warnings.warn(
                f"The UltraNest sampler option '{old_name}' is deprecated, "
                f"use options={{'{new_name}': ...}} instead.",
                GammapyDeprecationWarning,
                stacklevel=5,
            )
            value = sampler_opts.pop(old_name)
            run_opts.setdefault(new_name, value)
        return sampler_opts, run_opts

    def build(self, parameters, like):
        self._sampler = self._import_sampler_class()(
            parameters.names,
            like.fcn,
            transform=parameters._prior_inverse_cdf,
            **self.sampler_opts,
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
    default_options = {
        "n_live": 2000,
        "filepath": None,
        "resume": True,
        "f_live": 0.01,
        "n_eff": 2000,
        "verbose": True,
    }
    gammapy_filled_args = ("prior", "likelihood", "n_dim")

    @staticmethod
    def _import_sampler_class():
        import nautilus

        return nautilus.Sampler

    def build(self, parameters, like):
        self._sampler = self._import_sampler_class()(
            prior=parameters._prior_inverse_cdf,
            likelihood=like.fcn,
            n_dim=len(parameters),
            **self.sampler_opts,
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


class Sampler:
    """Sampler class.

    The sampler class provides a uniform interface to multiple sampler backends.
    Currently available: "ultranest" and "nautilus".

    Backend options are given in ``options``, with the backend's own names. When
    the sampler runs, Gammapy passes each of them to the backend sampler constructor
    or to its ``run`` method, based on the signatures of the installed backend
    library. An unknown option raises an error listing the valid ones.

    Parameters
    ----------
    backend : {"ultranest", "nautilus"}
        Global backend used for sampler. Default is "ultranest".
        UltraNest: Most options can be found in the
        `UltraNest doc <https://johannesbuchner.github.io/UltraNest/>`__.
        Nautilus uses neural-network-guided nested sampling. See the
        `Nautilus documentation <https://nautilus-sampler.readthedocs.io/>`__.
    sampler_opts : dict, optional
        Deprecated since v2.2, use ``options`` instead. Default is None.
    run_opts : dict, optional
        Deprecated since v2.2, use ``options`` instead. Default is None.
    options : dict, optional
        Backend options, routed automatically. Default is None. See the options of
        `UltraNest <https://johannesbuchner.github.io/UltraNest/ultranest.html#ultranest.integrator.ReactiveNestedSampler>`__
        (constructor and ``run``) and of
        `Nautilus <https://nautilus-sampler.readthedocs.io/en/latest/api.html#nautilus.Sampler>`__.

        Gammapy defaults for "ultranest": ``log_dir=None``, ``resume="subfolder"``,
        ``min_num_live_points=400``, ``frac_remain=0.5``.

        Gammapy defaults for "nautilus": ``n_live=2000``, ``filepath=None``,
        ``resume=True``, ``f_live=0.01``, ``n_eff=2000``, ``verbose=True``.

    Examples
    --------
    >>> from gammapy.modeling import Sampler
    >>> options = {"min_num_live_points": 300, "frac_remain": 0.3, "log_dir": None}
    >>> sampler = Sampler(backend="ultranest", options=options)

    For a complete usage example, see :doc:`/tutorials/details/nested_sampling_Crab` tutorial.

    Notes
    -----
    If you are using the "UltraNest" library, please follow its citation scheme:
    `Cite UltraNest <https://johannesbuchner.github.io/UltraNest/issues.html#how-should-i-cite-ultranest>`__.

    """

    # TODO: add "zeusmc", "emcee"

    @deprecated_renamed_argument(
        ["sampler_opts", "run_opts"],
        [None, None],
        ["2.2", "2.2"],
        alternative="options",
    )
    def __init__(
        self, backend="ultranest", sampler_opts=None, run_opts=None, options=None
    ):
        if backend not in SAMPLER_BACKENDS:
            raise ValueError(
                f"Sampler {backend} is not supported. "
                f"Available backends are: {sorted(SAMPLER_BACKENDS)}."
            )

        self.backend = backend
        self._backend = SAMPLER_BACKENDS[backend](
            options=options, sampler_opts=sampler_opts, run_opts=run_opts
        )

    @property
    def options(self):
        """Backend options, including the Gammapy defaults."""
        return self._backend.options

    @property
    def sampler_opts(self):
        """Options passed to the backend sampler constructor.

        Deprecated, use `options` instead. Accessing it imports the backend library.
        """
        warnings.warn(
            "Sampler.sampler_opts is deprecated, use Sampler.options instead.",
            GammapyDeprecationWarning,
            stacklevel=2,
        )
        return self._backend.resolve_options()[0]

    @property
    def run_opts(self):
        """Options passed to the backend sampler run method.

        Deprecated, use `options` instead. Accessing it imports the backend library.
        """
        warnings.warn(
            "Sampler.run_opts is deprecated, use Sampler.options instead.",
            GammapyDeprecationWarning,
            stacklevel=2,
        )
        return self._backend.resolve_options()[1]

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
        self._backend.resolve_options()
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
