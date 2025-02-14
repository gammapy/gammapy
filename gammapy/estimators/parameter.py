# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings

import logging
import numpy as np
from gammapy.datasets import Datasets
from gammapy.datasets.actors import DatasetsActor
from gammapy.modeling import Fit
from gammapy.modeling.selection import TestStatisticNested
from gammapy.modeling.parameter import restore_parameters_status
from gammapy.stats.utils import ts_to_sigma
from gammapy.utils.roots import find_roots
from .core import Estimator

log = logging.getLogger(__name__)

__all__ = ["ParameterEstimator", "ParameterSensitivityEstimator"]


class ParameterEstimator(Estimator):
    """Model parameter estimator.

    Estimates a model parameter for a group of datasets. Compute best fit value,
    symmetric and delta(TS) for a given null value. Additionally asymmetric errors
    as well as parameter upper limit and fit statistic profile can be estimated.

    Parameters
    ----------
    n_sigma : int
        Sigma to use for asymmetric error computation. Default is 1.
    n_sigma_ul : int
        Sigma to use for upper limit computation. Default is 2.
    n_sigma_sensitivity : int
        Sigma to use for sensitivity computation. Default is 5.
    null_value : float
        Which null value to use for the parameter.
    selection_optional : list of str, optional
        Which additional quantities to estimate. Available options are:

            * "all": all the optional steps are executed.
            * "errn-errp": estimate asymmetric errors on parameter best fit value.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        Default is None so the optional steps are not executed.
    fit : `~gammapy.modeling.Fit`
        Fit instance specifying the backend and fit options.
    reoptimize : bool
        Re-optimize other free model parameters. Default is True.

    Examples
    --------
    >>> from gammapy.datasets import SpectrumDatasetOnOff, Datasets
    >>> from gammapy.modeling.models import SkyModel, PowerLawSpectralModel
    >>> from gammapy.estimators import ParameterEstimator
    >>>
    >>> filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
    >>> dataset = SpectrumDatasetOnOff.read(filename)
    >>> datasets = Datasets([dataset])
    >>> spectral_model = PowerLawSpectralModel(amplitude="3e-11 cm-2s-1TeV-1", index=2.7)
    >>>
    >>> model = SkyModel(spectral_model=spectral_model, name="Crab")
    >>> model.spectral_model.amplitude.scan_n_values = 10
    >>>
    >>> for dataset in datasets:
    ...     dataset.models = model
    >>>
    >>> estimator = ParameterEstimator(selection_optional="all")
    >>> result = estimator.run(datasets, parameter="amplitude")
    """

    tag = "ParameterEstimator"
    _available_selection_optional = ["errn-errp", "ul", "scan", "sensitivity"]

    def __init__(
        self,
        n_sigma=1,
        n_sigma_ul=2,
        n_sigma_sensitivity=5,
        null_value=1e-150,
        selection_optional=None,
        fit=None,
        reoptimize=True,
    ):
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.n_sigma_sensitivity = n_sigma_sensitivity
        self.null_value = null_value
        self.selection_optional = selection_optional

        if fit is None:
            fit = Fit()

        self.fit = fit
        self.reoptimize = reoptimize

    def estimate_best_fit(self, datasets, parameter):
        """Estimate parameter asymmetric errors.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets.
        parameter : `Parameter`
            For which parameter to get the value.

        Returns
        -------
        result : dict
            Dictionary with the various parameter estimation values. Entries are:

                * parameter.name: best fit parameter value.
                * "stat": best fit total stat.
                * "success": boolean flag for fit success.
                * parameter.name_err: covariance-based error estimate on parameter value.
        """
        value, total_stat, success, error = np.nan, 0.0, False, np.nan

        if np.any(datasets.contributes_to_stat):
            result = self.fit.run(datasets=datasets)
            value, error = parameter.value, parameter.error
            total_stat = result.optimize_result.total_stat
            success = result.success

        return {
            f"{parameter.name}": value,
            "stat": total_stat,
            "success": success,
            f"{parameter.name}_err": error * self.n_sigma,
        }

    def estimate_ts(self, datasets, parameter):
        """Estimate parameter ts.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets.
        parameter : `Parameter`
            For which parameter to get the value.

        Returns
        -------
        result : dict
            Dictionary with the test statistic of the best fit value compared to the null hypothesis. Entries are:

                * "ts" : fit statistic difference with null hypothesis.
                * "npred" : predicted number of counts per dataset.
                * "stat_null" : total stat corresponding to the null hypothesis
        """
        npred = self.estimate_npred(datasets=datasets)

        if not np.any(datasets.contributes_to_stat):
            stat = np.nan
            npred["npred"][...] = np.nan
        else:
            stat = datasets.stat_sum()

        with datasets.parameters.restore_status():
            # compute ts value
            parameter.value = self.null_value

            if self.reoptimize:
                parameter.frozen = True
                _ = self.fit.optimize(datasets=datasets)

            ts = datasets.stat_sum() - stat
            stat_null = datasets.stat_sum()

        return {"ts": ts, "npred": npred["npred"], "stat_null": stat_null}

    def estimate_errn_errp(self, datasets, parameter):
        """Estimate parameter asymmetric errors.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets.
        parameter : `Parameter`
            For which parameter to get the value.

        Returns
        -------
        result : dict
            Dictionary with the parameter asymmetric errors. Entries are:

                * {parameter.name}_errp : positive error on parameter value.
                * {parameter.name}_errn : negative error on parameter value.
        """
        if not np.any(datasets.contributes_to_stat):
            return {
                f"{parameter.name}_errp": np.nan,
                f"{parameter.name}_errn": np.nan,
            }

        self.fit.optimize(datasets=datasets)

        res = self.fit.confidence(
            datasets=datasets,
            parameter=parameter,
            sigma=self.n_sigma,
            reoptimize=self.reoptimize,
        )

        return {
            f"{parameter.name}_errp": res["errp"],
            f"{parameter.name}_errn": res["errn"],
        }

    def estimate_scan(self, datasets, parameter):
        """Estimate parameter statistic scan.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter.
        parameter : `~gammapy.modeling.Parameter`
            For which parameter to get the value.

        Returns
        -------
        result : dict
            Dictionary with the parameter fit scan values. Entries are:

                * parameter.name_scan : parameter values scan.
                * "stat_scan" : fit statistic values scan.
        """
        scan_values = parameter.scan_values

        if not np.any(datasets.contributes_to_stat):
            return {
                f"{parameter.name}_scan": scan_values,
                "stat_scan": scan_values * np.nan,
            }

        self.fit.optimize(datasets=datasets)

        profile = self.fit.stat_profile(
            datasets=datasets, parameter=parameter, reoptimize=self.reoptimize
        )

        return {
            f"{parameter.name}_scan": scan_values,
            "stat_scan": profile["stat_scan"],
        }

    def estimate_ul(self, datasets, parameter):
        """Estimate parameter ul.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter.
        parameter : `~gammapy.modeling.Parameter`
            For which parameter to get the value.

        Returns
        -------
        result : dict
            Dictionary with the parameter upper limits. Entries are:

                * parameter.name_ul : upper limit on parameter value.
        """
        if not np.any(datasets.contributes_to_stat):
            return {f"{parameter.name}_ul": np.nan}

        self.fit.optimize(datasets=datasets)

        res = self.fit.confidence(
            datasets=datasets,
            parameter=parameter,
            sigma=self.n_sigma_ul,
            reoptimize=self.reoptimize,
        )
        return {f"{parameter.name}_ul": res["errp"] + parameter.value}

    def estimate_sensitivity(self, datasets, parameter):
        """Estimate norm sensitivity for the flux point.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets.

        Returns
        -------
        result : dict
            Dictionary with an array with one entry per dataset with the sum of the
            masked npred.
        """

        estimator = ParameterSensitivityEstimator(
            parameter, self.null_value, n_sigma=self.n_sigma_sensitivity
        )
        value = estimator.run(datasets)
        return {f"{parameter.name}_sensitivity": value}

    @staticmethod
    def estimate_counts(datasets):
        """Estimate counts for the flux point.

        Parameters
        ----------
        datasets : Datasets
            Datasets.

        Returns
        -------
        result : dict
            Dictionary with an array with one entry per dataset with the sum of the
            masked counts.
        """
        counts = []

        for dataset in datasets:
            mask = dataset.mask
            counts.append(dataset.counts.data[mask].sum())

        return {"counts": np.array(counts, dtype=int), "datasets": datasets.names}

    @staticmethod
    def estimate_npred(datasets):
        """Estimate npred for the flux point.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets.

        Returns
        -------
        result : dict
            Dictionary with an array with one entry per dataset with the sum of the
            masked npred.
        """
        npred = []

        for dataset in datasets:
            mask = dataset.mask
            npred.append(dataset.npred().data[mask].sum())

        return {"npred": np.array(npred), "datasets": datasets.names}

    def run(self, datasets, parameter):
        """Run the parameter estimator.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter.
        parameter : `str` or `~gammapy.modeling.Parameter`
            For which parameter to run the estimator.

        Returns
        -------
        result : dict
            Dictionary with the various parameter estimation values.
        """
        if not isinstance(datasets, DatasetsActor):
            datasets = Datasets(datasets)
        parameter = datasets.parameters[parameter]

        with datasets.parameters.restore_status():
            if not self.reoptimize:
                datasets.parameters.freeze_all()
                parameter.frozen = False

            result = self.estimate_best_fit(datasets, parameter)
            result.update(self.estimate_ts(datasets, parameter))

            if "errn-errp" in self.selection_optional:
                result.update(self.estimate_errn_errp(datasets, parameter))

            if "ul" in self.selection_optional:
                result.update(self.estimate_ul(datasets, parameter))

            if "scan" in self.selection_optional:
                result.update(self.estimate_scan(datasets, parameter))

            if "sensitivity" in self.selection_optional:
                result.update(self.estimate_sensitivity(datasets, parameter))

        result.update(self.estimate_counts(datasets))
        return result


class ParameterSensitivityEstimator:
    """Estimate the sensitivity to a given parameter

    Computes the TS distribution in the non-null hypothesis using the
    log likelihood of the Asimov dataset (i.e. a dataset with counts = npred)
    and the non central chi2 distribution.
    Once the TS distribution under the testing hypothesis is known,
    one can compute the required parameter value
    to have 50% of measurements above a given significance threshold.


    Parameters
    ----------
    parameter : `~gammapy.modeling.Parameter`
       Parameter to test
    null_value : float or `~gammapy.modeling.Parameter`
        Value of the parameter for the null hypothesis.
    n_sigma : int, default=5
        Number of required significance level.
    rtol : float
        Relative precision of the estimate. Used as a stopping criterion.
        Default is 0.01.
    max_niter : int
        Maximal number of iterations used by the root finding algorithm.
        Default is 100.

    References
    ----------
        * `Cowan et al. (2011), "Asymptotic formulae for likelihood-based tests of new physics"
        <https://arxiv.org/abs/1007.1727>`_

    """

    tag = "ParameterSensitivityEstimator"

    def __init__(
        self,
        parameter,
        null_value,
        n_sigma=5,
        n_free_parameters=None,
        rtol=0.01,
        max_niter=100,
    ):
        self.test = TestStatisticNested(
            [parameter], [null_value], n_free_parameters=n_free_parameters
        )
        self.parameter = parameter
        self.n_sigma = n_sigma
        self.rtol = rtol
        self.max_niter = max_niter

    def _fcn(self, value, datasets):
        """Call the Test Statistics function."""
        self.parameter.value = value
        ts_asimov = self.test.ts_asimov(datasets)
        return ts_to_sigma(ts_asimov, ts_asimov=ts_asimov) - self.n_sigma

    def parameter_matching_significance(self, datasets):
        """Parameter value  matching the target significance"""

        if ~np.isfinite(self.parameter.min):
            vmin = self.parameter.value / 1e3
        else:
            vmin = self.parameter.min
        if ~np.isfinite(self.parameter.max):
            vmax = self.parameter.value * 1e3
        else:
            vmax = self.parameter.max

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roots, res = find_roots(
                self._fcn,
                vmin,
                vmax,
                args=(datasets,),
                nbin=100,
                maxiter=self.max_niter,
                rtol=self.rtol,
                points_scale=self.parameter.interp,
            )
        # Where the root finding fails NaN is set as norm
        roots = roots[roots > 0]
        if roots.size > 0:
            return roots[0]
        else:
            return np.nan

    def run(self, datasets):
        """Parameter sensitivity
        given as the difference between value matching the target significance and the null value.
        """
        with restore_parameters_status(self.test.parameters):
            value = self.parameter_matching_significance(datasets)

        return value - self.test.null_values[0]
