# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.modeling import Fit, Parameter
from gammapy.stats.utils import sigma_to_ts
from .fit import FitResult, OptimizeResult

__all__ = ["select_nested_models"]


class TestStatisticNested:
    """Compute the test statistic (TS) between two nested hypothesis.

    The null hypothesis is the minimal one, for which a set of parameters
    are frozen to given values. The model is updated to the alternative hypothesis
    if there is a significant improvement (larger than the given threshold).

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters` or list of `~gammapy.modeling.Parameter`
        List of parameters frozen for the null hypothesis but free for the test hypothesis.
    null_values : list of float or `~gammapy.modeling.Parameters`
        Values of the parameters frozen for the null hypothesis.
        If a `Parameters` object or a list of `Parameters` is given
        the null hypothesis follows the values of these parameters,
        so this tests linked parameters versus unliked.
    n_sigma : float
        Threshold in number of sigma to switch from the null hypothesis
        to the alternative one. Default is 2.
        The TS is converted to sigma assuming that the Wilk's theorem is verified.
    n_free_parameters : int
        Number of free parameters to consider between the two hypothesis
        in order to estimate the `ts_threshold` from the `n_sigma` threshold.
        Default is len(parameters).
    fit : `Fit`
        Fit instance specifying the backend and fit options.
    """

    __test__ = False

    def __init__(
        self, parameters, null_values, n_sigma=2, n_free_parameters=None, fit=None
    ):
        self.parameters = parameters
        self.null_values = null_values
        self.n_sigma = n_sigma

        if n_free_parameters is None:
            n_free_parameters = len(parameters)
        self.n_free_parameters = n_free_parameters

        if fit is None:
            fit = Fit()
            minuit_opts = {"tol": 0.1, "strategy": 1}
            fit.backend = "minuit"
            fit.optimize_opts = minuit_opts
        self.fit = fit

    @property
    def ts_threshold(self):
        """Threshold value in TS corresponding to `n_sigma`.

        This assumes that the TS follows a chi squared distribution
        with a number of degree of freedom equal to `n_free_parameters`.
        """
        return np.sign(self.n_sigma) * sigma_to_ts(self.n_sigma, self.n_free_parameters)

    def ts_known_bkg(self, datasets):
        """Perform the alternative hypothesis testing assuming known background (all parameters frozen).
        This implicitly assumes that the non-null model is a good representation of the true model.
        If the assumption is true the ts_known_bkg should tend to the ts_asimov (deviation would indicate a bad fit of the data).
        Deviations between ts and frozen_ts can be used to identify potential sources of confusion depending on which parameters are let free for the ts computation
         (for example considereing diffuse background or nearby source).
        """
        stat = datasets.stat_sum()
        object_cache, prev_pars = self._apply_null_hypothesis(datasets)
        stat_null = datasets.stat_sum()
        self._restore_status(datasets, object_cache, prev_pars)
        return stat_null - stat

    def ts_asimov(self, datasets):
        """Perform the alternative hypothesis testing in the Asimov dataset.
        The Asimov dataset is defined by counts=npred such as the non-null model is the true model.
        """
        counts_cache = [d.counts for d in datasets]
        for d in datasets:
            d.counts = d.npred()

        ts = self.ts_known_bkg(datasets)

        for kd, d in enumerate(datasets):
            d.counts = counts_cache[kd]
        return ts

    def ts(self, datasets):
        """Perform the alternative hypothesis testing."""
        return self.run(datasets, apply_selection=False)["ts"]

    def run(self, datasets, apply_selection=True):
        """Perform the alternative hypothesis testing and apply model selection.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets.
        apply_selection : bool
            Apply or not the model selection. Default is True.

        Returns
        -------
        result : dict
            Dictionary with the TS of the best fit value compared to the null hypothesis
            and fit results for the two hypotheses. Entries are:

                * "ts" : fit statistic difference with null hypothesis
                * "fit_results" : results for the best fit
                * "fit_results_null" : fit results for the null hypothesis
        """

        for p in self.parameters:
            p.frozen = False
        fit_results = self.fit.run(datasets)
        stat = datasets.stat_sum()

        object_cache, prev_pars = self._apply_null_hypothesis(datasets)

        if len(datasets.models.parameters.free_parameters) > 0:
            fit_results_null = self.fit.run(datasets)
        else:
            fit_results_null = FitResult(
                OptimizeResult(
                    models=datasets.models.copy(),
                    nfev=0,
                    total_stat=datasets.stat_sum(),
                    trace=None,
                    backend=None,
                    method=None,
                    success=None,
                    message=None,
                )
            )
        stat_null = datasets.stat_sum()

        ts = stat_null - stat
        if not apply_selection or ts > self.ts_threshold:
            # restore default model if preferred against null hypothesis or if selection is ignored
            self._restore_status(datasets, object_cache, prev_pars)
        return dict(
            ts=ts,
            fit_results=fit_results,
            fit_results_null=fit_results_null,
        )

    def _apply_null_hypothesis(self, datasets):
        object_cache = [p.__dict__ for p in datasets.models.parameters]
        prev_pars = [p.value for p in datasets.models.parameters]
        for p, val in zip(self.parameters, self.null_values):
            if isinstance(val, Parameter):
                p.__dict__ = val.__dict__
            else:
                p.value = val
                p.frozen = True
        return object_cache, prev_pars

    def _restore_status(self, datasets, object_cache, prev_pars):
        """Restore parameters to given cached cached objects and values"""
        for p in self.parameters:
            p.frozen = False
        for kp, p in enumerate(datasets.models.parameters):
            p.__dict__ = object_cache[kp]
            p.value = prev_pars[kp]


def select_nested_models(
    datasets, parameters, null_values, n_sigma=2, n_free_parameters=None, fit=None
):
    """Compute the test statistic (TS) between two nested hypothesis.

    The null hypothesis is the minimal one, for which a set of parameters
    are frozen to given values. The model is updated to the alternative hypothesis
    if there is a significant improvement (larger than the given threshold).

    Parameters
    ----------
    datasets : `~gammapy.datasets.Datasets`
        Datasets.
    parameters : `~gammapy.modeling.Parameters` or list of `~gammapy.modeling.Parameter`
        List of parameters frozen for the null hypothesis but free for the test hypothesis.
    null_values : list of float or `~gammapy.modeling.Parameters`
        Values of the parameters frozen for the null hypothesis.
        If a `Parameters` object or a list of `Parameters` is given
        the null hypothesis follows the values of these parameters,
        so this tests linked parameters versus unliked.
    n_sigma : float, optional
        Threshold in number of sigma to switch from the null hypothesis
        to the alternative one. Default is 2.
        The TS is converted to sigma assuming that the Wilk's theorem is verified.
    n_free_parameters : int, optional
        Number of free parameters to consider between the two hypothesis
        in order to estimate the `ts_threshold` from the `n_sigma` threshold.
        Default is len(parameters).
    fit : `Fit`, optional
        Fit instance specifying the backend and fit options. Default is None.

    Returns
    -------
    result : dict
        Dictionary with the TS of the best fit value compared to the null hypothesis
        and fit results for the two hypotheses. Entries are:

            * "ts" : fit statistic difference with null hypothesis
            * "fit_results" : results for the best fit
            * "fit_results_null" : fit results for the null hypothesis

    Examples
    --------
    .. testcode::

        from gammapy.modeling.selection import select_nested_models
        from gammapy.datasets import Datasets, SpectrumDatasetOnOff
        from gammapy.modeling.models import SkyModel

        # Test if cutoff is significant
        dataset = SpectrumDatasetOnOff.read("$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits")
        datasets = Datasets(dataset)
        model = SkyModel.create(spectral_model="ecpl", spatial_model="point", name='hess')
        datasets.models = model
        result = select_nested_models(datasets,
                                      parameters=[model.spectral_model.lambda_],
                                      null_values=[0],
                                      )

        # Test if source is significant
        filename = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"
        filename_models = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"
        fermi_datasets = Datasets.read(filename=filename, filename_models=filename_models)
        model = fermi_datasets.models["Crab Nebula"]
        # Number of parameters previously fit for the source of interest
        n_free_parameters = len(model.parameters.free_parameters)
        # Freeze spatial parameters to ensure another weaker source does not move from its position
        # to replace the source of interest during the null hypothesis test.
        # (with all parameters free you test N vs. N+1 models and not the detection of a specific source.)
        fermi_datasets.models.freeze(model_type='spatial')
        results = select_nested_models(fermi_datasets,
                                      parameters=[model.spectral_model.amplitude],
                                      null_values=[0],
                                      n_free_parameters=n_free_parameters,
                                      n_sigma=4,
                                      )
    """
    test = TestStatisticNested(parameters, null_values, n_sigma, n_free_parameters, fit)
    return test.run(datasets)
