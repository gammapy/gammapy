# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.stats.utils import sigma_to_ts


class TestStatisticNested:
    """Compute the test statistic (TS) between two nested hypothesis.
    The null hypothesis is the minimal one, for which a set of parameters
    are frozen to given values.

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters` or list of `~gammapy.modeling.Parameter`
        List of parameters frozen for the null hypothesis.
    null_values : list of float
        Values of the parameters frozen for the null hypothesis.
    n_sigma : float
        Threshold in number of sigma to switch from the null hypothesis
        to the alternative one. Default is 2.
    n_free_parameters : int
        Number of free parameters to consider between the two hypothesis
        in order to estimate the `ts_threshold`. Default is len(parameters).
    fit : `Fit`
        Fit instance specifying the backend and fit options.
    """

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
            from gammapy.modeling import Fit

            fit = Fit()
            minuit_opts = {"tol": 0.1, "strategy": 1}
            fit.backend = "minuit"
            fit.optimize_opts = minuit_opts
        self.fit = fit

    @property
    def ts_threshold(self):
        """Threshold value in TS corresponding to `n_sigma`.
        This assumes that the TS follows a chi squared distribution
        with a number of degree of freedom equal to `len(parameters)`.
        """
        return sigma_to_ts(self.n_sigma, len(self.parameters))

    def run(self, datasets):
        """Perform the alternative hypothesis testing

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets

        Returns
        -------
        ts : float
            Test Statistic against the null positive.
        """

        for p in self.parameters:
            p.frozen = False
        self.fit.run(datasets)
        prev_pars = [p.value for p in datasets.models.parameters]
        stat = datasets.stat_sum()

        for p, val in zip(self.parameters, self.null_values):
            p.value = val
            p.frozen = True
        self.fit.run(datasets)
        stat_null = datasets.stat_sum()

        ts = stat_null - stat
        if ts > self.ts_threshold:
            # restore default model if prefered againt null hypothesis
            for p in self.parameters:
                p.frozen = False
            for kp, p in enumerate(datasets.models.parameters):
                p.value = prev_pars[kp]
        return ts
