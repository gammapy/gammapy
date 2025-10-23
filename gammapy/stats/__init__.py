# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Statistics."""

from .counts_statistic import CashCountsStatistic, WStatCountsStatistic
from .fit_statistics import (
    cash,
    cstat,
    get_wstat_gof_terms,
    get_wstat_mu_bkg,
    wstat,
    Chi2FitStatistic,
    CashFitStatistic,
    Chi2AsymmetricErrorFitStatistic,
    ProfileFitStatistic,
    WStatFitStatistic,
    WeightedCashFitStatistic,
    unbinned_likelihood,
    UnbinnedOnOffFitStatistic,
)

from .variability import (
    TimmerKonig_lightcurve_simulator,
    compute_chisq,
    compute_flux_doubling,
    compute_fpp,
    compute_fvar,
    discrete_correlation,
    structure_function,
)
from .utils import sigma_to_ts, ts_to_sigma

FIT_STATISTICS_REGISTRY = {
    "cash": CashFitStatistic,
    "wstat": WStatFitStatistic,
    "chi2": Chi2FitStatistic,
    "distrib": Chi2AsymmetricErrorFitStatistic,
    "profile": ProfileFitStatistic,
    "cash_weighted": WeightedCashFitStatistic,
    "unbinned_onoff": UnbinnedOnOffFitStatistic,
}

__all__ = [
    "cash",
    "CashCountsStatistic",
    "Chi2FitStatistic",
    "Chi2AsymmetricErrorFitStatistic",
    "cstat",
    "get_wstat_gof_terms",
    "get_wstat_mu_bkg",
    "wstat",
    "unbinned_likelihood",
    "UnbinnedOnOffFitStatistic",
    "WStatCountsStatistic",
    "compute_fvar",
    "compute_fpp",
    "compute_flux_doubling",
    "compute_chisq",
    "structure_function",
    "discrete_correlation",
    "TimmerKonig_lightcurve_simulator",
    "sigma_to_ts",
    "ts_to_sigma",
]
