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
)
from .fit_statistics_jit import (
    weighted_cash_sum_jit,
    cash_sum_jit,
    f_cash_root_jit,
    norm_bounds_jit,
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

FIT_STATISTICS_REGISTRY = {
    "cash": CashFitStatistic,
    "wstat": WStatFitStatistic,
    "chi2": Chi2FitStatistic,
    "distrib": Chi2AsymmetricErrorFitStatistic,
    "profile": ProfileFitStatistic,
    "cash_weighted": WeightedCashFitStatistic,
}

__all__ = [
    "cash",
    "cash_sum_jit",
    "CashCountsStatistic",
    "cstat",
    "f_cash_root_jit",
    "get_wstat_gof_terms",
    "get_wstat_mu_bkg",
    "norm_bounds_jit",
    "wstat",
    "WStatCountsStatistic",
    "compute_fvar",
    "compute_fpp",
    "compute_flux_doubling",
    "compute_chisq",
    "structure_function",
    "discrete_correlation",
    "TimmerKonig_lightcurve_simulator",
    "weighted_cash_sum_jit",
]
