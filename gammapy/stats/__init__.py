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
from .fit_statistics_cython import (
    weighted_cash_sum_cython,
    cash_sum_cython,
    f_cash_root_cython,
    norm_bounds_cython,
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
}

__all__ = [
    "cash",
    "cash_sum_cython",
    "CashCountsStatistic",
    "Chi2FitStatistic",
    "Chi2AsymmetricErrorFitStatistic",
    "cstat",
    "f_cash_root_cython",
    "get_wstat_gof_terms",
    "get_wstat_mu_bkg",
    "norm_bounds_cython",
    "wstat",
    "WStatCountsStatistic",
    "compute_fvar",
    "compute_fpp",
    "compute_flux_doubling",
    "compute_chisq",
    "structure_function",
    "discrete_correlation",
    "TimmerKonig_lightcurve_simulator",
    "weighted_cash_sum_cython",
    "sigma_to_ts",
    "ts_to_sigma",
]
