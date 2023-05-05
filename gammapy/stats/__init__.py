# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Statistics."""
from .counts_statistic import CashCountsStatistic, WStatCountsStatistic
from .fit_statistics import cash, cstat, get_wstat_gof_terms, get_wstat_mu_bkg, wstat
from .variability import (compute_fvar, weighted_fvar, compute_fpp,
                          compute_etime, compute_2time, compute_chisq,
                          lc_fpp, lc_fvar, eval_lc_timing)

from .fit_statistics_cython import (
    cash_sum_cython,
    f_cash_root_cython,
    norm_bounds_cython,
)

__all__ = [
    "cash",
    "cash_sum_cython",
    "CashCountsStatistic",
    "cstat",
    "f_cash_root_cython",
    "get_wstat_gof_terms",
    "get_wstat_mu_bkg",
    "norm_bounds_cython",
    "wstat",
    "WStatCountsStatistic",
    "compute_fvar",
    "weighted_fvar",
    "compute_fpp",
    "compute_2time",
    "compute_etime",
    "compute_chisq",
    "lc_fvar",
    "lc_fpp",
    "eval_lc_time",
]
