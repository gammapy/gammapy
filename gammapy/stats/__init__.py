# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Statistics."""
from .counts_statistic import CashCountsStatistic, WStatCountsStatistic
from .fit_statistics import cash, cstat, get_wstat_gof_terms, get_wstat_mu_bkg, wstat
from .fit_statistics_cython import (
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
    structure_function,
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
    "compute_fpp",
    "compute_flux_doubling",
    "compute_chisq",
    "structure_function",
    "TimmerKonig_lightcurve_simulator",
]
