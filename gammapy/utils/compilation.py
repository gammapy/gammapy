# Licensed under a 3-clause BSD style license - see LICENSE.rst

import importlib
from enum import Enum
import logging

log = logging.getLogger(__name__)


def is_numba_available():
    """Check if ray is available."""
    try:
        importlib.import_module("numba")
        return True
    except ModuleNotFoundError:
        return False


class CompilationBackendEnum(Enum):
    """Enum for parallel backend."""

    cython = "cython"
    jit = "jit"

    @classmethod
    def from_str(cls, value):
        """Get enum from string."""
        if value == "jit" and not is_numba_available():
            log.warning("numba is not installed, falling back to cython backend")
            value = "cython"

        return cls(value)


def _get_fit_statistics_cython():
    """Get fit_statistics module with cython."""
    from gammapy.stats.fit_statistics_cython import (
        TRUNCATION_VALUE,
        weighted_cash_sum_cython,
        cash_sum_cython,
        f_cash_root_cython,
        norm_bounds_cython,
    )

    return dict(
        TRUNCATION_VALUE=TRUNCATION_VALUE,
        weighted_cash_sum_compiled=weighted_cash_sum_cython,
        cash_sum_compiled=cash_sum_cython,
        f_cash_root_compiled=f_cash_root_cython,
        norm_bounds_compiled=norm_bounds_cython,
    )


def _get_fit_statistics_jit():
    """Get fit_statistics module with numba backend."""
    from gammapy.stats.fit_statistics_jit import (
        TRUNCATION_VALUE,
        weighted_cash_sum_jit,
        cash_sum_jit,
        f_cash_root_jit,
        norm_bounds_jit,
    )

    return dict(
        TRUNCATION_VALUE=TRUNCATION_VALUE,
        weighted_cash_sum_compiled=weighted_cash_sum_jit,
        cash_sum_compiled=cash_sum_jit,
        f_cash_root_compiled=f_cash_root_jit,
        norm_bounds_compiled=norm_bounds_jit,
    )


COMPILATION_BACKEND_DEFAULT = CompilationBackendEnum.cython

COMPILED_STATS_MODULES = {
    CompilationBackendEnum.cython: _get_fit_statistics_cython,
    CompilationBackendEnum.jit: _get_fit_statistics_jit,
}


def get_fit_statistics_compiled(backend=None):
    if backend is None:
        from gammapy.utils.compilation import COMPILATION_BACKEND_DEFAULT

        backend = COMPILATION_BACKEND_DEFAULT
    backend = CompilationBackendEnum.from_str(backend)
    return COMPILED_STATS_MODULES[backend]()
