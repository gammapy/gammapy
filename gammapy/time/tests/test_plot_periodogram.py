# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from ...utils.testing import requires_dependency
from ..period import robust_periodogram
from ..plot_periodogram import plot_periodogram
from .test_period import simulate_test_data, fap_astropy

pytest.importorskip("astropy", "3.0")


@requires_dependency("scipy")
def test_plot_periodogram():
    pars = dict(
        period=7,
        amplitude=2,
        t_length=100,
        n_data=1000,
        n_obs=500,
        n_outliers=50,
        loss="cauchy",
        scale=1,
    )
    test_data = simulate_test_data(
        pars["period"],
        pars["amplitude"],
        pars["t_length"],
        pars["n_data"],
        pars["n_obs"],
        pars["n_outliers"],
    )

    periodogram = robust_periodogram(
        test_data["t"],
        test_data["y"],
        test_data["dy"],
        loss=pars["loss"],
        scale=pars["scale"],
    )

    fap = fap_astropy(
        periodogram["power"],
        1. / periodogram["periods"],
        test_data["t"],
        test_data["y"],
        test_data["dy"],
    )

    plot_periodogram(
        test_data["t"],
        test_data["y"],
        periodogram["periods"],
        periodogram["power"],
        test_data["dy"],
        periodogram["best_period"],
        max(fap.values()),
    )
