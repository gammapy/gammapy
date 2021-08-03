# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Column, Table
from astropy.time import Time
from gammapy.stats.variability import compute_chisq, compute_fvar
from gammapy.utils.testing import assert_quantity_allclose


@pytest.fixture(scope="session")
def lc_table():
    meta = dict(TIMESYS="utc")

    table = Table(
        meta=meta,
        data=[
            Column(Time(["2010-01-01", "2010-01-03"]).mjd, "time_min"),
            Column(Time(["2010-01-03", "2010-01-10"]).mjd, "time_max"),
            Column([1e-11, 3e-11], "flux", unit="cm-2 s-1"),
            Column([0.1e-11, 0.3e-11], "flux_err", unit="cm-2 s-1"),
            Column([np.nan, 3.6e-11], "flux_ul", unit="cm-2 s-1"),
            Column([False, True], "is_ul"),
        ],
    )

    return table


def test_lightcurve_fvar(lc_table):
    flux = lc_table["flux"].astype("float64")
    flux_err = lc_table["flux_err"].astype("float64")
    fvar, fvar_err = compute_fvar(flux, flux_err)
    assert_allclose(fvar, 0.69821, rtol=1e-5)
    # Note: the following tolerance is very low in the next assert,
    # because results differ by ~ 1e-3 between different machines
    assert_allclose(fvar_err, 0.07956, rtol=1e-3)


def test_lightcurve_chisq(lc_table):
    flux = lc_table["flux"].astype("float64")
    chi2, pval = compute_chisq(flux)
    assert_quantity_allclose(chi2, 1e-11)
    assert_quantity_allclose(pval, 0.999997476867478)
