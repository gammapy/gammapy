# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Column, Table
from astropy.time import Time
from gammapy.estimators import FluxPoints
from gammapy.stats.variability import compute_chisq, compute_fpp, compute_fvar
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


def lc():
    meta = dict(TIMESYS="utc", SED_TYPE="flux")

    table = Table(
        meta=meta,
        data=[
            Column(Time(["2010-01-01", "2010-01-03", "2010-01-07"]).mjd, "time_min"),
            Column(Time(["2010-01-03", "2010-01-07", "2010-01-10"]).mjd, "time_max"),
            Column([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], "e_min", unit="TeV"),
            Column([[2.0, 5.0], [2.0, 5.0], [2.0, 5.0]], "e_max", unit="TeV"),
            Column(
                [[1e-11, 4e-12], [3e-11, np.nan], [1e-11, 1e-12]],
                "flux",
                unit="cm-2 s-1",
            ),
            Column(
                [[0.1e-11, 0.4e-12], [0.3e-11, np.nan], [0.1e-11, 0.1e-12]],
                "flux_err",
                unit="cm-2 s-1",
            ),
            Column(
                [[np.nan, np.nan], [3.6e-11, 1e-11], [1e-11, 1e-12]],
                "flux_ul",
                unit="cm-2 s-1",
            ),
            Column([[False, False], [True, True], [True, True]], "is_ul"),
            Column([[True, True], [True, True], [True, True]], "success"),
        ],
    )

    return FluxPoints.from_table(table=table, format="lightcurve")


def test_lightcurve_fvar():

    flux = np.array([[1e-11, 4e-12], [3e-11, np.nan], [1e-11, 1e-12]])
    flux_err = np.array([[0.1e-11, 0.4e-12], [0.3e-11, np.nan], [0.1e-11, 0.1e-12]])

    time_id = 0

    fvar, fvar_err = compute_fvar(flux, flux_err, axis=time_id)

    assert_allclose(fvar, [0.68322763, 0.84047606])
    assert_allclose(fvar_err, [0.06679978, 0.08285806])


def test_lightcurve_fpp():

    flux = np.array([[1e-11, 4e-12], [3e-11, np.nan], [1e-11, 1e-12]])
    flux_err = np.array([[0.1e-11, 0.4e-12], [0.3e-11, np.nan], [0.1e-11, 0.1e-12]])

    time_id = 0

    fpp, fpp_err = compute_fpp(flux, flux_err, axis=time_id)

    assert_allclose(fpp, [1.19448734, 0.11661904])
    assert_allclose(fpp_err, [0.06648574, 0.10099505])


def test_lightcurve_chisq(lc_table):
    flux = lc_table["flux"].astype("float64")
    chi2, pval = compute_chisq(flux)
    assert_quantity_allclose(chi2, 1e-11)
    assert_quantity_allclose(pval, 0.999997476867478)
