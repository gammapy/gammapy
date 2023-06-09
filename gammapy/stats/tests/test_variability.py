# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Column, Table
from astropy.time import Time
from gammapy.estimators import FluxPoints
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


def lc():
    meta = dict(TIMESYS="utc", SED_TYPE="flux")

    table = Table(
        meta=meta,
        data=[
            Column(Time(["2010-01-01", "2010-01-03"]).mjd, "time_min"),
            Column(Time(["2010-01-03", "2010-01-10"]).mjd, "time_max"),
            Column([[1.0, 2.0], [1.0, 2.0]], "e_min", unit="TeV"),
            Column([[2.0, 5.0], [2.0, 5.0]], "e_max", unit="TeV"),
            Column([[1e-11, 4e-12], [3e-11, np.nan]], "flux", unit="cm-2 s-1"),
            Column(
                [[0.1e-11, 0.4e-12], [0.3e-11, np.nan]], "flux_err", unit="cm-2 s-1"
            ),
            Column([[np.nan, np.nan], [3.6e-11, 1e-11]], "flux_ul", unit="cm-2 s-1"),
            Column([[False, False], [True, True]], "is_ul"),
            Column([[True, True], [True, True]], "success"),
        ],
    )

    return FluxPoints.from_table(table=table, format="lightcurve")


def test_lightcurve_fvar(lc_table):
    lightcurve = lc()

    quantity = "flux"

    flux = getattr(lightcurve, quantity)
    flux_err = getattr(lightcurve, quantity + "_err")

    time_id = flux.geom.axes.index_data("time")

    fvar, fvar_err = compute_fvar(flux.data, flux_err.data, axis=time_id)

    assert_allclose(fvar, np.asarray([[[0.68322763]], [[0.45946829]]]))
    assert_allclose(fvar_err, np.asarray([[[0.06679978]], [[0.07550996]]]))


def test_lightcurve_chisq(lc_table):
    flux = lc_table["flux"].astype("float64")
    chi2, pval = compute_chisq(flux)
    assert_quantity_allclose(chi2, 1e-11)
    assert_quantity_allclose(pval, 0.999997476867478)
