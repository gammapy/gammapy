# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Column, Table
from astropy.time import Time
from gammapy.datasets import MapDataset
from gammapy.estimators import ExcessMapEstimator, FluxPoints
from gammapy.estimators.utils import (
    compute_lightcurve_doublingtime,
    compute_lightcurve_fpp,
    compute_lightcurve_fvar,
    compute_lightcurve_discrete_correlation,
    find_peaks,
    find_peaks_in_flux_map,
    get_rebinned_axis,
    resample_energy_edges,
)
from gammapy.maps import Map, MapAxis
from gammapy.utils.testing import assert_time_allclose, requires_data


@pytest.fixture()
def lc():
    meta = dict(TIMESYS="utc", SED_TYPE="flux")

    table = Table(
        meta=meta,
        data=[
            Column(Time(["2010-01-01", "2010-01-03"]).mjd, "time_min"),
            Column(Time(["2010-01-03", "2010-01-10"]).mjd, "time_max"),
            Column([[1.0, 2.0], [1.0, 2.0]], "e_min", unit="TeV"),
            Column([[2.0, 5.0], [2.0, 5.0]], "e_max", unit="TeV"),
            Column([[1e-11, 4e-12], [3e-11, 7e-12]], "flux", unit="cm-2 s-1"),
            Column(
                [[0.1e-11, 0.4e-12], [0.3e-11, 0.7e-12]], "flux_err", unit="cm-2 s-1"
            ),
            Column([[np.nan, np.nan], [3.6e-11, 1e-11]], "flux_ul", unit="cm-2 s-1"),
            Column([[False, False], [True, True]], "is_ul"),
            Column([[True, True], [True, True]], "success"),
        ],
    )

    return FluxPoints.from_table(table=table, format="lightcurve")


@pytest.fixture(scope="session")
def lc2():
    meta = dict(TIMESYS="utc", SED_TYPE="flux")

    table = Table(
        meta=meta,
        data=[
            Column(Time(["2010-01-01", "2010-01-03", "2010-01-05"]).mjd, "time_min"),
            Column(Time(["2010-01-03", "2010-01-05", "2010-01-07"]).mjd, "time_max"),
            Column([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], "e_min", unit="GeV"),
            Column([[2.0, 5.0], [2.0, 5.0], [2.0, 5.0]], "e_max", unit="GeV"),
            Column(
                [[1.51e-7, 3.4e-8], [3.1e-7, 6.7e-8], [3.1e-7, 7.5e-8]],
                "flux",
                unit="m-2 s-1",
            ),
            Column(
                [[0.1e-7, 0.4e-8], [0.3e-7, 0.7e-8], [0.31e-7, 0.72e-8]],
                "flux_err",
                unit="m-2 s-1",
            ),
            Column(
                [[np.nan, np.nan], [3.6e-7, 1e-7], [3.6e-7, 1e-7]],
                "flux_ul",
                unit="m-2 s-1",
            ),
            Column([[False, False], [True, True], [True, True]], "is_ul"),
            Column([[True, True], [True, True], [True, True]], "success"),
        ],
    )

    return FluxPoints.from_table(table=table, format="lightcurve")


class TestFindPeaks:
    def test_simple(self):
        """Test a simple example"""
        image = Map.create(npix=(10, 5), unit="s")
        image.data[3, 3] = 11
        image.data[3, 4] = 10
        image.data[3, 5] = 12
        image.data[3, 6] = np.nan
        image.data[0, 9] = 1e20

        table = find_peaks(image, threshold=3)

        assert len(table) == 3
        assert table["value"].unit == "s"
        assert table["ra"].unit == "deg"
        assert table["dec"].unit == "deg"

        row = table[0]
        assert tuple((row["x"], row["y"])) == (9, 0)
        assert_allclose(row["value"], 1e20)
        assert_allclose(row["ra"], 359.55)
        assert_allclose(row["dec"], -0.2)

        row = table[1]
        assert tuple((row["x"], row["y"])) == (5, 3)
        assert_allclose(row["value"], 12)

    def test_no_peak(self):
        image = Map.create(npix=(10, 5))
        image.data[3, 5] = 12

        table = find_peaks(image, threshold=12.1)
        assert len(table) == 0

    def test_constant(self):
        image = Map.create(npix=(10, 5))

        table = find_peaks(image, threshold=3)
        assert len(table) == 0

    def test_flat_map(self):
        """Test a simple example"""
        axis1 = MapAxis.from_edges([1, 2], name="axis1")
        axis2 = MapAxis.from_edges([9, 10], name="axis2")
        image = Map.create(npix=(10, 5), unit="s", axes=[axis1, axis2])
        image.data[..., 3, 3] = 11
        image.data[..., 3, 4] = 10
        image.data[..., 3, 5] = 12
        image.data[..., 3, 6] = np.nan
        image.data[..., 0, 9] = 1e20

        table = find_peaks(image, threshold=3)
        row = table[0]

        assert len(table) == 3
        assert_allclose(row["value"], 1e20)
        assert_allclose(row["ra"], 359.55)
        assert_allclose(row["dec"], -0.2)


class TestFindFluxPeaks:
    """Tests for find_peaks_in_flux_map"""

    @requires_data()
    def test_find_peaks_in_flux_map(self):
        """Test a simple example"""
        dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
        estimator = ExcessMapEstimator(
            correlation_radius="0.1 deg", energy_edges=[0.1, 10] * u.TeV
        )
        maps = estimator.run(dataset)
        table = find_peaks_in_flux_map(maps, threshold=5, min_distance=0.1 * u.deg)

        assert table["ra"].unit == "deg"
        assert table["dec"].unit == "deg"

    @requires_data()
    def test_no_peak(self):
        dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
        position = SkyCoord(1.2, 1, frame="galactic", unit="deg")
        dataset_cutout = dataset.cutout(
            position=position, width=(0.2 * u.deg, 0.2 * u.deg)
        )

        estimator = ExcessMapEstimator(
            correlation_radius="0.1 deg", energy_edges=[0.1, 10] * u.TeV
        )
        maps = estimator.run(dataset_cutout)

        table = find_peaks_in_flux_map(maps, threshold=5, min_distance=0.1 * u.deg)

        assert len(table) == 0


def test_resample_energy_edges(spectrum_dataset):
    resampled_energy_edges = resample_energy_edges(spectrum_dataset, conditions={})
    assert (resampled_energy_edges == spectrum_dataset._geom.axes["energy"].edges).all()

    with pytest.raises(ValueError):
        resample_energy_edges(
            spectrum_dataset,
            conditions={"counts_min": spectrum_dataset.counts.data.sum() + 1},
        )

    resampled_energy_edges = resample_energy_edges(
        spectrum_dataset,
        conditions={"excess_min": spectrum_dataset.excess.data[-1] + 1},
    )
    grouped = spectrum_dataset.resample_energy_axis(
        MapAxis.from_edges(edges=resampled_energy_edges, name="energy")
    )

    assert grouped.counts.data.shape == (29, 1, 1)
    assert_allclose(np.squeeze(grouped.counts)[-1], 2518.0)
    assert_allclose(np.squeeze(grouped.background)[-1], 200)


def test_compute_lightcurve_fvar(lc):
    fvar = compute_lightcurve_fvar(lc)
    ffvar = fvar["fvar"].quantity
    ffvar_err = fvar["fvar_err"].quantity

    assert_allclose(ffvar, [[[0.698212]], [[0.37150576]]])
    assert_allclose(ffvar_err, [[[0.0795621]], [[0.074706]]])


def test_compute_lightcurve_fpp(lc):
    fpp = compute_lightcurve_fpp(lc)
    ffpp = fpp["fpp"].quantity
    ffpp_err = fpp["fpp_err"].quantity

    assert_allclose(ffpp, [[[0.99373035]], [[0.53551551]]])
    assert_allclose(ffpp_err, [[[0.07930673]], [[0.07397653]]])


def test_compute_lightcurve_doublingtime(lc):
    dtime = compute_lightcurve_doublingtime(lc)
    ddtime = dtime["doublingtime"].quantity
    ddtime_err = dtime["doubling_err"].quantity
    dcoord = dtime["doubling_coord"]

    assert_allclose(ddtime, [[[245305.49]], [[481572.59]]] * u.s)
    assert_allclose(ddtime_err, [[[45999.766]], [[11935.665]]] * u.s)
    assert_time_allclose(
        dcoord,
        Time([[[55197.99960648]], [[55197.99960648]]], format="mjd", scale="utc"),
    )


def test_compute_dcf(lc, lc2):
    dict = compute_lightcurve_discrete_correlation(lc, lc2, tau=3 * u.d)

    assert_allclose(dict["bins"], [-388800.0, -129600.0, 129600.0, 388800.0] * u.s)
    assert_allclose(
        dict["discrete_correlation"],
        [
            [-0.760599, -1.052783],
            [-0.760599, -0.537134],
            [1.014132, 1.059945],
            [-1.521198, -1.589918],
        ],
        rtol=1e-6,
    )
    assert_allclose(
        dict["discrete_correlation_err"],
        [[np.nan, np.nan], [np.nan, np.nan], [0.310513, 0.372241], [np.nan, np.nan]],
        rtol=1e-6,
    )

    dict2 = compute_lightcurve_discrete_correlation(lc2, tau=3 * u.d)
    assert_allclose(dict2["bins"], [-388800.0, -129600.0, 129600.0, 388800.0] * u.s)
    assert_allclose(
        dict2["discrete_correlation"],
        [
            [-1.11074, -1.448801],
            [-0.277685, -0.124862],
            [0.55537, 0.629465],
            [-1.11074, -1.448801],
        ],
        rtol=1e-5,
    )
    assert_allclose(
        dict2["discrete_correlation_err"],
        [[np.nan, np.nan], [1.178118, 0.868782], [0.589059, 0.53472], [np.nan, np.nan]],
        rtol=1e-6,
    )

    dict3 = compute_lightcurve_discrete_correlation(lc2)
    assert_allclose(dict3["bins"], [-345600.0, -115200.0, 115200.0, 345600.0] * u.s)


@requires_data()
def test_get_rebinned_axis():
    lc_1d = FluxPoints.read(
        "$GAMMAPY_DATA/estimators/pks2155_hess_lc/pks2155_hess_lc.fits",
        format="lightcurve",
    )
    axis_new = get_rebinned_axis(
        lc_1d, method="fixed-bins", group_size=2, axis_name="time"
    )
    assert_allclose(axis_new.bin_width[0], 20 * u.min)

    axis_new = get_rebinned_axis(
        lc_1d, method="min-ts", ts_threshold=2500.0, axis_name="time"
    )
    assert_allclose(axis_new.bin_width, [50, 30, 30, 50, 110, 70] * u.min)

    with pytest.raises(ValueError):
        get_rebinned_axis(lc_1d, method="error", value=2, axis_name="time")
