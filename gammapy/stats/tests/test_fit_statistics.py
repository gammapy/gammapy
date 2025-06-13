# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy import stats
from gammapy.stats.fit_statistics import (
    CashFitStatistic,
    WeightedCashFitStatistic,
    WStatFitStatistic,
    Chi2FitStatistic,
    Chi2AsymmetricErrorFitStatistic,
    GaussianPriorPenalty,
)


@pytest.fixture
def test_data():
    """Test data for fit statistics tests"""
    test_data = dict(
        mu_sig=[
            0.59752422,
            9.13666449,
            12.98288095,
            5.56974565,
            13.52509804,
            11.81725635,
            0.47963765,
            11.17708176,
            5.18504894,
            8.30202394,
        ],
        n_on=[0, 13, 7, 5, 11, 16, 0, 9, 3, 12],
        n_off=[0, 7, 4, 0, 18, 7, 1, 5, 12, 25],
        alpha=[
            0.83746243,
            0.17003354,
            0.26034507,
            0.69197751,
            0.89557033,
            0.34068848,
            0.0646732,
            0.86411967,
            0.29087245,
            0.74108241,
        ],
    )

    test_data["staterror"] = np.sqrt(test_data["n_on"])

    return test_data


@pytest.fixture
def reference_values():
    """Reference values for fit statistics test.

    Produced using sherpa stats module in dev/sherpa/stats/compare_wstat.py
    """
    return dict(
        wstat=[
            1.19504844,
            0.625311794002,
            4.25810886127,
            0.0603765381044,
            11.7285002468,
            0.206014834301,
            1.084611,
            2.72972381792,
            4.60602990838,
            7.51658734973,
        ],
        cash=[
            1.19504844,
            -39.24635098872072,
            -9.925081055136996,
            -6.034002586236575,
            -30.249839537105466,
            -55.39143500383233,
            0.9592753,
            -21.095413867175516,
            0.49542219758430406,
            -34.19193611846045,
        ],
        cstat=[
            1.19504844,
            1.4423323052792387,
            3.3176610316373925,
            0.06037653810442922,
            0.5038564644586838,
            1.3314041078406706,
            0.9592753,
            0.4546285248764317,
            1.0870959295929628,
            1.4458234764515652,
        ],
    )


def test_wstat(test_data, reference_values):
    statsvec = stats.wstat(
        n_on=test_data["n_on"],
        mu_sig=test_data["mu_sig"],
        n_off=test_data["n_off"],
        alpha=test_data["alpha"],
        extra_terms=True,
    )

    assert_allclose(statsvec, reference_values["wstat"])


def test_cash(test_data, reference_values):
    statsvec = stats.cash(n_on=test_data["n_on"], mu_on=test_data["mu_sig"])
    assert_allclose(statsvec, reference_values["cash"])


def test_cstat(test_data, reference_values):
    statsvec = stats.cstat(n_on=test_data["n_on"], mu_on=test_data["mu_sig"])
    assert_allclose(statsvec, reference_values["cstat"])


def test_cash_sum_cython(test_data):
    counts = np.array(test_data["n_on"], dtype=float)
    npred = np.array(test_data["mu_sig"], dtype=float)
    stat = stats.cash_sum_cython(counts=counts, npred=npred)
    ref = stats.cash(counts, npred).sum()
    assert_allclose(stat, ref)


def test_cash_bad_truncation():
    with pytest.raises(ValueError):
        stats.cash(10, 10, 0.0)


def test_cstat_bad_truncation():
    with pytest.raises(ValueError):
        stats.cstat(10, 10, 0.0)


def test_wstat_corner_cases():
    """test WSTAT formulae for corner cases"""
    n_on = 0
    n_off = 5
    mu_sig = 2.3
    alpha = 0.5

    actual = stats.wstat(n_on=n_on, mu_sig=mu_sig, n_off=n_off, alpha=alpha)
    desired = 2 * (mu_sig + n_off * np.log(1 + alpha))
    assert_allclose(actual, desired)

    actual = stats.get_wstat_mu_bkg(n_on=n_on, mu_sig=mu_sig, n_off=n_off, alpha=alpha)
    desired = n_off / (alpha + 1)
    assert_allclose(actual, desired)

    # n_off = 0 and mu_sig < n_on * (alpha / alpha + 1)
    n_on = 9
    n_off = 0
    mu_sig = 2.3
    alpha = 0.5

    actual = stats.wstat(n_on=n_on, mu_sig=mu_sig, n_off=n_off, alpha=alpha)
    desired = -2 * (mu_sig * (1.0 / alpha) + n_on * np.log(alpha / (1 + alpha)))
    assert_allclose(actual, desired)

    actual = stats.get_wstat_mu_bkg(n_on=n_on, mu_sig=mu_sig, n_off=n_off, alpha=alpha)
    desired = n_on / (1 + alpha) - (mu_sig / alpha)
    assert_allclose(actual, desired)

    # n_off = 0 and mu_sig > n_on * (alpha / alpha + 1)
    n_on = 5
    n_off = 0
    mu_sig = 5.3
    alpha = 0.5

    actual = stats.wstat(n_on=n_on, mu_sig=mu_sig, n_off=n_off, alpha=alpha)
    desired = 2 * (mu_sig + n_on * (np.log(n_on) - np.log(mu_sig) - 1))
    assert_allclose(actual, desired)

    actual = stats.get_wstat_mu_bkg(n_on=n_on, mu_sig=mu_sig, n_off=n_off, alpha=alpha)
    assert_allclose(actual, 0)


class MockDataset:
    @staticmethod
    def create_region(size, axis_name="energy"):
        from gammapy.maps import RegionGeom, MapAxis

        axis = MapAxis.from_nodes(np.arange(size), name=axis_name, unit="TeV")
        return RegionGeom.create(region=None, axes=[axis])


class MockMapDataset(MockDataset):
    """Mock dataset class"""

    def __init__(self, counts, npred, mask=None):
        from gammapy.maps import RegionNDMap

        geom = self.create_region(len(counts))
        self.counts = RegionNDMap(geom=geom, data=np.array(counts))
        self.npred = lambda: RegionNDMap(
            geom=geom, data=np.array(npred).astype("float")
        )
        self.mask = (
            RegionNDMap(geom=geom, data=np.array(mask)) if mask is not None else None
        )


class MockMapDatasetOnOff(MockDataset):
    """Mock dataset for ON-OFF Poisson measurements (WStat)."""

    def __init__(self, counts, npred_signal, counts_off, alpha, mask=None):
        from gammapy.maps import RegionNDMap

        geom = self.create_region(len(counts))
        self.counts = RegionNDMap(geom=geom, data=np.array(counts))
        self.npred_signal = lambda: RegionNDMap(
            geom=geom, data=np.array(npred_signal).astype("float")
        )
        self.mask = (
            RegionNDMap(geom=geom, data=np.array(mask)) if mask is not None else None
        )
        self.counts_off = RegionNDMap(geom=geom, data=np.array(counts_off))
        self.alpha = RegionNDMap(geom=geom, data=np.array(alpha).astype("float"))


class MockFluxPointsDataset(MockDataset):
    """Mock dataset for flux measurements (chi2, etc)."""

    def __init__(
        self,
        norm,
        norm_err,
        norm_pred,
        norm_errn=None,
        norm_errp=None,
        norm_ul=None,
        is_ul=None,
        model=None,
        mask=None,
    ):
        from gammapy.maps import RegionNDMap
        from gammapy.estimators import FluxPoints
        from gammapy.modeling.models import ConstantSpectralModel

        geom = self.create_region(len(norm))

        data = {}
        data["norm"] = RegionNDMap(geom=geom, data=np.array(norm))
        data["norm_err"] = RegionNDMap(geom=geom, data=np.array(norm_err))
        data["norm_errn"] = (
            RegionNDMap(geom=geom, data=np.array(norm_errn)) if norm_errn else None
        )
        data["norm_errp"] = (
            RegionNDMap(geom=geom, data=np.array(norm_errp)) if norm_errp else None
        )
        data["norm_ul"] = (
            RegionNDMap(geom=geom, data=np.array(norm_ul)) if norm_ul else None
        )
        data["is_ul"] = RegionNDMap(geom=geom, data=np.array(is_ul)) if is_ul else None

        model = model if model else ConstantSpectralModel()

        self.data = FluxPoints(data, model)
        new_geom = geom.copy(axes=geom.axes.rename_axes(["energy"], ["energy_true"]))
        ref_flux = self.data.reference_model.evaluate_geom(new_geom)
        self.flux_pred = lambda: np.array(norm_pred).reshape(ref_flux.shape) * ref_flux
        self.mask = np.array(mask) if mask is not None else None


@pytest.fixture
def mock_map_dataset():
    return MockMapDataset(counts=[1, 2, 3], npred=[1, 2, 3], mask=[True, False, True])


def test_cash_fit_statistic_stat_sum_nomask(mock_map_dataset):
    """Test the stat_sum_dataset method for CashFitStatistic."""
    mock_map_dataset.mask = None
    stat_sum = CashFitStatistic.stat_sum_dataset(mock_map_dataset)
    assert_allclose(stat_sum, 2.63573737)


def test_cash_fit_statistic_with_mask(mock_map_dataset):
    """Test CashFitStatistic with a mask."""
    stat_sum = CashFitStatistic.stat_sum_dataset(mock_map_dataset)
    assert_allclose(stat_sum, 1.40832626799)


def test_cash_fit_statistic_loglikelihood(mock_map_dataset):
    """Test loglikelihood_dataset method."""
    log_likelihood = CashFitStatistic.loglikelihood_dataset(mock_map_dataset)
    assert_allclose(log_likelihood, 1.40832626799 * -0.5)


def test_cash_fit_statistic_with_non_bool_mask(mock_map_dataset):
    """Ensure stat_sum_dataset handles non-bool masks gracefully."""
    mock_map_dataset.mask.data = np.array([1, 0, 2], dtype="float")

    stat_sum = CashFitStatistic.stat_sum_dataset(mock_map_dataset)
    assert_allclose(stat_sum, 1.40832626799)


def test_weightedcash_fit_statistic(mock_map_dataset):
    """Test WeightedCashFitStatistic."""
    mock_map_dataset.mask.data = np.array([0.5, 0.5, 0.5], dtype="float")

    stat_sum = WeightedCashFitStatistic.stat_sum_dataset(mock_map_dataset)
    assert_allclose(stat_sum, 2.63573737 * 0.5)


@pytest.fixture()
def mock_map_dataset_onoff():
    return MockMapDatasetOnOff(
        counts=[3, 6, 9],
        npred_signal=[1, 2, 3],
        counts_off=[10, 10, 30],
        alpha=[0.1, 0.2, 0.1],
        mask=[True, False, True],
    )


def test_wstat_fit_statistic_stat_sum_nomask(mock_map_dataset_onoff):
    """Test the stat_sum_dataset method for WStatFitStatistic."""
    mock_map_dataset_onoff.mask = None
    stat_sum = WStatFitStatistic.stat_sum_dataset(mock_map_dataset_onoff)
    assert_allclose(stat_sum, 2.4088762929)


def test_wstat_fit_statistic_with_mask(mock_map_dataset_onoff):
    """Test WStatFitStatistic with a mask."""
    stat_sum = WStatFitStatistic.stat_sum_dataset(mock_map_dataset_onoff)
    assert_allclose(stat_sum, 1.63526008301)


def test_wstat_fit_statistic_with_mask_false(mock_map_dataset_onoff):
    """Test WStatFitStatistic with mask_safe handling."""
    mask = [
        False,
    ] * 3
    mock_map_dataset_onoff.mask.data = np.array(mask)
    stat_sum = WStatFitStatistic.stat_sum_dataset(mock_map_dataset_onoff)
    assert stat_sum == 0


def test_wstat_fit_statistic_loglikelihood(mock_map_dataset_onoff):
    """Test loglikelihood_dataset method."""
    log_likelihood = WStatFitStatistic.loglikelihood_dataset(mock_map_dataset_onoff)
    assert_allclose(log_likelihood, -0.817630041)


@pytest.fixture()
def mock_fp_dataset():
    return MockFluxPointsDataset(
        norm=[1.1, 0.9, 1.2, 0.8],
        norm_err=[0.1, 0.1, 0.2, 0.2],
        norm_pred=[1, 1, 1, 1],
        norm_errn=[0.1, 0.1, 0.2, 0.2],
        norm_errp=[0.1, 0.1, 0.1, 0.1],
        norm_ul=[1.5, 1.5, 1.5, 1.5],
        is_ul=[False, False, False, True],
        mask=[True, True, False, True],
    )


def test_chi2_fit_statistic_stat_sum_nomask(mock_fp_dataset):
    mock_fp_dataset.mask = None
    stat_sum = Chi2FitStatistic.stat_sum_dataset(mock_fp_dataset)
    assert_allclose(stat_sum, 4)


def test_chi2_fit_statistic_with_mask(mock_fp_dataset):
    stat_sum = Chi2FitStatistic.stat_sum_dataset(mock_fp_dataset)
    assert_allclose(stat_sum, 3)


def test_chi2_fit_statistic_with_mask_false(mock_fp_dataset):
    mask = [False, False, False, False]
    mock_fp_dataset.mask = np.array(mask)
    stat_sum = Chi2FitStatistic.stat_sum_dataset(mock_fp_dataset)
    assert stat_sum == 0


def test_chi2_asym_fit_statistic_stat_sum_nomask(mock_fp_dataset):
    mock_fp_dataset.mask = None
    stat_sum = Chi2AsymmetricErrorFitStatistic.stat_sum_dataset(mock_fp_dataset)
    assert_allclose(stat_sum, 5.798344)


def test_chi2_asym_fit_statistic_with_mask(mock_fp_dataset):
    stat_sum = Chi2AsymmetricErrorFitStatistic.stat_sum_dataset(mock_fp_dataset)
    assert_allclose(stat_sum, 4.798344)


def test_gaussian_prior_penalty():
    from gammapy.modeling.models import PiecewiseNormSpectralModel

    norm_model = PiecewiseNormSpectralModel(energy=np.geomspace(0.1, 10, 5) * u.TeV)

    penalty = GaussianPriorPenalty.L2_penalty(
        norm_model.parameters, mean=0.0, lambda_=2
    )
    stat_sum = penalty.stat_sum()
    assert_allclose(stat_sum, 10)

    norm_model.parameters.value = [0.0, 1.0, 0.0, 1.0, 0]
    penalty = GaussianPriorPenalty.SmoothnessPenalty(norm_model.parameters, lambda_=0.5)
    stat_sum = penalty.stat_sum()
    assert_allclose(stat_sum, 2)
    assert_allclose(penalty._inverse_covariance[1], [-1, 2, -1, 0, 0], atol=1e-7)
    assert_allclose(penalty._inverse_covariance[3], [0, 0, -1, 2, -1], atol=1e-7)
    assert_allclose(penalty._inverse_covariance[4], [0, 0, 0, -1, 2], atol=1e-7)
