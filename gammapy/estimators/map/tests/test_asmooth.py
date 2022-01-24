# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.convolution import Tophat2DKernel
from gammapy.datasets import Datasets, MapDataset, MapDatasetOnOff
from gammapy.estimators import ASmoothMapEstimator
from gammapy.maps import Map, MapAxis, WcsNDMap
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def input_dataset_simple():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
    filename = "$GAMMAPY_DATA/tests/unbundled/poisson_stats_image/input_all.fits.gz"
    bkg_map = Map.read(filename, hdu="background")
    counts = Map.read(filename, hdu="counts")

    counts = counts.to_cube(axes=[axis])
    bkg_map = bkg_map.to_cube(axes=[axis])

    return MapDataset(counts=counts, background=bkg_map, name="test")


@pytest.fixture(scope="session")
def input_dataset():
    filename = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"
    filename_models = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"

    datasets = Datasets.read(filename=filename, filename_models=filename_models)

    dataset = datasets[0]
    dataset.psf = None
    return dataset


@requires_data()
def test_asmooth(input_dataset_simple):
    kernel = Tophat2DKernel
    scales = ASmoothMapEstimator.get_scales(3, factor=2, kernel=kernel) * 0.1 * u.deg

    asmooth = ASmoothMapEstimator(
        scales=scales, kernel=kernel, method="lima", threshold=2.5
    )
    smoothed = asmooth.estimate_maps(input_dataset_simple)

    desired = {
        "counts": 6.454327,
        "background": 1.0,
        "scale": 0.056419,
        "sqrt_ts": 18.125747,
    }

    for name in smoothed:
        actual = smoothed[name].data[0, 100, 100]
        assert_allclose(actual, desired[name], rtol=1e-5)


@requires_data()
def test_asmooth_dataset(input_dataset):
    kernel = Tophat2DKernel
    scales = ASmoothMapEstimator.get_scales(3, factor=2, kernel=kernel) * 0.1 * u.deg

    asmooth = ASmoothMapEstimator(
        scales=scales, kernel=kernel, method="lima", threshold=2.5
    )

    smoothed = asmooth.run(input_dataset)

    assert smoothed["flux"].data.shape == (1, 40, 50)
    assert smoothed["flux"].unit == u.Unit("cm-2s-1")
    assert smoothed["counts"].unit == u.Unit("")
    assert smoothed["background"].unit == u.Unit("")
    assert smoothed["scale"].unit == u.Unit("deg")

    desired = {
        "counts": 369.479167,
        "background": 0.184005,
        "scale": 0.056419,
        "sqrt_ts": 72.971513,
        "flux": 1.237119e-09,
    }

    for name in smoothed:
        actual = smoothed[name].data[0, 20, 25]
        assert_allclose(actual, desired[name], rtol=1e-2)


def test_asmooth_map_dataset_on_off():
    kernel = Tophat2DKernel
    scales = ASmoothMapEstimator.get_scales(3, factor=2, kernel=kernel) * 0.1 * u.deg

    asmooth = ASmoothMapEstimator(
        kernel=kernel, scales=scales, method="lima", threshold=2.5
    )

    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)

    counts = WcsNDMap.create(npix=(50, 50), binsz=0.02, unit="", axes=[axis])
    counts += 2
    counts_off = WcsNDMap.create(npix=(50, 50), binsz=0.02, unit="", axes=[axis])
    counts_off += 3

    acceptance = WcsNDMap.create(npix=(50, 50), binsz=0.02, unit="", axes=[axis])
    acceptance += 1

    acceptance_off = WcsNDMap.create(npix=(50, 50), binsz=0.02, unit="", axes=[axis])
    acceptance_off += 3

    dataset = MapDatasetOnOff(
        counts=counts,
        counts_off=counts_off,
        acceptance=acceptance,
        acceptance_off=acceptance_off,
    )

    smoothed = asmooth.run(dataset)
    assert_allclose(smoothed["counts"].data[0, 25, 25], 2)
    assert_allclose(smoothed["background"].data[0, 25, 25], 1)
    assert_allclose(smoothed["sqrt_ts"].data[0, 25, 25], 4.39, rtol=1e-2)
