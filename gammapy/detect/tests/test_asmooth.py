# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.convolution import Tophat2DKernel
from gammapy.detect import ASmoothMapEstimator
from gammapy.modeling import Datasets
from gammapy.maps import Map, WcsNDMap
from gammapy.cube import MapDatasetOnOff
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def input_maps():
    filename = "$GAMMAPY_DATA/tests/unbundled/poisson_stats_image/input_all.fits.gz"
    return {
        "counts": Map.read(filename, hdu="counts"),
        "background": Map.read(filename, hdu="background"),
    }


@pytest.fixture(scope="session")
def input_dataset():
    datasets = Datasets.read(
        filedata="$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml",
        filemodel="$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml",
    )
    dataset = datasets[0]
    dataset.psf = None
    return dataset


@requires_data()
def test_asmooth(input_maps):
    kernel = Tophat2DKernel
    scales = ASmoothMapEstimator.get_scales(3, factor=2, kernel=kernel) * 0.1 * u.deg

    asmooth = ASmoothMapEstimator(scales=scales, kernel=kernel,  method="simple", threshold=2.5)
    smoothed = asmooth.estimate_maps(input_maps["counts"], input_maps["background"])

    desired = {
        "counts": 6.454327,
        "background": 1.0,
        "scale": 0.056419,
        "significance": 18.125747,
    }

    for name in smoothed:
        actual = smoothed[name].data[100, 100]
        assert_allclose(actual, desired[name], rtol=1e-5)


@requires_data()
def test_asmooth_dataset(input_dataset):
    kernel = Tophat2DKernel
    scales = ASmoothMapEstimator.get_scales(3, factor=2, kernel=kernel) * 0.1 * u.deg

    asmooth = ASmoothMapEstimator(scales=scales, kernel=kernel,  method="simple", threshold=2.5)

    # First check that is fails if don't use to_image()
    with pytest.raises(ValueError):
        asmooth.run(input_dataset)
        
    smoothed = asmooth.run(input_dataset.to_image())

    assert smoothed["flux"].data.shape == (40, 50)
    assert smoothed["flux"].unit == u.Unit("cm-2s-1")
    assert smoothed["counts"].unit == u.Unit("")
    assert smoothed["background"].unit == u.Unit("")
    assert smoothed["scale"].unit == u.Unit("deg")

    desired = {
        "counts": 369.479167,
        "background": 0.13461,
        "scale": 0.056419,
        "significance": 74.677406,
        "flux": 1.237284e-09,
    }

    for name in smoothed:
        actual = smoothed[name].data[20, 25]
        assert_allclose(actual, desired[name], rtol=1e-5)

def test_asmooth_mapdatasetonoff():
    kernel = Tophat2DKernel
    scales = ASmoothMapEstimator.get_scales(3, factor=2, kernel=kernel) * 0.1 * u.deg

    asmooth = ASmoothMapEstimator(kernel=kernel, scales=scales, method="simple", threshold=2.5)

    counts = WcsNDMap.create(npix=(50,50), binsz=0.02, unit="")
    counts += 2
    counts_off = WcsNDMap.create(npix=(50,50), binsz=0.02, unit="")
    counts_off += 3
    acceptance = 1
    acceptance_off = 3

    dataset = MapDatasetOnOff(
        counts=counts,
        counts_off=counts_off,
        acceptance=acceptance,
        acceptance_off=acceptance_off
    )

    smoothed = asmooth.run(dataset)
    assert_allclose(smoothed["counts"].data[25,25], 2)
    assert_allclose(smoothed["background"].data[25,25], 1)
    assert_allclose(smoothed["significance"].data[25,25], 4.391334)

