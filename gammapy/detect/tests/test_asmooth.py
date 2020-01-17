# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.convolution import Tophat2DKernel
from gammapy.detect import ASmooth
from gammapy.modeling import Datasets
from gammapy.maps import Map
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
    # TODO : remove when issue 2717 solved
    dataset.mask_safe = dataset.counts.copy(
        data=np.ones_like(dataset.counts.data).astype("bool")
    )
    return dataset.to_image()


@requires_data()
def test_asmooth(input_maps):
    kernel = Tophat2DKernel
    scales = ASmooth.make_scales(3, factor=2, kernel=kernel) * 0.1 * u.deg

    asmooth = ASmooth(kernel=kernel, scales=scales, method="simple", threshold=2.5)
    smoothed = asmooth.run(input_maps["counts"], input_maps["background"])

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
    scales = ASmooth.make_scales(3, factor=2, kernel=kernel) * 0.1 * u.deg

    asmooth = ASmooth(kernel=kernel, scales=scales, method="simple", threshold=2.5)
    smoothed = asmooth.run(
        input_dataset.counts,
        input_dataset.background_model.evaluate(),
        input_dataset.exposure,
    )

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
