# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import matplotlib
from packaging import version
from gammapy.visualization import plot_spectrum_datasets_off_regions


@pytest.mark.skipif(
    version.parse(matplotlib.__version__) < version.parse("3.5"),
    reason="Requires matplotlib 3.5 or higher",
)
def test_plot_spectrum_datasets_off_regions():
    from gammapy.datasets import SpectrumDatasetOnOff
    from gammapy.maps import Map, RegionNDMap

    counts_off_1 = RegionNDMap.create("icrs;circle(0, 0.5, 0.2);circle(0.5, 0, 0.2)")

    counts_off_2 = RegionNDMap.create("icrs;circle(0.5, 0.5, 0.2);circle(0, 0, 0.2)")

    counts_off_3 = RegionNDMap.create("icrs;point(0.5, 0.5);point(0, 0)")

    m = Map.from_geom(geom=counts_off_1.geom.to_wcs_geom())
    ax = m.plot()

    dataset_1 = SpectrumDatasetOnOff(counts_off=counts_off_1)

    dataset_2 = SpectrumDatasetOnOff(counts_off=counts_off_2)

    dataset_3 = SpectrumDatasetOnOff(counts_off=counts_off_3)

    plot_spectrum_datasets_off_regions(
        ax=ax, datasets=[dataset_1, dataset_2, dataset_3]
    )

    actual = ax.patches[0].get_edgecolor()
    assert_allclose(actual, (0.121569, 0.466667, 0.705882, 1.0), rtol=1e-2)

    actual = ax.patches[2].get_edgecolor()
    assert_allclose(actual, (1.0, 0.498039, 0.054902, 1.0), rtol=1e-2)
    assert ax.lines[0].get_color() in ["green", "C0"]
