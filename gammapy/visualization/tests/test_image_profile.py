# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.utils.testing import requires_data, requires_dependency, mpl_plot_check
from astropy.coordinates import SkyCoord
from gammapy.datasets import Datasets
from gammapy.data import GTI
from gammapy.estimators import MapProfileEstimator, make_orthogonal_boxes


@requires_data()
def make_improf():
    datasets = Datasets.read("$GAMMAPY_DATA/fermi-3fhl-crab/",
                             "Fermi-LAT-3FHL_datasets.yaml", "Fermi-LAT-3FHL_models.yaml")
    datasets[0].gti = GTI.create("0s", "1e7s", "2010-01-01")

    start_line = SkyCoord(182.5, -5.8, unit='deg', frame='galactic')
    end_line = SkyCoord(186.5, -5.8, unit='deg', frame='galactic')
    boxes, axis = make_orthogonal_boxes(start_line,
                                        end_line,
                                        datasets[0].counts.geom.wcs,
                                        1. * u.deg,
                                        11)

    prof_maker = MapProfileEstimator(boxes, axis)
    fermi_prof = prof_maker.run(datasets[0])

    return fermi_prof


@requires_dependency("matplotlib")
def test_peek_plot():

    fermi_prof = make_improf()

    with mpl_plot_check():
        fermi_prof.peek()


@requires_dependency("matplotlib")
def test_flux_plot():
    import matplotlib.pyplot as plt
    fermi_prof = make_improf()

    with mpl_plot_check():
        ax = plt.gca()
        ax.set_yscale('log')
        ax = fermi_prof.plot("flux", ax=ax)


def test_content():
    fermi_prof = make_improf()

    assert_allclose(fermi_prof.profile("radiance")[10].value, 1.053396e-07, atol=1e-5)
    assert_allclose(fermi_prof.profile_ul("flux")[0].value, 4.5263849e-11, atol=1e-5)
    assert_allclose(fermi_prof.profile_err_p("excess")[4].value, 10.582960, atol=1e-5)
    assert_allclose(fermi_prof.profile_err_p("excess")[4].value, 10.582960, atol=1e-5)


def test_normalise():
    fermi_prof = make_improf()
    fermi_norm = fermi_prof.normalize(mode="peak", method="excess")

    assert_allclose(fermi_norm.profile("excess")[4].value, 0.047630, atol=1e-5)
