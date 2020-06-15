# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import astropy.units as u
import numpy as np
from gammapy.utils.testing import requires_dependency, mpl_plot_check
from gammapy.utils.table import table_from_row_data
from gammapy.estimators import ImageProfile


def make_improf():
    results = []
    bkg = 2
    counts = 5
    xmin = -2*u.deg
    xmax = -1*u.deg
    for ii in range(6):
        on = counts+ii
        sigma = np.sqrt(on+bkg)
        result =  {
              "xmax": xmin+ii*u.deg,
              "x_ref": xmin+ii*u.deg,
              "counts": on,
              "excess": on-bkg,
              "sqrt_ts": sigma,
              "err": (on-bkg)/sigma,
              "ul": 10,
              "exposure": 100 * u.m ** 2 * u.s,
              "solid_angle": 0.1 * u.sr,
              }
        results.append(result)

    ftable = table_from_row_data(results)
    return ImageProfile(ftable)


@requires_dependency("matplotlib")
def test_peek_plot():
    import matplotlib.pyplot as plt
    a_prof = make_improf()

    with mpl_plot_check():
        a_prof.peek()


@requires_dependency("matplotlib")
def test_flux_plot():
    import matplotlib.pyplot as plt
    a_prof = make_improf()

    with mpl_plot_check():
        ax = plt.gca()
        ax.set_yscale('log')
        ax = a_prof.plot("flux", ax=ax)


def test_content():
    a_prof = make_improf()

    assert_allclose(a_prof.profile("brightness")[4].value, 0.7, atol=1e-2)
    assert_allclose(a_prof.profile_ul("flux")[0].value, 0.1, atol=1e-2)
    assert_allclose(a_prof.profile_err_p("excess")[2].value, 1.666, atol=1e-3)
    assert_allclose(a_prof.profile_err("excess")[1].value, 1.414, atol=1e-3)


def test_normalise():
    a_prof = make_improf()
    a_norm = a_prof.normalize(mode="peak", method="excess")

    assert_allclose(a_norm.profile("excess")[4].value, 0.875, atol=1e-4)
