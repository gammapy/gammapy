# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from ...utils.testing import requires_dependency, requires_data, mpl_plot_check
from ...irf import PSF3D

pytest.importorskip("scipy")


@pytest.fixture(scope="session")
def psf_3d():
    filename = "$GAMMAPY_EXTRA/test_datasets/psf_table_023523.fits.gz"
    return PSF3D.read(filename)


@requires_data("gammapy-extra")
def test_psf_3d_basics(psf_3d):
    assert_allclose(psf_3d.rad_lo[-1].value, 0.6704476475715637)
    assert psf_3d.rad_lo.shape == (900,)
    assert psf_3d.rad_lo.unit == "deg"

    assert_allclose(psf_3d.energy_lo[0].value, 0.02)
    assert psf_3d.energy_lo.shape == (18,)
    assert psf_3d.energy_lo.unit == "TeV"

    assert psf_3d.psf_value.shape == (900, 6, 18)
    assert psf_3d.psf_value.unit == "sr-1"

    assert_allclose(psf_3d.energy_thresh_lo.value, 0.1)
    assert psf_3d.energy_lo.unit == "TeV"

    assert "PSF3D" in psf_3d.info()


@requires_data("gammapy-extra")
def test_psf_3d_evaluate(psf_3d):
    q = psf_3d.evaluate(energy="1 TeV", offset="0.3 deg", rad="0.1 deg")
    assert_allclose(q.value, 21417.213824)
    # TODO: is this the shape we want here?
    assert q.shape == (1, 1, 1)
    assert q.unit == "sr-1"


@requires_data("gammapy-extra")
def test_to_energy_dependent_table_psf(psf_3d):
    psf = psf_3d.to_energy_dependent_table_psf()
    assert psf.psf_value.shape == (18, 900)
    radius = psf.table_psf_at_energy("1 TeV").containment_radius(0.68).deg
    assert_allclose(radius, 0.171445, atol=1e-4)


@requires_data("gammapy-extra")
def test_psf_3d_containment_radius(psf_3d):
    q = psf_3d.containment_radius(energy="1 TeV")
    assert_allclose(q.value, 0.171445, rtol=1e-3)
    assert q.isscalar
    assert q.unit == "deg"

    q = psf_3d.containment_radius(energy=[1, 3] * u.TeV)
    assert_allclose(q.value, [0.171445, 0.157455], rtol=1e-3)
    assert q.shape == (2,)


@requires_data("gammapy-extra")
def test_psf_3d_write(psf_3d, tmpdir):
    filename = str(tmpdir / "temp.fits")
    psf_3d.write(filename)
    psf_3d = PSF3D.read(filename)

    assert_allclose(psf_3d.energy_lo[0].value, 0.02)


@requires_data("gammapy-extra")
@requires_dependency("matplotlib")
def test_psf_3d_plot_vs_rad(psf_3d):
    with mpl_plot_check():
        psf_3d.plot_psf_vs_rad()


@requires_data("gammapy-extra")
@requires_dependency("matplotlib")
def test_psf_3d_plot_containment(psf_3d):
    with mpl_plot_check():
        psf_3d.plot_containment(show_safe_energy=True)


@requires_data("gammapy-extra")
@requires_dependency("matplotlib")
def test_psf_3d_peek(psf_3d):
    with mpl_plot_check():
        psf_3d.peek()
