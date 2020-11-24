# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from gammapy.irf import PSF3D
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


@pytest.fixture(scope="session")
def psf_3d():
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    return PSF3D.read(filename, hdu="PSF")


@requires_data()
def test_psf_3d_basics(psf_3d):
    assert_allclose(psf_3d.rad_axis.edges[-2].value, 0.659048, rtol=1e-5)
    assert psf_3d.rad_axis.nbin == 144
    assert psf_3d.rad_axis.unit == "deg"

    assert_allclose(psf_3d.energy_axis_true.edges[0].value, 0.01)
    assert psf_3d.energy_axis_true.nbin == 32
    assert psf_3d.energy_axis_true.unit == "TeV"

    assert psf_3d.psf_value.shape == (32, 6, 144)
    assert psf_3d.psf_value.unit == "sr-1"

    assert_allclose(psf_3d.energy_thresh_lo.value, 0.01)

    assert "PSF3D" in str(psf_3d)

    with pytest.raises(ValueError):
        PSF3D(
            energy_axis_true=psf_3d.energy_axis_true,
            offset_axis=psf_3d.offset_axis,
            rad_axis=psf_3d.rad_axis,
            psf_value=psf_3d.psf_value.T,
        )


@requires_data()
def test_psf_3d_evaluate(psf_3d):
    q = psf_3d.evaluate(energy="1 TeV", offset="0.3 deg", rad="0.1 deg")
    assert_allclose(q.value, 25889.505886)
    # TODO: is this the shape we want here?
    assert q.shape == (1, 1, 1)
    assert q.unit == "sr-1"


@requires_data()
def test_to_energy_dependent_table_psf(psf_3d):
    psf = psf_3d.to_energy_dependent_table_psf()
    assert psf.psf_value.shape == (32, 144)
    radius = psf.table_psf_at_energy("1 TeV").containment_radius(0.68).deg
    assert_allclose(radius, 0.123352, atol=1e-4)


@requires_data()
def test_psf_3d_containment_radius(psf_3d):
    q = psf_3d.containment_radius(energy="1 TeV")
    assert_allclose(q.value, 0.123352, rtol=1e-2)
    assert q.isscalar
    assert q.unit == "deg"

    q = psf_3d.containment_radius(energy=[1, 3] * u.TeV)
    assert_allclose(q.value, [0.123261, 0.13131], rtol=1e-2)
    assert q.shape == (2,)


@requires_data()
def test_psf_3d_write(psf_3d, tmp_path):
    psf_3d.write(tmp_path / "tmp.fits")
    psf_3d = PSF3D.read(tmp_path / "tmp.fits", hdu=1)

    assert_allclose(psf_3d.energy_axis_true.edges[0].value, 0.01)


@requires_data()
@requires_dependency("matplotlib")
def test_psf_3d_plot_vs_rad(psf_3d):
    with mpl_plot_check():
        psf_3d.plot_psf_vs_rad()


@requires_data()
@requires_dependency("matplotlib")
def test_psf_3d_plot_containment(psf_3d):
    with mpl_plot_check():
        psf_3d.plot_containment()


@requires_data()
@requires_dependency("matplotlib")
def test_psf_3d_peek(psf_3d):
    with mpl_plot_check():
        psf_3d.peek()
