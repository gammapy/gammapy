# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.time import Time
from gammapy.modeling.models import LightCurveTemplateTemporalModel
from gammapy.modeling.models.utils import _template_model_from_cta_sdc, read_hermes_cube
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data, requires_dependency


@requires_data()
def test__template_model_from_cta_sdc(tmp_path):
    filename = "$GAMMAPY_DATA/gravitational_waves/GW_example_DC_file.fits.gz"
    mod = _template_model_from_cta_sdc(filename)

    assert isinstance(mod, LightCurveTemplateTemporalModel)
    t = Time(55555.6157407407, format="mjd")
    val = mod.evaluate(t, energy=[0.3, 2] * u.TeV)
    assert_allclose(val.data, [[2.281216e-21], [4.281390e-23]], rtol=1e-5)

    filename = make_path(tmp_path / "test.fits")
    mod.write(filename=filename, format="map", overwrite=True)
    mod1 = LightCurveTemplateTemporalModel.read(filename=filename, format="map")

    assert_allclose(mod1.t_ref.value, mod.t_ref.value, rtol=1e-7)


@requires_data()
def test__reference_time():
    filename = "$GAMMAPY_DATA/gravitational_waves/GW_example_DC_file.fits.gz"
    t_ref = Time("2028-01-01T00:00:00", format="isot", scale="utc")
    mod = _template_model_from_cta_sdc(filename, t_ref=t_ref)

    assert_allclose(mod.reference_time.mjd, t_ref.mjd, rtol=1e-7)


@requires_data()
@requires_dependency("healpy")
def test_read_hermes_cube():
    filename = make_path(
        "$GAMMAPY_DATA/tests/hermes/hermes-VariableMin-pi0-Htot_CMZ_nside256.fits.gz"
    )
    map_ = read_hermes_cube(filename)
    assert map_.geom.frame == "galactic"
    assert map_.geom.axes[0].unit == "GeV"
    assert_allclose(map_.geom.axes[0].center[3], 1 * u.GeV)
    assert_allclose(map_.get_by_coord((0 * u.deg, 0 * u.deg, 1 * u.GeV)), 2.6391575)
