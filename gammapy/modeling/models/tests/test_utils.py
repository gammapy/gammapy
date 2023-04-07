from numpy.testing import assert_allclose
import astropy.units as u
from astropy.time import Time
from gammapy.modeling.models import LightCurveTemplateTemporalModel
from gammapy.modeling.models.utils import _template_model_from_cta_sdc
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data


@requires_data()
def test__template_model_from_cta_sdc(tmp_path):
    filename = "$GAMMAPY_DATA/gravitational_waves/GW_example_DC_file.fits.gz"
    mod = _template_model_from_cta_sdc(filename)

    assert isinstance(mod, LightCurveTemplateTemporalModel)
    t = Time(55555.6157407407, format="mjd")
    val = mod.evaluate(t, energy=[0.3, 2] * u.TeV)
    assert_allclose(val.data, [[2.39329809e-21], [4.40027593e-23]], rtol=1e-5)

    filename = make_path(tmp_path / "test.fits")
    mod.write(filename=filename, format="map", overwrite=True)
    mod1 = LightCurveTemplateTemporalModel.read(filename=filename, format="map")

    assert_allclose(mod1.t_ref.value, mod.t_ref.value, rtol=1e-7)
