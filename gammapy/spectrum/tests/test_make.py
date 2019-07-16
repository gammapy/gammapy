# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_data
from ...spectrum import SpectrumDatasetMakerObs
from ...data import DataStore

@requires_data()
def test_spectrumdatasetmaker_cta_1dc_data():
    datastore = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    obs_ids = [110380, 111140]
    observations = datastore.get_observations(obs_ids)

    pos = SkyCoord(0.0, 0.0, unit="deg", frame="galactic")
    radius = Angle(0.11, "deg")
    on_region = CircleSkyRegion(pos, radius)

    # This will test non PSF3D input as well as absence of default thresholds
    extract = SpectrumDatasetMakerObs(
        observation=observations[0], on_region=on_region, containment_correction=True
    )
    extract.run()

    extract.compute_energy_threshold(method_lo="area_max", area_percent_lo=10)
    actual = extract.dataset.energy_range[0]
    assert_quantity_allclose(actual, 0.774263 * u.TeV, rtol=1e-3)
