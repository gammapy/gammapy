# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.coordinates import SkyCoord, Angle

from ...data import DataStore
from ...datasets import gammapy_extra
from ...image import ExclusionMask
from ...region import SkyCircleRegion
from ...spectrum import SpectrumExtraction
from ...utils.energy import EnergyBounds
from ...utils.testing import requires_dependency, requires_data


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_spectrum_extraction(tmpdir):
    # Construct w/o config file
    center = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
    radius = Angle('0.3 deg')
    on_region = SkyCircleRegion(pos=center, radius=radius)

    bkg_method = dict(type='reflected', n_min=2)

    exclusion_file = gammapy_extra.filename(
        "datasets/exclusion_masks/tevcat_exclusion.fits")
    excl = ExclusionMask.from_fits(exclusion_file)

    bounds = EnergyBounds.equal_log_spacing(1, 10, 40, unit='TeV')

    obs = [23523, 23559, 11111]
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    ds = DataStore.from_dir(store)

    ana = SpectrumExtraction(datastore=ds, obs_ids=obs, on_region=on_region,
                           bkg_method=bkg_method, exclusion=excl,
                           ebounds=bounds)

    #test methods on SpectrumObservationList
    obs = ana.observations
    assert len(obs) == 2
    obs23523 = obs.get_obs_by_id(23523)
    assert obs23523.on_vector.total_counts == 123

