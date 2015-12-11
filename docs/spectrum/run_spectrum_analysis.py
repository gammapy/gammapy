"""
This script illustrates how to use `~gammapy.spectrum.SpectrumAnalysis`
in order to create OGIP to be used for a Sherpa fit
"""

from astropy.coordinates import SkyCoord, Angle
from gammapy.datasets import gammapy_extra
from gammapy.image import ExclusionMask
from gammapy.obs import DataStore
from gammapy.region import SkyCircleRegion
from gammapy.spectrum import SpectrumAnalysis
from gammapy.utils.energy import EnergyBounds

center = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
radius = Angle('0.3 deg')
on_region = SkyCircleRegion(pos=center, radius=radius)

bkg_method = dict(type='reflected')

exclusion_file = gammapy_extra.filename("test_datasets/spectrum/"
                                        "dummy_exclusion.fits")
excl = ExclusionMask.from_fits(exclusion_file)

bounds = EnergyBounds.equal_log_spacing(1, 10, 40, unit='TeV')

store = gammapy_extra.filename("datasets/hess-crab4")
ds = DataStore.from_dir(store)
obs = [23523, 23559]

ana = SpectrumAnalysis(datastore=ds, obs=obs, on_region=on_region,
                       bkg_method=bkg_method, exclusion=excl, ebounds=bounds)

ana.write_ogip_data(outdir='ogip_data')
