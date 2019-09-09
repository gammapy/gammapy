"""Example how to compute and plot reflected regions."""
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import RectangleSkyRegion
import matplotlib.pyplot as plt
from gammapy.data import DataStore
from gammapy.spectrum import ReflectedRegionsBackgroundEstimator

data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
mask = data_store.obs_table["TARGET_NAME"] == "Crab"
obs_ids = data_store.obs_table["OBS_ID"][mask].data
observations = data_store.get_observations(obs_ids)

crab_position = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")

# The ON region center is defined in the icrs frame. The angle is defined w.r.t. to its axis.
on_region = RectangleSkyRegion(
    center=crab_position, width=0.5 * u.deg, height=0.4 * u.deg, angle=0 * u.deg
)


background_estimator = ReflectedRegionsBackgroundEstimator(
    observations=observations, on_region=on_region, min_distance=0.1 * u.rad
)

background_estimator.run()

# Let's inspect the data extracted in the first observation
print(background_estimator.result[0])

background_estimator.plot()

# Now we change the ON region, and use a center defined in the galactic frame
on_region_galactic = RectangleSkyRegion(
    center=crab_position.galactic,
    width=0.5 * u.deg,
    height=0.4 * u.deg,
    angle=0 * u.deg,
)

background_estimator = ReflectedRegionsBackgroundEstimator(
    observations=observations, on_region=on_region_galactic, min_distance=0.1 * u.rad
)

background_estimator.run()
# The reflected regions are rotated copies of a box aligned with the galactic frame
background_estimator.plot()

plt.show()
