"""Example how to compute and plot reflected regions."""
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.maps import WcsNDMap
from gammapy.background import ReflectedRegionsFinder

exclusion_mask = WcsNDMap.create(
    npix=(801, 701), binsz=0.01, skydir=(83.633, 23.014),
)

# Exclude a rectangular region
coords = exclusion_mask.geom.get_coord().skycoord
mask = (Angle('23d') < coords.dec) & (coords.dec < Angle('24d'))
exclusion_mask.data = np.invert(mask)

pos = SkyCoord(83.633, 22.014, unit='deg')
radius = Angle(0.3, 'deg')
on_region = CircleSkyRegion(pos, radius)
center = SkyCoord(83.633, 24, unit='deg')

finder = ReflectedRegionsFinder(
    region=on_region,
    center=center,
    exclusion_mask=exclusion_mask,
    min_distance_input='0.2 rad',
)
finder.run()
finder.plot()
plt.show()
