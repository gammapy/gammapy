from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS

from gammapy.region import find_reflected_regions, SkyCircleRegion
from gammapy.image import ExclusionMask
from gammapy.utils.testing import requires_data

mask = ExclusionMask.empty(name='Exclusion Mask', nxpix=801, nypix=701, binsz=0.01,
                       	   coordsys='CEL', xref=83.2, yref=22.7)
mask.fill_random_circles(n=8, min_rad=30, max_rad=80)

pos = SkyCoord(80.2, 23.5, unit='deg', frame='icrs')
radius = Angle(0.4, 'deg')
test_region = SkyCircleRegion(pos=pos, radius=radius)
center = SkyCoord(82.8, 22.5, unit='deg', frame='icrs')
regions = find_reflected_regions(test_region, center, mask)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 5), dpi=80)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=mask.wcs)
mask.plot(ax, fig)
for reg in regions:
    reg.plot(ax, facecolor='red')
test_region.plot(ax, facecolor='blue')

plt.show()
