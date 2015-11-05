from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS

from gammapy.region import find_reflected_regions, SkyCircleRegion
from gammapy.image import ExclusionMask, make_empty_image
from gammapy.utils.testing import requires_data

hdu = make_empty_image(nxpix=801, nypix=701, binsz=0.01,
                       coordsys='CEL', xref=83.2, yref=22.7)
mask = ExclusionMask.create_random(hdu, n=8, min_rad=30, max_rad=80)
pos = SkyCoord(80.2, 23.5, unit='deg', frame='icrs')
radius = Angle(0.4, 'deg')
test_region = SkyCircleRegion(pos=pos, radius=radius)
center = SkyCoord(82.8, 22.5, unit='deg', frame='icrs')
regions = find_reflected_regions(test_region, center, mask)

import matplotlib.pyplot as plt
wcs = WCS(hdu.header)
fig = plt.figure(figsize=(8, 5), dpi=80)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=wcs)
mask.plot(ax)
for reg in regions:
    patch = reg.plot(ax)
    patch.set_facecolor('red')
patch2 =test_region.plot(ax)
patch2.set_facecolor('blue')

plt.show()
