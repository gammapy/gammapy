"""Example how to compute and plot reflected regions."""
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.image import SkyImage
from gammapy.background import find_reflected_regions

mask = SkyImage.empty(
    name='Exclusion Mask', nxpix=801, nypix=701, binsz=0.01,
    coordsys='CEL', xref=83.633, yref=23.014, fill=1,
)

pos = SkyCoord(83.633, 22.014, unit='deg')
radius = Angle(0.3, 'deg')
on_region = CircleSkyRegion(pos, radius)
center = SkyCoord(83.633, 23.014, unit='deg')
regions = find_reflected_regions(on_region, center, mask)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 5))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=mask.wcs)
mask.plot(ax, fig)
for reg in regions:
    reg.to_pixel(mask.wcs).plot(ax, color='r')
on_region.to_pixel(mask.wcs).plot(ax)
ax.scatter(center.ra.deg, center.dec.deg, s=50, c='w', marker='+',
           transform=ax.get_transform('world'))

plt.show()
