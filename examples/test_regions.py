from astropy.coordinates import Angle, SkyCoord
from gammapy.region import SkyCircleRegion

pos1 = SkyCoord(2, 1, unit='deg', frame='galactic')
radius = Angle(1, 'deg')
reg = SkyCircleRegion(pos=pos1, radius=radius)
print(reg)


import IPython; IPython.embed()
