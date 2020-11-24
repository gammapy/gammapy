"""Plot the effective area at a given offset"""
from astropy.coordinates import Angle
import matplotlib.pyplot as plt
from gammapy.irf import EffectiveAreaTable2D

filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
aeff = EffectiveAreaTable2D.read(filename, hdu="AEFF")
# offset at which we want to examine the effective area
offset = Angle("0.5 deg")
aeff_table = aeff.to_effective_area_table(offset=offset)
aeff_table.plot()
plt.show()
