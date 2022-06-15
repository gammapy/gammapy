"""Plot an effective area from the HESS DL3 data release 1."""
import matplotlib.pyplot as plt
from gammapy.irf import EffectiveAreaTable2D

filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
aeff = EffectiveAreaTable2D.read(filename, hdu="AEFF")
aeff.peek()
plt.show()
