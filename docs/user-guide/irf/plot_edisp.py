"""Plot an energy dispersion from the HESS DL3 data release 1."""
import matplotlib.pyplot as plt
from gammapy.irf import EnergyDispersion2D

filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
edisp = EnergyDispersion2D.read(filename, hdu="EDISP")
edisp.peek()
plt.show()
