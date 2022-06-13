"""Plot the background rate from the HESS DL3 data release 1."""
import matplotlib.pyplot as plt
from gammapy.irf import Background3D

filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
bkg = Background3D.read(filename, hdu="BKG")
bkg.peek()
plt.show()
