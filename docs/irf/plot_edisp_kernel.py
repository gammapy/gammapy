"""Plot the energy dispersion at a given offset."""
from astropy.coordinates import Angle
import matplotlib.pyplot as plt
from gammapy.irf import EnergyDispersion2D

filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
edisp = EnergyDispersion2D.read(filename, hdu="EDISP")
# offset at which we want to examine the energy dispersion
offset = Angle("0.5 deg")
edisp_kernel = edisp.to_edisp_kernel(offset=offset)
edisp_kernel.peek()
plt.show()
