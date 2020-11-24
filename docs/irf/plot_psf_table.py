"""Plot the PSF at a given offset from the camera center"""
from astropy.coordinates import Angle
import matplotlib.pyplot as plt
from gammapy.irf import PSF3D

filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
psf = PSF3D.read(filename, hdu="PSF")
# offset at which we want to examine the PSF
offset = Angle("0.5 deg")
psf_table = psf.to_energy_dependent_table_psf(offset)
psf_table.plot_psf_vs_rad()
plt.show()
