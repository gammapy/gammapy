"""Example how to load and take a peek at an IACT observation.
"""
import matplotlib.pyplot as plt
from gammapy.data import DataStore
from gammapy.utils.mpl_style import gammapy_mpl_style

# TODO: Update once this issue is resolved:
# https://github.com/astropy/astropy/issues/4140
# plt.style.use(gammapy_mpl_style)
plt.rcParams.update(gammapy_mpl_style)

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flat

ds = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')

events = ds.load(obs_id=23523, filetype='events')
events.peek()

aeff2d = ds.load(obs_id=23523, filetype='aeff')
aeff2d.plot_energy_dependence(ax=next(axes))
aeff2d.plot_offset_dependence(ax=next(axes))

aeff = aeff2d.to_effective_area_table(offset='1 deg')
aeff.plot_area_vs_energy(ax=next(axes))

edisp2d = ds.load(obs_id=23523, filetype='edisp')
edisp2d.plot_bias(ax=next(axes))
edisp2d.plot_migration(ax=next(axes))

edisp = edisp2d.to_energy_dispersion(offset='1 deg')
edisp.plot_matrix(ax=next(axes))
# TODO: bias plot not implemented yet
# edisp.plot_bias(ax=next(axes))

# TODO: This PSF type isn't implemented yet.
# filename = gammapy_extra.filename('test_datasets/irf/hess/pa/hess_psf_king_023523.fits.gz')
# psf = irf.EnergyDependentMultiGaussPSF.read(filename)
# psf.plot_containment(ax=next(axes))

psf = ds.load(obs_id=23523, filetype='psf')
psf.plot_containment(0.68, show_safe_energy=False, ax=next(axes))

# TODO: hess_bkg_offruns_023523.fits.gz

plt.show()
