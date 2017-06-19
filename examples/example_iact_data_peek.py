"""Example how to load and take a peek at an IACT observation.
"""
import matplotlib.pyplot as plt
from gammapy.data import DataStore

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flat

data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')
obs = data_store.obs(obs_id=23523)

obs.events.peek()

obs.aeff.plot_energy_dependence(ax=next(axes))
obs.aeff.plot_offset_dependence(ax=next(axes))

aeff = obs.aeff.to_effective_area_table(offset='1 deg')
# import IPython; IPython.embed()
aeff.plot(ax=next(axes))

obs.edisp.plot_bias(ax=next(axes))
obs.edisp.plot_migration(ax=next(axes))

edisp = obs.edisp.to_energy_dispersion(offset='1 deg')
edisp.plot_matrix(ax=next(axes))
# TODO: bias plot not implemented yet
# edisp.plot_bias(ax=next(axes))

# TODO: This PSF type isn't implemented yet.
# filename = gammapy_extra.filename('test_datasets/irf/hess/pa/hess_psf_king_023523.fits.gz')
# psf = irf.EnergyDependentMultiGaussPSF.read(filename)
# psf.plot_containment(ax=next(axes))

obs.psf.plot_containment(0.68, show_safe_energy=False, ax=next(axes))

# TODO: hess_bkg_offruns_023523.fits.gz

plt.show()
