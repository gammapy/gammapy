"""Make plots of the IRFs we support.

This is not very useful for end-users,
more to check that all the plotting functions work,
until we implement proper tests for them.
"""
import matplotlib.pyplot as plt
from gammapy.datasets import gammapy_extra
from gammapy import irf
from gammapy.utils.mpl_style import gammapy_mpl_style

# TODO: Update once this issue is resolved:
# https://github.com/astropy/astropy/issues/4140
# plt.style.use(gammapy_mpl_style)
plt.rcParams.update(gammapy_mpl_style)

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flat

# TODO: event list summary plots not implemented yet:
# https://github.com/gammapy/gammapy/issues/297
# hess_events_023523.fits.gz

filename = gammapy_extra.filename('test_datasets/irf/hess/pa/hess_aeff_2d_023523.fits.gz')
aeff2d = irf.EffectiveAreaTable2D.read(filename)
aeff2d.plot_energy_dependence(ax=next(axes))
aeff2d.plot_offset_dependence(ax=next(axes))

aeff = aeff2d.to_effective_area_table(offset='1 deg')
aeff.plot_area_vs_energy(ax=next(axes))


filename = gammapy_extra.filename('test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz')
edisp2d = irf.EnergyDispersion2D.read(filename)
edisp2d.plot_bias(ax=next(axes))
edisp2d.plot_migration(ax=next(axes))

edisp = edisp2d.to_energy_dispersion(offset='1 deg')
edisp.plot(type='matrix', ax=next(axes))
# TODO: bias plot not implemented yet
# edisp.plot(type='bias', ax=next(axes))

# TODO: This PSF type isn't implemented yet.
# filename = gammapy_extra.filename('test_datasets/irf/hess/pa/hess_psf_king_023523.fits.gz')
# psf = irf.EnergyDependentMultiGaussPSF.read(filename)
# psf.plot_containment(ax=next(axes))

filename = gammapy_extra.filename('test_datasets/unbundled/irfs/psf.fits')
psf = irf.EnergyDependentMultiGaussPSF.read(filename)
psf.plot_containment(0.68, show_safe_energy=False, ax=next(axes))


# TODO: hess_bkg_offruns_023523.fits.gz



plt.show()