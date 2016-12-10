"""Example how to plot Fermi-LAT catalog spectra.
"""
import matplotlib.pyplot as plt
from gammapy.catalog import SourceCatalog3FGL, SourceCatalog2FHL
from gammapy.utils.energy import EnergyBounds

plt.style.use('ggplot')

# load catalogs
fermi_3fgl = SourceCatalog3FGL()
fermi_2fhl = SourceCatalog2FHL()

# access crab data by corresponding identifier
crab_3fgl = fermi_3fgl['3FGL J0534.5+2201']
crab_2fhl = fermi_2fhl['2FHL J0534.5+2201']

ax = crab_3fgl.spectral_model.plot(crab_3fgl.energy_range, energy_power=2,
                                   label='Fermi 3FGL', color='r',
                                   flux_unit='erg-1 cm-2 s-1')
ax.set_ylim(1e-12, 1E-9)

# set up an energy array to evaluate the butterfly
emin, emax = crab_3fgl.energy_range
energy = EnergyBounds.equal_log_spacing(emin, emax, 100)
butterfly_3fg = crab_3fgl.spectrum.butterfly(energy)

butterfly_3fg.plot(crab_3fgl.energy_range, ax=ax, energy_power=2, color='r',
                   flux_unit='erg-1 cm-2 s-1')

crab_3fgl.flux_points.plot(ax=ax, sed_type='eflux', color='r',
                           y_unit='erg cm-2 s-1')

crab_2fhl.spectral_model.plot(crab_2fhl.energy_range, ax=ax, energy_power=2,
                              c='g', label='Fermi 2FHL', flux_unit='erg-1 cm-2 s-1')

# set up an energy array to evaluate the butterfly using the 2FHL energy range
emin, emax = crab_2fhl.energy_range
energy = EnergyBounds.equal_log_spacing(emin, emax, 100)
butterfly_2fhl = crab_2fhl.spectrum.butterfly(energy)

# plot butterfly and flux points
butterfly_2fhl.plot(crab_2fhl.energy_range, ax=ax, energy_power=2, color='g',
                    flux_unit='erg-1 cm-2 s-1')
crab_2fhl.flux_points.plot(ax=ax, sed_type='dnde', energy_power=2, color='g',
                           y_unit='ph cm-2 s-1 erg-1')
plt.legend(loc=0)
plt.show()
