""" Example to show how to plot spectrum of Fermi/LAT sources
"""
import matplotlib.pyplot as plt

from gammapy.catalog import SourceCatalog3FGL, SourceCatalog2FHL, SourceCatalog1FHL, SourceCatalog3FHL
from gammapy.utils.energy import EnergyBounds

# # 2155
name = 'PKS 2155-304'

# load catalogs
fermi_3fgl = SourceCatalog3FGL()
fermi_2fhl = SourceCatalog2FHL()
fermi_1fhl = SourceCatalog1FHL()
fermi_3fhl = SourceCatalog3FHL()

# access crab data by corresponding identifier
src_3fgl = fermi_3fgl[name]
src_2fhl = fermi_2fhl[name]
src_1fhl = fermi_1fhl[name]
src_3fhl = fermi_3fhl[name]

# 3FHL
ax = src_3fgl.spectral_model.plot(src_3fgl.energy_range, energy_power=2,
                                  label='Fermi 3FGL', color='r',
                                  flux_unit='erg-1 cm-2 s-1')

# set up an energy array to evaluate the butterfly
emin, emax = src_3fgl.energy_range
energy = EnergyBounds.equal_log_spacing(emin, emax, 100)
butterfly_3fg = src_3fgl.spectrum.butterfly(energy)

butterfly_3fg.plot(src_3fgl.energy_range, ax=ax, energy_power=2, color='r',
                   flux_unit='erg-1 cm-2 s-1', facecolor='r')

src_3fgl.flux_points.plot(ax=ax, sed_type='eflux', color='r',
                          flux_unit='erg cm-2 s-1')

# 2FHL
src_2fhl.spectral_model.plot(src_2fhl.energy_range, ax=ax, energy_power=2,
                             c='g', label='Fermi 2FHL', flux_unit='erg-1 cm-2 s-1')

# set up an energy array to evaluate the butterfly
emin, emax = src_2fhl.energy_range
energy = EnergyBounds.equal_log_spacing(emin, emax, 100)
butterfly_2fhl = src_2fhl.spectrum.butterfly(energy)

# plot butterfly and flux points
butterfly_2fhl.plot(src_2fhl.energy_range, ax=ax, energy_power=2, color='g',
                    flux_unit='erg-1 cm-2 s-1', facecolor='g')
src_2fhl.flux_points.plot(ax=ax, sed_type='dnde', energy_power=2, color='g',
                          flux_unit='cm-2 s-1 erg-1')

# 1FHL
src_1fhl.spectral_model.plot(src_1fhl.energy_range, ax=ax, energy_power=2,
                             c='c', label='Fermi 1FHL',
                             flux_unit='erg-1 cm-2 s-1')

# set up an energy array to evaluate the butterfly 
emin, emax = src_1fhl.energy_range
energy = EnergyBounds.equal_log_spacing(emin, emax, 100)
butterfly_1fhl = src_1fhl.spectrum.butterfly(energy)

# plot butterfly and flux points
butterfly_1fhl.plot(src_1fhl.energy_range, ax=ax, energy_power=2, color='c',
                    flux_unit='erg-1 cm-2 s-1', facecolor='c')
src_1fhl.flux_points.plot(ax=ax, sed_type='dnde', energy_power=2, color='c',
                          flux_unit='cm-2 s-1 erg-1')

# 3FHL
src_3fhl.spectral_model.plot(src_3fhl.energy_range, ax=ax, energy_power=2,
                             c='b', label='Fermi 3FHL',
                             flux_unit='erg-1 cm-2 s-1')

# set up an energy array to evaluate the butterfly 
emin, emax = src_3fhl.energy_range
energy = EnergyBounds.equal_log_spacing(emin, emax, 100)
butterfly_3fhl = src_3fhl.spectrum.butterfly(energy)

# plot butterfly and flux points
butterfly_3fhl.plot(src_3fhl.energy_range, ax=ax, energy_power=2, color='b',
                    flux_unit='erg-1 cm-2 s-1', facecolor='b')

src_3fhl.flux_points.plot(ax=ax, sed_type='eflux', color='b',
                          flux_unit='erg cm-2 s-1')

ax.set_ylim(1.e-12, 7.e-11)
ax.set_xlim(1.e-4, 2.)
ax.set_ylabel('dN/dE [erg cm-2 s-1]')
plt.legend(loc=0)
plt.show()
