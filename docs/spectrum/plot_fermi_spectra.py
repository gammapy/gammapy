"""Example how to plot Fermi-LAT catalog spectra.
"""
import matplotlib.pyplot as plt
from gammapy.catalog import SourceCatalog3FGL, SourceCatalog2FHL, SourceCatalog1FHL

plt.style.use('ggplot')

# load catalogs
fermi_3fgl = SourceCatalog3FGL()
fermi_2fhl = SourceCatalog2FHL()
fermi_1fhl = SourceCatalog1FHL()

# access crab data by corresponding identifier
src_3fgl = fermi_3fgl['3FGL J2158.8-3013']
src_2fhl = fermi_2fhl['2FHL J2158.8-3013']
src_1fhl = fermi_1fhl['1FHL J2158.8-3013']

# 3FHL
ax = src_3fgl.spectral_model.plot(src_3fgl.energy_range, energy_power=2,
                                  label='Fermi 3FGL', color='r',
                                  flux_unit='erg-1 cm-2 s-1')
src_3fgl.spectral_model.plot_error(src_3fgl.energy_range, energy_power=2,
                                  color='r',
                                  flux_unit='erg-1 cm-2 s-1', ax=ax)
src_3fgl.flux_points.plot(ax=ax, sed_type='eflux', color='r',
                          flux_unit='erg cm-2 s-1')

# 2FHL
src_2fhl.spectral_model.plot(src_2fhl.energy_range, ax=ax, energy_power=2,
                             c='g', label='Fermi 2FHL', flux_unit='erg-1 cm-2 s-1')
src_2fhl.spectral_model.plot_error(src_2fhl.energy_range, ax=ax, energy_power=2,
                                    color='g', flux_unit='erg-1 cm-2 s-1')
src_2fhl.flux_points.plot(ax=ax, sed_type='dnde', energy_power=2, color='g',
                          flux_unit='cm-2 s-1 erg-1')

# 1FHL
src_1fhl.spectral_model.plot(src_1fhl.energy_range, ax=ax, energy_power=2,
                             c='b', label='Fermi 1FHL', flux_unit='erg-1 cm-2 s-1')
src_1fhl.spectral_model.plot_error(src_1fhl.energy_range, ax=ax, energy_power=2,
                             color='b', flux_unit='erg-1 cm-2 s-1')
src_1fhl.flux_points.plot(ax=ax, sed_type='dnde', energy_power=2, color='b',
                          flux_unit='cm-2 s-1 erg-1')

ax.set_ylim(5.e-12, 8.e-11)
ax.set_ylabel('dN/dE [erg cm-2 s-1]')
plt.legend(loc=0)
plt.show()
