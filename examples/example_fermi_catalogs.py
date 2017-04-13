"""Example to show how to plot spectrum of Fermi/LAT sources.
"""
import matplotlib.pyplot as plt
from gammapy.catalog import SourceCatalog3FGL, SourceCatalog2FHL, SourceCatalog1FHL, SourceCatalog3FHL


def plot_source_spectrum(source, label, color, sed_type):
    source.spectral_model.plot(
        energy_range=source.energy_range, energy_power=2, flux_unit='erg-1 cm-2 s-1',
        label=label, color=color,
    )
    source.spectral_model.plot_error(
        energy_range=source.energy_range, energy_power=2, flux_unit='erg-1 cm-2 s-1',
        facecolor=color,
    )
    energy_power = 0 if sed_type == 'eflux' else 2
    flux_unit = 'erg cm-2 s-1' if sed_type == 'eflux' else 'cm-2 s-1 erg-1'
    source.flux_points.plot(
        sed_type=sed_type, energy_power=energy_power, flux_unit=flux_unit,
        color=color,
    )


def plot_source_spectra(name):
    plot_source_spectrum(source=SourceCatalog3FGL()[name], label='Fermi 3FGL', color='r', sed_type='eflux')
    plot_source_spectrum(source=SourceCatalog2FHL()[name], label='Fermi 2FHL', color='g', sed_type='dnde')
    plot_source_spectrum(source=SourceCatalog1FHL()[name], label='Fermi 1FHL', color='c', sed_type='dnde')
    plot_source_spectrum(source=SourceCatalog3FHL()[name], label='Fermi 3FHL', color='b', sed_type='eflux')

    ax = plt.gca()
    ax.set_ylim(1.e-12, 7.e-11)
    ax.set_xlim(1.e-4, 2.)
    ax.set_xlabel('Energy (TeV)')
    ax.set_ylabel('E^2 dN/dE (erg cm-2 s-1])')
    plt.legend(loc=0)


if __name__ == '__main__':
    # Select your favourite source
    # (must be named like this in ASSOC columns of all catalogs)
    name = 'PKS 2155-304'
    plot_source_spectra(name)
    plt.show()
