"""Example to show how to plot spectrum of Fermi/LAT sources.
"""
import matplotlib.pyplot as plt
from gammapy.catalog import (
    SourceCatalog3FGL,
    SourceCatalog2FHL,
    SourceCatalog1FHL,
    SourceCatalog3FHL,
)


def plot_source_spectrum(source, label, color):
    opts = dict(energy_power=2, flux_unit="erg-1 cm-2 s-1")
    source.spectral_model.plot(
        energy_range=source.energy_range, label=label, color=color, **opts
    )
    source.spectral_model.plot_error(
        energy_range=source.energy_range, facecolor=color, **opts
    )
    source.flux_points.plot(sed_type="dnde", color=color, **opts)


def plot_source_spectra(name):
    plot_source_spectrum(
        source=SourceCatalog3FGL()[name], label="Fermi 3FGL", color="r"
    )
    plot_source_spectrum(
        source=SourceCatalog2FHL()[name], label="Fermi 2FHL", color="g"
    )
    plot_source_spectrum(
        source=SourceCatalog1FHL()[name], label="Fermi 1FHL", color="c"
    )
    plot_source_spectrum(
        source=SourceCatalog3FHL()[name], label="Fermi 3FHL", color="b"
    )

    ax = plt.gca()
    ax.set_ylim(1.e-12, 7.e-11)
    ax.set_xlim(1.e-4, 2.)
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel("E^2 dN/dE (erg cm-2 s-1])")
    plt.legend(loc=0)


if __name__ == "__main__":
    # Select your favourite source
    # (must be named like this in ASSOC columns of all catalogs)
    name = "PKS 2155-304"
    plot_source_spectra(name)
    plt.show()
