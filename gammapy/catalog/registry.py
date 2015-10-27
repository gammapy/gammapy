# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source catalog registry.

Provides simple and efficient access to source catalogs.

Catalog objects are cached in a module-level dict,
so that catalogs are not re-loaded from disk on each request.

You should use these catalogs read-only, if you modify
them you can get non-reproducible results if you access
the modified version later on.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from astropy.table import Table
from ..spectrum import EnergyBounds
from ..datasets import gammapy_extra
from ..datasets import fetch_fermi_catalog

__all__ = [
    'get_source_catalog',
    # TODO: I'm not sure if it's useful to make those part of the API docs:
    'SourceCatalog',
    'SourceCatalog2FHL',
    'SourceCatalog3FGL',
    'SourceCatalogATNF',
    'SourceCatalogObject2FHL',
    'SourceCatalogObject3FGL',
]

# The source catalog registry is a module-level dict
# that acts as a cache for `get_source_catalog`
_source_catalog_registry = {
    '3FGL': None,
    '2FHL': None
}
"""Built-in catalogs in Gammapy."""


def _load_catalog(name):
    """Load catalog into the cache.
    """
    if name == '3FGL':
        _source_catalog_registry['3FGL'] = SourceCatalog3FGL()
    elif name == '2FHL':
        _source_catalog_registry['3FHL'] = SourceCatalog2FHL()
    else:
        raise ValueError('Unknown catalog: {}'.format(name))


def get_source_catalog(name):
    """Get source catalog by name.
    """
    if name not in _source_catalog_registry.keys():
        msg = 'Unknown catalog: {} '.format(name)
        msg += 'Available catalogs: {}'.format(_source_catalog_registry.keys())
        raise ValueError(msg)

    if not _source_catalog_registry[name]:
        _load_catalog(name)

    return _source_catalog_registry[name]


class SourceCatalogObject3FGL(object):
    """One source from the Fermi-LAT 3FGL catalog.
    """

    x_bins_edges = Quantity([30, 100, 300, 1000, 3000, 10000, 100000], 'MeV')

    x_bins = Quantity(x_bins_edges, 'MeV')

    x_cens = EnergyBounds(x_bins).log_centers

    y_labels = ['Flux30_100', 'Flux100_300', 'Flux300_1000',
                'Flux1000_3000', 'Flux3000_10000', 'Flux10000_100000']

    def __init__(self, table, row_index):
        self.row_index = row_index
        cat_row = table.data[row_index]
        self.cat_row = cat_row
        self.source_name = cat_row['Source_Name']
        self.ra = cat_row['RAJ2000']
        self.dec = cat_row['DEJ2000']
        self.glon = cat_row['GLON']
        self.glat = cat_row['GLAT']
        self.flux_density = cat_row['Flux_Density']
        self.unc_flux_density = cat_row['Unc_Flux_Density']
        self.spec_type = cat_row['SpectrumType']
        self.pivot_en = cat_row['PIVOT_ENERGY']
        self.spec_index = cat_row['Spectral_Index']
        self.unc_spec_index = cat_row['Unc_Spectral_Index']
        self.beta = cat_row['beta']
        self.unc_beta = cat_row['unc_beta']
        self.cutoff = cat_row['Cutoff']
        self.unc_cutoff = cat_row['Unc_Cutoff']
        self.exp_index = cat_row['Exp_Index']
        self.unc_exp_index = cat_row['Unc_Exp_Index']
        self.signif = cat_row['Signif_Avg']

    def plot_lightcurve(self, ax=None):
        """Plot lightcurve.
        """
        from gammapy.time import plot_fermi_3fgl_light_curve

        ax = plot_fermi_3fgl_light_curve(self.source_name, ax=ax)
        return ax

    def plot_spectrum(self, ax=None):
        """Plot spectrum.
        """
        import matplotlib.pyplot as plt
        from gammapy.extern.stats import gmean
        from astropy.modeling.models import PowerLaw1D, LogParabola1D, ExponentialCutoffPowerLaw1D

        ax = plt.gca() if ax is None else ax

        # Only work with indices where we have a valid detection and a lower bound
        flux_bounds = [self.cat_row["Unc_" + self.y_labels[i]] for i in range(0, np.size(self.y_labels))]

        valid_indices = []

        for i in range(0, len(flux_bounds)):
            if np.size(flux_bounds[i]) == 2 and not np.isnan(flux_bounds[i][0]):
                valid_indices.append(i)

        y_vals = np.array([self.cat_row[i] for i in (self.y_labels[j] for j in valid_indices)])
        y_lower = np.array([self.cat_row["Unc_" + i][0] for i in (self.y_labels[j] for j in valid_indices)])
        y_upper = np.array([self.cat_row["Unc_" + i][1] for i in (self.y_labels[j] for j in valid_indices)])

        y_lower = y_vals + y_lower
        y_upper = y_vals + y_upper

        x_vals = [self.x_cens[i].value for i in valid_indices]
        bin_edges1 = [-(self.x_bins_edges[i] - self.x_cens[i]).value for i in valid_indices]
        bin_edges2 = [(self.x_bins_edges[i + 1] - self.x_cens[i]).value for i in valid_indices]

        y_vals = [y_vals[i] / x_vals[i] for i in range(0, np.size(y_vals))]
        y_upper = [y_upper[i] / x_vals[i] for i in range(0, np.size(y_vals))]
        y_lower = [y_lower[i] / x_vals[i] for i in range(0, np.size(y_vals))]

        y_cens = np.array([gmean([y_lower[i], y_upper[i]]) for i in range(0, np.size(y_lower))])

        y_upper = np.array([y_upper[i] - y_vals[i] for i in range(0, np.size(y_lower))])
        y_lower = np.array([y_vals[i] - y_lower[i] for i in range(0, np.size(y_lower))])

        ax.loglog()

        fmt = dict(elinewidth=1, linewidth=0, color='black')
        ax.errorbar(x_vals, y_vals, yerr=(y_lower, y_upper), **fmt)

        # Place the x-axis uncertainties in the center of the y-axis uncertainties.
        ax.errorbar(x_vals, y_cens, xerr=(bin_edges1, bin_edges2), **fmt)

        x_model = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 25)

        if self.spec_type == "PowerLaw":

            y_model = PowerLaw1D(amplitude=self.flux_density,
                                 x_0=self.pivot_en,
                                 alpha=self.spec_index)

        elif self.spec_type == "LogParabola":

            y_model = LogParabola1D(amplitude=self.flux_density,
                                    x_0=self.pivot_en,
                                    alpha=self.spec_index,
                                    beta=self.beta)

        elif self.spec_type == "PLExpCutoff":

            y_model = ExponentialCutoffPowerLaw1D(amplitude=self.flux_density,
                                                  x_0=self.pivot_en,
                                                  alpha=self.spec_index,
                                                  x_cutoff=self.cutoff)
        elif self.spec_type == "PLSuperExpCutoff":
            raise NotImplementedError
        else:
            raise NotImplementedError

        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Flux (ph/cm^2/s/MeV)')
        ax.plot(x_model, y_model(x_model))

        return ax

    def info(self):
        """Print summary info."""
        info = "\n"
        info += self.source_name + "\n"
        info += "\n"
        info += "RA (J2000) " + str(self.ra) + "\n"
        info += "Dec (J2000) " + str(self.dec) + "\n"
        info += "l " + str(self.glon) + "\n"
        info += "b " + str(self.glat) + "\n"
        # TODO: fix error: no attribute `energy_flux`
        # info += "Integrated Flux 100 MeV - 100 GeV: " + str(self.energy_flux) + \
        #         " +/- " + str(self.unc_energy_flux) + " erg /cm2 /s\n"
        info += "Detection significance: " + str(self.signif) + " sigma\n"

        return info


class SourceCatalogObject2FHL(object):
    """One source from the Fermi-LAT 2FHL catalog.
    """
    pass


class SourceCatalog(object):
    """Abstract base class for source catalogs.
    """

    def __getitem__(self, source_name):
        """Get source by name"""
        idx = np.where(self.table['Source_Name'] == source_name)[0][0]
        return self.source_index(idx)

    def source_index(self, idx):
        """Get source by row index.
        """
        return self._source_object_class(self, idx)


class SourceCatalog3FGL(SourceCatalog):
    """Fermi-LAT 3FGL source catalog.
    """
    name = '3FGL'
    _source_object_class = SourceCatalogObject3FGL

    def __init__(self, filename=None):
        # if not filename:
        #     filename =
        self.hdu_list = fetch_fermi_catalog(catalog='3FGL')
        self.table = Table(self.hdu_list['LAT_Point_Source_Catalog'].data)


class SourceCatalog2FHL(SourceCatalog):
    """Fermi-LAT 2FHL source catalog.
    """
    name = '2FHL'
    _source_object_class = SourceCatalogObject2FHL

    def __init__(self):
        self.hdu_list = fetch_fermi_catalog(catalog='2FGL')
        self.table = Table(self.hdu_list['LAT_Point_Source_Catalog'].data)


class SourceCatalogATNF(SourceCatalog):
    """ATNF pulsar catalog.

    The `ATNF pulsar catalog <http://www.atnf.csiro.au/people/pulsar/psrcat/>`__
    is **the** collection of information on all pulsars.

    Unfortunately it's only available in a database format that can only
    be read with their software.

    This function loads a FITS copy of version 1.51 of the ATNF catalog:
    http://www.atnf.csiro.au/research/pulsar/psrcat/catalogueHistory.html

    The ``ATNF_v1.51.fits.gz`` file and ``make_atnf.py`` script are available
    `here <https://github.com/gammapy/gammapy-extra/blob/master/datasets/catalogs/>`__.
    """
    name = 'ATNF'

    def __init__(self, filename=None):
        if not filename:
            filename = gammapy_extra.filename('datasets/catalogs/ATNF_v1.51.fits.gz')
            self.table = Table.read(filename)
