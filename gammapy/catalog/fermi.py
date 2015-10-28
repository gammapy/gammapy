# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fermi catalog and source classes.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tarfile
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file
from astropy.units import Quantity
from ..spectrum import EnergyBounds
from ..datasets import gammapy_extra
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    'fetch_fermi_catalog',
    'fetch_fermi_extended_sources',
    'SourceCatalog2FHL',
    'SourceCatalog3FGL',
    'SourceCatalogObject2FHL',
    'SourceCatalogObject3FGL',
]


def _is_galactic(source_class):
    """Re-group sources into rough categories.

    Categories:
    - 'galactic'
    - 'extra-galactic'
    - 'unknown'
    - 'other'

    Source identifications and associations are treated identically,
    i.e. lower-case and upper-case source classes are not distinguished.

    References:
    - Table 3 in 3FGL paper: http://adsabs.harvard.edu/abs/2015arXiv150102003T
    - Table 4 in the 1FHL paper: http://adsabs.harvard.edu/abs/2013ApJS..209...34A
    """
    source_class = source_class.lower().strip()

    gal_classes = ['psr', 'pwn', 'snr', 'spp', 'lbv', 'hmb',
                   'hpsr', 'sfr', 'glc', 'bin', 'nov']
    egal_classes = ['agn', 'agu', 'bzb', 'bzq', 'bll', 'gal', 'rdg', 'fsrq',
                    'css', 'sey', 'sbg', 'nlsy1', 'ssrq', 'bcu']

    if source_class in gal_classes:
        return 'galactic'
    elif source_class in egal_classes:
        return 'extra-galactic'
    elif source_class == '':
        return 'unknown'
    else:
        raise ValueError('Unknown source class: {}'.format(source_class))


def fetch_fermi_catalog(catalog, extension=None):
    """Fetch Fermi catalog data.

    Reference: http://fermi.gsfc.nasa.gov/ssc/data/access/lat/.

    The Fermi catalogs contain the following relevant catalog HDUs:

    * 3FGL Catalog : LAT 4-year Point Source Catalog
        * ``LAT_Point_Source_Catalog`` Point Source Catalog Table.
        * ``ExtendedSources`` Extended Source Catalog Table.
    * 2FGL Catalog : LAT 2-year Point Source Catalog
        * ``LAT_Point_Source_Catalog`` Point Source Catalog Table.
        * ``ExtendedSources`` Extended Source Catalog Table.
    * 1FGL Catalog : LAT 1-year Point Source Catalog
        * ``LAT_Point_Source_Catalog`` Point Source Catalog Table.
    * 2FHL Catalog : Second Fermi-LAT Catalog of High-Energy Sources
        * ``Count Map`` AIT projection 2D count image
        * ``2FHL Source Catalog`` Main catalog
        * ``Extended Sources`` Extended Source Catalog Table
        * ``ROIs`` Regions of interest
    * 1FHL Catalog : First Fermi-LAT Catalog of Sources above 10 GeV
        * ``LAT_Point_Source_Catalog`` Point Source Catalog Table.
        * ``ExtendedSources`` Extended Source Catalog Table.
    * 2PC Catalog : LAT Second Catalog of Gamma-ray Pulsars
        * ``PULSAR_CATALOG`` Pulsar Catalog Table.
        * ``SPECTRAL`` Table of Pulsar Spectra Parameters.
        * ``OFF_PEAK`` Table for further Spectral and Flux data for the Catalog.

    Parameters
    ----------
    catalog : {'3FGL', '2FGL', '1FGL', '1FHL', '2FHL', '2PC'}
       Specifies which catalog to display.
    extension : str
        Specifies which catalog HDU to provide as a table (optional).
        See list of catalog HDUs above.

    Returns
    -------
    hdu_list (Default) : `~astropy.io.fits.HDUList`
        Catalog FITS HDU list (for access to full catalog dataset).
    catalog_table : `~astropy.table.Table`
        Catalog table for a selected hdu extension.

    Examples
    --------
    >>> from gammapy.catalog import fetch_fermi_catalog
    >>> fetch_fermi_catalog('2FGL')
        [<astropy.io.fits.hdu.image.PrimaryHDU at 0x3330790>,
         <astropy.io.fits.hdu.table.BinTableHDU at 0x338b990>,
         <astropy.io.fits.hdu.table.BinTableHDU at 0x3396450>,
         <astropy.io.fits.hdu.table.BinTableHDU at 0x339af10>,
         <astropy.io.fits.hdu.table.BinTableHDU at 0x339ff10>]

    >>> from gammapy.catalog import fetch_fermi_catalog
    >>> fetch_fermi_catalog('2FGL', 'LAT_Point_Source_Catalog')
        <Table rows=1873 names= ... >
    """
    BASE_URL = 'http://fermi.gsfc.nasa.gov/ssc/data/access/lat/'

    if catalog == '3FGL':
        url = BASE_URL + '4yr_catalog/gll_psc_v16.fit'
    elif catalog == '2FGL':
        url = BASE_URL + '2yr_catalog/gll_psc_v08.fit'
    elif catalog == '1FGL':
        url = BASE_URL + '1yr_catalog/gll_psc_v03.fit'
    elif catalog == '1FHL':
        url = BASE_URL + '1FHL/gll_psch_v07.fit'
    elif catalog == '2FHL':
        url = 'https://github.com/gammapy/gammapy-extra/raw/master/datasets/catalogs/fermi/gll_psch_v08.fit.gz'
    elif catalog == '2PC':
        url = BASE_URL + '2nd_PSR_catalog/2PC_catalog_v03.fits'
    else:
        ss = 'Invalid catalog: {0}\n'.format(catalog)
        raise ValueError(ss)

    filename = download_file(url, cache=True)
    hdu_list = fits.open(filename)

    if extension is None:
        return hdu_list

    # TODO: 2FHL doesn't have a 'CLASS1' column, just 'CLASS'
    # It's probably better if we make a `SourceCatalog` class
    # and then sub-class `FermiSourceCatalog` and `Fermi2FHLSourceCatalog`
    # and handle catalog-specific stuff in these classes,
    # trying to provide an as-uniform as possible API to the common catalogs.
    table = Table(hdu_list[extension].data)
    table['IS_GALACTIC'] = [_is_galactic(_) for _ in table['CLASS1']]

    return table


def fetch_fermi_extended_sources(catalog):
    """Fetch Fermi catalog extended source images.

    Reference: http://fermi.gsfc.nasa.gov/ssc/data/access/lat/.

    Extended source are available for the following Fermi catalogs:

    * 3FGL Catalog : LAT 4-year Point Source Catalog
    * 2FGL Catalog : LAT 2-year Point Source Catalog
    * 1FHL Catalog : First Fermi-LAT Catalog of Sources above 10 GeV

    Parameters
    ----------
    catalog : {'3FGL', '2FGL', '1FHL'}
       Specifies which catalog extended sources to return.

    Returns
    -------
    hdu_list : `~astropy.io.fits.HDUList`
        FITS HDU list of FITS ImageHDUs for the extended sources.

    Examples
    --------
    >>> from gammapy.catalog import fetch_fermi_extended_sources
    >>> sources = fetch_fermi_extended_sources('2FGL')
    >>> len(sources)
    12
    """
    BASE_URL = 'http://fermi.gsfc.nasa.gov/ssc/data/access/lat/'
    if catalog == '3FGL':
        url = BASE_URL + '4yr_catalog/LAT_extended_sources_v15.tgz'
    elif catalog == '2FGL':
        url = BASE_URL + '2yr_catalog/gll_psc_v07_templates.tgz'
    elif catalog == '1FHL':
        url = BASE_URL + '1FHL/LAT_extended_sources_v12.tar'
    else:
        ss = 'Invalid catalog: {0}\n'.format(catalog)
        raise ValueError(ss)

    filename = download_file(url, cache=True)
    tar = tarfile.open(filename, 'r')

    hdu_list = []
    for member in tar.getmembers():
        if member.name.endswith(".fits"):
            file = tar.extractfile(member)
            hdu = fits.open(file)[0]
            hdu_list.append(hdu)
    hdu_list = fits.HDUList(hdu_list)

    return hdu_list


class SourceCatalogObject3FGL(SourceCatalogObject):
    """One source from the Fermi-LAT 3FGL catalog.
    """

    x_bins_edges = Quantity([30, 100, 300, 1000, 3000, 10000, 100000], 'MeV')

    x_bins = Quantity(x_bins_edges, 'MeV')

    x_cens = EnergyBounds(x_bins).log_centers

    y_labels = ['Flux30_100', 'Flux100_300', 'Flux300_1000',
                'Flux1000_3000', 'Flux3000_10000', 'Flux10000_100000']

    def plot_lightcurve(self, ax=None):
        """Plot lightcurve.
        """
        from gammapy.time import plot_fermi_3fgl_light_curve

        ax = plot_fermi_3fgl_light_curve(self.name, ax=ax)
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
        data = self.data
        info = "\n"
        info += data['Source_Name'] + "\n"
        info += "\n"
        info += "RA (J2000) " + str(data['RAJ2000']) + "\n"
        info += "Dec (J2000) " + str(data['DEJ2000']) + "\n"
        info += "l " + str(data['GLON']) + "\n"
        info += "b " + str(data['GLAT']) + "\n"
        info += "Energy Flux (100 MeV - 100 GeV): " + str(data['Energy_Flux100']) + \
                " +/- " + str('Unc_Energy_Flux100') + " erg /cm2 /s\n"
        info += "Detection significance: " + str(data['Signif_Avg']) + " sigma\n"

        return info


class SourceCatalogObject2FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 2FHL catalog.
    """
    pass


class SourceCatalog3FGL(SourceCatalog):
    """Fermi-LAT 3FGL source catalog.
    """
    name = '3fgl'
    description = 'LAT 4-year Point Source Catalog'
    source_object_class = SourceCatalogObject3FGL

    def __init__(self, filename=None):
        if not filename:
            filename = gammapy_extra.filename('datasets/catalogs/fermi/gll_psc_v16.fit.gz')

        self.hdu_list = fits.open(filename)
        table = Table(self.hdu_list['LAT_Point_Source_Catalog'].data)
        super(SourceCatalog3FGL, self).__init__(table=table)


class SourceCatalog2FHL(SourceCatalog):
    """Fermi-LAT 2FHL source catalog.
    """
    name = '2fhl'
    description = 'LAT Second High-Energy Source Catalog'
    source_object_class = SourceCatalogObject2FHL

    def __init__(self):
        self.hdu_list = fetch_fermi_catalog(catalog='2FGL')
        self.table = Table(self.hdu_list['LAT_Point_Source_Catalog'].data)
