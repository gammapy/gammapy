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
from ..spectrum import DifferentialFluxPoints, IntegralFluxPoints
from ..spectrum.models import PowerLaw, ExponentialCutoffPowerLaw, LogParabola
from ..spectrum.powerlaw import power_law_flux

from ..utils.energy import EnergyBounds
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
    """
    One source from the Fermi-LAT 3FGL catalog.
    """
    _ebounds = EnergyBounds([100, 300, 1000, 3000, 10000, 100000], 'MeV')
    _ebounds_suffix = ['100_300', '300_1000', '1000_3000', '3000_10000', '10000_100000']
    energy_range = Quantity([100, 100000], 'MeV')
    """Energy range of the catalog.

    Paper says that analysis uses data up to 300 GeV,
    but results are all quoted up to 100 GeV only to
    be consistent with previous catalogs.
    """

    def __str__(self):
        """Print default summary info string"""
        return self.summary()

    def summary(self):
        """Print summary info."""
        d = self.data

        ss = 'Source: {}\n'.format(d['Source_Name'])
        ss += '\n'

        ss += 'RA (J2000)  : {}\n'.format(d['RAJ2000'])
        ss += 'Dec (J2000) : {}\n'.format(d['DEJ2000'])
        ss += 'GLON        : {}\n'.format(d['GLON'])
        ss += 'GLAT        : {}\n'.format(d['GLAT'])
        ss += '\n'

        val, err = d['Energy_Flux100'], d['Unc_Energy_Flux100']
        ss += 'Energy flux (100 MeV - 100 GeV) : {} +- {} erg cm^-2 s^-1\n'.format(val, err)
        ss += 'Detection significance : {}\n'.format(d['Signif_Avg'])

        return ss

    @property
    def spectral_model(self):
        """
        Best fit spectral model `~gammapy.spectrum.SpectralModel`.
        """
        spec_type = self.data['SpectrumType'].strip()
        pars = {}
        pars['amplitude'] = Quantity(self.data['Flux_Density'], 'MeV-1 cm-2 s-1')
        pars['reference'] = Quantity(self.data['Pivot_Energy'], 'MeV')

        if spec_type == 'PowerLaw':
            pars['index'] = Quantity(self.data['Spectral_Index'], '')
            return PowerLaw(**pars)

        elif spec_type == 'PLExpCutoff':
            pars['index'] = Quantity(self.data['Spectral_Index'], '')
            pars['lambda_'] = Quantity(1. / self.data['Cutoff'], 'MeV-1')
            return ExponentialCutoffPowerLaw(**pars)

        elif spec_type == 'LogParabola':
            pars['alpha'] = Quantity(self.data['Spectral_Index'], '')
            pars['beta'] = Quantity(self.data['beta'], '')
            return  LogParabola(**pars)

        elif spec_type == "PLSuperExpCutoff":
            # TODO Implement super exponential cut off
            raise NotImplementedError

        else:
            raise ValueError('Spectral model {} not available'.format(spec_type))

    @property
    def flux_points(self):
        """
        Differential flux points (`~gammapy.spectrum.DifferentialFluxPoints`).
        """

        energy = self._ebounds.log_centers

        nuFnu = self._get_flux_values('nuFnu', 'erg cm-2 s-1')
        diff_flux = (nuFnu * energy ** -2).to('erg-1 cm-2 s-1')

        # Get relativ error on integral fluxes
        int_flux_points = self.flux_points_integral
        diff_flux_err_hi = diff_flux * int_flux_points['INT_FLUX_ERR_HI_%'] / 100
        diff_flux_err_lo = diff_flux * int_flux_points['INT_FLUX_ERR_LO_%'] / 100

        return DifferentialFluxPoints.from_arrays(energy=energy, diff_flux=diff_flux,
                                                  diff_flux_err_lo=diff_flux_err_lo,
                                                  diff_flux_err_hi=diff_flux_err_hi)

    @property
    def flux_points_integral(self):
        """
        Integral flux points (`~gammapy.spectrum.IntegralFluxPoints`).
        """
        flux = self._get_flux_values()
        flux_err = self._get_flux_values('Unc_Flux')

        return IntegralFluxPoints.from_arrays(self._ebounds, flux, flux + flux_err[:, 1],
                                              flux + flux_err[:, 0])

    def _get_flux_values(self, prefix='Flux', unit='cm-2 s-1'):
        if prefix not in ['Flux', 'Unc_Flux', 'nuFnu']:
            raise ValueError("Must be one of the following: 'Flux', 'Unc_Flux', 'nuFnu'")

        values = [self.data[prefix + _] for _ in self._ebounds_suffix]
        return Quantity(values, unit)


    def plot_lightcurve(self, ax=None):
        """Plot lightcurve.
        """
        from gammapy.time import plot_fermi_3fgl_light_curve

        ax = plot_fermi_3fgl_light_curve(self.name, ax=ax)
        return ax


class SourceCatalogObject2FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 2FHL catalog.
    """
    _ebounds = EnergyBounds([50, 171, 585, 2000], 'GeV')
    _ebounds_suffix = ['50_171', '171_585', '585_2000']
    energy_range = Quantity([0.05, 2], 'TeV')
    """Energy range of the Fermi 2FHL source catalog"""

    def __str__(self):
        """Print default summary info string"""
        return self.summary()

    def summary(self):
        """Print summary info."""
        # TODO: can we share code with 3FGL summary funtion?
        d = self.data

        ss = 'Source: {}\n'.format(d['Source_Name'])
        ss += '\n'

        ss += 'RA (J2000)  : {}\n'.format(d['RAJ2000'])
        ss += 'Dec (J2000) : {}\n'.format(d['DEJ2000'])
        ss += 'GLON        : {}\n'.format(d['GLON'])
        ss += 'GLAT        : {}\n'.format(d['GLAT'])
        ss += '\n'

        # val, err = d['Energy_Flux100'], d['Unc_Energy_Flux100']
        # ss += 'Energy flux (100 MeV - 100 GeV) : {} +- {} erg cm^-2 s^-1\n'.format(val, err)
        # ss += 'Detection significance : {}\n'.format(d['Signif_Avg'])

        return ss

    def _get_flux_values(self, prefix='Flux', unit='cm-2 s-1'):
        if prefix not in ['Flux', 'Unc_Flux']:
            raise ValueError("Must be one of the following: 'Flux', 'Unc_Flux'")

        values = [self.data[prefix + _ + 'GeV'] for _ in self._ebounds_suffix]
        return Quantity(values, unit)

    @property
    def flux_points(self):
        """
        Differential flux points (`~gammapy.spectrum.DifferentialFluxPoints`).
        """
        int_flux_points = self.flux_points_integral
        gamma = self.data['Spectral_Index']
        return int_flux_points.to_differential_flux_points(
            x_method='log_center',
            spectral_index=gamma,
        )

    @property
    def flux_points_integral(self):
        """
        Integral flux points (`~gammapy.spectrum.IntegralFluxPoints`).
        """
        flux = self._get_flux_values()
        flux_err = self._get_flux_values('Unc_Flux')
        return IntegralFluxPoints.from_arrays(self._ebounds, flux, flux + flux_err[:, 1],
                                              flux + flux_err[:, 0])

    @property
    def spectral_model(self):
        """
        Best fit spectral model `~gammapy.spectrum.SpectralModel`.
        """
        emin, emax = self.energy_range
        g = Quantity(self.data['Spectral_Index'], '')

        # The pivot energy information is missing in the 2FHL catalog. Set it to
        # 100 GeV per default.
        ref = Quantity(100, 'GeV')

        pars = {}
        flux = Quantity(self.data['Flux50'], 'cm-2 s-1')
        pars['amplitude'] = power_law_flux(flux, g, ref, emin, emax).to('cm-2 s-1 GeV-1')
        pars['reference'] = ref
        pars['index'] = g
        return PowerLaw(**pars)


class SourceCatalog3FGL(SourceCatalog):
    """Fermi-LAT 3FGL source catalog.
    """
    name = '3fgl'
    description = 'LAT 4-year point source catalog'
    source_object_class = SourceCatalogObject3FGL

    def __init__(self, filename=None):
        if not filename:
            filename = gammapy_extra.filename('datasets/catalogs/fermi/gll_psc_v16.fit.gz')

        self.hdu_list = fits.open(filename)
        self.extended_sources_table = Table(self.hdu_list['ExtendedSources'].data)

        table = Table(self.hdu_list['LAT_Point_Source_Catalog'].data)
        super(SourceCatalog3FGL, self).__init__(table=table)


class SourceCatalog2FHL(SourceCatalog):
    """Fermi-LAT 2FHL source catalog.
    """
    name = '2fhl'
    description = 'LAT second high-energy source catalog'
    source_object_class = SourceCatalogObject2FHL

    def __init__(self, filename=None):
        if not filename:
            filename = gammapy_extra.filename('datasets/catalogs/fermi/gll_psch_v08.fit.gz')

        self.hdu_list = fits.open(filename)
        self.count_map_hdu = self.hdu_list['Count Map']
        self.extended_sources_table = Table(self.hdu_list['Extended Sources'].data)
        self.rois = Table(self.hdu_list['ROIs'].data)

        table = Table(self.hdu_list['2FHL Source Catalog'].data)
        super(SourceCatalog2FHL, self).__init__(table=table)
