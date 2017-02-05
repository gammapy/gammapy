# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fermi catalog and source classes.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tarfile
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from astropy.utils.data import download_file
from astropy.units import Quantity
from ..utils.energy import EnergyBounds
from ..spectrum import (
    FluxPoints,
    SpectrumFitResult,
    compute_flux_points_dnde
)
from ..spectrum.models import (PowerLaw, PowerLaw2, ExponentialCutoffPowerLaw,
                               ExponentialCutoffPowerLaw3FGL, LogParabola)
from ..datasets import gammapy_extra
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    'fetch_fermi_catalog',
    'fetch_fermi_extended_sources',
    'SourceCatalog1FHL',
    'SourceCatalog2FHL',
    'SourceCatalog3FGL',
    'SourceCatalog3FHL',
    'SourceCatalogObject1FHL',
    'SourceCatalogObject2FHL',
    'SourceCatalogObject3FGL',
    'SourceCatalogObject3FHL',
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
    _ebounds_suffix = ['100_300', '300_1000',
                       '1000_3000', '3000_10000', '10000_100000']
    energy_range = Quantity([100, 100000], 'MeV')
    """Energy range of the catalog.

    Paper says that analysis uses data up to 300 GeV,
    but results are all quoted up to 100 GeV only to
    be consistent with previous catalogs.
    """

    def __str__(self):
        """Summary info string."""
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
        pars['amplitude'] = Quantity(
            self.data['Flux_Density'], 'MeV-1 cm-2 s-1')
        pars['reference'] = Quantity(self.data['Pivot_Energy'], 'MeV')

        if spec_type == 'PowerLaw':
            pars['index'] = Quantity(self.data['Spectral_Index'], '')
            return PowerLaw(**pars)

        elif spec_type == 'PLExpCutoff':
            pars['index'] = Quantity(self.data['Spectral_Index'], '')
            pars['ecut'] = Quantity(self.data['Cutoff'], 'MeV')
            return ExponentialCutoffPowerLaw3FGL(**pars)

        elif spec_type == 'LogParabola':
            pars['alpha'] = Quantity(self.data['Spectral_Index'], '')
            pars['beta'] = Quantity(self.data['beta'], '')
            return LogParabola(**pars)

        elif spec_type == "PLSuperExpCutoff":
            # TODO Implement super exponential cut off
            raise NotImplementedError

        else:
            raise ValueError(
                'Spectral model {} not available'.format(spec_type))

    @property
    def flux_points(self):
        """
        Differential flux points (`~gammapy.spectrum.FluxPoints`).
        """
        table = Table()
        table.meta['SED_TYPE'] = 'flux'
        e_ref = self._ebounds.log_centers
        table['e_ref'] = e_ref
        table['e_min'] = self._ebounds.lower_bounds
        table['e_max'] = self._ebounds.upper_bounds

        flux = self._get_flux_values()
        flux_err = self._get_flux_values('Unc_Flux')
        table['flux'] = flux
        table['flux_errn'] = np.abs(flux_err[:, 0])
        table['flux_errp'] = flux_err[:, 1]

        nuFnu = self._get_flux_values('nuFnu', 'erg cm-2 s-1')
        table['eflux'] = nuFnu
        table['eflux_errn'] = np.abs(nuFnu * flux_err[:, 0] / flux)
        table['eflux_errp'] = nuFnu * flux_err[:, 1] / flux

        is_ul = np.isnan(table['flux_errn'])
        table['is_ul'] = is_ul

        # handle upper limits
        table['flux_ul'] = np.nan * flux_err.unit
        table['flux_ul'][is_ul] = table['flux_errp'][is_ul]

        for column in ['flux', 'flux_errp', 'flux_errn']:
            table[column][is_ul] = np.nan

        # handle upper limits
        table['eflux_ul'] = np.nan * nuFnu.unit
        table['eflux_ul'][is_ul] = table['eflux_errp'][is_ul]

        for column in ['eflux', 'eflux_errp', 'eflux_errn']:
            table[column][is_ul] = np.nan

        table['dnde'] = (nuFnu * e_ref ** -2).to('TeV-1 cm-2 s-1')
        return FluxPoints(table)

    @property
    def spectrum(self):
        """Spectrum model fit result (`~gammapy.spectrum.SpectrumFitResult`)
        """
        data = self.data
        model = self.spectral_model

        spec_type = self.data['SpectrumType'].strip()

        if spec_type == 'PowerLaw':
            par_names = ['index', 'amplitude']
            par_errs = [data['Unc_Spectral_Index'],
                        data['Unc_Flux_Density']]
        elif spec_type == 'PLExpCutoff':
            par_names = ['index', 'amplitude', 'ecut']
            par_errs = [data['Unc_Spectral_Index'],
                        data['Unc_Flux_Density'],
                        data['Unc_Cutoff']]
        elif spec_type == 'LogParabola':
            par_names = ['amplitude', 'alpha', 'beta']
            par_errs = [data['Unc_Flux_Density'],
                        data['Unc_Spectral_Index'],
                        data['Unc_beta']]
        elif spec_type == "PLSuperExpCutoff":
            # TODO Implement super exponential cut off
            raise NotImplementedError
        else:
            raise ValueError(
                'Spectral model {} not available'.format(spec_type))

        covariance = np.diag(par_errs) ** 2

        return SpectrumFitResult(
            model=model,
            fit_range=self.energy_range,
            covariance=covariance,
            covar_axis=par_names,
        )

    def _get_flux_values(self, prefix='Flux', unit='cm-2 s-1'):
        if prefix not in ['Flux', 'Unc_Flux', 'nuFnu']:
            raise ValueError(
                "Must be one of the following: 'Flux', 'Unc_Flux', 'nuFnu'")

        values = [self.data[prefix + _] for _ in self._ebounds_suffix]
        return Quantity(values, unit)

    def plot_lightcurve(self, ax=None):
        """Plot lightcurve.
        """
        # TODO: move that function here and change to method
        # that returns a `gammapy.time.LightCurve` object
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
            raise ValueError(
                "Must be one of the following: 'Flux', 'Unc_Flux'")

        values = [self.data[prefix + _ + 'GeV'] for _ in self._ebounds_suffix]
        return Quantity(values, unit)

    @property
    def flux_points(self):
        """
        Integral flux points (`~gammapy.spectrum.FluxPoints`).
        """
        table = Table()
        table.meta['SED_TYPE'] = 'flux'
        table['e_min'] = self._ebounds.lower_bounds
        table['e_max'] = self._ebounds.upper_bounds
        table['flux'] = self._get_flux_values()
        flux_err = self._get_flux_values('Unc_Flux')
        table['flux_errn'] = np.abs(flux_err[:, 0])
        table['flux_errp'] = flux_err[:, 1]

        # handle upper limits
        is_ul = np.isnan(table['flux_errn'])
        table['is_ul'] = is_ul
        table['flux_ul'] = np.nan * flux_err.unit
        table['flux_ul'][is_ul] = table['flux_errp'][is_ul]

        for column in ['flux', 'flux_errp', 'flux_errn']:
            table[column][is_ul] = np.nan

        flux_points = FluxPoints(table)

        flux_points_dnde = compute_flux_points_dnde(
            flux_points, model=self.spectral_model)
        return flux_points_dnde

    @property
    def spectral_model(self):
        """
        Best fit spectral model `~gammapy.spectrum.models.SpectralModel`.
        """
        emin, emax = self.energy_range
        g = Quantity(self.data['Spectral_Index'], '')

        pars = {}
        pars['amplitude'] = Quantity(self.data['Flux50'], 'cm-2 s-1')
        pars['emin'], pars['emax'] = self.energy_range
        pars['index'] = g
        return PowerLaw2(**pars)

    @property
    def spectrum(self):
        """Spectrum information (`~gammapy.spectrum.SpectrumFitResult`)
        """
        data = self.data
        model = self.spectral_model

        covariance = np.diag([
            data['Unc_Spectral_Index'] ** 2,
            data['Unc_Flux50'] ** 2,
        ])

        covar_axis = ['index', 'amplitude']

        fit = SpectrumFitResult(
            model=model,
            fit_range=self.energy_range,
            covariance=covariance,
            covar_axis=covar_axis,
        )

        return fit


class SourceCatalog3FGL(SourceCatalog):
    """Fermi-LAT 3FGL source catalog.
    """
    name = '3fgl'
    description = 'LAT 4-year point source catalog'
    source_object_class = SourceCatalogObject3FGL

    def __init__(self, filename=None):
        if not filename:
            filename = gammapy_extra.filename(
                'datasets/catalogs/fermi/gll_psc_v16.fit.gz')

        self.hdu_list = fits.open(filename)
        self.extended_sources_table = Table(
            self.hdu_list['ExtendedSources'].data)

        table = Table(self.hdu_list['LAT_Point_Source_Catalog'].data)

        source_name_key = 'Source_Name'
        source_name_alias = ('Extended_Source_Name', '0FGL_Name', '1FGL_Name',
                             '2FGL_Name', '1FHL_Name', 'ASSOC_TEV', 'ASSOC1',
                             'ASSOC2')
        super(SourceCatalog3FGL, self).__init__(table=table,
                                                source_name_key=source_name_key,
                                                source_name_alias=source_name_alias)


class SourceCatalog2FHL(SourceCatalog):
    """Fermi-LAT 2FHL source catalog.
    """
    name = '2fhl'
    description = 'LAT second high-energy source catalog'
    source_object_class = SourceCatalogObject2FHL

    def __init__(self, filename=None):
        if not filename:
            filename = gammapy_extra.filename(
                'datasets/catalogs/fermi/gll_psch_v08.fit.gz')

        self.hdu_list = fits.open(filename)
        self.count_map_hdu = self.hdu_list['Count Map']
        self.extended_sources_table = Table(
            self.hdu_list['Extended Sources'].data)
        self.rois = Table(self.hdu_list['ROIs'].data)
        table = Table(self.hdu_list['2FHL Source Catalog'].data)

        source_name_key = 'Source_Name'
        source_name_alias = ('ASSOC', '3FGL_Name', '1FHL_Name', 'TeVCat_Name')
        super(SourceCatalog2FHL, self).__init__(table=table,
                                                source_name_key=source_name_key,
                                                source_name_alias=source_name_alias)


class SourceCatalogObject1FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 1FHL catalog.
    """
    _ebounds = EnergyBounds([10, 30, 100, 500], 'GeV')
    _ebounds_suffix = ['10_30', '30_100', '100_500']
    energy_range = Quantity([0.01, 0.5], 'TeV')
    """Energy range of the Fermi 1FHL source catalog"""

    def __str__(self):
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
            raise ValueError(
                "Must be one of the following: 'Flux', 'Unc_Flux'")

        values = [self.data[prefix + _ + 'GeV'] for _ in self._ebounds_suffix]
        return Quantity(values, unit)

    @property
    def flux_points(self):
        """
        Integral flux points (`~gammapy.spectrum.FluxPoints`).
        """
        table = Table()
        table.meta['SED_TYPE'] = 'flux'
        table['e_min'] = self._ebounds.lower_bounds
        table['e_max'] = self._ebounds.upper_bounds
        table['flux'] = self._get_flux_values()
        flux_err = self._get_flux_values('Unc_Flux')
        table['flux_errn'] = np.abs(flux_err[:, 0])
        table['flux_errp'] = flux_err[:, 1]

        # handle upper limits
        is_ul = np.isnan(table['flux_errn'])
        table['is_ul'] = is_ul
        table['flux_ul'] = np.nan * flux_err.unit
        table['flux_ul'][is_ul] = table['flux_errp'][is_ul]

        for column in ['flux', 'flux_errp', 'flux_errn']:
            table[column][is_ul] = np.nan

        flux_points = FluxPoints(table)

        flux_points_dnde = compute_flux_points_dnde(
            flux_points, model=self.spectral_model)
        return flux_points_dnde

    @property
    def spectral_model(self):
        """
        Best fit spectral model `~gammapy.spectrum.models.SpectralModel`.
        """
        emin, emax = self.energy_range
        g = Quantity(self.data['Spectral_Index'], '')

        pars = {}
        pars['amplitude'] = Quantity(self.data['Flux'], 'cm-2 s-1')
        pars['emin'], pars['emax'] = self.energy_range
        pars['index'] = g
        return PowerLaw2(**pars)

    @property
    def spectrum(self):
        """Spectrum information (`~gammapy.spectrum.SpectrumFitResult`)
        """
        data = self.data
        model = self.spectral_model

        covariance = np.diag([
            data['Unc_Spectral_Index'] ** 2,
            data['Unc_Flux'] ** 2,
        ])

        covar_axis = ['index', 'amplitude']

        fit = SpectrumFitResult(
            model=model,
            fit_range=self.energy_range,
            covariance=covariance,
            covar_axis=covar_axis,
        )

        return fit


class SourceCatalog1FHL(SourceCatalog):
    """Fermi-LAT 1FHL source catalog."""
    name = '1fhl'
    description = 'First Fermi-LAT Catalog of Sources above 10 GeV'
    source_object_class = SourceCatalogObject1FHL

    def __init__(self, filename=None):
        if not filename:
            filename = gammapy_extra.filename(
                'datasets/catalogs/fermi/gll_psch_v07.fit.gz')

        self.hdu_list = fits.open(filename)
        # self.count_map_hdu = self.hdu_list['Count Map']
        self.extended_sources_table = Table(
            self.hdu_list['ExtendedSources'].data)
        table = Table(self.hdu_list['LAT_Point_Source_Catalog'].data)

        source_name_key = 'Source_Name'
        source_name_alias = ('ASSOC1', 'ASSOC2', 'ASSOC_TEV', 'ASSOC_GAM')
        super(SourceCatalog1FHL, self).__init__(table=table,
                                                source_name_key=source_name_key,
                                                source_name_alias=source_name_alias)



class SourceCatalogObject3FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 3FHL catalog.
    """
    _ebounds = EnergyBounds([10, 20, 50, 150, 500, 2000], 'GeV')
    _ebounds_suffix = ['10_20', '20_50', '50_150', '150_500', '500_2000']
    energy_range = Quantity([0.01, 2], 'TeV')
    """Energy range of the Fermi 1FHL source catalog"""

    def __str__(self):
        """Print summary info."""
        d = self.data

        ss = 'Source: {}\n'.format(d['Source_Name'])
        ss += '\n'

        ss += 'RA (J2000)  : {}\n'.format(d['RAJ2000'])
        ss += 'Dec (J2000) : {}\n'.format(d['DEJ2000'])
        ss += 'GLON        : {}\n'.format(d['GLON'])
        ss += 'GLAT        : {}\n'.format(d['GLAT'])
        ss += '\n'
        ss += 'Detection significance : {}\n'.format(d['Signif_Avg'])

        return ss

    def _get_flux_values(self, prefix='Flux', unit='cm-2 s-1'):
        if prefix not in ['Flux', 'Unc_Flux']:
            raise ValueError(
                "Must be one of the following: 'Flux', 'Unc_Flux'")

        values = [self.data[prefix + _ + 'GeV'] for _ in self._ebounds_suffix]
        return Quantity(values, unit)


    @property
    def spectral_model(self):
        """
        Best fit spectral model `~gammapy.spectrum.models.SpectralModel`.
        """
        spec_type = self.data['SpectrumType'].strip()
        pars = {}
        pars['amplitude'] = Quantity(
            self.data['Flux_Density'], 'GeV-1 cm-2 s-1')
        pars['reference'] = Quantity(self.data['Pivot_Energy'], 'GeV')

        if spec_type == 'PowerLaw':
            pars['index'] = Quantity(self.data['Spectral_Index'], '')
            return PowerLaw(**pars)

        elif spec_type == 'LogParabola':
            pars['alpha'] = Quantity(self.data['Spectral_Index'], '')
            pars['beta'] = Quantity(self.data['beta'], '')
            return LogParabola(**pars)

        else:
            raise ValueError(
                'Spectral model {} not available'.format(spec_type))

    @property
    def flux_points(self):
        """
        Differential flux points (`~gammapy.spectrum.FluxPoints`).
        """
        table = Table()
        table.meta['SED_TYPE'] = 'flux'
        e_ref = self._ebounds.log_centers
        table['e_ref'] = e_ref
        table['e_min'] = self._ebounds.lower_bounds
        table['e_max'] = self._ebounds.upper_bounds

        flux = self._get_flux_values()
        flux_err = self._get_flux_values('Unc_Flux')
        table['flux'] = flux
        table['flux_errn'] = np.abs(flux_err[:, 0])
        table['flux_errp'] = flux_err[:, 1]

        nuFnu = self._get_flux_values('nuFnu', 'erg cm-2 s-1')
        table['eflux'] = nuFnu
        table['eflux_errn'] = np.abs(nuFnu * flux_err[:, 0] / flux)
        table['eflux_errp'] = nuFnu * flux_err[:, 1] / flux

        is_ul = np.isnan(table['flux_errn'])
        table['is_ul'] = is_ul

        # handle upper limits
        table['flux_ul'] = np.nan * flux_err.unit
        table['flux_ul'][is_ul] = table['flux_errp'][is_ul]

        for column in ['flux', 'flux_errp', 'flux_errn']:
            table[column][is_ul] = np.nan

        # handle upper limits
        table['eflux_ul'] = np.nan * nuFnu.unit
        table['eflux_ul'][is_ul] = table['eflux_errp'][is_ul]

        for column in ['eflux', 'eflux_errp', 'eflux_errn']:
            table[column][is_ul] = np.nan

        table['dnde'] = (nuFnu * e_ref ** -2).to('TeV-1 cm-2 s-1')
        return FluxPoints(table)
    
    @property
    def spectrum(self):
        """Spectrum model fit result (`~gammapy.spectrum.SpectrumFitResult`)
        """
        data = self.data
        model = self.spectral_model

        spec_type = self.data['SpectrumType'].strip()

        if spec_type == 'PowerLaw':
            par_names = ['index', 'amplitude']
            par_errs = [data['Unc_Spectral_Index'],
                        data['Unc_Flux_Density']]
        elif spec_type == 'LogParabola':
            par_names = ['amplitude', 'alpha', 'beta']
            par_errs = [data['Unc_Flux_Density'],
                        data['Unc_Spectral_Index'],
                        data['Unc_beta']]
        else:
            raise ValueError(
                'Spectral model {} not available'.format(spec_type))

        covariance = np.diag(par_errs) ** 2

        return SpectrumFitResult(
            model=model,
            fit_range=self.energy_range,
            covariance=covariance,
            covar_axis=par_names,
        )


    def _get_flux_values(self, prefix='Flux', unit='cm-2 s-1'):
        if prefix not in ['Flux', 'Unc_Flux', 'nuFnu']:
            raise ValueError(
                "Must be one of the following: 'Flux', 'Unc_Flux', 'nuFnu'")

        values = [self.data[prefix + _ + 'GeV'] for _ in self._ebounds_suffix]
        return Quantity(values, unit)


class SourceCatalog3FHL(SourceCatalog):
    """Fermi-LAT 3FHL source catalog.
    """
    name = '3fhl'
    description = 'LAT third high-energy source catalog'
    source_object_class = SourceCatalogObject3FHL

    def __init__(self, filename=None):
        if not filename:
            filename = gammapy_extra.filename(
                'datasets/catalogs/fermi/gll_psch_v11.fit.gz')

        self.hdu_list = fits.open(filename)
        self.extended_sources_table = Table(
            self.hdu_list['ExtendedSources'].data)
        self.rois = Table(self.hdu_list['ROIs'].data)
        table = Table(self.hdu_list['LAT_Point_Source_Catalog'].data)
        # Definition of energy bounds
        self.energy_bounds_table = Table(self.hdu_list['EnergyBounds'].data)

        # Add integrated flux columns (defined in the same way as in the
        # other Fermi catalogs (e.g. FluxY_ZGeV))
        for i, band in enumerate(self.energy_bounds_table):
            col_flux_name = 'Flux{:d}_{:d}GeV'.format(int(band['LowerEnergy']),
                                                      int(band['UpperEnergy']))
            col_flux_value = table['Flux_Band'][:,i].data
            col_flux = Column(col_flux_value, name=col_flux_name)

            col_unc_flux_name = 'Unc_' + col_flux_name
            col_unc_flux_value = table['Unc_Flux_Band'][:,i].data
            col_unc_flux = Column(col_unc_flux_value, name=col_unc_flux_name)

            col_nufnu_name = 'nuFnu{:d}_{:d}GeV'.format(int(band['LowerEnergy']),
                                                        int(band['UpperEnergy']))
            col_nufnu_value = table['nuFnu'][:,i].data
            col_nufnu = Column(col_nufnu_value, name=col_nufnu_name)

            table.add_column(col_flux)
            table.add_column(col_unc_flux)
            table.add_column(col_nufnu)
            
        source_name_key = 'Source_Name'
        source_name_alias = ('ASSOC1', 'ASSOC2', 'ASSOC_TEV', 'ASSOC_GAM')
        super(SourceCatalog3FHL, self).__init__(table=table,
                                                source_name_key=source_name_key,
                                                source_name_alias=source_name_alias)
