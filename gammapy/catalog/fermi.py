# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fermi catalog and source classes.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tarfile
import numpy as np
from astropy.io import fits
from astropy.table import QTable, Table, Column
from astropy.time import Time
from astropy.utils.data import download_file
from astropy.units import Quantity, Unit
from ..time import LightCurve
from ..utils.scripts import make_path
from ..utils.energy import EnergyBounds
from ..spectrum import (
    FluxPoints,
    SpectrumFitResult,
    compute_flux_points_dnde
)
from ..spectrum.models import (
    PowerLaw,
    PowerLaw2,
    ExponentialCutoffPowerLaw3FGL,
    LogParabola,
)
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    'SourceCatalogObject3FGL',
    'SourceCatalogObject1FHL',
    'SourceCatalogObject2FHL',
    'SourceCatalogObject3FHL',
    'SourceCatalog3FGL',
    'SourceCatalog1FHL',
    'SourceCatalog2FHL',
    'SourceCatalog3FHL',
    'fetch_fermi_catalog',
    'fetch_fermi_extended_sources',
]


class SourceCatalogObject3FGL(SourceCatalogObject):
    """
    One source from the Fermi-LAT 3FGL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog3FGL`.
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

    def __str__(self, info='all'):
        """
        Summary info string.

        Parameters
        ----------
        info : {'all', 'basic', 'position', 'spectral', 'other'}
            Comma separated list of options
        """

        if info == 'all':
            info = 'basic,position,spectral,other'

        ss = ''
        ops = info.split(',')
        if 'basic' in ops:
            ss += self._info_basic()
        if 'position' in ops:
            ss += self._info_position()
        if 'spectral' in ops:
            ss += self._info_spectral()
        if 'other' in ops:
            ss += self._info_other()
        return ss

    def _info_basic(self):
        """Print basic info."""
        d = self.data
        ss = '\n*** Basic info ***\n\n'
        ss += '{:<20s}           : {}\n'.format('Source', d['Source_Name'])
        ss += '{:<20s} : {}\n'.format('Catalog row index (zero-based)', d['catalog_row_index'])
        ss += '{:<20s}           : {}\n'.format('Extended name', d['Extended_Source_Name'])

        associations = []
        if d['ASSOC1'].isspace() == False:
            associations.append(d['ASSOC1'].rstrip())
        if d['ASSOC2'].isspace() == False:
            associations.append(d['ASSOC2'].rstrip())
        if d['ASSOC_TEV'].isspace() == False:
            associations.append(d['ASSOC_TEV'].rstrip())
        if d['ASSOC_GAM1'].isspace() == False:
            associations.append(d['ASSOC_GAM1'].rstrip())
        if d['ASSOC_GAM2'].isspace() == False:
            associations.append(d['ASSOC_GAM2'].rstrip())
        if d['ASSOC_GAM3'].isspace() == False:
            associations.append(d['ASSOC_GAM3'].rstrip())

        associations = ', '.join(associations)
        ss += '{:<20s}           : {}\n'.format('Associations', associations)

        otherNames = []
        if d['0FGL_Name'].isspace() == False:
            otherNames.append(d['0FGL_Name'].rstrip())
        if d['1FGL_Name'].isspace() == False:
            otherNames.append(d['1FGL_Name'].rstrip())
        if d['2FGL_Name'].isspace() == False:
            otherNames.append(d['2FGL_Name'].rstrip())
        if d['1FHL_Name'].isspace() == False:
            otherNames.append(d['1FHL_Name'].rstrip())

        otherNames = ', '.join(otherNames)
        ss += '{:<20s}           : {}\n'.format('Other names', otherNames)

        ss += '{:<20s}           : {}\n'.format('Class', d['CLASS1'])

        return ss

    def _info_position(self):
        """Print position info."""
        d = self.data
        ss = '\n*** Position info ***\n\n'
        ss += '{:<20s} : {:.3f} deg\n'.format('RA (J2000)', d['RAJ2000'])
        ss += '{:<20s} : {:.3f} deg\n'.format('Dec (J2000)', d['DEJ2000'])
        ss += '{:<20s} : {:.3f} deg\n'.format('GLON', d['GLON'])
        ss += '{:<20s} : {:.3f} deg\n'.format('GLAT', d['GLAT'])

        ss += '\n'
        ss += '{:<20s} : {:.0f}\n'.format('ROI number', d['ROI_num'])
        ss += '{:<20s} : {:.4f} deg\n'.format('Semimajor (68%)', d['Conf_68_SemiMajor'])
        ss += '{:<20s} : {:.4f} deg\n'.format('Semiminor (68%)', d['Conf_68_SemiMinor'])
        ss += '{:<20s} : {:.2f} deg\n'.format('Position angle (68%)', d['Conf_68_PosAng'])

        ss += '{:<20s} : {:.4f} deg\n'.format('Semimajor (95%)', d['Conf_95_SemiMajor'])
        ss += '{:<20s} : {:.4f} deg\n'.format('Semiminor (95%)', d['Conf_95_SemiMinor'])
        ss += '{:<20s} : {:.2f} deg\n'.format('Position angle (95%)', d['Conf_95_PosAng'])

        return ss

    def _info_spectral(self):
        """Print spectral info."""
        d = self.data
        ss = '\n*** Spectral info ***\n\n'
        ss += '{:<20s}            : {:.3} +- {:.3} erg cm^-2 s^-1\n'.format('Energy flux (100 MeV - 100 GeV)',
                                                                            d['Energy_Flux100'],
                                                                            d['Unc_Energy_Flux100'])
        ss += '{:<20s} : {:.3f} Sigma\n'.format('Detection significance (100 MeV - 300 GeV)', d['Signif_Avg'])
        ss += '{:<20s}                       : {}\n'.format('Spectrum type', d['SpectrumType'])
        if d['SpectrumType'].rstrip() == 'LogParabola':
            ss += '{:<20s}                       : {} +- {}\n'.format('beta', d['beta'], d['Unc_beta'])
        if d['SpectrumType'].rstrip() in ['PLExpCutoff', 'PlSuperExpCutoff']:
            ss += '{:<20s}                       : {:.0f} +- {:.0f} MeV\n'.format('Cutoff energy',
                                                                                  d['Cutoff'], d['Unc_Cutoff'])
        if d['SpectrumType'].rstrip() == 'PLSuperExpCutoff':
            ss += '{:<20s}                       : {} +- {}\n'.format('Exponential index', d['Exp_Index'],
                                                                      d['Unc_Exp_Index'])
        ss += '{:<20s}                       : {:.3f}\n'.format('Power law index', d['PowerLaw_Index'])

        ss += '{:<20s}                       : {:.3f} +- {:.3f}\n'.format('Spectral index', d['Spectral_Index'],
                                                                          d['Unc_Spectral_Index'])
        ss += '{:<20s}                       : {:.0f} MeV\n'.format('Pivot energy', d['Pivot_Energy'])
        ss += '{:<20s}           : {:.3} +- {:.3} cm^-2 MeV^-1 s^-1\n'.format('Flux Density (100 MeV - 100 GeV)',
                                                                              d['Flux_Density'],
                                                                              d['Unc_Flux_Density'])
        ss += '{:<20s}                : {:.3} +- {:.3} cm^-2 s^-1\n'.format('Integral flux (1 - 100 GeV)',
                                                                            d['Flux1000'], d['Unc_Flux1000'])
        ss += '{:<20s}                     : {:.1f}\n'.format('Significance curvature', d['Signif_Curve'])

        return ss

    def _info_other(self):
        """
        Other items - I'm not sure if they belong in __str__ or not. Or I'm not sure what they are.
        """
        d = self.data
        ss = '\n*** Other info (omitted) ***\n\n'

        ss += 'Flux<energy range>\n'
        ss += 'Unc_Flux<energy range>\n'
        ss += 'nuFnu<energy range>\n'
        ss += 'Sqrt_TS<energy range>\n'
        # I think the above are all spectral points info? If so, they shouldn't be in __str__

        ss += 'Variability_Index\n'
        ss += 'Flux_Peak\n'
        ss += 'Unc_Flux_Peak\n'
        ss += 'Time_Peak\n'
        ss += 'Peak_Interval\n'
        ss += 'Flux_History\n'
        ss += 'Unc_Flux_History\n'

        ss += 'TEVCAT_FLAG\n'
        ss += 'Flags\n'

        return ss

    @property
    def spectral_model(self):
        """
        Best fit spectral model `~gammapy.spectrum.SpectralModel`.
        """
        spec_type = self.data['SpectrumType'].strip()
        pars, errs = {}, {}
        pars['amplitude'] = Quantity(self.data['Flux_Density'], 'cm-2 s-1 MeV-1')
        errs['amplitude'] = Quantity(self.data['Unc_Flux_Density'], 'cm-2 s-1 MeV-1')
        pars['reference'] = Quantity(self.data['Pivot_Energy'], 'MeV')

        if spec_type == 'PowerLaw':
            pars['index'] = Quantity(self.data['Spectral_Index'], '')
            errs['index'] = Quantity(self.data['Unc_Spectral_Index'], '')
            model = PowerLaw(**pars)
        elif spec_type == 'PLExpCutoff':
            pars['index'] = Quantity(self.data['Spectral_Index'], '')
            pars['ecut'] = Quantity(self.data['Cutoff'], 'MeV')
            errs['index'] = Quantity(self.data['Unc_Spectral_Index'], '')
            errs['ecut'] = Quantity(self.data['Unc_Cutoff'], 'MeV')
            model = ExponentialCutoffPowerLaw3FGL(**pars)
        elif spec_type == 'LogParabola':
            pars['alpha'] = Quantity(self.data['Spectral_Index'], '')
            pars['beta'] = Quantity(self.data['beta'], '')
            errs['alpha'] = Quantity(self.data['Unc_Spectral_Index'], '')
            errs['beta'] = Quantity(self.data['Unc_beta'], '')
            model = LogParabola(**pars)
        elif spec_type == "PLSuperExpCutoff":
            # TODO Implement super exponential cut off
            raise NotImplementedError
        else:
            raise ValueError('Spectral model {} not available'.format(spec_type))

        model.parameters.set_parameter_errors(errs)
        return model
    
    @property
    def flux_points(self):
        """
        Flux points (`~gammapy.spectrum.FluxPoints`).
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

    def _get_flux_values(self, prefix='Flux', unit='cm-2 s-1'):
        if prefix not in ['Flux', 'Unc_Flux', 'nuFnu']:
            raise ValueError(
                "Must be one of the following: 'Flux', 'Unc_Flux', 'nuFnu'")

        values = [self.data[prefix + _] for _ in self._ebounds_suffix]
        return Quantity(values, unit)

    @property
    def lightcurve(self):
        """Lightcurve (`~gammapy.time.LightCurve`).
        """
        flux = self.data['Flux_History']

        # Flux error is given as asymmetric high/low
        flux_err_lo = self.data['Unc_Flux_History'][:, 0]
        flux_err_hi = self.data['Unc_Flux_History'][:, 1]

        # TODO: Change lightcurve class to support this,
        # then fill appropriately here
        # for now, we just use the mean
        flux_err = 0.5 * (-flux_err_lo + flux_err_hi)
        flux_unit = Unit('cm-2 s-1')

        # Really the time binning is stored in a separate HDU in the FITS
        # catalog file called `Hist_Start`, with a single column `Hist_Start`
        # giving the time binning in MET (mission elapsed time)
        # This is not available here for now.
        # TODO: read that info in `SourceCatalog3FGL` and pass it down to the
        # `SourceCatalogObject3FGL` object somehow.

        # For now, we just hard-code the start and stop time and assume
        # equally-spaced time intervals. This is roughly correct,
        # for plotting the difference doesn't matter, only for analysis
        time_start = Time('2008-08-02T00:33:19')
        time_end = Time('2012-07-31T22:45:47')

        n_points = len(flux)
        time_step = (time_end - time_start) / n_points
        time_bounds = time_start + np.arange(n_points + 1) * time_step
        table = QTable()
        table['TIME_MIN'] = time_bounds[:-1]
        table['TIME_MAX'] = time_bounds[1:]
        table['FLUX'] = flux * flux_unit
        table['FLUX_ERR'] = flux_err * flux_unit
        lc = LightCurve(table)
        return lc


class SourceCatalogObject1FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 1FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog1FHL`.
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
        table['e_max'] =    self._ebounds.upper_bounds
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
        pars, errs = {}, {}
        pars['amplitude'] = Quantity(self.data['Flux'], 'cm-2 s-1')
        pars['emin'], pars['emax'] = self.energy_range
        pars['index'] = Quantity(self.data['Spectral_Index'], '')
        errs['amplitude'] = Quantity(self.data['Unc_Flux'], 'cm-2 s-1')
        errs['index'] = Quantity(self.data['Unc_Spectral_Index'], '')
        pwl = PowerLaw2(**pars)
        pwl.parameters.set_parameter_errors(errs)
        return pwl


class SourceCatalogObject2FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 2FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog2FHL`.
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
            raise ValueError("Must be one of the following: 'Flux', 'Unc_Flux'")

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
        pars, errs = {}, {}
        pars['amplitude'] = Quantity(self.data['Flux50'], 'cm-2 s-1')
        pars['emin'], pars['emax'] = self.energy_range
        pars['index'] = Quantity(self.data['Spectral_Index'], '')
        errs['amplitude'] = Quantity(self.data['Unc_Flux50'], 'cm-2 s-1')
        errs['index'] = Quantity(self.data['Unc_Spectral_Index'], '')
        pwl = PowerLaw2(**pars)
        pwl.parameters.set_parameter_errors(errs)
        return pwl


class SourceCatalogObject3FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 3FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog3FHL`.
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

    @property
    def spectral_model(self):
        """
        Best fit spectral model `~gammapy.spectrum.models.SpectralModel`.
        """
        spec_type = self.data['SpectrumType'].strip()
        pars, errs = {}, {}
        pars['amplitude'] = Quantity(self.data['Flux_Density'], 'cm-2 s-1 GeV-1')
        errs['amplitude'] = Quantity(self.data['Unc_Flux_Density'], 'cm-2 s-1 GeV-1')
        pars['reference'] = Quantity(self.data['Pivot_Energy'], 'GeV')

        if spec_type == 'PowerLaw':
            pars['index'] = Quantity(self.data['Spectral_Index'], '')
            errs['index'] = Quantity(self.data['Unc_Spectral_Index'], '')
            model = PowerLaw(**pars)

        elif spec_type == 'LogParabola':
            pars['alpha'] = Quantity(self.data['Spectral_Index'], '')
            pars['beta'] = Quantity(self.data['beta'], '')
            errs['alpha'] = Quantity(self.data['Unc_Spectral_Index'], '')
            errs['beta'] = Quantity(self.data['Unc_beta'], '')
            model = LogParabola(**pars)

        else:
            raise ValueError('Spectral model {} not available'.format(spec_type))

        model.parameters.set_parameter_errors(errs)
        return model

    @property
    def flux_points(self):
        """
        Flux points (`~gammapy.spectrum.FluxPoints`).
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

        e2dnde = self._get_flux_values('nuFnu', 'erg cm-2 s-1')
        table['eflux'] = e2dnde
        table['eflux_errn'] = np.abs(e2dnde * flux_err[:, 0] / flux)
        table['eflux_errp'] = e2dnde * flux_err[:, 1] / flux

        is_ul = np.isnan(table['flux_errn'])
        table['is_ul'] = is_ul

        # handle upper limits
        table['flux_ul'] = np.nan * flux_err.unit
        table['flux_ul'][is_ul] = table['flux_errp'][is_ul]

        for column in ['flux', 'flux_errp', 'flux_errn']:
            table[column][is_ul] = np.nan

        # handle upper limits
        table['eflux_ul'] = np.nan * e2dnde.unit
        table['eflux_ul'][is_ul] = table['eflux_errp'][is_ul]

        for column in ['eflux', 'eflux_errp', 'eflux_errn']:
            table[column][is_ul] = np.nan

        table['dnde'] = (e2dnde * e_ref ** -2).to('cm-2 s-1 TeV-1')
        return FluxPoints(table)

    def _get_flux_values(self, prefix='Flux', unit='cm-2 s-1'):
        if prefix not in ['Flux', 'Unc_Flux', 'nuFnu']:
            raise ValueError("Must be one of the following: 'Flux', 'Unc_Flux', 'nuFnu'")

        values = [self.data[prefix + _ + 'GeV'] for _ in self._ebounds_suffix]
        return Quantity(values, unit)


class SourceCatalog3FGL(SourceCatalog):
    """Fermi-LAT 3FGL source catalog.

    One source is represented by `~gammapy.catalog.SourceCatalogObject3FGL`.
    """
    name = '3fgl'
    description = 'LAT 4-year point source catalog'
    source_object_class = SourceCatalogObject3FGL

    def __init__(self, filename='$GAMMAPY_EXTRA/datasets/catalogs/fermi/gll_psc_v16.fit.gz'):
        filename = str(make_path(filename))
        self.hdu_list = fits.open(filename)
        self.extended_sources_table = Table(self.hdu_list['ExtendedSources'].data)

        table = Table(self.hdu_list['LAT_Point_Source_Catalog'].data)

        source_name_key = 'Source_Name'
        source_name_alias = ('Extended_Source_Name', '0FGL_Name', '1FGL_Name',
                             '2FGL_Name', '1FHL_Name', 'ASSOC_TEV', 'ASSOC1',
                             'ASSOC2')
        super(SourceCatalog3FGL, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )


class SourceCatalog1FHL(SourceCatalog):
    """Fermi-LAT 1FHL source catalog.

    One source is represented by `~gammapy.catalog.SourceCatalogObject1FHL`.
    """
    name = '1fhl'
    description = 'First Fermi-LAT Catalog of Sources above 10 GeV'
    source_object_class = SourceCatalogObject1FHL

    def __init__(self, filename='$GAMMAPY_EXTRA/datasets/catalogs/fermi/gll_psch_v07.fit.gz'):
        filename = str(make_path(filename))
        self.hdu_list = fits.open(filename)
        # self.count_map_hdu = self.hdu_list['Count Map']
        self.extended_sources_table = Table(self.hdu_list['ExtendedSources'].data)
        table = Table(self.hdu_list['LAT_Point_Source_Catalog'].data)

        source_name_key = 'Source_Name'
        source_name_alias = ('ASSOC1', 'ASSOC2', 'ASSOC_TEV', 'ASSOC_GAM')
        super(SourceCatalog1FHL, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )


class SourceCatalog2FHL(SourceCatalog):
    """Fermi-LAT 2FHL source catalog.

    One source is represented by `~gammapy.catalog.SourceCatalogObject2FHL`.
    """
    name = '2fhl'
    description = 'LAT second high-energy source catalog'
    source_object_class = SourceCatalogObject2FHL

    def __init__(self, filename='$GAMMAPY_EXTRA/datasets/catalogs/fermi/gll_psch_v08.fit.gz'):
        filename = str(make_path(filename))
        self.hdu_list = fits.open(filename)
        self.count_map_hdu = self.hdu_list['Count Map']
        self.extended_sources_table = Table(self.hdu_list['Extended Sources'].data)
        self.rois = Table(self.hdu_list['ROIs'].data)
        table = Table(self.hdu_list['2FHL Source Catalog'].data)

        source_name_key = 'Source_Name'
        source_name_alias = ('ASSOC', '3FGL_Name', '1FHL_Name', 'TeVCat_Name')
        super(SourceCatalog2FHL, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )


class SourceCatalog3FHL(SourceCatalog):
    """Fermi-LAT 3FHL source catalog.

    One source is represented by `~gammapy.catalog.SourceCatalogObject3FHL`.
    """
    name = '3fhl'
    description = 'LAT third high-energy source catalog'
    source_object_class = SourceCatalogObject3FHL

    def __init__(self, filename='$GAMMAPY_EXTRA/datasets/catalogs/fermi/gll_psch_v11.fit.gz'):
        filename = str(make_path(filename))
        self.hdu_list = fits.open(filename)
        self.extended_sources_table = Table(self.hdu_list['ExtendedSources'].data)
        self.rois = Table(self.hdu_list['ROIs'].data)
        table = Table(self.hdu_list['LAT_Point_Source_Catalog'].data)

        self.energy_bounds_table = Table(self.hdu_list['EnergyBounds'].data)
        self._add_flux_point_columns(
            table=table,
            energy_bounds_table=self.energy_bounds_table,
        )

        source_name_key = 'Source_Name'
        source_name_alias = ('ASSOC1', 'ASSOC2', 'ASSOC_TEV', 'ASSOC_GAM')
        super(SourceCatalog3FHL, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

    @staticmethod
    def _add_flux_point_columns(table, energy_bounds_table):
        """
        Add integrated flux columns (defined in the same way as in the
        other Fermi catalogs (e.g. FluxY_ZGeV))
        """
        for idx, band in enumerate(energy_bounds_table):
            col_flux_name = 'Flux{:d}_{:d}GeV'.format(int(band['LowerEnergy']),
                                                      int(band['UpperEnergy']))
            col_flux_value = table['Flux_Band'][:, idx].data
            col_flux = Column(col_flux_value, name=col_flux_name)

            col_unc_flux_name = 'Unc_' + col_flux_name
            col_unc_flux_value = table['Unc_Flux_Band'][:, idx].data
            col_unc_flux = Column(col_unc_flux_value, name=col_unc_flux_name)

            col_nufnu_name = 'nuFnu{:d}_{:d}GeV'.format(int(band['LowerEnergy']),
                                                        int(band['UpperEnergy']))
            col_nufnu_value = table['nuFnu'][:, idx].data
            col_nufnu = Column(col_nufnu_value, name=col_nufnu_name)

            table.add_column(col_flux)
            table.add_column(col_unc_flux)
            table.add_column(col_nufnu)


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
