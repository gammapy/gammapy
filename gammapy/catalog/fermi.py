# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fermi catalog and source classes.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tarfile
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.table import QTable, Table
from astropy.time import Time
from astropy.utils.data import download_file
from astropy.tests.helper import ignore_warnings
from astropy.modeling.models import Gaussian2D, Disk2D
from astropy.coordinates import Angle
from ..utils.scripts import make_path
from ..utils.energy import EnergyBounds
from ..utils.table import table_standardise_units_inplace
from ..image import SkyImage
from ..image.models import Delta2D, Template2D
from ..spectrum import FluxPoints
from ..spectrum.models import (
    PowerLaw,
    PowerLaw2,
    ExponentialCutoffPowerLaw3FGL,
    PLSuperExpCutoff3FGL,
    LogParabola,
)
from ..time import LightCurve
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


def compute_flux_points_ul(quantity, quantity_errp):
    """Compute UL value for fermi flux points.

    See https://arxiv.org/pdf/1501.02003.pdf (page 30)
    """
    return 2 * quantity_errp + quantity


class SourceCatalogObject3FGL(SourceCatalogObject):
    """One source from the Fermi-LAT 3FGL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog3FGL`.
    """
    _ebounds = EnergyBounds([100, 300, 1000, 3000, 10000, 100000], 'MeV')
    _ebounds_suffix = ['100_300', '300_1000',
                       '1000_3000', '3000_10000', '10000_100000']
    energy_range = u.Quantity([100, 100000], 'MeV')
    """Energy range of the catalog.

    Paper says that analysis uses data up to 300 GeV,
    but results are all quoted up to 100 GeV only to
    be consistent with previous catalogs.
    """

    def info(self, info='all'):
        """Print info.

        Parameters
        ----------
        info : {'all', 'basic', 'position', 'spectral', 'lightcurve'}
            Comma separated list of options
        """
        ss = self.__str__(info=info)
        print(ss)

    def __str__(self, info='all'):
        """Summary info string.

        Parameters
        ----------
        info : {'all', 'basic', 'position', 'spectral', 'lightcurve'}
            Comma separated list of options
        """
        if info == 'all':
            info = 'basic,position,spectral,lightcurve'

        ss = ''
        ops = info.split(',')
        if 'basic' in ops:
            ss += self._info_basic()
        if 'position' in ops:
            ss += self._info_position()
        if 'spectral' in ops:
            ss += self._info_spectral_fit()
            ss += self._info_spectral_points()
        if 'lightcurve' in ops:
            ss += self._info_lightcurve()
        return ss

    def _info_basic(self):
        """Print basic info."""
        d = self.data
        ss = '\n*** Basic info ***\n\n'
        ss += 'Catalog row index (zero-based) : {}\n'.format(d['catalog_row_index'])
        ss += '{:<20s} : {}\n'.format('Source name', d['Source_Name'])
        ss += '{:<20s} : {}\n'.format('Extended name', d['Extended_Source_Name'])

        def get_nonentry_keys(keys):
            vals = [d[_].strip() for _ in keys]
            return ', '.join([_ for _ in vals if _ != ''])

        keys = ['ASSOC1', 'ASSOC2', 'ASSOC_TEV', 'ASSOC_GAM1', 'ASSOC_GAM2', 'ASSOC_GAM3']
        associations = get_nonentry_keys(keys)
        ss += '{:<20s} : {}\n'.format('Associations', associations)

        keys = ['0FGL_Name', '1FGL_Name', '2FGL_Name', '1FHL_Name']
        other_names = get_nonentry_keys(keys)
        ss += '{:<20s} : {}\n'.format('Other names', other_names)

        ss += '{:<20s} : {}\n'.format('Class', d['CLASS1'])

        tevcat_flag = d['TEVCAT_FLAG']
        if tevcat_flag == 'N':
            tevcat_message = 'No TeV association'
        elif tevcat_flag == 'P':
            tevcat_message = 'Small TeV source'
        elif tevcat_flag == 'E':
            tevcat_message = 'Extended TeV source (diameter > 40 arcmins)'
        else:
            tevcat_message = 'N/A'
        ss += '{:<20s} : {}\n'.format('TeVCat flag', tevcat_message)

        flag_message = {
            0: 'None',
            1: 'Source with TS > 35 which went to TS < 25 when changing the diffuse model. Note that sources with TS < '
               '35 are not flagged with this bit because normal statistical fluctuations can push them to TS < 25.',
            3: 'Flux (> 1 GeV) or energy flux (> 100 MeV) changed by more than 3 sigma when changing the diffuse model.'
               ' Requires also that the flux change by more than 35% (to not flag strong sources).',
            4: 'Source-to-background ratio less than 10% in highest band in which TS > 25. Background is integrated '
               'over the 68%-confidence area (pi*r_682) or 1 square degree, whichever is smaller.',
            5: 'Closer than theta_ref from a brighter neighbor, where theta_ref is defined in the highest band in which'
               ' source TS > 25, or the band with highest TS if all are < 25. theta_ref is set to 2.17 degrees (FWHM)'
               ' below 300 MeV, 1.38 degrees between 300 MeV and 1 GeV, 0.87 degrees between 1 GeV and 3 GeV, 0.67'
               ' degrees between 3 and 10 GeV and 0.45 degrees about 10 GeV (2*r_68).',
            6: 'On top of an interstellar gas clump or small-scale defect in the model of diffuse emission. This flag '
               'is equivalent to the "c" suffix in the source name.',
            7: 'Unstable position determination; result from gtfindsrc outside the 95% ellipse from pointlike.',
            9: 'Localization Quality > 8 in pointlike (see Section 3.1 in catalog paper) or long axis of 95% ellipse >'
               ' 0.25.',
            10: 'Spectral Fit Quality > 16.3 (see Equation 3 in 2FGL catalog paper).',
            11: 'Possibly due to the Sun (see Section 3.6 in catalog paper).',
            12: 'Highly curved spectrum; LogParabola beta fixed to 1 or PLExpCutoff Spectral Index fixed to 0 (see '
                'Section 3.3 in catalog paper).'
        }
        ss += '{:<20s} : {}\n'.format('Other flags', flag_message.get(d['Flags'], 'N/A'))

        return ss

    def _info_position(self):
        """Print position info."""
        d = self.data
        ss = '\n*** Position info ***\n\n'
        ss += '{:<20s} : {:.3f}\n'.format('RA', d['RAJ2000'])
        ss += '{:<20s} : {:.3f}\n'.format('DEC', d['DEJ2000'])
        ss += '{:<20s} : {:.3f}\n'.format('GLON', d['GLON'])
        ss += '{:<20s} : {:.3f}\n'.format('GLAT', d['GLAT'])

        ss += '\n'
        ss += '{:<20s} : {:.4f}\n'.format('Semimajor (68%)', d['Conf_68_SemiMajor'])
        ss += '{:<20s} : {:.4f}\n'.format('Semiminor (68%)', d['Conf_68_SemiMinor'])
        ss += '{:<20s} : {:.2f}\n'.format('Position angle (68%)', d['Conf_68_PosAng'])
        ss += '{:<20s} : {:.4f}\n'.format('Semimajor (95%)', d['Conf_95_SemiMajor'])
        ss += '{:<20s} : {:.4f}\n'.format('Semiminor (95%)', d['Conf_95_SemiMinor'])
        ss += '{:<20s} : {:.2f}\n'.format('Position angle (95%)', d['Conf_95_PosAng'])
        ss += '{:<20s} : {:.0f}\n'.format('ROI number', d['ROI_num'])

        return ss

    def _info_spectral_fit(self):
        """Print spectral info."""
        d = self.data
        ss = '\n*** Spectral info ***\n\n'

        ss += '{:<45s} : {}\n'.format('Spectrum type', d['SpectrumType'])
        fmt = '{:<45s} : {:.3f}\n'
        args = ('Detection significance (100 MeV - 300 GeV)', d['Signif_Avg'])
        ss += fmt.format(*args)
        ss += '{:<45s} : {:.1f}\n'.format('Significance curvature', d['Signif_Curve'])

        spec_type = d['SpectrumType'].strip()
        if spec_type == 'LogParabola':
            ss += '{:<45s} : {} +- {}\n'.format('beta', d['beta'], d['Unc_beta'])
        if spec_type in ['PLExpCutoff', 'PlSuperExpCutoff']:
            fmt = '{:<45s} : {:.0f} +- {:.0f} {}\n'
            args = ('Cutoff energy', d['Cutoff'].value, d['Unc_Cutoff'].value, d['Cutoff'].unit)
            ss += fmt.format(*args)
        if spec_type == 'PLSuperExpCutoff':
            ss += '{:<45s} : {} +- {}\n'.format('Exponential index', d['Exp_Index'], d['Unc_Exp_Index'])

        ss += '{:<45s} : {:.0f} {}\n'.format('Pivot energy', d['Pivot_Energy'].value, d['Pivot_Energy'].unit)

        ss += '{:<45s} : {:.3f}\n'.format('Power law index', d['PowerLaw_Index'])

        fmt = '{:<45s} : {:.3f} +- {:.3f}\n'
        args = ('Spectral index', d['Spectral_Index'], d['Unc_Spectral_Index'])
        ss += fmt.format(*args)

        unit = 'cm-2 MeV-1 s-1'
        fmt = '{:<45s} : {:.3} +- {:.3} {}\n'
        args = ('Flux Density at pivot energy', d['Flux_Density'].value, d['Unc_Flux_Density'].value, unit)
        ss += fmt.format(*args)

        unit = 'cm-2 s-1'
        fmt = '{:<45s} : {:.3} +- {:.3} {}\n'
        args = ('Integral flux (1 - 100 GeV)', d['Flux1000'].value, d['Unc_Flux1000'].value, unit)
        ss += fmt.format(*args)

        unit = 'erg cm-2 s-1'
        fmt = '{:<45s} : {:.3} +- {:.3} {}\n'
        args = ('Energy flux (100 MeV - 100 GeV)', d['Energy_Flux100'].value, d['Unc_Energy_Flux100'].value, unit)
        ss += fmt.format(*args)

        return ss

    def _info_spectral_points(self):
        """Print spectral points."""
        d = self.data
        ss = '\n*** Spectral points ***\n\n'
        ss += '\n'.join(self._flux_points_table_formatted.pformat(max_width=-1))

        return ss + '\n'

    def _info_lightcurve(self):
        """Print lightcurve info."""
        d = self.data
        ss = '\n*** Lightcurve info ***\n\n'
        ss += 'Lightcurve measured in the energy band: 100 MeV - 100 GeV\n\n'

        ss += '{:<15s} : {:.3f}\n'.format('Variability index', d['Variability_Index'])

        if d['Signif_Peak'] == np.nan:
            ss += '{:<40s} : {:.3f}\n'.format('Significance peak (100 MeV - 100 GeV)', d['Signif_Peak'])

            fmt = '{:<40s} : {:.3} +- {:.3} cm^-2 s^-1\n'
            args = ('Integral flux peak (100 MeV - 100 GeV)', d['Flux_Peak'], d['Unc_Flux_Peak'])
            ss += fmt.format(*args)

            # TODO: give time as UTC string, not MET
            ss += '{:<40s} : {:.3} s (Mission elapsed time)\n'.format('Time peak', d['Time_Peak'])
            peak_interval = d['Peak_Interval'].to('day').value
            ss += '{:<40s} : {:.3} day\n'.format('Peak interval', peak_interval)
        else:
            ss += '\nNo peak measured for this source.\n'

        # TODO: Add a lightcurve table with d['Flux_History'] and d['Unc_Flux_History']

        return ss

    @property
    def spectral_model(self):
        """Best fit spectral model (`~gammapy.spectrum.SpectralModel`)."""
        spec_type = self.data['SpectrumType'].strip()
        pars, errs = {}, {}
        pars['amplitude'] = self.data['Flux_Density']
        errs['amplitude'] = self.data['Unc_Flux_Density']
        pars['reference'] = self.data['Pivot_Energy']

        if spec_type == 'PowerLaw':
            pars['index'] = self.data['Spectral_Index'] * u.dimensionless_unscaled
            errs['index'] = self.data['Unc_Spectral_Index'] * u.dimensionless_unscaled
            model = PowerLaw(**pars)
        elif spec_type == 'PLExpCutoff':
            pars['index'] = self.data['Spectral_Index'] * u.dimensionless_unscaled
            pars['ecut'] = self.data['Cutoff']
            errs['index'] = self.data['Unc_Spectral_Index'] * u.dimensionless_unscaled
            errs['ecut'] = self.data['Unc_Cutoff']
            model = ExponentialCutoffPowerLaw3FGL(**pars)
        elif spec_type == 'LogParabola':
            pars['alpha'] = self.data['Spectral_Index'] * u.dimensionless_unscaled
            pars['beta'] = self.data['beta'] * u.dimensionless_unscaled
            errs['alpha'] = self.data['Unc_Spectral_Index'] * u.dimensionless_unscaled
            errs['beta'] = self.data['Unc_beta'] * u.dimensionless_unscaled
            model = LogParabola(**pars)
        elif spec_type == "PLSuperExpCutoff":
            # TODO: why convert to GeV here? Remove?
            pars['reference'] = pars['reference'].to('GeV')
            pars['index_1'] = self.data['Spectral_Index'] * u.dimensionless_unscaled
            pars['index_2'] = self.data['Exp_Index'] * u.dimensionless_unscaled
            pars['ecut'] = self.data['Cutoff'].to('GeV')
            errs['index_1'] = self.data['Unc_Spectral_Index'] * u.dimensionless_unscaled
            errs['index_2'] = self.data['Unc_Exp_Index'] * u.dimensionless_unscaled
            errs['ecut'] = self.data['Unc_Cutoff'].to('GeV')
            model = PLSuperExpCutoff3FGL(**pars)
        else:
            raise ValueError('Spectral model {} not available'.format(spec_type))

        model.parameters.set_parameter_errors(errs)
        return model

    @property
    def _flux_points_table_formatted(self):
        """Returns formatted version of self.flux_points.table"""
        table = self.flux_points.table.copy()
        flux_cols = ['flux', 'flux_errn', 'flux_errp', 'e2dnde', 'e2dnde_errn',
                     'e2dnde_errp', 'flux_ul', 'e2dnde_ul', 'dnde']
        table['sqrt_TS'].format = '.1f'
        table['e_ref'].format = '.1f'
        for _ in flux_cols:
            table[_].format = '.3'

        return table

    @property
    def flux_points(self):
        """Flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta['SED_TYPE'] = 'flux'
        e_ref = self._ebounds.log_centers
        table['e_ref'] = e_ref
        table['e_min'] = self._ebounds.lower_bounds
        table['e_max'] = self._ebounds.upper_bounds

        flux = self._get_flux_values('Flux')
        flux_err = self._get_flux_values('Unc_Flux')
        table['flux'] = flux
        table['flux_errn'] = np.abs(flux_err[:, 0])
        table['flux_errp'] = flux_err[:, 1]

        nuFnu = self._get_flux_values('nuFnu', 'erg cm-2 s-1')
        table['e2dnde'] = nuFnu
        table['e2dnde_errn'] = np.abs(nuFnu * flux_err[:, 0] / flux)
        table['e2dnde_errp'] = nuFnu * flux_err[:, 1] / flux

        is_ul = np.isnan(table['flux_errn'])
        table['is_ul'] = is_ul

        # handle upper limits
        table['flux_ul'] = np.nan * flux_err.unit
        flux_ul = compute_flux_points_ul(table['flux'], table['flux_errp'])
        table['flux_ul'][is_ul] = flux_ul[is_ul]

        # handle upper limits
        table['e2dnde_ul'] = np.nan * nuFnu.unit
        e2dnde_ul = compute_flux_points_ul(table['e2dnde'], table['e2dnde_errp'])
        table['e2dnde_ul'][is_ul] = e2dnde_ul[is_ul]

        # Square root of test statistic
        table['sqrt_TS'] = [self.data['Sqrt_TS' + _] for _ in self._ebounds_suffix]

        table['dnde'] = (nuFnu * e_ref ** -2).to('TeV-1 cm-2 s-1')
        return FluxPoints(table)

    def _get_flux_values(self, prefix, unit='cm-2 s-1'):
        values = [self.data[prefix + _] for _ in self._ebounds_suffix]
        return u.Quantity(values, unit)

    @property
    def lightcurve(self):
        """Lightcurve (`~gammapy.time.LightCurve`)."""
        flux = self.data['Flux_History']

        # Flux error is given as asymmetric high/low
        flux_err_lo = self.data['Unc_Flux_History'][:, 0]
        flux_err_hi = self.data['Unc_Flux_History'][:, 1]

        # TODO: Change lightcurve class to support this,
        # then fill appropriately here
        # for now, we just use the mean
        flux_err = 0.5 * (-flux_err_lo + flux_err_hi)

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
        table['FLUX'] = flux
        table['FLUX_ERR'] = flux_err
        lc = LightCurve(table)
        return lc


class SourceCatalogObject1FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 1FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog1FHL`.
    """
    _ebounds = EnergyBounds([10, 30, 100, 500], 'GeV')
    _ebounds_suffix = ['10_30', '30_100', '100_500']
    energy_range = u.Quantity([0.01, 0.5], 'TeV')
    """Energy range of the Fermi 1FHL source catalog"""

    def __str__(self):
        """Print summary info."""
        # TODO: can we share code with 3FGL summary function?
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

    def _get_flux_values(self, prefix, unit='cm-2 s-1'):
        values = [self.data[prefix + _ + 'GeV'] for _ in self._ebounds_suffix]
        return u.Quantity(values, unit)

    @property
    def flux_points(self):
        """Integral flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta['SED_TYPE'] = 'flux'
        table['e_min'] = self._ebounds.lower_bounds
        table['e_max'] = self._ebounds.upper_bounds
        table['flux'] = self._get_flux_values('Flux')
        flux_err = self._get_flux_values('Unc_Flux')
        table['flux_errn'] = np.abs(flux_err[:, 0])
        table['flux_errp'] = flux_err[:, 1]

        # handle upper limits
        is_ul = np.isnan(table['flux_errn'])
        table['is_ul'] = is_ul
        table['flux_ul'] = np.nan * flux_err.unit
        flux_ul = compute_flux_points_ul(table['flux'], table['flux_errp'])
        table['flux_ul'][is_ul] = flux_ul[is_ul]

        flux_points = FluxPoints(table)

        # TODO: change this and leave it up to the caller to convert to dnde
        # See https://github.com/gammapy/gammapy/issues/1034
        return flux_points.to_sed_type('dnde', model=self.spectral_model)

    @property
    def spectral_model(self):
        """Best fit spectral model `~gammapy.spectrum.models.SpectralModel`."""
        pars, errs = {}, {}
        pars['amplitude'] = self.data['Flux']
        pars['emin'], pars['emax'] = self.energy_range
        pars['index'] = self.data['Spectral_Index'] * u.dimensionless_unscaled
        errs['amplitude'] = self.data['Unc_Flux']
        errs['index'] = self.data['Unc_Spectral_Index'] * u.dimensionless_unscaled
        model = PowerLaw2(**pars)
        model.parameters.set_parameter_errors(errs)
        return model


class SourceCatalogObject2FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 2FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog2FHL`.
    """
    _ebounds = EnergyBounds([50, 171, 585, 2000], 'GeV')
    _ebounds_suffix = ['50_171', '171_585', '585_2000']
    energy_range = u.Quantity([0.05, 2], 'TeV')
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

    def _get_flux_values(self, prefix, unit='cm-2 s-1'):
        values = [self.data[prefix + _ + 'GeV'] for _ in self._ebounds_suffix]
        return u.Quantity(values, unit)

    @property
    def flux_points(self):
        """Integral flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta['SED_TYPE'] = 'flux'
        table['e_min'] = self._ebounds.lower_bounds
        table['e_max'] = self._ebounds.upper_bounds
        table['flux'] = self._get_flux_values('Flux')
        flux_err = self._get_flux_values('Unc_Flux')
        table['flux_errn'] = np.abs(flux_err[:, 0])
        table['flux_errp'] = flux_err[:, 1]

        # handle upper limits
        is_ul = np.isnan(table['flux_errn'])
        table['is_ul'] = is_ul
        table['flux_ul'] = np.nan * flux_err.unit
        flux_ul = compute_flux_points_ul(table['flux'], table['flux_errp'])
        table['flux_ul'][is_ul] = flux_ul[is_ul]

        flux_points = FluxPoints(table)

        # TODO: change this and leave it up to the caller to convert to dnde
        # See https://github.com/gammapy/gammapy/issues/1034
        return flux_points.to_sed_type('dnde', model=self.spectral_model)

    @property
    def spectral_model(self):
        """Best fit spectral model (`~gammapy.spectrum.models.SpectralModel`)."""
        pars, errs = {}, {}
        pars['amplitude'] = self.data['Flux50']
        pars['emin'], pars['emax'] = self.energy_range
        pars['index'] = self.data['Spectral_Index'] * u.dimensionless_unscaled

        errs['amplitude'] = self.data['Unc_Flux50']
        errs['index'] = self.data['Unc_Spectral_Index'] * u.dimensionless_unscaled

        model = PowerLaw2(**pars)
        model.parameters.set_parameter_errors(errs)
        return model


class SourceCatalogObject3FHL(SourceCatalogObject):
    """One source from the Fermi-LAT 3FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog3FHL`.
    """
    energy_range = u.Quantity([0.01, 2], 'TeV')
    """Energy range of the Fermi 1FHL source catalog"""

    _ebounds = EnergyBounds([10, 20, 50, 150, 500, 2000], 'GeV')

    def info(self, info='all'):
        """Print info.

        Parameters
        ----------
        info : {'all', 'basic', 'position', 'spectral'}
            Comma separated list of options
        """
        ss = self.__str__(info=info)
        print(ss)

    def __str__(self, info='all'):
        """Summary info string.

        Parameters
        ----------
        info : {'all', 'basic', 'position', 'spectral'}
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
            ss += self._info_spectral_fit()
            ss += self._info_spectral_points()
        if 'other' in ops:
            ss += self._info_other()

        return ss

    def _info_basic(self):
        """Print basic info."""
        d = self.data
        ss = '\n*** Basic info ***\n\n'
        ss += 'Catalog row index (zero-based) : {}\n'.format(d['catalog_row_index'])
        ss += '{:<20s} : {}\n'.format('Source name', d['Source_Name'])
        ss += '{:<20s} : {}\n'.format('Extended name', d['Extended_Source_Name'])

        def get_nonentry_keys(keys):
            vals = [d[_].strip() for _ in keys]
            return ', '.join([_ for _ in vals if _ != ''])

        keys = ['ASSOC1', 'ASSOC2', 'ASSOC_TEV', 'ASSOC_GAM']
        associations = get_nonentry_keys(keys)
        ss += '{:<16s} : {}\n'.format('Associations', associations)
        ss += '{:<16s} : {:.3f}\n'.format('ASSOC_PROB_BAY', d['ASSOC_PROB_BAY'])
        ss += '{:<16s} : {:.3f}\n'.format('ASSOC_PROB_LR', d['ASSOC_PROB_LR'])

        ss += '{:<16s} : {}\n'.format('Class', d['CLASS'])

        tevcat_flag = d['TEVCAT_FLAG']
        if tevcat_flag == 'N':
            tevcat_message = 'No TeV association'
        elif tevcat_flag == 'P':
            tevcat_message = 'Small TeV source'
        elif tevcat_flag == 'E':
            tevcat_message = 'Extended TeV source (diameter > 40 arcmins)'
        else:
            tevcat_message = 'N/A'
        ss += '{:<16s} : {}\n'.format('TeVCat flag', tevcat_message)

        fmt = '\n{:<32s} : {:.3f}\n'
        args = ('Significance (10 GeV - 2 TeV)', d['Signif_Avg'])
        ss += fmt.format(*args)
        ss += '{:<32s} : {:.1f}\n'.format('Npred', d['Npred'])

        return ss

    def _info_position(self):
        """Print position info."""
        d = self.data
        ss = '\n*** Position info ***\n\n'
        ss += '{:<20s} : {:.3f}\n'.format('RA', d['RAJ2000'])
        ss += '{:<20s} : {:.3f}\n'.format('DEC', d['DEJ2000'])
        ss += '{:<20s} : {:.3f}\n'.format('GLON', d['GLON'])
        ss += '{:<20s} : {:.3f}\n'.format('GLAT', d['GLAT'])

        # TODO: All sources are non-elliptical; just give one number for radius?
        ss += '\n'
        ss += '{:<20s} : {:.4f}\n'.format('Semimajor (95%)', d['Conf_95_SemiMajor'])
        ss += '{:<20s} : {:.4f}\n'.format('Semiminor (95%)', d['Conf_95_SemiMinor'])
        ss += '{:<20s} : {:.2f}\n'.format('Position angle (95%)', d['Conf_95_PosAng'])
        ss += '{:<20s} : {:.0f}\n'.format('ROI number', d['ROI_num'])

        return ss

    def _info_spectral_fit(self):
        """Print model data."""
        d = self.data
        ss = '\n*** Model info ***\n\n'

        if not self.is_pointlike:
            e = self.data_extended
            ss += 'Extended source information:\n'
            ss += '{:<16s} : {}\n'.format('Model form', e['Model_Form'])
            ss += '{:<16s} : {:.4f}\n'.format('Model semimajor', e['Model_SemiMajor'])
            ss += '{:<16s} : {:.4f}\n'.format('Model semiminor', e['Model_SemiMinor'])
            ss += '{:<16s} : {:.4f}\n'.format('Position angle', e['Model_PosAng'])
            ss += '{:<16s} : {}\n'.format('Spatial function', e['Spatial_Function'])
            ss += '{:<16s} : {}\n\n'.format('Spatial filename', e['Spatial_Filename'])

        ss += '{:<32s} : {}\n'.format('Spectrum type', d['SpectrumType'])
        ss += '{:<32s} : {:.1f}\n'.format('Significance curvature', d['Signif_Curve'])
        fmt = '{:<32s} : {:.3f} +- {:.3f}\n'
        args = ('Spectral index', d['Spectral_Index'], d['Unc_Spectral_Index'])
        ss += fmt.format(*args)

        spec_type = d['SpectrumType'].strip()
        if spec_type == 'LogParabola':
            # ss += '{:<32s} : {:.3f} +- {:.3f}\n'.format('alpha', d['Spectral_Index'], d['Unc_Spectral_Index'])
            ss += '{:<32s} : {:.3f} +- {:.3f}\n'.format('beta', d['beta'], d['Unc_beta'])

        ss += '{:<32s} : {:.0f} {}\n'.format('Pivot energy', d['Pivot_Energy'].value, d['Pivot_Energy'].unit)

        fmt = '{:<32s} : {:.3f} +- {:.3f}\n'
        args = ('Power Law index', d['PowerLaw_Index'], d['Unc_PowerLaw_Index'], d['Unc_PowerLaw_Index'])
        ss += fmt.format(*args)

        unit = 'cm-2 GeV-1 s-1'
        fmt = '{:<32s} : {:.3} +- {:.3} {}\n'
        args = ('Flux Density at pivot energy', d['Flux_Density'].value, d['Unc_Flux_Density'].value, unit)
        ss += fmt.format(*args)

        unit = 'cm-2 s-1'
        fmt = '{:<32s} : {:.3} +- {:.3} {}\n'
        args = ('Integral flux (10 GeV - 1 TeV)', d['Flux'].value, d['Unc_Flux'].value, unit)
        ss += fmt.format(*args)

        unit = 'erg cm-2 s-1'
        fmt = '{:<32s} : {:.3} +- {:.3} {}\n'
        args = ('Energy flux (10 GeV - TeV)', d['Energy_Flux'].value, d['Unc_Energy_Flux'].value, unit)
        ss += fmt.format(*args)

        return ss

    def _info_spectral_points(self):
        """Print spectral points."""
        ss = '\n*** Spectral points ***\n\n'
        ss += '\n'.join(self._flux_points_table_formatted.pformat(max_width=-1))

        return ss + '\n'

    def _info_other(self):
        """Print other info."""
        d = self.data
        ss = '\n*** Other info ***\n\n'
        ss += '{:<16s} : {:.3f} {}\n'.format('HEP Energy', d['HEP_Energy'].value, d['HEP_Energy'].unit)
        ss += '{:<16s} : {:.3f}\n'.format('HEP Probability', d['HEP_Prob'])

        # This is the number of Bayesian blocks for most sources,
        # except -1 means "could not be tested"
        msg = d['Variability_BayesBlocks']
        if msg == 1:
            msg = '1 (not variable)'
        elif msg == -1:
            msg = 'Could not be tested'
        ss += '{:<16s} : {}\n'.format('Bayesian Blocks', msg)

        ss += '{:<16s} : {:.3f}\n'.format('Redshift', d['Redshift'])
        ss += '{:<16s} : {:.3} {}\n'.format('NuPeak_obs', d['NuPeak_obs'].value, d['NuPeak_obs'].unit)

        return ss

    @property
    def spectral_model(self):
        """Best fit spectral model (`~gammapy.spectrum.models.SpectralModel`)."""
        d = self.data
        spec_type = d['SpectrumType'].strip()
        pars, errs = {}, {}
        pars['amplitude'] = d['Flux_Density']
        errs['amplitude'] = d['Unc_Flux_Density']
        pars['reference'] = d['Pivot_Energy']

        if spec_type == 'PowerLaw':
            pars['index'] = d['Spectral_Index'] * u.dimensionless_unscaled
            errs['index'] = d['Unc_Spectral_Index'] * u.dimensionless_unscaled
            model = PowerLaw(**pars)
        elif spec_type == 'LogParabola':
            pars['alpha'] = d['Spectral_Index'] * u.dimensionless_unscaled
            pars['beta'] = d['beta'] * u.dimensionless_unscaled
            errs['alpha'] = d['Unc_Spectral_Index'] * u.dimensionless_unscaled
            errs['beta'] = d['Unc_beta'] * u.dimensionless_unscaled
            model = LogParabola(**pars)
        else:
            raise ValueError('Spectral model {} not available'.format(spec_type))

        model.parameters.set_parameter_errors(errs)
        return model

    @property
    def _flux_points_table_formatted(self):
        """Returns formatted version of self.flux_points.table"""
        table = self.flux_points.table.copy()
        flux_cols = ['flux', 'flux_errn', 'flux_errp', 'e2dnde', 'e2dnde_errn',
                     'e2dnde_errp', 'flux_ul', 'e2dnde_ul', 'dnde']
        table['sqrt_ts'].format = '.1f'
        table['e_ref'].format = '.1f'
        for _ in flux_cols:
            table[_].format = '.3'

        return table

    @property
    def flux_points(self):
        """Flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta['SED_TYPE'] = 'flux'
        e_ref = self._ebounds.log_centers
        table['e_ref'] = e_ref
        table['e_min'] = self._ebounds.lower_bounds
        table['e_max'] = self._ebounds.upper_bounds

        flux = self.data['Flux_Band']
        flux_err = self.data['Unc_Flux_Band']
        e2dnde = self.data['nuFnu']

        table['flux'] = flux
        table['flux_errn'] = np.abs(flux_err[:, 0])
        table['flux_errp'] = flux_err[:, 1]

        table['e2dnde'] = e2dnde
        table['e2dnde_errn'] = np.abs(e2dnde * flux_err[:, 0] / flux)
        table['e2dnde_errp'] = e2dnde * flux_err[:, 1] / flux

        is_ul = np.isnan(table['flux_errn'])
        table['is_ul'] = is_ul

        # handle upper limits
        table['flux_ul'] = np.nan * flux_err.unit
        flux_ul = compute_flux_points_ul(table['flux'], table['flux_errp'])
        table['flux_ul'][is_ul] = flux_ul[is_ul]

        table['e2dnde_ul'] = np.nan * e2dnde.unit
        e2dnde_ul = compute_flux_points_ul(table['e2dnde'], table['e2dnde_errp'])
        table['e2dnde_ul'][is_ul] = e2dnde_ul[is_ul]

        # Square root of test statistic
        table['sqrt_ts'] = self.data['Sqrt_TS_Band']

        # TODO: remove this computation here.
        # # Instead provide a method on the FluxPoints class like `to_dnde()` or something.
        table['dnde'] = (e2dnde * e_ref ** -2).to('cm-2 s-1 TeV-1')

        return FluxPoints(table)

    def spatial_model(self, emin=1 * u.TeV, emax=10 * u.TeV):
        """
        Source spatial model.
        """
        d = self.data
        flux = self.spectral_model.integral(emin, emax)
        amplitude = flux.to('cm-2 s-1').value

        pars = {}
        glon = Angle(d['GLON']).wrap_at('180d')
        glat = Angle(d['GLAT']).wrap_at('180d')

        if self.is_pointlike:
            pars['amplitude'] = amplitude
            pars['x_0'] = glon.value
            pars['y_0'] = glat.value
            return Delta2D(**pars)
        else:
            de = self.data_extended
            morph_type = de['Spatial_Function'].strip()

            if morph_type == 'RadialDisk':
                pars['x_0'] = glon.value
                pars['y_0'] = glat.value
                pars['R_0'] = de['Model_SemiMajor'].to('deg').value
                pars['amplitude'] = amplitude / (np.pi * pars['R_0'] ** 2)
                return Disk2D(**pars)
            elif morph_type == 'SpatialMap':
                filename = de['Spatial_Filename'].strip()
                base = '$GAMMAPY_EXTRA/datasets/catalogs/fermi/Extended_archive_v17/Templates/'
                template = Template2D.read(base + filename)
                template.amplitude = amplitude
                return template
            elif morph_type == 'RadialGauss':
                pars['x_mean'] = glon.value
                pars['y_mean'] = glat.value
                pars['x_stddev'] = de['Model_SemiMajor'].to('deg').value
                pars['y_stddev'] = de['Model_SemiMajor'].to('deg').value
                pars['amplitude'] = amplitude * 1 / (2 * np.pi * pars['x_stddev'] ** 2)
                return Gaussian2D(**pars)
            else:
                raise ValueError('Not a valid spatial model{}'.format(morph_type))

    @property
    def is_pointlike(self):
        return self.data['Extended_Source_Name'].strip() == ''


class SourceCatalog3FGL(SourceCatalog):
    """Fermi-LAT 3FGL source catalog.

    One source is represented by `~gammapy.catalog.SourceCatalogObject3FGL`.
    """
    name = '3fgl'
    description = 'LAT 4-year point source catalog'
    source_object_class = SourceCatalogObject3FGL

    def __init__(self, filename='$GAMMAPY_EXTRA/datasets/catalogs/fermi/gll_psc_v16.fit.gz'):
        filename = str(make_path(filename))

        with ignore_warnings():  # ignore FITS units warnings
            table = Table.read(filename, hdu='LAT_Point_Source_Catalog')
        table_standardise_units_inplace(table)

        source_name_key = 'Source_Name'
        source_name_alias = ('Extended_Source_Name', '0FGL_Name', '1FGL_Name',
                             '2FGL_Name', '1FHL_Name', 'ASSOC_TEV', 'ASSOC1',
                             'ASSOC2')
        super(SourceCatalog3FGL, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

        self.extended_sources_table = Table.read(filename, hdu='ExtendedSources')


class SourceCatalog1FHL(SourceCatalog):
    """Fermi-LAT 1FHL source catalog.

    One source is represented by `~gammapy.catalog.SourceCatalogObject1FHL`.
    """
    name = '1fhl'
    description = 'First Fermi-LAT Catalog of Sources above 10 GeV'
    source_object_class = SourceCatalogObject1FHL

    def __init__(self, filename='$GAMMAPY_EXTRA/datasets/catalogs/fermi/gll_psch_v07.fit.gz'):
        filename = str(make_path(filename))

        with ignore_warnings():  # ignore FITS units warnings
            table = Table.read(filename, hdu='LAT_Point_Source_Catalog')
        table_standardise_units_inplace(table)

        source_name_key = 'Source_Name'
        source_name_alias = ('ASSOC1', 'ASSOC2', 'ASSOC_TEV', 'ASSOC_GAM')
        super(SourceCatalog1FHL, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

        self.extended_sources_table = Table.read(filename, hdu='ExtendedSources')


class SourceCatalog2FHL(SourceCatalog):
    """Fermi-LAT 2FHL source catalog.

    One source is represented by `~gammapy.catalog.SourceCatalogObject2FHL`.
    """
    name = '2fhl'
    description = 'LAT second high-energy source catalog'
    source_object_class = SourceCatalogObject2FHL

    def __init__(self, filename='$GAMMAPY_EXTRA/datasets/catalogs/fermi/gll_psch_v08.fit.gz'):
        filename = str(make_path(filename))

        with ignore_warnings():  # ignore FITS units warnings
            table = Table.read(filename, hdu='2FHL Source Catalog')
        table_standardise_units_inplace(table)

        source_name_key = 'Source_Name'
        source_name_alias = ('ASSOC', '3FGL_Name', '1FHL_Name', 'TeVCat_Name')
        super(SourceCatalog2FHL, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

        self.counts_image = SkyImage.read(filename, hdu='Count Map')
        self.extended_sources_table = Table.read(filename, hdu='Extended Sources')
        self.rois = Table.read(filename, hdu='ROIs')


class SourceCatalog3FHL(SourceCatalog):
    """Fermi-LAT 3FHL source catalog.

    One source is represented by `~gammapy.catalog.SourceCatalogObject3FHL`.
    """
    name = '3fhl'
    description = 'LAT third high-energy source catalog'
    source_object_class = SourceCatalogObject3FHL

    def __init__(self, filename='$GAMMAPY_EXTRA/datasets/catalogs/fermi/gll_psch_v11.fit.gz'):
        filename = str(make_path(filename))

        with ignore_warnings():  # ignore FITS units warnings
            table = Table.read(filename, hdu='LAT_Point_Source_Catalog')
        table_standardise_units_inplace(table)

        source_name_key = 'Source_Name'
        source_name_alias = ('ASSOC1', 'ASSOC2', 'ASSOC_TEV', 'ASSOC_GAM')
        super(SourceCatalog3FHL, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

        self.extended_sources_table = Table.read(filename, hdu='ExtendedSources')
        self.rois = Table.read(filename, hdu='ROIs')
        self.energy_bounds_table = Table.read(filename, hdu='EnergyBounds')


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
    elif (source_class == '') or (source_class == 'unknown'):
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
        ss = 'Invalid catalog: {}\n'.format(catalog)
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
        ss = 'Invalid catalog: {}\n'.format(catalog)
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
