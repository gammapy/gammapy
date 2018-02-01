# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""HESS Galactic plane survey (HGPS) catalog.

(Not released yet.)

TODO:
- [ ] Load HGPS maps
- [ ] Source object should contain info on components and associations
- [ ] Links to SNRCat
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from collections import OrderedDict
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import Angle
from astropy.modeling.models import Gaussian2D
from ..extern.pathlib import Path
from ..utils.scripts import make_path
from ..spectrum import FluxPoints
from ..spectrum.models import PowerLaw, ExponentialCutoffPowerLaw
from ..image.models import Delta2D, Shell2D
from ..background.models import GaussianBand2D
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    'SourceCatalogHGPS',
    'SourceCatalogObjectHGPS',
]

# Flux factor, used for printing
FF = 1e-12

# Multiplicative factor to go from cm^-2 s^-1 to % Crab for integral flux > 1 TeV
# Here we use the same Crab reference that's used in the HGPS paper
# CRAB = crab_integral_flux(energy_min=1, reference='hess_ecpl')
FLUX_TO_CRAB = 100 / 2.26e-11
FLUX_TO_CRAB_DIFF = 100 / 3.5060459323111307e-11


class HGPSGaussComponent(object):
    """One Gaussian component from the HGPS catalog.
    """
    _source_name_key = 'Source_Name'
    _source_index_key = 'catalog_row_index'

    def __init__(self, data):
        self.data = OrderedDict(data)

    def __str__(self):
        """Pretty-print source data"""
        d = self.data
        ss = 'Component {}:\n'.format(d['Component_ID'])
        fmt = '{:<20s} : {:8.3f} +/- {:.3f} deg\n'
        ss += fmt.format('GLON', d['GLON'].value, d['GLON_Err'].value)
        ss += fmt.format('GLAT', d['GLAT'].value, d['GLAT_Err'].value)
        fmt = '{:<20s} : {:.3f} +/- {:.3f} deg\n'
        ss += fmt.format('Size', d['Size'].value, d['Size_Err'].value)
        val, err = d['Flux_Map'].value, d['Flux_Map_Err'].value
        fmt = '{:<20s} : ({:.2f} +/- {:.2f}) x 10^-12 cm^-2 s^-1 = ({:.1f} +/- {:.1f}) % Crab'
        ss += fmt.format('Flux (>1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB)
        return ss

    @property
    def name(self):
        """Source name"""
        name = self.data[self._source_name_key]
        return name.strip()

    @property
    def index(self):
        """Row index of source in catalog"""
        return self.data[self._source_index_key]

    @property
    def spatial_model(self):
        """
        Component spatial model.
        """
        d = self.data
        # At the moment spatial models are normalised by deg^2
        amplitude = d['Flux_Map'].value / (2 * np.pi * d['Size'].value ** 2)
        glon = Angle(d['GLON']).wrap_at('180d')
        glat = Angle(d['GLAT']).wrap_at('180d')

        return Gaussian2D(
            x_mean=glon.value,
            y_mean=glat.value,
            x_stddev=d['Size'].value,
            y_stddev=d['Size'].value,
            amplitude=amplitude,
        )


class SourceCatalogObjectHGPS(SourceCatalogObject):
    """One object from the HGPS catalog."""

    def __str__(self):
        return self.info()

    def info(self, info='all'):
        """Info string.

        Parameters
        ----------
        info : {'all', 'basic', 'map', 'spec', 'flux_points', 'components', 'associations'}
            Comma separated list of options
        """
        if info == 'all':
            info = 'basic,associations,map,spec,flux_points,components'

        ss = ''
        ops = info.split(',')
        if 'basic' in ops:
            ss += self._info_basic()
        if 'map' in ops:
            ss += self._info_map()
        if 'spec' in ops:
            ss += self._info_spec()
        if 'flux_points' in ops:
            ss += self._info_flux_points()
        if 'components' in ops:
            ss += self._info_components()
        if 'associations' in ops:
            ss += self._info_associations()
        return ss

    def _info_basic(self):
        """Print basic info."""
        d = self.data
        ss = '\n*** Basic info ***\n\n'
        ss += 'Catalog row index (zero-based) : {}\n'.format(d['catalog_row_index'])
        ss += '{:<20s} : {}\n'.format('Source name', d['Source_Name'])

        ss += '{:<20s} : {}\n'.format('Analysis reference', d['Analysis_Reference'])
        ss += '{:<20s} : {}\n'.format('Source class', d['Source_Class'])
        ss += '{:<20s} : {}\n'.format('Identified object', d['Identified_Object'])
        ss += '{:<20s} : {}\n'.format('Gamma-Cat id', d['Gamma_Cat_Source_ID'])
        ss += '\n'
        return ss

    def _info_associations(self):
        ss = '\n*** Source associations info ***\n\n'
        associations = ', '.join(self.associations)
        ss += 'List of associated objects: {}\n'.format(associations)
        return ss

    def _info_map(self):
        """Print info from map analysis."""
        d = self.data
        ss = '\n*** Info from map analysis ***\n\n'

        ra_str = Angle(d['RAJ2000'], 'deg').to_string(unit='hour', precision=0)
        dec_str = Angle(d['DEJ2000'], 'deg').to_string(unit='deg', precision=0)
        ss += '{:<20s} : {:8.3f} = {}\n'.format('RA', d['RAJ2000'], ra_str)
        ss += '{:<20s} : {:8.3f} = {}\n'.format('DEC', d['DEJ2000'], dec_str)

        ss += '{:<20s} : {:8.3f} +/- {:.3f} deg\n'.format('GLON', d['GLON'].value, d['GLON_Err'].value)
        ss += '{:<20s} : {:8.3f} +/- {:.3f} deg\n'.format('GLAT', d['GLAT'].value, d['GLAT_Err'].value)

        ss += '{:<20s} : {:.3f}\n'.format('Position Error (68%)', d['Pos_Err_68'])
        ss += '{:<20s} : {:.3f}\n'.format('Position Error (95%)', d['Pos_Err_95'])

        ss += '{:<20s} : {:.0f}\n'.format('ROI number', d['ROI_Number'])
        ss += '{:<20s} : {}\n'.format('Spatial model', d['Spatial_Model'])
        ss += '{:<20s} : {}\n'.format('Spatial components', d['Components'])

        ss += '{:<20s} : {:.1f}\n'.format('TS', d['Sqrt_TS'] ** 2)
        ss += '{:<20s} : {:.1f}\n'.format('sqrt(TS)', d['Sqrt_TS'])

        ss += '{:<20s} : {:.3f} +/- {:.3f} (UL: {:.3f}) deg\n'.format(
            'Size', d['Size'].value, d['Size_Err'].value, d['Size_UL'].value)

        ss += '{:<20s} : {:.3f}\n'.format('R70', d['R70'])
        ss += '{:<20s} : {:.3f}\n'.format('RSpec', d['RSpec'])

        ss += '{:<20s} : {:.1f}\n'.format('Total model excess', d['Excess_Model_Total'])
        ss += '{:<20s} : {:.1f}\n'.format('Excess in RSpec', d['Excess_RSpec'])
        ss += '{:<20s} : {:.1f}\n'.format('Model Excess in RSpec', d['Excess_RSpec_Model'])
        ss += '{:<20s} : {:.1f}\n'.format('Background in RSpec', d['Background_RSpec'])

        ss += '{:<20s} : {:.1f} hours\n'.format('Livetime', d['Livetime'].value)

        ss += '{:<20s} : {:.2f}\n'.format('Energy threshold', d['Energy_Threshold'])

        val, err = d['Flux_Map'].value, d['Flux_Map_Err'].value
        ss += '{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1 = ({:.2f} +/- {:.2f}) % Crab\n'.format(
            'Source flux (>1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB)

        ss += '\nFluxes in RSpec (> 1 TeV):\n'

        ss += '{:<30s} : {:.3f} x 10^-12 cm^-2 s^-1 = {:5.2f} % Crab\n'.format(
            'Map measurement',
            d['Flux_Map_RSpec_Data'].value / FF,
            d['Flux_Map_RSpec_Data'].value * FLUX_TO_CRAB,
        )

        ss += '{:<30s} : {:.3f} x 10^-12 cm^-2 s^-1 = {:5.2f} % Crab\n'.format(
            'Source model',
            d['Flux_Map_RSpec_Source'].value / FF,
            d['Flux_Map_RSpec_Source'].value * FLUX_TO_CRAB,
        )

        ss += '{:<30s} : {:.3f} x 10^-12 cm^-2 s^-1 = {:5.2f} % Crab\n'.format(
            'Other component model',
            d['Flux_Map_RSpec_Other'].value / FF,
            d['Flux_Map_RSpec_Other'].value * FLUX_TO_CRAB,
        )

        ss += '{:<30s} : {:.3f} x 10^-12 cm^-2 s^-1 = {:5.2f} % Crab\n'.format(
            'Large scale component model',
            d['Flux_Map_RSpec_LS'].value / FF,
            d['Flux_Map_RSpec_LS'].value * FLUX_TO_CRAB,
        )

        ss += '{:<30s} : {:.3f} x 10^-12 cm^-2 s^-1 = {:5.2f} % Crab\n'.format(
            'Total model',
            d['Flux_Map_RSpec_Total'].value / FF,
            d['Flux_Map_RSpec_Total'].value * FLUX_TO_CRAB,
        )

        ss += '{:<35s} : {:5.1f} %\n'.format('Containment in RSpec', 100 * d['Containment_RSpec'])
        ss += '{:<35s} : {:5.1f} %\n'.format('Contamination in RSpec', 100 * d['Contamination_RSpec'])
        label, val = 'Flux correction (RSpec -> Total)', 100 * d['Flux_Correction_RSpec_To_Total']
        ss += '{:<35s} : {:5.1f} %\n'.format(label, val)
        label, val = 'Flux correction (Total -> RSpec)', 100 * (1 / d['Flux_Correction_RSpec_To_Total'])
        ss += '{:<35s} : {:5.1f} %\n'.format(label, val)

        return ss

    def _info_spec(self):
        """Print info from spectral analysis."""
        d = self.data
        ss = '\n*** Info from spectral analysis ***\n\n'

        ss += '{:<20s} : {:.1f} hours\n'.format('Livetime', d['Livetime_Spec'].value)

        lo = d['Energy_Range_Spec_Min'].value
        hi = d['Energy_Range_Spec_Max'].value
        ss += '{:<20s} : {:.2f} to {:.2f} TeV\n'.format('Energy range:', lo, hi)

        ss += '{:<20s} : {:.1f}\n'.format('Background', d['Background_Spec'])
        ss += '{:<20s} : {:.1f}\n'.format('Excess', d['Excess_Spec'])
        ss += '{:<20s} : {}\n'.format('Spectral model', d['Spectral_Model'])

        val = d['TS_ECPL_over_PL']
        ss += '{:<20s} : {:.1f}\n'.format('TS ECPL over PL', val)

        val = d['Flux_Spec_Int_1TeV'].value
        err = d['Flux_Spec_Int_1TeV_Err'].value
        ss += '{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1  = ({:.2f} +/- {:.2f}) % Crab\n'.format(
            'Best-fit model flux(> 1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB)

        val = d['Flux_Spec_Energy_1_10_TeV'].value
        err = d['Flux_Spec_Energy_1_10_TeV_Err'].value
        ss += '{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 erg cm^-2 s^-1\n'.format(
            'Best-fit model energy flux(1 to 10 TeV)', val / FF, err / FF)

        # TODO: can we just use the Gammapy model classes here instead of duplicating the code?
        ss += self._info_spec_pl()
        ss += self._info_spec_ecpl()

        return ss

    def _info_spec_pl(self):
        d = self.data
        ss = '{:<20s} : {:.2f}\n'.format('Pivot energy', d['Energy_Spec_PL_Pivot'])

        val = d['Flux_Spec_PL_Diff_Pivot'].value
        err = d['Flux_Spec_PL_Diff_Pivot_Err'].value
        ss += '{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1 TeV^-1  = ({:.2f} +/- {:.2f}) % Crab\n'.format(
            'Flux at pivot energy', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB_DIFF)

        val = d['Flux_Spec_PL_Int_1TeV'].value
        err = d['Flux_Spec_PL_Int_1TeV_Err'].value
        ss += '{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1  = ({:.2f} +/- {:.2f}) % Crab\n'.format(
            'PL   Flux(> 1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB)

        val = d['Flux_Spec_PL_Diff_1TeV'].value
        err = d['Flux_Spec_PL_Diff_1TeV_Err'].value
        ss += '{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1 TeV^-1  = ({:.2f} +/- {:.2f}) % Crab\n'.format(
            'PL   Flux(@ 1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB_DIFF)

        val = d['Index_Spec_PL']
        err = d['Index_Spec_PL_Err']
        ss += '{:<20s} : {:.2f} +/- {:.2f}\n'.format('PL   Index', val, err)

        return ss

    def _info_spec_ecpl(self):
        d = self.data
        ss = ''
        # ss = '{:<20s} : {:.1f}\n'.format('Pivot energy', d['Energy_Spec_ECPL_Pivot'])

        val = d['Flux_Spec_ECPL_Diff_1TeV'].value
        err = d['Flux_Spec_ECPL_Diff_1TeV_Err'].value
        ss += '{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1 TeV^-1  = ({:.2f} +/- {:.2f}) % Crab\n'.format(
            'ECPL   Flux(@ 1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB_DIFF)

        val = d['Flux_Spec_ECPL_Int_1TeV'].value
        err = d['Flux_Spec_ECPL_Int_1TeV_Err'].value
        ss += '{:<20s} : ({:.3f} +/- {:.3f}) x 10^-12 cm^-2 s^-1  = ({:.2f} +/- {:.2f}) % Crab\n'.format(
            'ECPL   Flux(> 1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB)

        val = d['Index_Spec_ECPL']
        err = d['Index_Spec_ECPL_Err']
        ss += '{:<20s} : {:.2f} +/- {:.2f}\n'.format('ECPL Index', val, err)

        val = d['Lambda_Spec_ECPL'].value
        err = d['Lambda_Spec_ECPL_Err'].value
        ss += '{:<20s} : {:.3f} +/- {:.3f} TeV^-1\n'.format('ECPL Lambda', val, err)

        # Use Gaussian analytical error propagation,
        # tested against the uncertainties package
        err = err / val ** 2
        val = 1. / val

        ss += '{:<20s} : {:.2f} +/- {:.2f} TeV\n'.format('ECPL E_cut', val, err)

        return ss

    def _info_flux_points(self):
        """Print flux point results"""
        d = self.data
        ss = '\n*** Flux points info ***\n\n'
        ss += 'Number of flux points: {}\n'.format(d['N_Flux_Points'])
        ss += 'Flux points table: \n\n\t'

        flux_points = self.flux_points.table.copy()

        energy_cols = ['e_ref', 'e_min', 'e_max']
        flux_cols = ['dnde', 'dnde_errn', 'dnde_errp', 'dnde_ul']
        cols = energy_cols + flux_cols + ['is_ul']
        flux_points = flux_points[cols]

        for _ in energy_cols:
            flux_points[_].format = '.3f'

        for _ in flux_cols:
            flux_points[_].format = '.3e'

        # convert table to string
        lines = flux_points.pformat(max_width=-1, max_lines=-1)
        ss += '\n'.join(lines)
        return ss + '\n'

    def _info_components(self):
        """Print info about the components."""
        if not hasattr(self, 'components'):
            return ''

        ss = '\n*** Gaussian component info ***\n\n'
        ss += 'Number of components: {}\n'.format(len(self.components))
        ss += '{:<20s} : {}\n\n'.format('Spatial components', self.data['Components'])

        for component in self.components:
            ss += str(component)
            ss += '\n\n'

        return ss

    @property
    def energy_range(self):
        """Spectral model energy range (`~astropy.units.Quantity` with length 2)."""
        e = u.Quantity([
            self.data['Energy_Range_Spec_Min'],
            self.data['Energy_Range_Spec_Max'],
        ])
        # Some EXTERN sources have no energy range information.
        # In those cases, we put a default
        use_default = np.isnan(e)
        e_default = [0.2, 50] * u.TeV
        e[use_default] = e_default[use_default]

        return e

    @property
    def spectral_model(self):
        """Spectral model (`~gammapy.spectrum.models.SpectralModel`).

        One of the following models:

        - ``Spectral_Model="PL"`` : `~gammapy.spectrum.models.PowerLaw`
        - ``Spectral_Model="ECPL"`` : `~gammapy.spectrum.models.ExponentialCutoffPowerLaw`
        """
        data = self.data
        spec_type = data['Spectral_Model'].strip()
        return self._spectral_model(spec_type)

    def _spectral_model(self, spec_type):
        """
        Create spectral model from source parameters.

        Parameters
        ----------
        spec_type : str
            Spectral model type either 'PL' or 'ECPL'.

        Returns
        -------
        model : `~gammapy.spectrum.models.SpectralModel`
            Spectral model instance.
        """
        data = self.data

        pars, errs = {}, {}

        if spec_type == 'PL':
            pars['index'] = data['Index_Spec_PL']
            pars['amplitude'] = data['Flux_Spec_PL_Diff_Pivot']
            pars['reference'] = data['Energy_Spec_PL_Pivot']
            errs['amplitude'] = data['Flux_Spec_PL_Diff_Pivot_Err']
            errs['index'] = data['Index_Spec_PL_Err'] * u.dimensionless_unscaled
            model = PowerLaw(**pars)
        elif spec_type == 'ECPL':
            pars['index'] = data['Index_Spec_ECPL']
            pars['amplitude'] = data['Flux_Spec_ECPL_Diff_Pivot']
            pars['reference'] = data['Energy_Spec_ECPL_Pivot']
            pars['lambda_'] = data['Lambda_Spec_ECPL']
            errs['index'] = data['Index_Spec_ECPL_Err'] * u.dimensionless_unscaled
            errs['amplitude'] = data['Flux_Spec_ECPL_Diff_Pivot_Err']
            errs['lambda_'] = data['Lambda_Spec_ECPL_Err']
            model = ExponentialCutoffPowerLaw(**pars)
        else:
            raise ValueError('Invalid spectral model: {}'.format(spec_type))

        model.parameters.set_parameter_errors(errs)
        return model

    def spatial_model(self, emin=1 * u.TeV, emax=1E5 * u.TeV):
        """
        Source spatial model.
        """
        d = self.data
        flux = self.spectral_model.integral(emin, emax)
        amplitude = flux.to('cm-2 s-1').value

        pars = {}
        glon = Angle(d['GLON']).wrap_at('180d')
        glat = Angle(d['GLAT']).wrap_at('180d')

        spatial_model = d['Spatial_Model']

        if self.is_pointlike:
            pars['amplitude'] = amplitude
            pars['x_0'] = glon.value
            pars['y_0'] = glat.value
            return Delta2D(**pars)

        elif spatial_model == 'Shell':
            # HGPS contains no information on shell width
            # Here we assuma a 5% shell width for all shells.
            shell_width_fraction = 0.05
            pars['x_0'] = glon.value
            pars['y_0'] = glat.value
            r_out = d['Size'].to('deg').value
            pars['width'] = r_out * shell_width_fraction
            pars['r_in'] = r_out * (1 - shell_width_fraction)
            pars['amplitude'] = amplitude
            return Shell2D(**pars)

        elif spatial_model in ['Gaussian', '2-Gaussian', '3-Gaussian']:
            amplitude_total = d['Flux_Map'].to('cm-2 s-1').value
            try:
                models = [component.spatial_model for component in self.components]
            except AttributeError:
                # there is one external source (HESS J1801-233) where there is no
                # component info available, so we create it here locally
                models = [HGPSGaussComponent(d).spatial_model]

            for model in models:
                # weight total flux according to relative amplitude
                model.amplitude = model.amplitude / amplitude_total * amplitude

            if spatial_model == 'Gaussian':
                return models[0]

            elif spatial_model == '2-Gaussian':
                model = models[0] + models[1]

                # use generic bounding box of 2 deg width and height
                model.bounding_box = ((glat.deg - 2, glat.deg + 2),
                                      (glon.deg - 2, glon.deg + 2))
                return model

            elif spatial_model == '3-Gaussian':
                model = models[0] + models[1] + models[2]

                # use generic bounding box of 2 deg width and height
                model.bounding_box = ((glat.deg - 2, glat.deg + 2),
                                      (glon.deg - 2, glon.deg + 2))
                return model

        else:
            raise ValueError('Not a valid spatial model{}'.format(spatial_model))

    @property
    def is_pointlike(self):
        """Source is pointlike"""
        d = self.data
        has_size_ul = np.isfinite(d['Size_UL'])
        pointlike = d['Spatial_Model'] == 'Point-Like'
        return pointlike or has_size_ul

    @property
    def flux_points(self):
        """Flux points (`~gammapy.spectrum.FluxPoints`)."""
        table = Table()
        table.meta['SED_TYPE'] = 'dnde'
        mask = ~np.isnan(self.data['Flux_Points_Energy'])

        table['e_ref'] = self.data['Flux_Points_Energy'][mask]
        table['e_min'] = self.data['Flux_Points_Energy_Min'][mask]
        table['e_max'] = self.data['Flux_Points_Energy_Max'][mask]

        table['dnde'] = self.data['Flux_Points_Flux'][mask]
        table['dnde_errn'] = self.data['Flux_Points_Flux_Err_Lo'][mask]
        table['dnde_errp'] = self.data['Flux_Points_Flux_Err_Hi'][mask]
        table['dnde_ul'] = self.data['Flux_Points_Flux_UL'][mask]
        table['is_ul'] = self.data['Flux_Points_Flux_Is_UL'][mask].astype('bool')

        return FluxPoints(table)


class SourceCatalogHGPS(SourceCatalog):
    """HESS Galactic plane survey (HGPS) source catalog.

    Note: this catalog isn't publicly available yet.
    H.E.S.S. members have access and can use this.
    """
    name = 'hgps'
    description = 'H.E.S.S. Galactic plane survey (HGPS) source catalog'
    source_object_class = SourceCatalogObjectHGPS

    def __init__(self, filename=None, hdu='HGPS_SOURCES'):
        if not filename:
            filename = Path(os.environ['HGPS_ANALYSIS']) / 'data/catalogs/HGPS3/release/HGPS_v0.4.fits'

        filename = str(make_path(filename))
        table = Table.read(filename, hdu=hdu)

        source_name_alias = ('Identified_Object',)
        super(SourceCatalogHGPS, self).__init__(
            table=table,
            source_name_alias=source_name_alias,
        )

        self.components = Table.read(filename, hdu='HGPS_GAUSS_COMPONENTS')
        self.associations = Table.read(filename, hdu='HGPS_ASSOCIATIONS')
        self.identifications = Table.read(filename, hdu='HGPS_IDENTIFICATIONS')
        self._large_scale_component = Table.read(filename, hdu='HGPS_LARGE_SCALE_COMPONENT')

    @property
    def large_scale_component(self):
        """Large sclae component model (`~gammapy.background.models.GaussianBand2D`).
        """
        table = self._large_scale_component
        return GaussianBand2D(table, spline_kwargs=dict(k=3, s=0, ext=0))

    def _make_source_object(self, index):
        """Make one source object.

        Parameters
        ----------
        index : int
            Row index

        Returns
        -------
        source : `SourceCatalogObject`
            Source object
        """
        source = super(SourceCatalogHGPS, self)._make_source_object(index)
        if hasattr(self, 'components'):
            if source.data['Components'] != '':
                self._attach_component_info(source)
        if hasattr(self, 'associations'):
            self._attach_association_info(source)
        # TODO: implement or remove:
        # if source.data['Source_Class'] != 'Unid':
        #    self._attach_identification_info(source)
        return source

    def _attach_component_info(self, source):
        source.components = []
        lookup = SourceCatalog(self.components, source_name_key='Component_ID')
        for name in source.data['Components'].split(', '):
            component = HGPSGaussComponent(data=lookup[name].data)
            source.components.append(component)

    def _attach_association_info(self, source):
        source.associations = []
        _ = source.data['Source_Name'] == self.associations['Source_Name']

        source.associations = list(self.associations['Association_Name'][_])
