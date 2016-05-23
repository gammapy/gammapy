# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""HESS Galactic plane survey (HGPS) catalog.

(Not released yet.)

TODO:
- [ ] Comparison with previous publication
- [ ] Links to SNRCat
- [ ] Show image in ds9 or js9
- [ ] Take units automatically from table?
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from collections import OrderedDict
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import Angle
from ..extern.pathlib import Path
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

    @property
    def name(self):
        """Source name"""
        name = self.data[self._source_name_key]
        return name.strip()

    @property
    def index(self):
        """Row index of source in catalog"""
        return self.data[self._source_index_key]

    def __str__(self):
        """Pretty-print source data"""
        d = self.data
        ss = 'Component {}:\n'.format(d['Component_ID'])
        ss += '{:<20s} : {:8.3f} \u00B1 {:.3f} deg\n'.format('GLON', d['GLON'],
                                                             d['GLON_Err'])
        ss += '{:<20s} : {:8.3f} \u00B1 {:.3f} deg\n'.format('GLAT', d['GLAT'],
                                                             d['GLAT_Err'])
        ss += '{:<20s} : {:.3f} \u00B1 {:.3f} deg\n'.format('Size', d['Size'], d['Size_Err'])
        val, err = d['Flux_Map'], d['Flux_Map_Err']
        ss += '{:<20s} : ({:.2f} \u00B1 {:.2f}) x 10^-12 cm^-2 s^-1 = ({:.1f} \u00B1 {:.1f}) % Crab'.format(
            'Flux (>1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB)
        return ss


class SourceCatalogObjectHGPS(SourceCatalogObject):
    """One object from the HGPS catalog.
    """

    def __str__(self):
        """Print default summary info string"""
        return self.summary()

    def summary(self, info='all'):
        """Print summary info string.

        Parameters
        ----------
        info : {'all', 'basic', 'map', 'spec', 'flux_points', 'components', 'associations'}
            Comma separated list of options
        """
        if info == 'all':
            info = 'basic,map,spec,flux_points,components,associations'

        ss = ''
        ops = info.split(',')
        if 'basic' in ops:
            ss += self._summary_basic()
        if 'map' in ops:
            ss += self._summary_map()
        if 'spec' in ops:
            ss += self._summary_spec()
        if 'flux_points' in ops:
            ss += self._summary_flux_points()
        if 'components' in ops:
            ss += self._summary_components()
        if 'associations' in ops:
            ss += self._summary_associations()
        return ss

    def _summary_basic(self):
        """Print basic info."""
        d = self.data
        ss = '\n*** Basic info ***\n\n'
        # ss += 'Catalog row index (zero-based) : {}\n'.format(d['catalog_row_index'])
        ss += '{:<20s} : {}\n'.format('Source name', d['Source_Name'])
        ss += '{:<20s} : {}\n'.format('Analysis reference', d['Analysis_Reference'])
        ss += '{:<20s} : {}\n'.format('Source class', d['Source_Class'])
        ss += '{:<20s} : {}\n'.format('Associated object', d['Associated_Object'])
        ss += '{:<20s} : {}\n'.format('TeVCat reference', d['TeVCat_Reference'])
        ss += '\n'

        return ss

    def _summary_map(self):
        """Print info from map analysis."""
        d = self.data
        ss = '\n*** Info from map analysis ***\n\n'

        ra_str = Angle(d['RAJ2000'], 'deg').to_string(unit='hour', precision=0)
        dec_str = Angle(d['DEJ2000'], 'deg').to_string(unit='deg', precision=0)
        ss += '{:<20s} : {:8.3f} deg = {}\n'.format('RA', d['RAJ2000'], ra_str)
        ss += '{:<20s} : {:8.3f} deg = {}\n'.format('DEC', d['DEJ2000'], dec_str)

        ss += '{:<20s} : {:8.3f} \u00B1 {:.3f} deg\n'.format('GLON', d['GLON'], d['GLON_Err'])
        ss += '{:<20s} : {:8.3f} \u00B1 {:.3f} deg\n'.format('GLAT', d['GLAT'], d['GLAT_Err'])

        ss += '{:<20s} : {:.3f} deg\n'.format('Position Error (68%)', d['Pos_Err_68'])
        ss += '{:<20s} : {:.3f} deg\n'.format('Position Error (95%)', d['Pos_Err_95'])

        ss += '{:<20s} : {}\n'.format('ROI number', d['ROI_Number'])
        ss += '{:<20s} : {}\n'.format('Spatial model', d['Spatial_Model'])
        ss += '{:<20s} : {}\n'.format('Spatial components', d['Components'])

        ss += '{:<20s} : {:.1f}\n'.format('TS', d['Sqrt_TS'] ** 2)
        ss += '{:<20s} : {:.1f}\n'.format('sqrt(TS)', d['Sqrt_TS'])

        ss += '{:<20s} : {:.3f} \u00B1 {:.3f} (UL: {:.3f}) deg\n'.format(
            'Size', d['Size'], d['Size_Err'], d['Size_UL'])

        ss += '{:<20s} : {:.3f} deg\n'.format('R70', d['R70'])
        ss += '{:<20s} : {:.3f} deg\n'.format('RSpec', d['RSpec'])

        ss += '{:<20s} : {:.1f}\n'.format('Total model excess', d['Excess_Model_Total'])
        ss += '{:<20s} : {:.1f}\n'.format('Excess in r_spec', d['Excess_RSpec'])

        # TODO: listed in paper, but not present in catalog:
        # ss += '{:<20s} : {:.1f}\n'.format('Excess_RSpec_Model', d['Excess_RSpec_Model'])

        ss += '{:<20s} : {:.1f}\n'.format('Background in r_spec', d['Background_RSpec'])

        # TODO: listed in paper, but not present in catalog:
        # ss += '{:<20s} : {:.1f} hours\n'.format('Livetime', d['Livetime'])

        # TODO: listed in paper, but not present in catalog:
        # ss += '{:<20s} : {:.1f} TeV\n'.format('Energy threshold', d['Energy_Threshold'])

        val, err = d['Flux_Map'], d['Flux_Map_Err']
        ss += '{:<20s} : ({:.2f} \u00B1 {:.2f}) x 10^-12 cm^-2 s^-1 = ({:.1f} \u00B1 {:.1f}) % Crab\n'.format(
            'Source flux (>1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB)

        ss += '\nFluxes in RSpec (> 1 TeV):\n'

        ss += '{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab\n'.format(
            'Map measurement', d['Flux_Map_RSpec_Data'] / FF, d['Flux_Map_RSpec_Data'] * FLUX_TO_CRAB)

        ss += '{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab\n'.format(
            'Source model', d['Flux_Map_RSpec_Source'] / FF, d['Flux_Map_RSpec_Source'] * FLUX_TO_CRAB)

        ss += '{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab\n'.format(
            'Other component model', d['Flux_Map_RSpec_Other'] / FF, d['Flux_Map_RSpec_Other'] * FLUX_TO_CRAB)

        ss += '{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab\n'.format(
            'Diffuse component model', d['Flux_Map_RSpec_Diffuse'] / FF, d['Flux_Map_RSpec_Diffuse'] * FLUX_TO_CRAB)

        ss += '{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab\n'.format(
            'Total model', d['Flux_Map_RSpec_Total'] / FF, d['Flux_Map_RSpec_Total'] * FLUX_TO_CRAB)

        ss += '{:<35s} : {:5.1f} %\n'.format('Containment in RSpec', 100 * d['Containment_RSpec'])
        ss += '{:<35s} : {:5.1f} %\n'.format('Contamination in RSpec', 100 * d['Contamination_RSpec'])
        label, val = 'Flux correction (RSpec -> Total)', 100 * d['Flux_Correction_RSpec_To_Total']
        ss += '{:<35s} : {:5.1f} %\n'.format(label, val)
        label, val = 'Flux correction (Total -> RSpec)', 100 * (1 / d['Flux_Correction_RSpec_To_Total'])
        ss += '{:<35s} : {:5.1f} %\n'.format(label, val)

        return ss

    def _summary_spec(self):
        """Print info from spectral analysis."""
        d = self.data
        ss = '\n*** Info from spectral analysis ***\n\n'

        ss += '{:<20s} : {:.1f} hours\n'.format('Livetime', d['Livetime_Spec'])

        lo = d['Energy_Range_Spec_Lo']
        hi = d['Energy_Range_Spec_Hi']
        ss += '{:<20s} : {:.1f} TeV\n'.format('Energy range: {} to {}', lo, hi)

        ss += '{:<20s} : {:.1f}\n'.format('Background', d['Background_Spec'])
        ss += '{:<20s} : {:.1f}\n'.format('Excess', d['Excess_Spec'])
        ss += '{:<20s} : {}\n'.format('Spectral model', d['Spectral_Model'])

        # TODO: can we just use the Gammapy model classes here instead of duplicating the code?
        ss += self._summary_spec_pl()
        ss += self._summary_spec_ecpl()

        return ss

    def _summary_spec_pl(self):
        d = self.data
        ss = '{:<20s} : {:.1f} TeV\n'.format('Pivot energy', d['Energy_Spec_PL_Pivot'])

        val = d['Flux_Spec_PL_Diff_Pivot']
        err = d['Flux_Spec_PL_Diff_Pivot_Err']
        ss += '{:<20s} : ({:.1f} \u00B1 {:.1f}) x 10^-12 cm^-2 s^-1 TeV^-1  = ({:.1f} \u00B1 {:.1f}) % Crab\n'.format(
            'Flux at pivot energy', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB_DIFF)

        val = d['Flux_Spec_PL_Int_1TeV']
        err = d['Flux_Spec_PL_Int_1TeV_Err']
        ss += '{:<20s} : ({:.1f} \u00B1 {:.1f}) x 10^-12 cm^-2 s^-1  = ({:.1f} \u00B1 {:.1f}) % Crab\n'.format(
            'PL   Flux(> 1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB)

        val = d['Flux_Spec_PL_Diff_1TeV']
        err = d['Flux_Spec_PL_Diff_1TeV_Err']
        ss += '{:<20s} : ({:.1f} \u00B1 {:.1f}) x 10^-12 cm^-2 s^-1 TeV^-1  = ({:.1f} \u00B1 {:.1f}) % Crab\n'.format(
            'PL   Flux(@ 1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB_DIFF)

        val = d['Index_Spec_PL']
        err = d['Index_Spec_PL_Err']
        ss += '{:<20s} : {:.2f} \u00B1 {:.2f}\n'.format('PL   Index', val, err)

        return ss

    def _summary_spec_ecpl(self):
        d = self.data
        ss = ''
        # ss = '{:<20s} : {:.1f} TeV\n'.format('Pivot energy', d['Energy_Spec_ECPL_Pivot'])

        val = d['Flux_Spec_ECPL_Diff_1TeV']
        err = d['Flux_Spec_ECPL_Diff_1TeV_Err']
        ss += '{:<20s} : ({:.1f} \u00B1 {:.1f}) x 10^-12 cm^-2 s^-1 TeV^-1  = ({:.1f} \u00B1 {:.1f}) % Crab\n'.format(
            'ECPL   Flux(@ 1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB_DIFF)

        val = d['Flux_Spec_ECPL_Int_1TeV']
        err = d['Flux_Spec_ECPL_Int_1TeV_Err']
        ss += '{:<20s} : ({:.1f} \u00B1 {:.1f}) x 10^-12 cm^-2 s^-1  = ({:.1f} \u00B1 {:.1f}) % Crab\n'.format(
            'ECPL   Flux(> 1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB)

        val = d['Index_Spec_ECPL']
        err = d['Index_Spec_ECPL_Err']
        ss += '{:<20s} : {:.2f} \u00B1 {:.2f}\n'.format('ECPL Index', val, err)

        val = d['Lambda_Spec_ECPL']
        err = d['Lambda_Spec_ECPL_Err']
        ss += '{:<20s} : {:.3f} \u00B1 {:.3f} TeV^-1\n'.format('ECPL Lambda', val, err)

        val = d['Energy_Cutoff_Spec_ECPL']
        err = d['Energy_Cutoff_Spec_ECPL_Err']

        import uncertainties
        energy = 1 / uncertainties.ufloat(val, err)
        val, err = energy.nominal_value, energy.std_dev
        ss += '{:<20s} : {:.1f} \u00B1 {:.1f} TeV\n'.format('ECPL E_cut', val, err)

        val = d['TS_ECPL_over_PL']
        ss += '{:<20s} : {:.1f}\n'.format('TS ECPL over PL', val)

        return ss

    def _summary_flux_points(self):
        """Print flux point results"""
        d = self.data
        ss = '\n*** TODO: print flux points info ***\n\n'
        return ss

    def _summary_components(self):
        """Print info about the components."""
        if not hasattr(self, 'components'):
            return ''

        ss = '\n*** Gaussian component info ***\n\n'
        ss += 'Number of components: {}\n'.format(len(self.components))
        ss += '{:<20s} : {}\n\n'.format('Spatial components', self.data['Components'])

        for component in self.components:
            # Call __str__ directly to 
            ss += component.__str__()
            ss += '\n\n'
        return ss

    def _summary_associations(self):
        ss = '\n*** Source associations info ***\n\n'
        associations = ', '.join(self.associations)
        ss += 'List of associated objects: {}\n'.format(associations)
        return ss


class SourceCatalogHGPS(SourceCatalog):
    """HESS Galactic plane survey (HGPS) source catalog.

    Note: this catalog isn't publicly available yet.
    For now you need to be a H.E.S.S. member with an account
    at MPIK to fetch it.
    """
    name = 'hgps'
    description = 'H.E.S.S. Galactic plane survey (HGPS) source catalog'
    source_object_class = SourceCatalogObjectHGPS

    def __init__(self, filename=None):
        if not filename:
            filename = Path(os.environ['HGPS_ANALYSIS']) / 'data/catalogs/HGPS3/release/HGPS_v0.4.fits'
        self.filename = str(filename)
        self.hdu_list = fits.open(str(filename))
        table = Table(self.hdu_list['HGPS_SOURCES'].data)
        self.components = Table(self.hdu_list['HGPS_COMPONENTS'].data)
        self.associations = Table(self.hdu_list['HGPS_ASSOCIATIONS'].data)
        self.identifications = Table(self.hdu_list['HGPS_IDENTIFICATIONS'].data)
        super(SourceCatalogHGPS, self).__init__(table=table)

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

        if source.data['Components'] != '':
            self._attach_component_info(source)
        self._attach_association_info(source)
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
