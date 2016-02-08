# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
TODO:

- [ ] URL link to TeVCat
- [ ] Comparison with previous publication
- [ ] Links to SNRCat
- [ ] Show image in ds9 or js9
- [ ] HGPS Component and Association as separate classes?
- [ ] Take units automatically from table?
- [ ] Fluxes in Crab units
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
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

    def print_info(self, file=None):
        """Pretty-print source data"""
        if not file:
            file = sys.stdout
        d = self.data   
        print('\nComponent {}:'.format(d['Component_ID']), file=file)
        print('{:<20s} : {:8.3f} \u00B1 {:.3f} deg'.format('GLON', d['GLON'],
              d['GLON_Err']), file=file)
        print('{:<20s} : {:8.3f} \u00B1 {:.3f} deg'.format('GLAT', d['GLAT'],
              d['GLAT_Err']), file=file)
        print('{:<20s} : {:.3f} \u00B1 {:.3f} deg'
              ''.format('Size', d['Size'], d['Size_Err']), file=file)
        val, err = d['Flux_Map'], d['Flux_Map_Err']
        print('{:<20s} : ({:.2f} \u00B1 {:.2f}) x 10^-12 cm^-2 s^-1 = ({:.1f} \u00B1 {:.1f}) % Crab'
              ''.format('Flux (>1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB), file=file)



class SourceCatalogObjectHGPS(SourceCatalogObject):
    """One object from the HGPS catalog.
    """

    def print_info(self, file=None):
        """Print all info."""
        self.print_info_basic(file=file)
        self.print_info_map(file=file)
        self.print_info_spec(file=file)
        self.print_info_components(file=file)
        self.print_info_associations(file=file)

    def print_info_basic(self, file=None):
        """Print basic info."""
        if file is None:
            file = sys.stdout

        d = self.data
        print('\n*** Basic info ***\n', file=file)

        print('{:<20s} : {}'.format('Source name', d['Source_Name']), file=file)
        print('{:<20s} : {}'.format('Analysis reference', d['Analysis_Reference']), file=file)
        print('{:<20s} : {}'.format('Source class', d['Source_Class']), file=file)
        print('{:<20s} : {}'.format('Spatial model', d['Spatial_Model']), file=file)
        print('{:<20s} : {}'.format('Spatial components', d['Components']), file=file)
        print('{:<20s} : {}'.format('Spectral model', d['Spectral_Model']), file=file)
        print('{:<20s} : {}'.format('Associated_Object', d['Associated_Object']), file=file)
        print('Catalog row index (zero-based) : {}'.format(d['catalog_row_index']), file=file)

    def print_info_map(self, file=None):
        """Print info from map analysis."""
        if file is None:
            file = sys.stdout

        d = self.data
        print('\n*** Info from map analysis ***\n', file=file)

        ra_str = Angle(d['RAJ2000'], 'deg').to_string(unit='hour', precision=0)
        dec_str = Angle(d['DEJ2000'], 'deg').to_string(unit='deg', precision=0)
        print('{:<20s} : {:8.3f} deg = {}'.format('RA', d['RAJ2000'], ra_str), file=file)
        print('{:<20s} : {:8.3f} deg = {}'.format('DEC', d['DEJ2000'], dec_str), file=file)

        print('{:<20s} : {:8.3f} \u00B1 {:.3f} deg'.format('GLON', d['GLON'], d['GLON_Err']), file=file)
        print('{:<20s} : {:8.3f} \u00B1 {:.3f} deg'.format('GLAT', d['GLAT'], d['GLAT_Err']), file=file)

        print('{:<20s} : {:.3f} deg'.format('Position Error (68%)', d['Pos_Err_68']), file=file)
        print('{:<20s} : {:.3f} deg'.format('Position Error (95%)', d['Pos_Err_95']), file=file)

        print('{:<20s} : {}'.format('Spatial model', d['Spatial_Model']), file=file)
        print('{:<20s} : {:.1f}'.format('TS', d['Sqrt_TS'] ** 2), file=file)
        print('{:<20s} : {:.1f}'.format('sqrt(TS)', d['Sqrt_TS']), file=file)

        print('{:<20s} : {:.3f} \u00B1 {:.3f} (UL: {:.3f}) deg'
              ''.format('Size', d['Size'], d['Size_Err'], d['Size_UL']), file=file)
        print('{:<20s} : {:.3f} deg'.format('R70', d['R70']), file=file)
        print('{:<20s} : {:.3f} deg'.format('RSpec', d['RSpec']), file=file)

        print('{:<20s} : {:.1f}'.format('Excess_Model_Total', d['Excess_Model_Total']), file=file)
        print('{:<20s} : {:.1f}'.format('Excess_RSpec', d['Excess_RSpec']), file=file)
        # TODO: column still missing in catalog:
        # print('{:<20s} : {:.1f}'.format('Excess_RSpec_Model', d['Excess_RSpec_Model']), file=file)
        print('{:<20s} : {:.1f}'.format('Background_RSpec', d['Background_RSpec']), file=file)
        # TODO: column still missing in catalog:
        # print('{:<20s} : {:.1f} hours'.format('Livetime', d['Livetime']), file=file)

        # TODO: column still missing in catalog:
        # print('{:<20s} : {:.1f} TeV'.format('Energy_Threshold', d['Energy_Threshold']), file=file)
        print('{:<20s} : {}'.format('ROI_Number', d['ROI_Number']), file=file)

        val, err = d['Flux_Map'], d['Flux_Map_Err']
        print('{:<20s} : ({:.2f} \u00B1 {:.2f}) x 10^-12 cm^-2 s^-1 = ({:.1f} \u00B1 {:.1f}) % Crab'
              ''.format('Source flux (>1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB), file=file)

        print('\nFluxes in RSpec (> 1 TeV):\n', file=file)

        print('{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab'
              ''.format('Map measurement', d['Flux_Map_RSpec_Data'] / FF, d['Flux_Map_RSpec_Data'] * FLUX_TO_CRAB), file=file)

        print('{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab'
              ''.format('Total model', d['Flux_Map_RSpec_Total'] / FF, d['Flux_Map_RSpec_Total'] * FLUX_TO_CRAB), file=file)

        print('{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab'
              ''.format('Source model', d['Flux_Map_RSpec_Source'] / FF, d['Flux_Map_RSpec_Source'] * FLUX_TO_CRAB), file=file)

        print('{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab'
              ''.format('Other component model', d['Flux_Map_RSpec_Other'] / FF, d['Flux_Map_RSpec_Other'] * FLUX_TO_CRAB), file=file)

        print('{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab'
              ''.format('Diffuse component model', d['Flux_Map_RSpec_Diffuse'] / FF, d['Flux_Map_RSpec_Diffuse'] * FLUX_TO_CRAB), file=file)

        print('{:<35s} : {:5.1f} %'.format('Containment in RSpec', 100 * d['Containment_RSpec']), file=file)
        print('{:<35s} : {:5.1f} %'.format('Contamination in RSpec', 100 * d['Contamination_RSpec']), file=file)
        label, val = 'Flux correction (RSpec -> Total)', 100 * d['Flux_Correction_RSpec_To_Total']
        print('{:<35s} : {:5.1f} %'.format(label, val), file=file)
        label, val = 'Flux correction (Total -> RSpec)', 100 * (1 / d['Flux_Correction_RSpec_To_Total'])
        print('{:<35s} : {:5.1f} %'.format(label, val, file=file))

    def print_info_spec(self, file=None):
        """Print info from spectral analysis."""
        if file is None:
            file = sys.stdout

        d = self.data
        print('\n*** Info from spectral analysis ***\n', file=file)

        val = d['Flux_Spec_PL_Int_1TeV']
        err = d['Flux_Spec_PL_Int_1TeV_Err']
        print('PL   Flux(>1 TeV) : {:.1f} \u00B1 {:.1f}'.format(val / FF, err / FF), file=file)
        val = d['Flux_Spec_ECPL_Int_1TeV']
        err = d['Flux_Spec_ECPL_Int_1TeV_Err']
        print('ECPL Flux(>1 TeV) : {:.1f} \u00B1 {:.1f}'.format(val / FF, err / FF), file=file)

        val = d['Index_Spec_PL']
        err = d['Index_Spec_PL_Err']
        print('PL   Index    : {:.2f} \u00B1 {:.2f}'.format(val, err), file=file)
        val = d['Index_Spec_ECPL']
        err = d['Index_Spec_ECPL_Err']
        print('ECPL Index    : {:.2f} \u00B1 {:.2f}'.format(val, err), file=file)

        val = d['Lambda_Spec_ECPL']
        err = d['Lambda_Spec_ECPL_Err']
        print('ECPL Lambda   : {:.3f} \u00B1 {:.3f}'.format(val, err), file=file)
        # TODO: change to catalog parameters here as soon as they are filled!
        # val = d['Energy_Cutoff_Spec_ECPL']
        # err = d['Energy_Cutoff_Spec_ECPL_Err']
        import uncertainties
        energy = 1 / uncertainties.ufloat(val, err)
        val, err = energy.nominal_value, energy.std_dev
        print('ECPL E_cut    : {:.1f} \u00B1 {:.1f}'.format(val, err), file=file)

    def print_info_components(self, file=None):
        """Print info about the components."""
        if file is None:
            file = sys.stdout

        # import IPython; IPython.embed(); 1/0

        if not hasattr(self, 'components'):
            return

        print('\n*** Gaussian component info ***\n', file=file)
        print('Number of components: {}'.format(len(self.components)))
        print('{:<20s} : {}\n'.format('Spatial components', self.data['Components']), file=file)

        for component in self.components:
            component.print_info(file=file)

    def print_info_associations(self, file=None):
        print('\n*** Source associations info ***\n', file=file)
        associations = ', '.join(self.associations)
        print('List of associated objects: {}'.format(associations), file=file)




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
            filename = Path(os.environ['HGPS_ANALYSIS']) / 'data/catalogs/HGPS3/HGPS_v0.3.1.fits'
        self.filename = str(filename)
        self._hdu_list = fits.open(str(filename))
        table = Table(self._hdu_list['HGPS_SOURCES'].data)
        self.components = Table(self._hdu_list['HGPS_COMPONENTS'].data)
        self.associations = Table(self._hdu_list['HGPS_ASSOCIATIONS'].data)
        self.identifications = Table(self._hdu_list['HGPS_IDENTIFICATIONS'].data)
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
        #if source.data['Source_Class'] != 'Unid':
        #    self._attach_identification_info(source)
        return source

    def _attach_component_info(self, source):
        source.components = []
        lookup = SourceCatalog(self.components, source_name_key='Component_ID')
        for name in source.data['Components'].split(', '):
            component = HGPSGaussComponent(data=lookup[name].data)
            source.components.append(component)

    def _attach_association_info(self, source):
        from IPython import embed; embed()
        source.associations = []
        _ = source.data['Source_Name'] == self.associations['Source_Name']

        source.associations = list(self.associations['Association_Name'][_])


