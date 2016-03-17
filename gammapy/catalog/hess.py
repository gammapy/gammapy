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

    def __str__(self):
        """Pretty-print source data"""
        d = self.data
        ss = '\nComponent {}:\n'.format(d['Component_ID'])
        ss += '{:<20s} : {:8.3f} \u00B1 {:.3f} deg\n'.format('GLON', d['GLON'],
              d['GLON_Err'])
        ss += '{:<20s} : {:8.3f} \u00B1 {:.3f} deg\n'.format('GLAT', d['GLAT'],
              d['GLAT_Err'])
        ss += '{:<20s} : {:.3f} \u00B1 {:.3f} deg\n'.format(
        'Size', d['Size'], d['Size_Err'])
        val, err = d['Flux_Map'], d['Flux_Map_Err']
        ss += '{:<20s} : ({:.2f} \u00B1 {:.2f}) x 10^-12 cm^-2 s^-1 = ({:.1f} \u00B1 {:.1f}) % Crab'.format(
            'Flux (>1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB)


class SourceCatalogObjectHGPS(SourceCatalogObject):
    """One object from the HGPS catalog.
    """
    def __str__(self):
        """Print default summary info string"""
        return self.summary()

    def summary(self, info='basic'):
        """Print summary info string

        Parameters
        ----------
        info : str {all, basic, map, spec, components, associations, references}
            Comma separated list of options
        """
        if info == 'all':
            info = 'basic,maps,spec,components,associations,references'

        ss = ''
        ops = info.split(',')
        if 'basic' in ops:
            ss += self._info_basic()
        if 'map' in ops:
            ss += self._info_map()
        if 'spec' in ops:
            ss += self._info_spec()
        if 'components' in ops:
            ss += self._info_components()
        if 'assiciations' in ops:
            ss += self._info_associations()

        return ss

    def _info_basic(self):
        """Print basic info."""
        d = self.data
        ss = '\n*** Basic info ***\n\n'
        ss += '{:<20s} : {}\n'.format('Source name', d['Source_Name'])
        ss += '{:<20s} : {}\n'.format('Analysis reference', d['Analysis_Reference'])
        ss += '{:<20s} : {}\n'.format('Source class', d['Source_Class'])
        ss += '{:<20s} : {}\n'.format('Spatial model', d['Spatial_Model'])
        ss += '{:<20s} : {}\n'.format('Spatial components', d['Components'])
        ss += '{:<20s} : {}\n'.format('Spectral model', d['Spectral_Model'])
        ss += '{:<20s} : {}\n'.format('Associated_Object', d['Associated_Object'])
        ss += 'Catalog row index (zero-based) : {}\n'.format(d['catalog_row_index'])
        return ss

    def _info_map(self):
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

        ss += '{:<20s} : {}\n'.format('Spatial model', d['Spatial_Model'])
        ss += '{:<20s} : {:.1f}\n'.format('TS', d['Sqrt_TS'] ** 2)
        ss += '{:<20s} : {:.1f}\n'.format('sqrt(TS)', d['Sqrt_TS'])

        ss += '{:<20s} : {:.3f} \u00B1 {:.3f} (UL: {:.3f}) deg\n'.format(
            'Size', d['Size'], d['Size_Err'], d['Size_UL'])
        ss += '{:<20s} : {:.3f} deg\n'.format('R70', d['R70'])
        ss += '{:<20s} : {:.3f} deg\n'.format('RSpec', d['RSpec'])

        ss += '{:<20s} : {:.1f}\n'.format('Excess_Model_Total', d['Excess_Model_Total'])
        ss += '{:<20s} : {:.1f}\n'.format('Excess_RSpec', d['Excess_RSpec'])
        # TODO: column still missing in catalog:
        # ss += '{:<20s} : {:.1f}\n'.format('Excess_RSpec_Model', d['Excess_RSpec_Model'])
        ss += '{:<20s} : {:.1f}\n'.format('Background_RSpec', d['Background_RSpec'])
        # TODO: column still missing in catalog:
        # ss += '{:<20s} : {:.1f} hours\n'.format('Livetime', d['Livetime'])

        # TODO: column still missing in catalog:
        # ss += '{:<20s} : {:.1f} TeV\n'.format('Energy_Threshold', d['Energy_Threshold'])
        ss += '{:<20s} : {}\n'.format('ROI_Number', d['ROI_Number'])

        val, err = d['Flux_Map'], d['Flux_Map_Err']
        ss += '{:<20s} : ({:.2f} \u00B1 {:.2f}) x 10^-12 cm^-2 s^-1 = ({:.1f} \u00B1 {:.1f}) % Crab\n'.format(
        'Source flux (>1 TeV)', val / FF, err / FF, val * FLUX_TO_CRAB, err * FLUX_TO_CRAB)

        ss += '\nFluxes in RSpec (> 1 TeV):\n'

        ss += '{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab\n'.format(
        'Map measurement', d['Flux_Map_RSpec_Data'] / FF, d['Flux_Map_RSpec_Data'] * FLUX_TO_CRAB)

        ss += '{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab\n'.format(
        'Total model', d['Flux_Map_RSpec_Total'] / FF, d['Flux_Map_RSpec_Total'] * FLUX_TO_CRAB)

        ss += '{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab\n'.format(
            'Source model', d['Flux_Map_RSpec_Source'] / FF, d['Flux_Map_RSpec_Source'] * FLUX_TO_CRAB)

        ss += '{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab\n'.format(
        'Other component model', d['Flux_Map_RSpec_Other'] / FF, d['Flux_Map_RSpec_Other'] * FLUX_TO_CRAB)

        ss += '{:<30s} : {:.2f} x 10^-12 cm^-2 s^-1 = {:5.1f} % Crab\n'.format(
        'Diffuse component model', d['Flux_Map_RSpec_Diffuse'] / FF, d['Flux_Map_RSpec_Diffuse'] * FLUX_TO_CRAB)

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

        val = d['Flux_Spec_PL_Int_1TeV']
        err = d['Flux_Spec_PL_Int_1TeV_Err']
        ss += 'PL   Flux(>1 TeV) : {:.1f} \u00B1 {:.1f}\n'.format(val / FF, err / FF)
        val = d['Flux_Spec_ECPL_Int_1TeV']
        err = d['Flux_Spec_ECPL_Int_1TeV_Err']
        ss += 'ECPL Flux(>1 TeV) : {:.1f} \u00B1 {:.1f}\n'.format(val / FF, err / FF)

        val = d['Index_Spec_PL']
        err = d['Index_Spec_PL_Err']
        ss += 'PL   Index    : {:.2f} \u00B1 {:.2f}\n'.format(val, err)
        val = d['Index_Spec_ECPL']
        err = d['Index_Spec_ECPL_Err']
        ss += 'ECPL Index    : {:.2f} \u00B1 {:.2f}\n'.format(val, err)

        val = d['Lambda_Spec_ECPL']
        err = d['Lambda_Spec_ECPL_Err']
        ss +='ECPL Lambda   : {:.3f} \u00B1 {:.3f}\n'.format(val, err)
        # TODO: change to catalog parameters here as soon as they are filled!
        # val = d['Energy_Cutoff_Spec_ECPL']
        # err = d['Energy_Cutoff_Spec_ECPL_Err']
        import uncertainties
        energy = 1 / uncertainties.ufloat(val, err)
        val, err = energy.nominal_value, energy.std_dev
        ss +='ECPL E_cut    : {:.1f} \u00B1 {:.1f}\n'.format(val, err)
        return ss

    def _info_components(self, file=None):
        """Print info about the components."""
        if file is None:
            file = sys.stdout

        if not hasattr(self, 'components'):
            return

        ss = '\n*** Gaussian component info ***\n\n'
        ss += 'Number of components: {}\n'.format(len(self.components))
        ss += '{:<20s} : {}\n\n'.format('Spatial components', self.data['Components'])

        for component in self.components:
            ss += str(component)
        return ss

    def _info_associations(self, file=None):
        ss = '\n*** Source associations info ***\n\n'
        associations = ', '.join(self.associations)
        ss += 'List of associated objects: {}\n'.format(associations)
        return ss

    def _info_references(self, file=None):
        ss = '\n*** Further source info ***\n\n'
        ss += self.data['TeVCat_Reference']
        ss += '\n'
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
            filename = Path(os.environ['HGPS_ANALYSIS']) / 'data/catalogs/HGPS3/HGPS_v0.3.1.fits'
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
        source.associations = []
        _ = source.data['Source_Name'] == self.associations['Source_Name']

        source.associations = list(self.associations['Association_Name'][_])


