# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
HAWC catalogs (https://www.hawc-observatory.org)
"""

from .core import SourceCatalog, SourceCatalogObject
from ..utils.scripts import make_path
from astropy.table import Table
import astropy.units as u
import numpy as np
from ..spectrum.models import PowerLaw


class SourceCatalogObject2HWC(SourceCatalogObject):
    """One source from the HAWC 2FHL catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalog2HWC`.
    """

    _source_name_key = 'source_name'
    _source_index_key = 'catalog_row_index'

    def __str__(self):
        return self.info()

    def info(self, info='all'):
        """Summary info string.

        Parameters
        ----------
        info : {'all', 'basic', 'position', 'spectrum}
            Comma separated list of options
        """

        if info == 'all':
            info = 'basic,position,spectrum'

        ss = ''
        ops = info.split(',')
        if 'basic' in ops:
            ss += self._info_basic()
        if 'position' in ops:
            ss += self._info_position()
        if 'spectrum' in ops:
            ss += self._info_spectrum()

        return ss

    def _info_basic(self):
        """Print basic info."""
        d = self.data
        ss = '\n*** Basic info ***\n\n'
        ss += 'Catalog row index (zero-based) : {}\n'.format(d['catalog_row_index'])
        ss += '{:<15s} : {}\n'.format('Source name:', d['source_name'])

        return ss

    def _info_position(self):
        """Print position info."""
        d = self.data
        ss = '\n*** Position info ***\n\n'
        ss += 'Measurement:\n'
        ss += '{:20s} : {:.3f}\n'.format('RA', d['ra'])
        ss += '{:20s} : {:.3f}\n'.format('DEC', d['dec'])
        ss += '{:20s} : {:.3f}\n'.format('GLON', d['glon'])
        ss += '{:20s} : {:.3f}\n'.format('GLAT', d['glat'])
        ss += '{:20s} : {:.3f}\n'.format('Position error', d['pos_err'])

        return ss

    def _info_spectrum(self):
        """Print spectral info."""
        d = self.data
        ss = '\n*** Spectral info ***\n\n'

        ss += 'Spectrum 1:\n'
        args1 = 'Flux at 7 TeV', d['spec0_dnde'].value, d['spec0_dnde_err'].value, 'cm-2 s-1 TeV-1'
        ss += '{:20s} : {:.3} +- {:.3} {}\n'.format(*args1)
        args2 = 'Spectral index', d['spec0_index'], d['spec0_index_err']
        ss += '{:20s} : {:.3f} +- {:.3f}\n'.format(*args2)
        ss += '{:20s} : {:1}\n\n'.format('Test radius', d['spec0_radius'])

        if(np.isnan(d['spec1_dnde'])):
            ss += 'No second spectrum available for this source'
        else:
            ss += 'Spectrum 2:\n'
            args3 = 'Flux at 7 TeV', d['spec1_dnde'].value, d['spec1_dnde_err'].value, 'cm-2 s-1 TeV-1'
            ss += '{:20s} : {:.3} +- {:.3} {}\n'.format(*args3)
            args4 = 'Spectral index', d['spec1_index'], d['spec1_index_err']
            ss += '{:20s} : {:.3f} +- {:.3f}\n'.format(*args4)
            ss += '{:20s} : {:1}'.format('Test radius', d['spec1_radius'])

        return ss

    @property
    def n_spectra(self):
        """Number of measured spectra (1 or 2)."""

        return 1 if np.isnan(self.data['spec1_dnde']) else 2

    def _get_spec_pars(self, id):
        pars, errs = {}, {}
        data = self.data
        label = 'spec{}_'.format(id)

        pars['amplitude'] = data[label + 'dnde']
        errs['amplitude'] = data[label + 'dnde_err']
        pars['index'] = data[label + 'index'] * u.Unit('')
        errs['index'] = data[label + 'index_err'] * u.Unit('')
        pars['reference'] = 7 * u.TeV

        return [pars, errs]

    def spectral_model(self):
        """Returns a list of `~gammapy.spectrum.models.SpectralModel`
        objects. Either with one or with two entries depending on the
        source in the catalog.
        """

        models = []

        model1 = PowerLaw(**(self._get_spec_pars(0)[0]))
        model1.parameters.set_parameter_errors(self._get_spec_pars(0)[1])
        models = [model1]

        if(self.n_spectra == 2):
            model2 = PowerLaw(**(self._get_spec_pars(1)[0]))
            model2.parameters.set_parameter_errors(self._get_spec_pars(1)[1])
            models.append(model2)

        return models


class SourceCatalog2HWC(SourceCatalog):
    """ HAWC 2FHL catalog

    One source is represented by `~gammapy.catalog.SourceCatalogObjectGammaCat`

    See: http://adsabs.harvard.edu/abs/2017ApJ...843...40A

    References
    -----------
    .. [1] Abeysekara et al, "The 2HWC HAWC Observatory Gamma Ray Catalog",
     `Link <https://arxiv.org/abs/1702.02992>_
    """

    name = '2hwc'
    description = 'Second HWC FHL catalog from the HAWC observatory'
    source_object_class = SourceCatalogObject2HWC

    def __init__(self, filename='$GAMMAPY_EXTRA/datasets/catalogs/2HWC.ecsv'):
        filename = str(make_path(filename))
        table = Table.read(filename, format='ascii.ecsv')

        source_name_key = 'source_name'

        super(SourceCatalog2HWC, self).__init__(
            table=table,
            source_name_key=source_name_key)
