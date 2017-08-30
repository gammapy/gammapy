# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""HAWC catalogs (https://www.hawc-observatory.org)."""
import numpy as np
from astropy.table import Table
import astropy.units as u
from ..utils.scripts import make_path
from ..spectrum.models import PowerLaw
from .core import SourceCatalog, SourceCatalogObject


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
        info : {'all', 'basic', 'position', 'spectrum'}
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

        if self.n_spectra == 2:
            ss += 'Spectrum 2:\n'
            args3 = 'Flux at 7 TeV', d['spec1_dnde'].value, d['spec1_dnde_err'].value, 'cm-2 s-1 TeV-1'
            ss += '{:20s} : {:.3} +- {:.3} {}\n'.format(*args3)
            args4 = 'Spectral index', d['spec1_index'], d['spec1_index_err']
            ss += '{:20s} : {:.3f} +- {:.3f}\n'.format(*args4)
            ss += '{:20s} : {:1}'.format('Test radius', d['spec1_radius'])
        else:
            ss += 'No second spectrum available for this source'

        return ss

    @property
    def n_spectra(self):
        """Number of measured spectra (1 or 2)."""
        return 1 if np.isnan(self.data['spec1_dnde']) else 2

    def _get_spectral_model(self, idx_model):
        pars, errs = {}, {}
        data = self.data
        label = 'spec{}_'.format(idx_model)

        pars['amplitude'] = data[label + 'dnde']
        errs['amplitude'] = data[label + 'dnde_err']
        pars['index'] = data[label + 'index'] * u.Unit('')
        errs['index'] = data[label + 'index_err'] * u.Unit('')
        pars['reference'] = 7 * u.TeV

        model = PowerLaw(**pars)
        model.parameters.set_parameter_errors(errs)

        return model

    @property
    def spectral_models(self):
        """Spectral models (either one or two).

        The HAWC catalog has one or two spectral measurements for each source.

        Returns
        -------
        models : list
            List of `~gammapy.spectrum.models.SpectralModel`
        """
        models = [self._get_spectral_model(0)]

        if self.n_spectra == 2:
            models.append(self._get_spectral_model(1))

        return models


class SourceCatalog2HWC(SourceCatalog):
    """HAWC 2FHL catalog.

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
            source_name_key=source_name_key,
        )
