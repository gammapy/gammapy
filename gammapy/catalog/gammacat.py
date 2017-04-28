# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Gammacat open TeV source catalog.

Meow!!!!

https://github.com/gammapy/gamma-cat
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import numpy as np
from astropy.tests.helper import ignore_warnings
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import Angle
from astropy.modeling.models import Gaussian2D
from ..utils.modeling import SourceModel, SourceLibrary
from ..utils.scripts import make_path
from ..spectrum import FluxPoints
from ..spectrum.models import PowerLaw, PowerLaw2, ExponentialCutoffPowerLaw
from ..image.models import Shell2D
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    'SourceCatalogGammaCat',
    'SourceCatalogObjectGammaCat',
]

log = logging.getLogger(__name__)


class NoDataAvailableError(LookupError):
    """Generic error used in Gammapy, when some data isn't available.
    """
    pass


class GammaCatNotFoundError(OSError):
    """The gammapy-cat repo is not available.

    You have to set the GAMMA_CAT environment variable so that it's found.
    """
    pass


class SourceCatalogObjectGammaCat(SourceCatalogObject):
    """One object from the gamma-cat source catalog.

    Catalog is represented by `~gammapy.catalog.SourceCatalogGammaCat`.
    """
    _source_name_key = 'common_name'
    _source_index_key = 'catalog_row_index'

    def __str__(self):
        """Print default summary info string"""
        d = self.data

        ss = 'Source: {}\n'.format(d['common_name'])
        ss += 'source_id: {}\n'.format(d['source_id'])
        ss += 'reference_id: {}\n'.format(d['reference_id'])
        ss += '\n'

        ss += 'RA          : {:.2f}\n'.format(d['ra'])
        ss += 'DEC         : {:.2f}\n'.format(d['dec'])
        ss += 'GLON        : {:.2f}\n'.format(d['glon'])
        ss += 'GLAT        : {:.2f}\n'.format(d['glat'])
        ss += '\n'
        return ss

    def info(self):
        """Print summary info."""
        print(self)

    @property
    def spectral_model(self):
        """Source spectral model (`~gammapy.spectrum.models.SpectralModel`)."""
        d = self.data
        spec_type = d['spec_type']
        pars, errs = {}, {}
        pars['index'] = d['spec_index'] * u.dimensionless_unscaled
        errs['index'] = d['spec_index_err'] * u.dimensionless_unscaled

        if spec_type == 'pl':
            pars['reference'] = d['spec_ref']
            pars['amplitude'] = d['spec_norm']
            errs['amplitude'] = d['spec_norm_err']
            model = PowerLaw(**pars)
        elif spec_type == 'ecpl':
            pars['amplitude'] = d['spec_norm']
            pars['reference'] = d['spec_ref']
            pars['lambda_'] = 1. / d['spec_ecut']
            errs['amplitude'] = d['spec_norm_err']
            errs['lambda_'] = d['spec_ecut_err'] / d['spec_ecut'] ** 2
            model = ExponentialCutoffPowerLaw(**pars)
        elif spec_type == 'pl2':
            pars['emin'] = d['spec_erange_min']
            # The PowerLaw2 model needs an `emax` to work.
            # If none is available, we put a default value here
            # that is effectively infinity
            DEFAULT_EMAX = 1e3 * u.TeV
            if np.isnan(d['spec_erange_max']):
                pars['emax'] = DEFAULT_EMAX
            else:
                pars['emax'] = d['spec_erange_max']

            # TODO: remove this hack once this issue is resolved in gamma-cat
            # https://github.com/gammapy/gamma-cat/issues/101
            pars['amplitude'] = d['spec_norm'].value * u.Unit('cm-2 s-1')
            errs['amplitude'] = d['spec_norm_err'].value * u.Unit('cm-2 s-1')
            model = PowerLaw2(**pars)
        elif spec_type == 'none':
            raise NoDataAvailableError('No spectral model available: {}'.format(self.name))
        else:
            raise NotImplementedError('Unknown spectral model: {!r}'.format(spec_type))

        model.parameters.set_parameter_errors(errs)
        return model

    def spatial_model(self, emin=1 * u.TeV, emax=10 * u.TeV):
        """Source spatial model."""
        d = self.data
        morph_type = d['morph_type']
        pars = {}
        flux = self.spectral_model.integral(emin, emax)

        glon = Angle(d['glon']).wrap_at('180d')
        glat = Angle(d['glat']).wrap_at('180d')

        if morph_type == 'gauss':
            pars['x_mean'] = glon.value
            pars['y_mean'] = glat.value
            pars['x_stddev'] = d['morph_sigma'].value
            pars['y_stddev'] = d['morph_sigma'].value
            if not np.isnan(d['morph_sigma2']):
                pars['y_stddev'] = d['morph_sigma2'].value
            if not np.isnan(d['morph_pa']):
                # TODO: handle reference frame for rotation angle
                pars['theta'] = Angle(d['morph_pa'], 'deg').rad
            ampl = flux.to('cm-2 s-1').value
            pars['amplitude'] = ampl * 1 / (2 * np.pi * pars['x_stddev'] * pars['y_stddev'])
            return Gaussian2D(**pars)
        elif morph_type == 'shell':
            pars['amplitude'] = flux.to('cm-2 s-1').value
            pars['x_0'] = glon.value
            pars['y_0'] = glat.value
            pars['r_in'] = d['morph_sigma'].value * 0.8
            pars['width'] = 0.2 * d['morph_sigma'].value
            return Shell2D(**pars)
        elif morph_type == 'point':
            DEFAULT_POINT_EXTENSION = Angle('0.05 deg')
            pars['amplitude'] = flux.to('cm-2 s-1').value
            pars['x_mean'] = glon.value
            pars['y_mean'] = glat.value
            pars['x_stddev'] = DEFAULT_POINT_EXTENSION
            pars['y_stddev'] = DEFAULT_POINT_EXTENSION
            # TODO: make Delta2D work and use it here.
            return Gaussian2D(**pars)
        elif morph_type == 'none':
            raise NoDataAvailableError('No spatial model available: {}'.format(self.name))
        else:
            raise NotImplementedError('Unknown spatial model: {!r}'.format(morph_type))

    def _add_source_meta(self, table):
        """Copy over some info to table.meta"""
        d = self.data
        m = table.meta
        m['origin'] = 'Data from gamma-cat'
        m['source_id'] = d['source_id']
        m['common_name'] = d['common_name']
        m['reference_id'] = d['reference_id']

    @property
    def flux_points(self):
        """Differential flux points (`~gammapy.spectrum.FluxPoints`)."""
        d = self.data
        table = Table()
        table.meta['SED_TYPE'] = 'dnde'
        self._add_source_meta(table)

        valid = np.isfinite(d['sed_e_ref'].value)

        if valid.sum() == 0:
            raise NoDataAvailableError('No flux points available: {}'.format(self.name))

        table['e_ref'] = d['sed_e_ref']
        table['e_min'] = d['sed_e_min']
        table['e_max'] = d['sed_e_max']

        table['dnde'] = d['sed_dnde']
        table['dnde_err'] = d['sed_dnde_err']
        table['dnde_errn'] = d['sed_dnde_errn']
        table['dnde_errp'] = d['sed_dnde_errp']
        table['dnde_ul'] = d['sed_dnde_ul']

        # Only keep rows that actually contain information
        table = table[valid]

        # Only keep columns that actually contain information
        def _del_nan_col(table, colname):
            if np.isfinite(table[colname]).sum() == 0:
                del table[colname]
        for colname in table.colnames:
            _del_nan_col(table, colname)

        return FluxPoints(table)


class SourceCatalogGammaCat(SourceCatalog):
    """Gammacat open TeV source catalog.

    See: https://github.com/gammapy/gamma-cat

    One source is represented by `~gammapy.catalog.SourceCatalogObjectGammaCat`.

    Parameters
    ----------
    filename : str
        Path to the gamma-cat fits file.

    Examples
    --------
    Load the catalog data:

    >>> from gammapy.catalog import SourceCatalogGammaCat
    >>> cat = SourceCatalogGammaCat()

    Access a source by name:

    >>> source = cat['Vela Junior']

    Access source spectral data and plot it:

    >>> source.spectral_model.plot()
    >>> source.spectral_model.plot_error()
    >>> source.flux_points.plot()
    """
    name = 'gamma-cat'
    description = 'An open catalog of gamma-ray sources'
    source_object_class = SourceCatalogObjectGammaCat

    def __init__(self, filename='$GAMMA_CAT/docs/data/gammacat.fits.gz'):
        filename = str(make_path(filename))

        if 'GAMMA_CAT' not in os.environ:
            msg = 'The gamma-cat repo is not available. '
            msg += 'You have to set the GAMMA_CAT environment variable '
            msg += 'to point to the location for it to be found.'
            raise GammaCatNotFoundError(msg)

        with ignore_warnings():  # ignore FITS units warnings
            table = Table.read(filename, hdu=1)
        self.filename = filename

        source_name_key = 'common_name'
        source_name_alias = ('other_names', 'gamma_names')
        super(SourceCatalogGammaCat, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

    def to_source_library(self):
        """Convert to a `~gammapy.utils.modeling.SourceLibrary`.

        TODO: add an option whether to skip or raise on missing models or data.
        """
        source_list = []

        for source_idx in range(len(self.table)):
            source = self[source_idx]
            try:
                source_model = SourceModel.from_gammacat(source)
            except NoDataAvailableError:
                log.warning('Skipping source {} (missing data in gamma-cat)'.format(source.name))
                continue
            source_list.append(source_model)

        return SourceLibrary(source_list=source_list)
