# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Meow!!!!

Gammacat open TeV source catalog

https://github.com/gammapy/gamma-cat
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import numpy as np
from astropy import units as u
from astropy.table import Table, QTable
from astropy.coordinates import Angle
from astropy.modeling.models import Gaussian2D
from ..utils.modeling import SourceModel, SourceLibrary
from ..spectrum import FluxPoints, SpectrumFitResult
from ..spectrum.models import PowerLaw, PowerLaw2, ExponentialCutoffPowerLaw
from ..image.models import Shell2D, Delta2D
from ..utils.scripts import make_path
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
    """
    The gammapy-cat repo is not available.

    You have to set the GAMMA_CAT environment variable so that it's found.
    """
    pass


class SourceCatalogObjectGammaCat(SourceCatalogObject):
    """
    One object from the gamma-cat source catalog.

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

        ss += 'RA (J2000)  : {:.2f}\n'.format(d['ra'])
        ss += 'Dec (J2000) : {:.2f}\n'.format(d['dec'])
        ss += 'GLON        : {:.2f}\n'.format(d['glon'])
        ss += 'GLAT        : {:.2f}\n'.format(d['glat'])
        ss += '\n'
        return ss

    def info(self):
        """Print summary info."""
        print(self)

    @property
    def spectral_model(self):
        """
        Source spectral model `~gammapy.spectrum.models.SpectralModel`.
        """
        d = self.data
        spec_type = d['spec_type']
        pars, errs = {}, {}
        pars['index'] = u.Quantity(d['spec_index'])
        errs['index'] = u.Quantity(d['spec_index_err'])

        if spec_type == 'pl':
            pars['reference'] = d['spec_ref']
            pars['amplitude'] = d['spec_norm'] * u.Unit('TeV-1 cm-2 s-1')
            errs['amplitude'] = d['spec_norm_err'] * u.Unit('TeV-1 cm-2 s-1')
            model = PowerLaw(**pars)

        elif spec_type == 'ecpl':
            pars['amplitude'] = d['spec_norm'] * u.Unit('TeV-1 cm-2 s-1')
            pars['reference'] = d['spec_ref']
            pars['lambda_'] = 1. / d['spec_ecut']
            errs['amplitude'] = d['spec_norm_err'] * u.Unit('TeV-1 cm-2 s-1')
            errs['lambda_'] = d['spec_ecut_err'] * u.TeV / d['spec_ecut'] ** 2
            model = ExponentialCutoffPowerLaw(**pars)

        elif spec_type == 'pl2':
            pars['emin'] = d['spec_ref']
            # TODO: I'd be better to put np.inf, but uncertainties can't handle it
            pars['emax'] = 1e10 * u.TeV
            pars['amplitude'] = d['spec_norm'] * u.Unit('cm-2 s-1')
            errs['amplitude'] = d['spec_norm_err'] * u.Unit('cm-2 s-1')
            model = PowerLaw2(**pars)
        else:
            raise ValueError('Spectral model {} not available'.format(spec_type))

        model.parameters.set_parameter_errors(errs)
        return model

    def spatial_model(self, emin=1 * u.TeV, emax=10 * u.TeV):
        """
        Source spatial model.
        """
        d = self.data
        morph_type = d['morph_type']
        pars = {}
        flux = self.spectral_model.integral(emin, emax)

        glon = Angle(d['glon'], 'deg').wrap_at('180d').deg
        glat = Angle(d['glat'], 'deg').wrap_at('180d').deg

        if morph_type == 'gauss':
            pars['x_mean'] = glon
            pars['y_mean'] = glat
            pars['x_stddev'] = d['morph_sigma']
            pars['y_stddev'] = d['morph_sigma']
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
            pars['x_0'] = glon
            pars['y_0'] = glat
            pars['r_in'] = d['morph_sigma'] * 0.8
            pars['width'] = 0.2 * d['morph_sigma']
            return Shell2D(**pars)

        elif morph_type == 'point':
            pars['amplitude'] = flux.to('cm-2 s-1').value
            pars['x_mean'] = glon
            pars['y_mean'] = glat
            pars['x_stddev'] = 0.05
            pars['y_stddev'] = 0.05
            # TODO: make Delta2D work and use it here.
            return Gaussian2D(**pars)
        else:
            raise ValueError('Spatial model {} not available'.format(morph_type))

    @property
    def flux_points(self):
        """
        Differential flux points (`~gammapy.spectrum.FluxPoints`).
        """
        d = self.data
        table = Table()
        table.meta['SED_TYPE'] = 'dnde'

        e_ref = d['sed_e_ref']
        valid = ~np.isnan(e_ref)

        table['e_ref'] = e_ref[valid]
        table['dnde'] = d['sed_dnde'][valid]
        table['dnde_errp'] = d['sed_dnde_errp'][valid]
        table['dnde_errn'] = d['sed_dnde_errn'][valid]

        if len(e_ref) == 0:
            raise NoDataAvailableError('No flux points available.')

        return FluxPoints(table)


class SourceCatalogGammaCat(SourceCatalog):
    """
    Gammacat open TeV sources catalog.

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

    >>> source.spectrum.butterfly().plot()
    >>> # source.spectral_model.plot(energy_range=energy_range)
    >>> source.flux_points.plot()
    """
    name = 'gamma-cat'
    description = 'An open catalog of gamma-ray sources'
    source_object_class = SourceCatalogObjectGammaCat

    def __init__(self, filename='$GAMMA_CAT/docs/data/gammacat.fits.gz'):
        filename = make_path(filename)

        if 'GAMMA_CAT' not in os.environ:
            msg = 'The gamma-cat repo is not available. '
            msg += 'You have to set the GAMMA_CAT environment variable '
            msg += 'to point to the location for it to be found.'
            raise GammaCatNotFoundError(msg)

        self.filename = str(filename)
        table = QTable.read(self.filename)
        source_name_key = 'common_name'
        source_name_alias = ('other_names', 'gamma_names')
        super(SourceCatalogGammaCat, self).__init__(
            table=table,
            source_name_key=source_name_key,
            source_name_alias=source_name_alias,
        )

    def to_source_library(self):
        """
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