# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Meow!!!!

Gammacat open TeV source catalog see https://github.com/gammapy/gamma-cat for details
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from collections import OrderedDict
import numpy as np
from astropy import units as u
from astropy.table import Table, QTable
from ..extern.pathlib import Path
from ..spectrum import DifferentialFluxPoints, SpectrumFitResult
from ..spectrum.models import PowerLaw, PowerLaw2, ExponentialCutoffPowerLaw
from ..utils.scripts import make_path
from .core import SourceCatalog, SourceCatalogObject

__all__ = [
    'SourceCatalogGammaCat',
    'SourceCatalogObjectGammaCat',
]


class SourceCatalogObjectGammaCat(SourceCatalogObject):
    """
    One object from the gamma-cat source catalog.
    """
    _source_name_key = 'common_name'
    _source_index_key = 'catalog_row_index'

    def __str__(self):
        """Print default summary info string"""
        d = self.data

        ss = 'Source: {}\n'.format(d['common_name'])
        ss += 'Paper ID: {}\n'.format(d['paper_id'])
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
        Best fit spectral model `~gammapy.spectrum.SpectralModel`.
        """
        d = self.data
        spec_type = d['spec_type']
        pars = {}
        pars['index'] = u.Quantity(d['spec_index'])

        if spec_type == 'pl':
            pars['reference'] = d['spec_ref']
            pars['amplitude'] = d['spec_norm'] * u.Unit('TeV-1 cm-2 s-1')
            return PowerLaw(**pars)

        elif spec_type == 'ecpl':
            pars['amplitude'] = d['spec_norm'] * u.Unit('TeV-1 cm-2 s-1')
            pars['reference'] = d['spec_ref']
            pars['lambda_'] = 1. / d['spec_ecut']
            return ExponentialCutoffPowerLaw(**pars)

        elif spec_type == 'pl2':
            pars['emin'] = d['spec_ref']
            # TODO: I'd be better to put np.inf, but uncertainties can't handle it
            pars['emax'] = 1E10 * u.TeV
            pars['amplitude'] = d['spec_norm'] * u.Unit('cm-2 s-1')
            return PowerLaw2(**pars)
        else:
            raise ValueError('Spectral model {} not available'.format(spec_type))

    @property
    def flux_points(self):
        """
        Differential flux points (`~gammapy.spectrum.DifferentialFluxPoints`).
        """
        d = self.data
        e_ref = d['sed_e_ref']
        valid = ~np.isnan(e_ref)

        e_ref = e_ref[valid]
        dnde = d['sed_dnde'][valid]
        dnde_errp = d['sed_dnde_errp'][valid]
        dnde_errn = d['sed_dnde_errn'][valid]

        if len(e_ref) == 0:
            raise DataMissingError('No flux points available.')

        return DifferentialFluxPoints.from_arrays(energy=e_ref,
                                                  diff_flux=dnde,
                                                  diff_flux_err_lo=dnde_errn,
                                                  diff_flux_err_hi=dnde_errp)

    @property
    def spectrum(self):
        """
        Spectrum model fit result (`~gammapy.spectrum.SpectrumFitResult`)

        TODO: remove!???
        """
        d = self.data
        model = self.spectral_model

        spec_type = d['spec_type']
        erange = d['spec_erange_min'], d['spec_erange_max']

        if spec_type == 'pl':
            par_names = ['index', 'amplitude']
            par_errs = [d['spec_index_err'], d['spec_norm_err']]
        elif spec_type == 'ecpl':
            par_names = ['index', 'amplitude', 'lambda_']
            lambda_err = d['spec_ecut_err'] / d['spec_ecut'] ** 2
            par_errs = [d['spec_index_err'],
                        d['spec_norm_err'],
                        lambda_err.value]
        elif spec_type == 'pl2':
            par_names = ['amplitude', 'index']
            par_errs = [d['spec_norm_err'],
                        d['spec_index_err'],
                        ]
        else:
            raise ValueError('Spectral model {} not available'.format(spec_type))

        covariance = np.diag(par_errs) ** 2

        return SpectrumFitResult(
            model=model,
            fit_range=erange,
            covariance=covariance,
            covar_axis=par_names,
        )


class GammaCatNotFoundError(OSError):
    """
    The gammapy-cat repo is not available.

    You have to set the GAMMA_CAT environment variable so that it's found.
    """
    pass


class SourceCatalogGammaCat(SourceCatalog):
    """
    Gammacat open TeV sources catalog.

    See: https://github.com/gammapy/gamma-cat

    Parameters
    ----------
    filename : str
        Path to the gamma-cat fits file.
    """
    name = 'gamma-cat'
    description = 'An open catalog of gamma-ray sources'
    source_object_class = SourceCatalogObjectGammaCat

    def __init__(self, filename='$GAMMA_CAT/docs/data/gammacat.fits.gz'):
        filename = make_path(filename)
        if not 'GAMMA_CAT' in os.environ:
            msg = 'The gamma-cat repo is not available. '
            msg += 'You have to set the GAMMA_CAT environment variable '
            msg += 'to point to the location for it to be found.'
            raise GammaCatNotFoundError(msg)

        self.filename = str(filename)
        table = QTable.read(self.filename)
        super(SourceCatalogGammaCat, self).__init__(table=table, source_name_key='common_name')

    def _make_source_dict(self, index):
        """Make one source data dict.

        Parameters
        ----------
        index : int
            Row index

        Returns
        -------
        data : dict
            Source data dict
        """
        row = self.table[index]
        return row