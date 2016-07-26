# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from astropy.table import Table
from ..utils.fits import get_hdu
from ..irf import EffectiveAreaTable2D
from ..background import Cube
from ..irf import EnergyDispersion2D
from ..irf import EnergyDependentMultiGaussPSF

__all__ = [
    'CTAIrf',
]


class CTAIrf(object):
    """Contain CTA instrument response functions.

    Class handling CTA instrument response function. For now
    we use the production 2 of the CTA IRF
    (https://portal.cta-observatory.org/Pages/CTA-Performance.aspx)
    adapted from the ctools
    (http://cta.irap.omp.eu/ctools/user_manual/getting_started/response.html).
    The IRF format should be compliant with the one discussed
    at http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/.
    Waiting for a new public production of the CTA IRF,
    we'll fix the missing pieces.

    This class is similar to `~gammapy.data.DataStoreObservation`,
    but only contains IRFs (no event data or livetime info).

    TODO: maybe re-factor code somehow to avoid code duplication.

    Parameters
    ----------
    area_2d : `~gammapy.irf.EffectiveAreaTable2D`
        2D Effective Area Table
    bcg_cube : `~gammapy.background.Cube`
        Cube container class with scheme fixed to `bg_cube`
    edisp_2d : `~gammapy.irf.EnergyDispersion2D`
        Offset-dependent energy dispersion matrix
    psf_mgauss : `~gammapy.irf.EnergyDependentMultiGaussPSF`
        Position-dependent multi-Gauss PSF
    """

    def __init__(self, area_2d=None, bkg_cube=None, e_disp_2d=None, psf_mgauss=None):
        self.area_2d = area_2d
        self.bkg_cube = bkg_cube
        self.e_disp_2d = e_disp_2d
        self.psf_mgauss = psf_mgauss

    @classmethod
    def read(cls, filename):
        """
        Auxiliary constructor from a fits file

        Parameters
        ----------
        filename : `str`
            Path of the file containing the IRF
        """
        # Dealing with effective area and fix headers
        table_hdu_area = get_hdu(filename + '[EFFECTIVE AREA]')
        table_area = Table(table_hdu_area.data)

        table_area['ENERG_LO'].unit = u.TeV
        table_area['ENERG_HI'].unit = u.TeV
        table_area['THETA_LO'].unit = u.deg
        table_area['THETA_HI'].unit = u.deg
        table_area['EFFAREA'].unit = u.m * u.m
        table_area['EFFAREA_RECO'].unit = u.m * u.m

        area_2d = EffectiveAreaTable2D.from_table(table_area)

        # Dealing with background and fix headers
        table_hdu_bkg = get_hdu(filename + '[BACKGROUND]')

        table_hdu_bkg.header['TTYPE7'] = 'Bgd'
        table_hdu_bkg.header['TUNIT7'] = '1 / (MeV s sr)'

        bkg_cube = Cube.from_fits_table(table_hdu_bkg, scheme='bg_cube')

        # Dealing energy dispersion matrix
        table_hdu_disp = get_hdu(filename + '[ENERGY DISPERSION]')
        e_disp_2d = EnergyDispersion2D.from_fits(table_hdu_disp)

        # Dealing with psf and fix values to get a single gaussian component
        # by setting the amplitudes and the sigmas of the last two components
        # to 0 and 1, respectively
        table_hdu_psf = get_hdu(filename + '[POINT SPREAD FUNCTION]')

        table_hdu_psf.data[0]['AMPL_2'] = 0
        table_hdu_psf.data[0]['AMPL_3'] = 0

        table_hdu_psf.data[0]['SIGMA_2'] = 1
        table_hdu_psf.data[0]['SIGMA_3'] = 1

        psf_mgauss = EnergyDependentMultiGaussPSF.from_fits(table_hdu_psf)

        # Building object
        irf = cls(area_2d=area_2d,
                  bkg_cube=bkg_cube,
                  e_disp_2d=e_disp_2d,
                  psf_mgauss=psf_mgauss)

        return irf
