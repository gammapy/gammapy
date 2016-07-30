# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.io import fits
from ..utils.scripts import make_path
from ..irf import EffectiveAreaTable2D
from ..background import Cube
from ..irf import EnergyDispersion2D
from ..irf import EnergyDependentMultiGaussPSF

__all__ = [
    'CTAIrf',
]


class CTAIrf(object):
    """CTA instrument response function container.

    Class handling CTA instrument response function.

    For now we use the production 2 of the CTA IRF
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
    aeff : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area
    edisp : `~gammapy.irf.EnergyDispersion2D`
        Energy dispersion
    psf : `~gammapy.irf.EnergyDependentMultiGaussPSF`
        Point spread function
    bkg : `~gammapy.background.Cube`
        Background rate
    """

    def __init__(self, aeff=None, edisp=None, psf=None, bkg=None):
        self.aeff = aeff
        self.edisp = edisp
        self.psf = psf
        self.bkg = bkg

    @classmethod
    def read(cls, filename):
        """
        Read from a FITS file.

        Parameters
        ----------
        filename : `str`
            File containing the IRFs
        """
        filename = str(make_path(filename))

        # table = Table.read(filename, hdu='EFFECTIVE AREA')
        # aeff = EffectiveAreaTable2D.from_table(table)
        aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')

        # TODO: fix `Cube.read`, then use it directly here.
        table = fits.open(filename)['BACKGROUND']
        table.columns.change_name(str('BGD'), str('Bgd'))
        table.header['TUNIT7'] = '1 / (MeV s sr)'
        bkg = Cube.from_fits_table(table, scheme='bg_cube')

        # Dealing energy dispersion matrix
        # table_hdu_disp = get_hdu(filename + '[ENERGY DISPERSION]')
        # edisp = EnergyDispersion2D.from_fits(table_hdu_disp)
        edisp = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')

        # Dealing with psf and fix values to get a single gaussian component
        # by setting the amplitudes and the sigmas of the last two components
        # to 0 and 1, respectively
        table_hdu_psf = fits.open(filename)['POINT SPREAD FUNCTION']

        table_hdu_psf.data[0]['AMPL_2'] = 0
        table_hdu_psf.data[0]['AMPL_3'] = 0

        table_hdu_psf.data[0]['SIGMA_2'] = 1
        table_hdu_psf.data[0]['SIGMA_3'] = 1

        psf = EnergyDependentMultiGaussPSF.from_fits(table_hdu_psf)

        return cls(
            aeff=aeff,
            bkg=bkg,
            edisp=edisp,
            psf=psf,
        )
