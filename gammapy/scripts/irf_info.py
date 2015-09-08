# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import logging
from astropy.units import Quantity
from astropy.io import fits
from ..utils.scripts import get_parser

__all__ = ['irf_info']

log = logging.getLogger(__name__)


def retrieve_psf_info(hdu_list, energies, theta, fractions, plot=False):
    """
    Retrieve psf information from psf fits file.

    Parameters
    ----------
    hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``POINT SPREAD FUNCTION``extension.
    plot : bool
        Make a containment radius vs. theta and energy plot.
    """
    from gammapy.irf import EnergyDependentMultiGaussPSF
    psf = EnergyDependentMultiGaussPSF.from_fits(hdu_list)
    energies = Quantity(energies, 'TeV')
    thetas = Quantity(theta, 'deg')
    print(psf.info(fractions=fractions, energies=energies, thetas=thetas))

    if plot:
        for fraction in fractions:
            filename = 'containment_R{0:.0f}_energy_theta.png'.format(100 * fraction)
            psf.plot_containment(fraction, filename)


def retrieve_arf_info(hdu_list, energies, plot=False):
    """
    Retrieve effective area information from arf fits file.

    Parameters
    ----------
    hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``SPECRESP``extension.
    plot : bool
        Make an effective area vs. energy plot.
    """
    from gammapy.irf import EffectiveAreaTable
    arf = EffectiveAreaTable.from_fits(hdu_list)
    energies = Quantity(energies, 'TeV')
    print(arf.info(energies=energies))

    if plot:
        arf.plot_area_vs_energy('effective_area.png')


def main(args=None):
    parser = get_parser(irf_info)
    parser.add_argument('infiles', type=str, nargs='+',
                        help='Input FITS file name')
    parser.add_argument('--plot', action='store_true',
                        help='Make info plot. Containment vs. theta and '
                             'energy for PSF or effective area vs. energy for ARF.')
    parser.add_argument('--thetas', type=float, default=[0.], nargs='+',
                        help='Thetas where to evaluate PSF info.')
    parser.add_argument('--energies', type=float, default=[1., 10.], nargs='+',
                        help='Energies where to evaluate PSF and ARF info.')
    parser.add_argument('--fractions', type=float, default=[0.68, 0.95], nargs='+',
                        help='Containment fractions to compute for the PSF info.')
    args = parser.parse_args(args)
    irf_info(**vars(args))


def irf_info(infiles,
             plot,
             thetas,
             energies,
             fractions):
    """Print or plot info about instrument response function (IRF) files.
    """
    for infile in infiles:
        hdu_list = fits.open(infile)
        hdu_names = [hdu.name for hdu in hdu_list]
        if 'POINT SPREAD FUNCTION' in hdu_names:
            log.info('Auto detected PSF FITS file.')
            log.info('Retrieving PSF info for {0}'.format(os.path.split(infile)[1]))
            retrieve_psf_info(hdu_list, energies, thetas, fractions, plot)
        elif 'SPECRESP' in hdu_names:
            log.info('Auto detected ARF FITS file.')
            log.info('Retrieving ARF info for {0}'.format(os.path.split(infile)[1]))
            retrieve_arf_info(hdu_list, energies, plot)
        else:
            log.error('No valid FITS file found.')
