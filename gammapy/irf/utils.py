# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ['load_irf']


def load_irf(filename):
    """Load instrument response function (IRF) from file.

    This is a helper function that does some poking around and guesswork
    to figure out what kind of IRF this is, and then loads it
    using the appropriate class.
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
