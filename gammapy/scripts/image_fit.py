# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from ..utils.scripts import get_parser, set_up_logging_from_args

__all__ = ['run_image_fit_sherpa']

log = logging.getLogger(__name__)


def image_fit_main(args=None):
    parser = get_parser(run_image_fit_sherpa)
    parser.add_argument('--counts', type=str, default='counts.fits',
                        help='Counts FITS file name')
    parser.add_argument('--exposure', type=str, default='exposure.fits',
                        help='Exposure FITS file name')
    parser.add_argument('--background', type=str, default='background.fits',
                        help='Background FITS file name')
    parser.add_argument('--psf', type=str, default=None,
                        help='PSF JSON file name')
    parser.add_argument('--sources', type=str, default='sources.json',
                        help='Sources JSON file name (contains start '
                             'values for fit of Gaussians)')
    parser.add_argument('--roi', type=str, default=None,
                        help='Region of interest (ROI) file name (ds9 reg format)')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")
    parser.add_argument('outfile', type=str, default='fit_results.json',
                        help='Output JSON file with fit results')
    args = parser.parse_args(args)
    set_up_logging_from_args(args)
    run_image_fit_sherpa(**vars(args))


def run_image_fit_sherpa(counts,
                         exposure,
                         background,
                         psf,
                         sources,
                         roi,
                         outfile):
    """Fit the morphology of a number of sources.

    Uses initial parameters from a JSON file (for now only Gaussians).
    """
    import sherpa.astro.ui
    from ..image.models.utils import read_json, write_all
    from ..irf import SherpaMultiGaussPSF

    # ---------------------------------------------------------
    # Load images, PSF and sources
    # ---------------------------------------------------------
    log.info('Clearing the sherpa session')
    sherpa.astro.ui.clean()

    log.info('Reading counts: {0}'.format(counts))
    sherpa.astro.ui.load_image(counts)

    log.info('Reading exposure: {0}'.format(exposure))
    sherpa.astro.ui.load_table_model('exposure', exposure)

    log.info('Reading background: {0}'.format(background))
    sherpa.astro.ui.load_table_model('background', background)

    if psf:
        log.info('Reading PSF: {0}'.format(psf))
        SherpaMultiGaussPSF(psf).set()
    else:
        log.warning("No PSF convolution.")

    if roi:
        log.info('Reading ROI: {0}'.format(roi))
        sherpa.astro.ui.notice2d(roi)
    else:
        log.info('No ROI selected.')

    log.info('Reading sources: {0}'.format(sources))
    read_json(sources, sherpa.astro.ui.set_source)

    # ---------------------------------------------------------
    # Set up the full model and freeze PSF, exposure, background
    # ---------------------------------------------------------
    # Scale exposure by 1e-10 to get ampl or order unity and avoid some fitting problems
    name = sherpa.astro.ui.get_source().name
    if psf:
        full_model = 'background + 1e-12 * exposure * psf ({})'.format(name)
        sherpa.astro.ui.set_full_model(full_model)
        sherpa.astro.ui.freeze('background', 'exposure', 'psf')
    else:
        full_model = 'background + 1e-12 * exposure * {}'.format(name)
        sherpa.astro.ui.set_full_model(full_model)
        sherpa.astro.ui.freeze('background', 'exposure')

    # ---------------------------------------------------------
    # Set up the fit
    # ---------------------------------------------------------
    sherpa.astro.ui.set_coord('image')
    sherpa.astro.ui.set_stat('cash')
    sherpa.astro.ui.set_method('levmar')  # levmar, neldermead, moncar
    sherpa.astro.ui.set_method_opt('maxfev', int(1e3))
    sherpa.astro.ui.set_method_opt('verbose', 10)

    # ---------------------------------------------------------
    # Fit and save information we care about
    # ---------------------------------------------------------
    # sherpa.astro.ui.show_all() # Prints info about data and model
    sherpa.astro.ui.fit()  # Does the fit
    # sherpa.astro.ui.covar()  # Computes symmetric errors (fast)
    # conf() # Computes asymmetric errors (slow)
    # image_fit() # Shows data, model, residuals in ds9
    log.info('Writing {}'.format(outfile))
    write_all(outfile)

    # Save model image
    sherpa.astro.ui.set_par('background.ampl', 0)
    sherpa.astro.ui.notice2d()
    log.info('Writing model.fits')
    sherpa.astro.ui.save_model('model.fits', clobber=True)
    sherpa.astro.ui.clean()
