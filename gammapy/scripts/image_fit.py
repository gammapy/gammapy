# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click
from ..irf._utils_old import read_json, write_all

log = logging.getLogger(__name__)


@click.command('fit')
@click.option('--counts', default='counts.fits',
              help='Counts FITS file name')
@click.option('--exposure', default='exposure.fits',
              help='Exposure FITS file name')
@click.option('--background', default='background.fits',
              help='Background FITS file name')
@click.option('--psf', type=str, default=None,
              help='PSF JSON file name')
@click.option('--sources', default='sources.json',
              help='Sources JSON file name (contains start '
                   'values for fit of Gaussians)')
@click.option('--roi', type=str, default=None,
              help='Region of interest (ROI) file name (ds9 reg format)')
@click.option('--outfile', default='fit_results.json',
              help='Output JSON file with fit results')
def cli_image_fit(counts, exposure, background, psf,
                  sources, roi, outfile):
    """Fit morphology model to image using Sherpa.

    Uses initial parameters from a JSON file (for now only Gaussians).
    """
    import sherpa.astro.ui
    from ..irf import SherpaMultiGaussPSF

    # ---------------------------------------------------------
    # Load images, PSF and sources
    # ---------------------------------------------------------
    log.info('Clearing the sherpa session')
    sherpa.astro.ui.clean()

    log.info('Reading counts: {}'.format(counts))
    sherpa.astro.ui.load_image(counts)

    log.info('Reading exposure: {}'.format(exposure))
    sherpa.astro.ui.load_table_model('exposure', exposure)

    log.info('Reading background: {}'.format(background))
    sherpa.astro.ui.load_table_model('background', background)

    if psf:
        log.info('Reading PSF: {}'.format(psf))
        SherpaMultiGaussPSF(psf).set()
    else:
        log.warning("No PSF convolution.")

    if roi:
        log.info('Reading ROI: {}'.format(roi))
        sherpa.astro.ui.notice2d(roi)
    else:
        log.info('No ROI selected.')

    log.info('Reading sources: {}'.format(sources))
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
