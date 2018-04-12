# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import logging
import click

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


def write_all(filename='results.json'):
    """Dump source, fit results and conf results to a JSON file.
    http://www.astropython.org/snippet/2010/7/Save-sherpa-fit-and-conf-results-to-a-JSON-file
    """
    import sherpa.astro.ui as sau
    out = dict()

    if 0:
        src = sau.get_source()
        src_par_attrs = ('name', 'frozen', 'modelname', 'units', 'val', 'fullname')
        out['src'] = dict(name=src.name,
                          pars=[dict((attr, getattr(par, attr)) for attr in src_par_attrs)
                                for par in src.pars])

    try:
        fit_attrs = ('methodname', 'statname', 'succeeded', 'statval', 'numpoints', 'dof',
                     'rstat', 'qval', 'nfev', 'message', 'parnames', 'parvals')
        fit = sau.get_fit_results()
        out['fit'] = dict((attr, getattr(fit, attr)) for attr in fit_attrs)
    except Exception as err:
        print(err)

    try:
        conf_attrs = ('datasets', 'methodname', 'fitname', 'statname', 'sigma', 'percent',
                      'parnames', 'parvals', 'parmins', 'parmaxes', 'nfits')
        conf = sau.get_conf_results()
        out['conf'] = dict((attr, getattr(conf, attr)) for attr in conf_attrs)
    except Exception as err:
        print(err)

    try:
        covar_attrs = ('datasets', 'methodname', 'fitname', 'statname', 'sigma', 'percent',
                       'parnames', 'parvals', 'parmins', 'parmaxes', 'nfits')
        covar = sau.get_covar_results()
        out['covar'] = dict((attr, getattr(covar, attr)) for attr in covar_attrs)
    except Exception as err:
        print(err)

    if 0:
        out['pars'] = []
        for par in src.pars:
            fullname = par.fullname
            if any(fullname == x['name'] for x in out['pars']):
                continue  # Parameter was already processed
            outpar = dict(name=fullname, kind=par.name)

            # None implies no calculated confidence interval for Measurement
            parmin = None
            parmax = None
            try:
                if fullname in conf.parnames:  # Confidence limits available from conf
                    i = conf.parnames.index(fullname)
                    parval = conf.parvals[i]
                    parmin = conf.parmins[i]
                    parmax = conf.parmaxes[i]
                if parmin is None:
                    parmin = -float('inf')  # None from conf means infinity, so set accordingly
                if parmax is None:
                    parmax = float('inf')
                elif fullname in fit.parnames:  # Conf failed or par is uninteresting and wasn't sent to conf
                    i = fit.parnames.index(fullname)
                    parval = fit.parvals[i]
                else:  # No fit or conf value (maybe frozen)
                    parval = par.val
            except Exception as err:
                print(err)

            out['pars'].append(outpar)
    if filename is None:
        return out
    else:
        json.dump(out, open(filename, 'w'), sort_keys=True, indent=4)


def _set(name, par, val):
    """Set a source parameter."""
    import sherpa.astro.ui as sau
    sau.set_par('{name}.{par}'.format(**locals()), val)


def _model(source_names):
    """Build additive model string for Gaussian sources."""
    return ' + '.join(['normgauss2d.' + name for name in source_names])


def read_json(source, setter):
    """Read from JSON file."""
    if isinstance(source, dict):
        # Assume source is a dict with correct format
        d = source
    else:
        # Assume source is a filename with correct format
        d = json.load(open(source))
    source_names = d.keys()
    model = _model(source_names)
    setter(model)
    for name, pars in d.items():
        for par, val in pars.items():
            _set(name, par, val)
