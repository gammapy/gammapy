# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click
from astropy.io import fits

click.disable_unicode_literals_warning = True

__all__ = ['data_simulate_main']

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('model')
@click.option('--runlist', default='runs.lis', help='Run list filename')
@click.option('--indir', default='.', help='Input folder')
@click.option('--outdir', default='out', help='Output folder')
def data_simulate_main(model, runlist, indir, outdir):
    """Simulate event data.

    Wrapper for ctobssim:
    http://cta.irap.omp.eu/ctools/reference_manual/ctobssim.html

    \b
    Examples
    --------
    \b
    cd $GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2
    gammapy-data-simulate model.xml --runs runs.lis --indir . --outdir out
    """
    click.echo('model: {}'.format(model))
    click.echo('runlist: {}'.format(runlist))
    click.echo('indir: {}'.format(indir))
    click.echo('outdir: {}'.format(outdir))

    # TODO: get obs list from `runlist` input file
    obs_ids = [23523]
    make_obsdef(obs_ids)
    run_obssim(model)
    # cleanup()

    infile = 'hess_events_023523.fits.gz'
    outfile='hess_events_023523_empty.fits'
    select_event_subset(infile=infile, outfile=outfile)



def make_obsdef(obs_ids):
    """Make observation definition XML file."""
    template = """
<observation_list title="observation library">
    <observation name="Crab" id="{obs_id}" instrument="HESS">
        <parameter name="EventList" file="{event_file}"/>
        <parameter name="EffectiveArea" file="{aeff_file}[AEFF_2D]"/>
        <!--<parameter name="EnergyDispersion" file="{edisp_file}[EDISP_2D]"/>-->
        <parameter name="PointSpreadFunction" file="{psf_file}[PSF_2D_GAUSS]"/>
        <!--<parameter name="Background" file="hess_bkg_offruns_023523.fits.gz"/>-->
    </observation>
</observation_list>
    """
    filename = 'obsdef.xml'
    context = dict()
    context['obs_id'] = obs_ids[0]
    context['event_file'] = 'run023400-023599/run023523/hess_events_023523.fits.gz'
    context['aeff_file'] = 'run023400-023599/run023523/hess_aeff_2d_023523.fits.gz'
    context['edisp_file'] = 'run023400-023599/run023523/hess_edisp_2d_023523.fits.gz'
    context['psf_file'] = 'run023400-023599/run023523/hess_psf_3gauss_023523.fits.gz'
    text = template.format(**context)
    with open(filename, 'w') as fh:
        log.info('Writing {}'.format(filename))
        fh.write(text)


def run_obssim(model):
    import ctools
    sim = ctools.ctobssim()
    sim['inmodel'] = 'model.xml'
    sim['inobs'] = 'obsdef.xml'
    sim['outevents'] = 'outevents.xml'
    sim['prefix'] = 'hess_events_'
    sim['emin'] = 0.5
    sim['emax'] = 70
    sim['rad'] = 3
    sim['logfile'] = 'simulation_output.log'
    sim.execute()


def select_event_subset(infile, outfile, n_event_max=5):
    """Select subset of events.

    We use this for H.E.S.S. event lists which we're not allowed
    to share publicly, e.g. for Gammalib or Gammapy tests and bug reports.
    But after just keeping a few (5 by default) it's not an issue any more.
    """
    log.info('Reading {}'.format(infile))
    hdu = fits.open(infile)['EVENTS']

    log.info('Selecting first {} events...'.format(n_event_max))
    hdu.data = hdu.data[:n_event_max]

    log.info('Writing {}'.format(outfile))
    hdu.writeto(outfile, clobber=True)


def cleanup():
    import os
    os.system("rm *temp*")
    os.system("mv hess_events_0.fits hess_events_simulated_023523.fits")
