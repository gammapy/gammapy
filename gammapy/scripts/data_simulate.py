# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click

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

    # obs_ids = [23523]
    # make_obsdef(obs_ids)
    # run_obssim()
    # cleanup()


def make_obsdef(obs_ids):
    """Make observation definition XML file."""
    pass


def run_obssim():
    import ctools
    sim = ctools.ctobssim()
    sim['inmodel'] = 'model.xml'
    sim['inobs'] = 'observation_definition.xml'
    sim['outevents'] = 'outevents.xml'
    sim['prefix'] = 'hess_events_'
    sim['emin'] = 0.5
    sim['emax'] = 70
    sim['rad'] = 3
    sim['logfile'] = 'simulation_output.log'
    sim.execute()


def cleanup():
    import os
    os.system("rm *temp*")
    os.system("mv hess_events_0.fits hess_events_simulated_023523.fits")
