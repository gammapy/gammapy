import logging

import click

from ..extern.pathlib import Path

click.disable_unicode_literals_warning = True
from ..spectrum import SpectrumExtraction
from ..spectrum.spectrum_fit import SpectrumFit
from ..spectrum.results import SpectrumResult, SpectrumFitResult
from ..spectrum.results import SpectrumResultDict
from ..spectrum.spectrum_pipe import run_spectrum_analysis_using_config
from ..utils.scripts import read_yaml, make_path
from ..data import Target

__all__ = [
           'SpectrumFitResult',
           'SpectrumStats',
           'FluxPoints',
           'SpectrumResult',
           'SpectrumResultDict',
           ]

log = logging.getLogger(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """Gammapy tool for spectrum extraction and fitting.

    \b
    Examples
    --------
    \b
    gammapy-spectrum extract config.yaml
    gammapy-spectrum fit config.yaml
    gammapy-spectrum all config.yaml
    """
    pass


@cli.command('extract')
@click.option('--interactive', is_flag=True, default=False, help='Open IPython session at the end')
@click.option('--dry-run', is_flag=True, default=False, help='Do no extract spectrum')
@click.argument('configfile')
def extract_spectrum(configfile, interactive, dry_run):
    """Extract 1D spectral information from event list and 2D IRFs"""
    config = read_yaml(configfile)
    target = Target.from_config(config)
    if dry_run:
        return target
    else:
        target.run_spectral_analysis()

    if interactive:
        import IPython;
        IPython.embed()


@cli.command('fit')
@click.option('--interactive', is_flag=True, default=False)
@click.argument('configfile')
def fit_spectrum(configfile, interactive):
    """Fit spectral model to 1D spectrum"""
    config = read_yaml(configfile)
    fit = SpectrumFit.from_config(config)
    fit.run()
    if interactive:
        import IPython;
        IPython.embed()


@cli.command('all')
@click.argument('configfile')
@click.pass_context
def all_spectrum(ctx, configfile):
    """Run all steps"""
    ctx.invoke(extract_spectrum, configfile=configfile)
    ctx.invoke(fit_spectrum, configfile=configfile)
