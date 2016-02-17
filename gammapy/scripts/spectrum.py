import logging

import click

from gammapy.extern.pathlib import Path

click.disable_unicode_literals_warning = True
from ..spectrum import SpectrumExtraction
from ..spectrum.spectrum_fit import SpectrumFit
from ..spectrum.results import SpectrumResult, SpectrumFitResult
from ..spectrum.results import SpectrumResultDict
from ..spectrum.spectrum_pipe import run_spectrum_analysis_using_config
from ..utils.scripts import read_yaml

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
    analysis = SpectrumExtraction.from_configfile(configfile)
    if dry_run:
        return analysis
    else:
        analysis.run()

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
    """Fit spectral model to 1D spectrum"""
    ctx.invoke(extract_spectrum, configfile=configfile)
    ctx.invoke(fit_spectrum, configfile=configfile)


@cli.command('display')
def display_spectrum():
    """Display results of spectrum fit"""
    stats = SpectrumResult.from_yaml('total_spectrum_stats.yaml')
    stats_table = stats.to_table()['n_on', 'n_off', 'alpha', 'n_bkg', 'excess', 'energy_range']
    files = [str(l) for l in Path.cwd().glob('fit*.yaml')]
    fit = SpectrumResultDict.from_files(files)
    fit_table = fit.to_table()
    fit_table.remove_column('analysis')

    print('\n\n')
    print('\t\tSpectrum Stats')
    print('\t\t--------------')
    print(stats_table)
    print('\n\n')
    print('\t\tSpectral Fit')
    print('\t\t------------')
    print(fit_table)
    print('\n\n')


@cli.command('plot')
def plot_spectrum():
    """Plot spectrum results file"""
    raise NotImplementedError
