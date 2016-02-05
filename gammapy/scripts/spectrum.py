import logging
import click

from gammapy.spectrum import run_spectrum_extraction_using_config
from gammapy.spectrum.results import SpectrumResult
from gammapy.spectrum.results import SpectrumResultDict
from gammapy.spectrum.spectrum_fit import run_spectrum_fit_using_config
from gammapy.utils.scripts import read_yaml

__all__ = []

log = logging.getLogger(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """Gammapy tool for spectrum extraction and fitting.

    \b
    Examples
    --------
    \b
    TODO
    """
    pass

@cli.command('extract')
@click.option('--interactive', is_flag=True, default=False)
@click.option('--dry-run', is_flag=True, default=False)
@click.argument('configfile')
def extract_spectrum(configfile, interactive, dry_run):
    """Extract 1D spectral information from event list and 2D IRFs"""
    config = read_yaml(configfile)
    analysis = run_spectrum_extraction_using_config(config, dry_run=dry_run)
    if interactive:
        import IPython; IPython.embed()


@cli.command('fit')
@click.option('--interactive', is_flag=True, default=False)
@click.argument('configfile')
def extract_spectrum(configfile, interactive):
    """Fit spectral model to 1D spectrum"""
    config = read_yaml(configfile)
    fit = run_spectrum_fit_using_config(config)
    if interactive:
        import IPython; IPython.embed()

@cli.command('display')
@click.argument('files', nargs=-1, required=True)
@click.option('--short', is_flag=True, default=False)
@click.option('--browser', is_flag=True, default=False)
def display_spectrum(files, short, browser):
    """Display results table of one or several spectrum results files"""

    master = SpectrumResultDict.from_files(files)
    t = master.to_table(format='.3g')

    if short:
        t = t['analysis', 'index', 'index_err', 'flux [1TeV]', 'flux_err [1TeV]']

    if browser:
        t.show_in_browser(jsviewer=True)
    else:
        print(t)

@cli.command('plot')
@click.argument('file', nargs=1, required=True)
@click.option('--flux_unit', default='m-2 s-1 TeV-1', help='Unit of flux axis')
@click.option('--energy_unit', default='TeV', help='Unit of energy axis')
@click.option('--energy_power', default=0, help='Energy power to multiply flux with')
def overplot_spectrum(file, flux_unit, energy_unit, energy_power):
    """Plot spectrum results file"""

    import matplotlib.pyplot as plt
    spec = SpectrumResult.from_all(file)
    spec.fit.plot_spectrum(flux_unit=flux_unit, energy_unit=energy_unit,
                           e_power=energy_power)
    plt.show()