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

@cli.command('compare')
@click.argument('files', nargs=-1, required=True)
@click.option('--short', is_flag=True, default=False)
def compare_spectrum(files, short):
    """Create comparison table from several spectrum results files"""
    if len(files) == 0:
        raise ValueError("")

    master = SpectrumResultDict()
    for f in files:
        val = SpectrumResult.from_all(f)
        master[f] = val

    if short:
        master = master['index', 'index_err', 'flux [1TeV]', 'flux_err [1TeV]']

    print(master.to_table(format='.3g'))


