import logging
import click
from gammapy.spectrum import run_spectrum_extraction_using_config
from gammapy.spectrum.results import SpectrumResult, SpectrumFitResult
from gammapy.spectrum.results import SpectrumResultDict
from gammapy.spectrum.spectrum_fit import run_spectrum_fit_using_config
from gammapy.spectrum.spectrum_pipe import run_spectrum_analysis_using_config
from gammapy.utils.scripts import read_yaml

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
        import IPython;
        IPython.embed()


@cli.command('fit')
@click.option('--interactive', is_flag=True, default=False)
@click.argument('configfile')
def fit_spectrum(configfile, interactive):
    """Fit spectral model to 1D spectrum"""
    config = read_yaml(configfile)
    fit = run_spectrum_fit_using_config(config)
    if interactive:
        import IPython;
        IPython.embed()


@cli.command('all')
@click.option('--interactive', is_flag=True, default=False)
@click.argument('configfile')
def all_spectrum(configfile, interactive):
    """Fit spectral model to 1D spectrum"""
    config = read_yaml(configfile)
    fit, analysis = run_spectrum_analysis_using_config(config)
    if interactive:
        import IPython;
        IPython.embed()


@cli.command('pipe')
@click.argument('configfile')
def spectrum_pipe(configfile):
    """Run spectrum analysis pipeline"""
    raise NotImplementedError


@cli.command('display')
@click.argument('files', nargs=-1, required=True)
@click.option('--browser', is_flag=True, default=False,
              help='Display in browser')
@click.option('--keys', is_flag=True, default=False,
              help='Print available column names')
@click.option('--format', default='.3g', help='Column format')
@click.option('--identifiers', '-id', type=str, default=None,
              help='Comma separated list of file identifiers')
@click.option('--cols', '-c', type=str,
              help='Comma separated list of parameters to display')
@click.option('--sort', '-s', type=str,
              help='Column to sort by')
def display_spectrum(files, cols, browser, format, keys, identifiers, sort):
    """Display results table of one or several spectrum results files"""

    id = identifiers.split(',') if identifiers is not None else None
    master = SpectrumResultDict.from_files(files, id)
    t = master.to_table(format=format)
    if keys:
        print '\n'.join(t.keys())
        return
    if cols:
        t = t[cols.split(',')]
    if sort:
        t.sort(sort)
    if browser:
        t.show_in_browser(jsviewer=True)
    else:
        print(t)


@cli.command('plot')
@click.option('--function', '-f', multiple=True)
@click.option('--points', '-p', multiple=True)
@click.option('--residuals', '-r', multiple=True)
@click.option('--butterfly', '-b', multiple=True)
@click.option('--flux_unit', default='m-2 s-1 TeV-1', help='Unit of flux axis')
@click.option('--energy_unit', default='TeV', help='Unit of energy axis')
@click.option('--energy_power', default=0,
              help='Energy power to multiply flux with')
def plot_spectrum(function, butterfly, residuals, points, flux_unit,
                  energy_unit, energy_power):
    """Plot spectrum results file"""
    if butterfly or residuals:
        raise NotImplementedError

    import matplotlib.pyplot as plt
    for f in function:
        res = SpectrumResult.from_all(f)
        if res.fit is None:
            raise ValueError('File {} does not contain a fit function'.format(f))
        res.fit.plot(flux_unit=flux_unit, energy_unit=energy_unit,
        e_power=energy_power)

    for p in points:
        res = SpectrumResult.from_all(p)
        if res.points is None:
            raise ValueError('File {} does not contain flux points'.format(f))
        res.points.plot(flux_unit=flux_unit, energy_unit=energy_unit,
        e_power=energy_power)


    plt.loglog()
    plt.show()
