# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A quick look command line tool to check data file contents.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from astropy.io import fits
from astropy.units import Quantity
from astropy.table import Table
import click
click.disable_unicode_literals_warning = True
from .. import irf
from ..spectrum import CountsSpectrum
from ..data import EventList

__all__ = ['data_show_main']

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FILETYPES = ['events', 'aeff', 'edisp', 'psf', 'arf', 'pha']

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('filename')
@click.argument('filetype')
@click.option('--plot', '-p', 'do_plot', is_flag=True, help='Show plots?')
def data_show_main(filename, filetype, do_plot):
    """A quick look command line tool to check a single data or IRF file.

    Use the `gammapy-data-browser` to browse / check lots of files.
    """
    if do_plot:
        import matplotlib.pyplot as plt
        plt.style.use('fivethirtyeight')

    if filetype == 'events':
        show_events(filename, do_plot)
    elif filetype == 'aeff':
        show_aeff2d(filename, do_plot)
    elif filetype == 'edisp':
        show_edisp2d(filename, do_plot)
    elif filetype == 'psf':
        show_psf2d(filename, do_plot)
    elif filetype == 'arf':
        show_arf(filename, do_plot)
    elif filetype == 'pha':
        show_pha(filename, do_plot)
    else:
        msg = 'Invalid filetype: {} '.format(filetype)
        msg += 'Valid filetypes are: {}'.format(FILETYPES)
        raise ValueError(msg)


def show_events(filename, do_plot=False):
    """Print / plot some basic info about an event list file.
    """
    events = EventList.read(filename, hdu='EVENTS')
    events.info(['attributes', 'stats'])

    hdulist = fits.open(filename)
    print('\n\n')
    hdulist.info()

    print('\n\n')
    print(events.summary)

    if do_plot:
        events.peek()
        import matplotlib.pyplot as plt
        plt.show()


def show_aeff2d(filename, do_plot=False):
    """Print / plot some basic info about an AEFF2D format file.
    """
    hdulist = fits.open(filename)
    print('\n\n')
    hdulist.info()

    table = Table.read(filename, hdu='AEFF_2D')
    table.info(['attributes', 'stats'])

    aeff2d = irf.EffectiveAreaTable2D.read(filename)

    if do_plot:
        aeff2d.peek()


def show_edisp2d(filename, do_plot=False):
    """Print / plot some basic info about an EDISP2D format file.
    """
    edisp2d = irf.EnergyDispersion2D.read(filename)
    print(edisp2d.info())

    if do_plot:
        edisp2d.peek()


def show_psf2d(filename, do_plot=False):
    """Print / plot some basic info about a PSF2D format file.
    """
    psf = irf.EnergyDependentMultiGaussPSF.read(filename)
    print(psf.info())

    if do_plot:
        psf.peek()


def show_arf(hdu_list, energies, do_plot=False):
    """Print / plot some basic info about an ARF format file.

    Parameters
    ----------
    hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``SPECRESP``extension.
    plot : bool
        Make an effective area vs. energy plot.
    """
    arf = irf.EffectiveAreaTable.from_fits(hdu_list)
    energies = Quantity(energies, 'TeV')
    print(arf.info(energies=energies))

    if do_plot:
        arf.plot_area_vs_energy('effective_area.png')


def show_pha(filename, do_plot=True):
    """Print / plot some basic info about a PHA format file.
    """
    pha = CountsSpectrum.read(filename)
    print(pha.info())

    if do_plot:
        pha.peek()


def show_background(filename, do_plot=False):
    """Print / plot some basic info about a background cube.
    """
    raise NotImplementedError
