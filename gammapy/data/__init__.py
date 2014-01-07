# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Example and test datasets.

Example how to load a dataset from file:

    >>> from gammapy import data
    >>> image = data.poisson_stats_image()

To get a summary table of available datasets::

    >>> from gammapy import data
    >>> data.list_datasets()

To download all datasets into a local cache::

    >>> from gammapy import data
    >>> data.download_datasets()
"""
from astropy.utils.data import get_pkg_data_filename
from astropy.units import Quantity
from astropy.io import fits
from astropy.table import Table

included_datasets = ['poisson_stats_image',
                     'tev_spectrum',
                     'diffuse_gamma_spectrum',
                     'electron_spectrum']

remote_datasets = ['fermi_galactic_center'
                   ]

datasets = included_datasets + remote_datasets

__all__ = ['list_datasets',
           'download_datasets',
           ] + datasets


def list_datasets():
    """List available datasets."""
    for name in datasets:
        docstring = eval('{0}.__doc__'.format(name))
        summary = docstring.split('\n')[0]
        print('{0:>25s} : {1}'.format(name, summary))


def download_datasets(names='all'):
    """Download all datasets in to a local cache.
    
    TODO: set this up and test
    """
    for name in remote_datasets:
        raise NotImplementedError
        # Check if available in cache
        # if not download to cache


def poisson_stats_image(extra_info=False, return_filenames=False):
    """Poisson statistics counts image of a Gaussian source on flat background.

    See poissson_stats_image/README.md for further info.
    TODO: add better description (extract from README?)

    Parameters
    ----------
    extra_info : bool
        If true, a dict of images is returned.
    
    return_filenames : bool
        If true, return filenames instead of images

    Returns
    -------
    data : numpy array or dict of arrays or filenames
        Depending on the `extra_info` and `return_filenames` options.
    """
    if extra_info:
        out = dict()
        for name in ['counts', 'model', 'source', 'background']:
            filename = 'poisson_stats_image/{0}.fits.gz'.format(name)
            filename = get_pkg_data_filename(filename)
            if return_filenames:
                out[name] = filename
            else:
                data = fits.getdata(filename)
                out[name] = data
    else:
        filename = 'poisson_stats_image/counts.fits.gz'
        filename = get_pkg_data_filename(filename)
        if return_filenames:
            out = filename
        else:
            out = fits.getdata(filename)

    return out


def fermi_galactic_center():
    """Fermi high-energy counts image of the Galactic center region.
    
    TODO: document energy band, region, ... add script to produce the image 
    TODO: download from Dropbox
    """
    raise NotImplementedError


def tev_spectrum(source_name):
    """Get published TeV flux point measurements.

    TODO: give references to publications and describe the returned table.

    Parameters
    ----------
    source_name : str
        Source name

    Returns
    -------
    spectrum : `astropy.table.Table`
        Energy spectrum as a table (one flux point per row).
    """
    if source_name == 'crab':
        filename = 'tev_spectra/crab_hess_spec.txt'
    else:
        raise ValueError('Data not available for source: {0}'.format(source_name))

    filename = get_pkg_data_filename(filename)
    table = Table.read(filename, format='ascii',
                       names = ['energy', 'flux', 'flux_lo', 'flux_hi'])
    table['flux_err'] = 0.5 * (table['flux_lo'] + table['flux_hi'])
    return table


def diffuse_gamma_spectrum(reference):
    """Get published diffuse gamma-ray spectrum.
    
    TODO: give references to publications and describe the returned table.
    
    Parameters
    ----------
    reference : {'Fermi', 'Fermi2'}
        Which publication.
    
    Returns
    -------
    spectrum : `astropy.table.Table`
        Energy spectrum as a table (one flux point per row).    
    """
    if reference == 'Fermi':
        filename = 'tev_spectra/diffuse_isotropic_gamma_spectrum_fermi.txt'
    elif reference == 'Fermi2':
        filename = 'tev_spectra/diffuse_isotropic_gamma_spectrum_fermi2.txt'
    else:
        raise ValueError('Data not available for reference: {0}'.format(reference))

    return _read_diffuse_gamma_spectrum_fermi(filename)


def _read_diffuse_gamma_spectrum_fermi(filename):
    filename = get_pkg_data_filename(filename)
    table = Table.read(filename, format='ascii',
                       names = ['energy', 'flux', 'flux_hi', 'flux_lo'])
    table['flux_err'] = 0.5 * (table['flux_lo'] + table['flux_hi'])

    table['energy'] = Quantity(table['energy'], 'MeV').to('TeV')

    for colname in table.colnames:
        if 'flux' in colname:
            energy = Quantity(table['energy'], 'TeV')
            energy2_flux = Quantity(table[colname], 'MeV cm^-2 s^-1 sr^-1')
            table[colname] = (energy2_flux / energy ** 2).to('m^-2 s^-1 TeV^-1 sr^-1')

    return table


def electron_spectrum(reference):
    """Get published electron spectrum.
    
    TODO: give references to publications and describe the returned table.
    
    Parameters
    ----------
    reference : {'HESS', 'HESS low energy', 'Fermi'}
        Which publication.
    
    Returns
    -------
    spectrum : `astropy.table.Table`
        Energy spectrum as a table (one flux point per row).    
    """
    if reference == 'HESS':
        filename = 'tev_spectra/electron_spectrum_hess.txt'
        return _read_electron_spectrum_hess(filename)
    elif reference == 'HESS low energy':
        filename = 'tev_spectra/electron_spectrum_hess_low_energy.txt'
        return _read_electron_spectrum_hess(filename)
    elif reference == 'Fermi':
        filename = 'tev_spectra/electron_spectrum_fermi.txt'
        return _read_electron_spectrum_fermi(filename)
    else:
        raise ValueError('Data not available for reference: {0}'.format(reference))


def _read_electron_spectrum_hess(filename):
    filename = get_pkg_data_filename(filename)
    table = Table.read(filename, format='ascii',
                       names = ['energy', 'flux', 'flux_lo', 'flux_hi'])
    table['flux_err'] = 0.5 * (table['flux_lo'] + table['flux_hi'])
    
    table['energy'] = Quantity(table['energy'], 'GeV').to('TeV')

    # The ascii files store fluxes as (E ** 3) * dN / dE.
    # Here we change this to dN / dE.
    for colname in table.colnames:
        if 'flux' in colname:
            energy = Quantity(table['energy'], 'TeV')
            energy3_flux = Quantity(table[colname], 'GeV^2 m^-2 s^-1 sr^-1')
            table[colname] = (energy3_flux / energy ** 3).to('m^-2 s^-1 TeV^-1 sr^-1')

    return table


def _read_electron_spectrum_fermi(filename):
    filename = get_pkg_data_filename(filename)
    t = Table.read(filename, format='ascii')

    table = Table()
    table['energy'] = Quantity(t['E'], 'GeV').to('TeV')
    table['flux'] = Quantity(t['y'], 'm^-2 s^-1 GeV^-1 sr^-1').to('m^-2 s^-1 TeV^-1 sr^-1')
    flux_err = 0.5 * (t['yerrtot_lo'] + t['yerrtot_up'])
    table['flux_err'] = Quantity(flux_err, 'm^-2 s^-1 GeV^-1 sr^-1').to('m^-2 s^-1 TeV^-1 sr^-1')

    return table
