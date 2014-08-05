# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Example and test datasets.

Example how to load a dataset from file::

    from gammapy import datasets
    image = datasets.poisson_stats_image()

To get a summary table of available datasets::

    from gammapy import datasets
    datasets.list_datasets()

To download all datasets into a local cache::

    from gammapy import datasets
    datasets.download_datasets()
"""
import numpy as np
from astropy.utils import data
from astropy.utils.data import get_pkg_data_filename
from astropy.units import Quantity
from astropy.io import fits
from astropy.table import Table
from astropy.utils import data
import tarfile
import os
from ..spectral_cube import GammaSpectralCube


included_datasets = ['poisson_stats_image',
                     'tev_spectrum',
                     'load_crab_flux_points',
                     'diffuse_gamma_spectrum',
                     'electron_spectrum',
                     'FermiGalacticCenter',
                     'fetch_fermi_catalog',
                     'arf_fits_table',
                     'psf_fits_table',
                     ]

remote_datasets = ['fetch_fermi_extended_sources',
                   'FermiVelaRegion',
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


def arf_fits_table():
    """Read arf fits table"""
    filename = get_pkg_data_filename('data/irfs/arf.fits')
    return fits.open(filename)


def psf_fits_table():
    """Read psf fits table"""
    filename = get_pkg_data_filename('data/irfs/psf.fits')
    return fits.open(filename)


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
        Depending on the ``extra_info`` and ``return_filenames`` options.
    """
    if extra_info:
        out = dict()
        for name in ['counts', 'model', 'source', 'background']:
            filename = 'data/poisson_stats_image/{0}.fits.gz'.format(name)
            filename = get_pkg_data_filename(filename)
            if return_filenames:
                out[name] = filename
            else:
                data = fits.getdata(filename)
                out[name] = data
    else:
        filename = 'data/poisson_stats_image/counts.fits.gz'
        filename = get_pkg_data_filename(filename)
        if return_filenames:
            out = filename
        else:
            out = fits.getdata(filename)

    return out


class FermiGalacticCenter(object):
    """Fermi high-energy data for the Galactic center region.

    TODO: document energy band, region, content of the files. 
    TODO: document
    """

    @staticmethod
    def filenames():
        """Dictionary of available file names."""
        result = dict()
        result['psf'] = get_pkg_data_filename('data/fermi/psf.fits')
        result['counts'] = get_pkg_data_filename('data/fermi/fermi_counts.fits.gz')
        result['diffuse_model'] = get_pkg_data_filename('data/fermi/gll_iem_v02_cutout.fits')
        result['exposure_cube'] = get_pkg_data_filename('data/fermi/fermi_exposure.fits.gz')

        return result

    @staticmethod
    def counts():
        """Counts image as `astropy.io.fits.ImageHDU`."""
        filename = FermiGalacticCenter.filenames()['counts']
        return fits.open(filename)[1]

    @staticmethod
    def psf():
        """PSF as `astropy.io.fits.HDUList`."""
        filename = FermiGalacticCenter.filenames()['psf']
        return fits.open(filename)

    @staticmethod
    def diffuse_model():
        """Diffuse model spectral cube.

        Returns
        -------
        spectral_cube : `~gammapy.spectral_cube.GammaSpectralCube`
            Diffuse model spectral cube
        """
        filename = FermiGalacticCenter.filenames()['diffuse_model']
        return GammaSpectralCube.read(filename)

    @staticmethod
    def exposure_cube():
        """Exposure cube.

        Returns
        -------
        spectral_cube : `~gammapy.spectral_cube.GammaSpectralCube`
            Exposure cube
        """
        filename = FermiGalacticCenter.filenames()['exposure_cube']
        return GammaSpectralCube.read(filename)


class FermiVelaRegion(object):
    """Fermi high-energy data for the Vela region.

    * 5 years of observation time (2008-08-05 to 2013-08-05)
    * Event class and IRF: P7REP_CLEAN_V15
    * Max zenith angle cut: 105 deg
    * 10 GeV < Energy < 500 GeV in 3 equal log10 bins
    * CenterPosition: GLON = 263.05836967702709; GLAT = 3.9298511274632784
    * Coordinate System: GALACTIC
    * Image Size: 5 x 5 degrees (Search center 3 degrees)
    * Pixel size: 0.1 degrees/pixel
    * Image Shape: (50, 50) pixels

    For further details see:
    https://github.com/gammapy/gammapy-extra/tree/master/datasets/vela_region
    """

    @staticmethod
    def filenames():
        """Dictionary of available file names."""
        result = dict()

        BASE_URL = 'https://github.com/gammapy/gammapy-extra/raw/master/datasets/vela_region/'
        url_counts = BASE_URL + 'counts_vela.fits'
        url_exposure = BASE_URL + 'exposure_vela.fits'
        url_background = BASE_URL + 'background_vela.fits'
        url_diffuse = BASE_URL + 'gll_iem_v05_rev1_cutout.fit'
        url_psf = BASE_URL + 'psf_vela.fits'

        result['counts_cube'] = data.download_file(url_counts, cache=True)
        result['exposure_cube'] = data.download_file(url_exposure, cache=True)
        result['background_image'] = data.download_file(url_background, cache=True)
        result['diffuse_model'] = data.download_file(url_diffuse, cache=True)
        result['psf'] = data.download_file(url_psf, cache=True)
        return result

    @staticmethod
    def counts_cube():
        """Counts cube.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            * Counts cube `~astropy.io.fits.PrimaryHDU`.
            * Energy bins `~astropy.io.fits.BinTableHDU`.
            * MET bins `~astropy.io.fits.BinTableHDU`.
        """
        filename = FermiVelaRegion.filenames()['counts_cube']
        return fits.open(filename)
    
    @staticmethod
    def psf():
        """Fermi PSF for the Vela region.

        Returns
        -------
        hdu_list : `astropy.io.fits.HDUList`.
            PSF information as hdu_list
        """
        filename = FermiVelaRegion.filenames()['psf']
        return fits.open(filename)
    
    @staticmethod
    def diffuse_model():
        """Diffuse model spectral cube cutout for Vela region.

        Returns
        -------
        spectral_cube : `~gammapy.spectral_cube.GammaSpectralCube`
            Diffuse model spectral cube
        """
        filename = FermiVelaRegion.filenames()['diffuse_model']
        return GammaSpectralCube.read(filename)

    @staticmethod
    def background_image():
        """Predicted background counts spectral cube.
        Based on the Fermi Diffuse model (see class docstring).

        Returns
        -------
        background_cube : `~astropy.io.fits.PrimaryHDU`
            Diffuse model spectral cube
        """
        filename = FermiVelaRegion.filenames()['background_image']
        return fits.open(filename)[0]

    @staticmethod
    def exposure_cube():
        """Exposure cube.

        Returns
        -------
        exposure_cube : `~gammapy.spectral_cube.GammaSpectralCube`
            Exposure cube
        """
        filename = FermiVelaRegion.filenames()['exposure_cube']
        return GammaSpectralCube.read(filename)


def tev_spectrum(source_name):
    """Get published TeV flux point measurements.

    TODO: give references to publications and describe the returned table.

    Parameters
    ----------
    source_name : str
        Source name

    Returns
    -------
    spectrum : `~astropy.table.Table`
        Energy spectrum as a table (one flux point per row).
    """
    if source_name == 'crab':
        filename = 'tev_spectra/crab_hess_spec.txt'
    else:
        raise ValueError('Data not available for source: {0}'.format(source_name))

    filename = get_pkg_data_filename(filename)
    table = Table.read(filename, format='ascii',
                       names=['energy', 'flux', 'flux_lo', 'flux_hi'])
    table['flux_err'] = 0.5 * (table['flux_lo'] + table['flux_hi'])
    return table


def load_crab_flux_points(component='both', with_fermi_flare=False):
    """Load published Crab pulsar and nebula flux points.

    Besides the usual flux point columns, this table contains
    the following two columns:
    * component : {'pulsar', 'nebula'}
    * paper : Short string describing which point originates from which paper.

    TODO:

    * Add link to Crab flux point tutorial in Gammapy where these points are plotted.
    * Add html links to ADS directly in the docstring and as a table column.

    Parameters
    ----------
    component : {'pulsar', 'nebula', 'both'}
        Which emission component to include

    Returns
    -------
    flux_points : `~astropy.table.Table`
        Flux point table

    Notes
    -----
    This data compilation is from Buehler and Blandford, Rep. Prog. Phys. 77, 2014.
    It was contributed to Gammapy directly by Rolf Buehler via a pull request.

    The data for the nebula were taken from Meyer et al. Astron. Astrophys. 523 2010
    with the addition of the Fermi-LAT measurement reported in Buehler et al. ApJ 749 2012.

    The pulsar spectrum is reproduced from Kuiper et al Astron. Astrophys. 378 2001 .
    Additionally shown are infrared measurements reported in Sollerman et al. ApJ 537 2000
    and Tziamtzis et al. Astron. Astrophys. 508 2009, radio measurements referenced in
    Thompson et al. ApJ 516 1999 and gamma-ray measurements referenced in
    Aleksic et al. ApJ 742 2011, Aliu et al. Science 334 2011,
    Aleksic et al. Astron. Astrophys. 540 2012
    and Abdo et al. Astrophys. J. Suppl. Ser. 208 2013.

    """
    filename = 'data/tev_spectra/crab_mwl.fits.gz'
    filename = get_pkg_data_filename(filename)
    table = Table.read(filename)

    if component == 'pulsar':
        mask = table['component'] == 'pulsar'
        table = table[mask]
    elif component == 'nebula':
        mask = table['component'] == 'nebula'
        table = table[mask]
    elif component == 'both':
        pass
    else:
        raise ValueError('Invalid component: {0}'.format(component))

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
    spectrum : `~astropy.table.Table`
        Energy spectrum as a table (one flux point per row).
    """
    if reference == 'Fermi':
        filename = 'data/tev_spectra/diffuse_isotropic_gamma_spectrum_fermi.txt'
    elif reference == 'Fermi2':
        filename = 'data/tev_spectra/diffuse_isotropic_gamma_spectrum_fermi2.txt'
    else:
        raise ValueError('Data not available for reference: {0}'.format(reference))

    return _read_diffuse_gamma_spectrum_fermi(filename)


def _read_diffuse_gamma_spectrum_fermi(filename):
    filename = get_pkg_data_filename(filename)
    table = Table.read(filename, format='ascii',
                       names=['energy', 'flux', 'flux_hi', 'flux_lo'])
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
    spectrum : `~astropy.table.Table`
        Energy spectrum as a table (one flux point per row).
    """
    if reference == 'HESS':
        filename = 'data/tev_spectra/electron_spectrum_hess.txt'
        return _read_electron_spectrum_hess(filename)
    elif reference == 'HESS low energy':
        filename = 'data/tev_spectra/electron_spectrum_hess_low_energy.txt'
        return _read_electron_spectrum_hess(filename)
    elif reference == 'Fermi':
        filename = 'data/tev_spectra/electron_spectrum_fermi.txt'
        return _read_electron_spectrum_fermi(filename)
    else:
        raise ValueError('Data not available for reference: {0}'.format(reference))


def _read_electron_spectrum_hess(filename):
    filename = get_pkg_data_filename(filename)
    table = Table.read(filename, format='ascii',
                       names=['energy', 'flux', 'flux_lo', 'flux_hi'])
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

FERMI_CATALOGS = '2FGL 1FGL 1FHL 2PC'.split()


def fetch_fermi_catalog(catalog, extension=None):
    """Get Fermi catalog data.

    Reference: http://fermi.gsfc.nasa.gov/ssc/data/access/lat/.

    The Fermi catalogs contain the following relevant catalog HDUs:

    * 2FGL Catalog : LAT 2-year Point Source Catalog
        * ``LAT_Point_Source_Catalog`` Point Source Catalog Table.
        * ``ExtendedSources`` Extended Source Catalog Table.
    * 1FGL Catalog : LAT 1-year Point Source Catalog
        * ``LAT_Point_Source_Catalog`` Point Source Catalog Table.
    * 1FHL Catalog : First Fermi-LAT Catalog of Sources above 10 GeV
        * ``LAT_Point_Source_Catalog`` Point Source Catalog Table.
        * ``ExtendedSources`` Extended Source Catalog Table.
    * 2PC Catalog : LAT Second Catalog of Gamma-ray Pulsars
        * ``PULSAR_CATALOG`` Pulsar Catalog Table.
        * ``SPECTRAL`` Table of Pulsar Spectra Parameters.
        * ``OFF_PEAK`` Table for further Spectral and Flux data for the Catalog.

    Parameters
    ----------
    catalog : {'2FGL', '1FGL', '1FHL', '2PC'}
       Specifies which catalog to display.
    extension : str
        Specifies which catalog HDU to provide as a table (optional).
        See list of catalog HDUs above.

    Returns
    -------
    hdu_list (Default) : `~astropy.io.fits.HDUList`
        Catalog FITS HDU list (for access to full catalog dataset).
    catalog_table : `~astropy.table.Table`
        Catalog table for a selected hdu extension.

    Examples
    --------
    >>> from gammapy.datasets import fetch_fermi_catalog
    >>> fetch_fermi_catalog('2FGL')  # doctest: +REMOTE_DATA
        [<astropy.io.fits.hdu.image.PrimaryHDU at 0x3330790>,
         <astropy.io.fits.hdu.table.BinTableHDU at 0x338b990>,
         <astropy.io.fits.hdu.table.BinTableHDU at 0x3396450>,
         <astropy.io.fits.hdu.table.BinTableHDU at 0x339af10>,
         <astropy.io.fits.hdu.table.BinTableHDU at 0x339ff10>]

    >>> from gammapy.datasets import fetch_fermi_catalog
    >>> fetch_fermi_catalog('2FGL', 'LAT_Point_Source_Catalog')  # doctest: +REMOTE_DATA
        <Table rows=1873 names= ... >
    """
    BASE_URL = 'http://fermi.gsfc.nasa.gov/ssc/data/access/lat/'

    if catalog == '2FGL':
        url = BASE_URL + '2yr_catalog/gll_psc_v08.fit'
    elif catalog == '1FGL':
        url = BASE_URL + '/1yr_catalog/gll_psc_v03.fit'
    elif catalog == '1FHL':
        url = BASE_URL + '/1FHL/gll_psch_v07.fit'
    elif catalog == '2PC':
        url = BASE_URL + '2nd_PSR_catalog/2PC_catalog_v03.fits'
    else:
        ss = 'Invalid catalog: {0}\n'.format(catalog)
        ss += 'Available: {0}'.format(', '.join(FERMI_CATALOGS))
        raise ValueError(ss)

    filename = data.download_file(url, cache=True)
    hdu_list = fits.open(filename)

    if extension != None:
        catalog_table = Table(hdu_list[extension].data)
        return catalog_table
    else:
        return hdu_list

FERMI_EXTENDED = '2FGL 1FHL'.split()


def fetch_fermi_extended_sources(catalog):
    """Get Fermi catalog extended source images.

    Reference: http://fermi.gsfc.nasa.gov/ssc/data/access/lat/.

    Extended source are available for the following Fermi catalogs:

    * 2FGL Catalog : LAT 2-year Point Source Catalog
    * 1FHL Catalog : First Fermi-LAT Catalog of Sources above 10 GeV

    Parameters
    ----------
    catalog : {'2FGL', '1FHL'}
       Specifies which catalog extended sources to return.

    Returns
    -------
    hdu_list : `~astropy.io.fits.HDUList`
        FITS HDU list of FITS ImageHDUs for the extended sources.

    Examples
    --------
    >>> from gammapy.datasets import fetch_fermi_extended_sources
    >>> sources = fetch_fermi_extended_sources('2FGL')
    >>> len(sources) = 12
    """
    BASE_URL = 'http://fermi.gsfc.nasa.gov/ssc/data/access/lat/'
    if catalog == '2FGL':
        url = BASE_URL + '2yr_catalog/gll_psc_v07_templates.tgz'
    elif catalog == '1FHL':
        url = BASE_URL + '1FHL/LAT_extended_sources_v12.tar'
    else:
        ss = 'Invalid catalog: {0}\n'.format(catalog)
        ss += 'Available: {0}'.format(', '.join(FERMI_EXTENDED))
        raise ValueError(ss)

    filename = data.download_file(url, cache=True)
    tar = tarfile.open(filename, 'r')

    hdu_list = []
    for member in tar.getmembers():
        if member.name.endswith(".fits"):
            file = tar.extractfile(member)
            hdu = fits.open(file)[0]
            hdu_list.append(hdu)
    hdu_list = fits.HDUList(hdu_list)

    return hdu_list


def get_fermi_diffuse_background_model(filename='gll_iem_v02.fit'):
    """Get Fermi diffuse background model.

    Parameters
    ----------
    filename : str
        Diffuse model file name

    Returns
    -------
    filename : str
        Full local path name
    """
    BASE_URL = 'http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/'

    url = BASE_URL + filename
    filename = data.download_file(url, cache=True)

    return filename
