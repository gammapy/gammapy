# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fermi datasets.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import tarfile
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file
from ..irf import EnergyDependentTablePSF
from ..data import SpectralCube
from ..datasets import get_path

__all__ = ['FermiGalacticCenter',
           'FermiVelaRegion',
           'fetch_fermi_catalog',
           'fetch_fermi_extended_sources',
           'fetch_fermi_diffuse_background_model',
           ]


FERMI_CATALOGS = '2FGL 1FGL 1FHL 2PC'.split()


def fetch_fermi_catalog(catalog, extension=None):
    """Fetch Fermi catalog data.

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

    filename = download_file(url, cache=True)
    hdu_list = fits.open(filename)

    if extension != None:
        catalog_table = Table(hdu_list[extension].data)
        return catalog_table
    else:
        return hdu_list

FERMI_EXTENDED = '2FGL 1FHL'.split()


def fetch_fermi_extended_sources(catalog):
    """Fetch Fermi catalog extended source images.

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

    filename = download_file(url, cache=True)
    tar = tarfile.open(filename, 'r')

    hdu_list = []
    for member in tar.getmembers():
        if member.name.endswith(".fits"):
            file = tar.extractfile(member)
            hdu = fits.open(file)[0]
            hdu_list.append(hdu)
    hdu_list = fits.HDUList(hdu_list)

    return hdu_list


def fetch_fermi_diffuse_background_model(filename='gll_iem_v02.fit'):
    """Fetch Fermi diffuse background model.

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
    filename = download_file(url, cache=True)

    return filename


class FermiGalacticCenter(object):
    """Fermi high-energy data for the Galactic center region.

    For details, see this
    `README file
    <https://github.com/gammapy/gammapy/blob/master/gammapy/datasets/data/fermi/README.rst>`_.
    """

    @staticmethod
    def filenames():
        """Dictionary of available file names."""
        result = dict()
        result['psf'] = get_path('fermi/psf.fits')
        result['counts'] = get_path('fermi/fermi_counts.fits.gz')
        result['diffuse_model'] = get_path('fermi/gll_iem_v02_cutout.fits')
        result['exposure_cube'] = get_path('fermi/fermi_exposure.fits.gz')

        return result

    @staticmethod
    def counts():
        """Counts image as `astropy.io.fits.ImageHDU`."""
        filename = FermiGalacticCenter.filenames()['counts']
        return fits.open(filename)[1]

    @staticmethod
    def psf():
        """PSF as `~gammapy.irf.EnergyDependentTablePSF`."""
        filename = FermiGalacticCenter.filenames()['psf']
        return EnergyDependentTablePSF.read(filename)

    @staticmethod
    def diffuse_model():
        """Diffuse model spectral cube.

        Returns
        -------
        spectral_cube : `~gammapy.data.SpectralCube`
            Diffuse model spectral cube
        """
        filename = FermiGalacticCenter.filenames()['diffuse_model']
        return SpectralCube.read(filename)

    @staticmethod
    def exposure_cube():
        """Exposure cube.

        Returns
        -------
        spectral_cube : `~gammapy.data.SpectralCube`
            Exposure cube
        """
        filename = FermiGalacticCenter.filenames()['exposure_cube']
        return SpectralCube.read(filename)


class FermiVelaRegion(object):
    """Fermi high-energy data for the Vela region.

    For details, see
    `README file for FermiVelaRegion
    <https://github.com/gammapy/gammapy-extra/blob/master/datasets/vela_region/README.rst>`_.
    """

    @staticmethod
    def filenames():
        """Dictionary of available file names."""
        def get(filename):
            return get_path('vela_region/' + filename, location='remote')

        result = dict()
        result['counts_cube'] = get('counts_vela.fits')
        result['exposure_cube'] = get('exposure_vela.fits')
        result['background_image'] = get('background_vela.fits')
        result['total_image'] = get('total_vela.fits')
        result['diffuse_model'] = get('gll_iem_v05_rev1_cutout.fits')
        result['events'] = get('events_vela.fits')
        result['psf'] = get('psf_vela.fits')
        result['livetime_cube'] = get('livetime_vela.fits')
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
        psf : `~gammapy.irf.EnergyDependentTablePSF`.
            PSF
        """
        filename = FermiVelaRegion.filenames()['psf']
        return EnergyDependentTablePSF.read(filename)

    @staticmethod
    def diffuse_model():
        """Diffuse model spectral cube cutout for Vela region.

        Returns
        -------
        spectral_cube : `~gammapy.data.SpectralCube`
            Diffuse model spectral cube
        """
        filename = FermiVelaRegion.filenames()['diffuse_model']
        return SpectralCube.read(filename)

    @staticmethod
    def background_image():
        """Predicted background counts image.
        Based on the Fermi Diffuse model (see class docstring).

        Returns
        -------
        background_cube : `~astropy.io.fits.PrimaryHDU`
            Diffuse model image.
        """
        filename = FermiVelaRegion.filenames()['background_image']
        return fits.open(filename)[0]

    @staticmethod
    def predicted_image():
        """Predicted counts spectral image including Vela Pulsar.
        Based on the Fermi Diffuse model (see class docstring) and
        Vela Point source model.

        Returns
        -------
        background_cube : `~astropy.io.fits.PrimaryHDU`
            Predicted model image.
        """
        filename = FermiVelaRegion.filenames()['total_image']
        return fits.open(filename)[0]

    @staticmethod
    def events():
        """Fermi Events list for Vela Region.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`.
            Events list.
        """
        filename = FermiVelaRegion.filenames()['events']
        return fits.open(filename)

    @staticmethod
    def exposure_cube():
        """Exposure cube.

        Returns
        -------
        exposure_cube : `~gammapy.data.SpectralCube`
            Exposure cube
        """
        filename = FermiVelaRegion.filenames()['exposure_cube']
        return SpectralCube.read(filename)

    @staticmethod
    def livetime_cube():
        """Livetime cube.

        Returns
        -------
        livetime_cube : `~astropy.io.fits.HDUList`
            Livetime cube hdu_list
        """
        filename = FermiVelaRegion.filenames()['livetime_cube']
        return fits.open(filename)
