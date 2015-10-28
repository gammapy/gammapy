# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fermi datasets.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file
from .core import gammapy_extra

__all__ = [
    'FermiGalacticCenter',
    'FermiVelaRegion',
    'fetch_fermi_diffuse_background_model',
    'load_lat_psf_performance',
]


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
        base_dir = gammapy_extra.dir / 'test_datasets/unbundled/fermi'
        result = dict()
        result['psf'] = str(base_dir / 'psf.fits')
        result['counts'] = str(base_dir / 'fermi_counts.fits.gz')
        result['diffuse_model'] = str(base_dir / 'gll_iem_v02_cutout.fits')
        result['exposure_cube'] = str(base_dir / 'fermi_exposure.fits.gz')

        return result

    @staticmethod
    def counts():
        """Counts image (`astropy.io.fits.ImageHDU`)"""
        filename = FermiGalacticCenter.filenames()['counts']
        return fits.open(filename)[1]

    @staticmethod
    def psf():
        """PSF as `~gammapy.irf.EnergyDependentTablePSF`"""
        from ..irf import EnergyDependentTablePSF
        filename = FermiGalacticCenter.filenames()['psf']
        return EnergyDependentTablePSF.read(filename)

    @staticmethod
    def diffuse_model():
        """Diffuse model (`~gammapy.data.SpectralCube`)"""
        from ..data import SpectralCube
        filename = FermiGalacticCenter.filenames()['diffuse_model']
        return SpectralCube.read(filename)

    @staticmethod
    def exposure_cube():
        """Exposure cube (`~gammapy.data.SpectralCube`)"""
        from ..data import SpectralCube
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

        base_dir = gammapy_extra.dir / 'datasets/vela_region'

        result = dict()
        result['counts_cube'] = str(base_dir / 'counts_vela.fits')
        result['exposure_cube'] = str(base_dir / 'exposure_vela.fits')
        result['background_image'] = str(base_dir / 'background_vela.fits')
        result['total_image'] = str(base_dir / 'total_vela.fits')
        result['diffuse_model'] = str(base_dir / 'gll_iem_v05_rev1_cutout.fits')
        result['events'] = str(base_dir / 'events_vela.fits')
        result['psf'] = str(base_dir / 'psf_vela.fits')
        result['livetime_cube'] = str(base_dir / 'livetime_vela.fits')
        return result

    @staticmethod
    def counts_cube():
        """Counts cube information (`~astropy.io.fits.HDUList`).

        The HDU list contains:

        * Counts cube `~astropy.io.fits.PrimaryHDU`.
        * Energy bins `~astropy.io.fits.BinTableHDU`.
        * MET bins `~astropy.io.fits.BinTableHDU`.
        """
        filename = FermiVelaRegion.filenames()['counts_cube']
        return fits.open(filename)

    @staticmethod
    def psf():
        """Point spread function (`~gammapy.irf.EnergyDependentTablePSF`)"""
        from ..irf import EnergyDependentTablePSF
        filename = FermiVelaRegion.filenames()['psf']
        return EnergyDependentTablePSF.read(filename)

    @staticmethod
    def diffuse_model():
        """Diffuse model (`~gammapy.data.SpectralCube`)"""
        from ..data import SpectralCube
        filename = FermiVelaRegion.filenames()['diffuse_model']
        return SpectralCube.read(filename)

    @staticmethod
    def background_image():
        """Predicted background counts image (`~astropy.io.fits.PrimaryHDU`).

        Based on the Fermi Diffuse model (see class docstring).
        """
        filename = FermiVelaRegion.filenames()['background_image']
        return fits.open(filename)[0]

    @staticmethod
    def predicted_image():
        """Predicted total counts image (`~astropy.io.fits.PrimaryHDU`).

        Based on the Fermi diffuse model (see class docstring) and
        Vela Point source model.
        """
        filename = FermiVelaRegion.filenames()['total_image']
        return fits.open(filename)[0]

    @staticmethod
    def events():
        """Events list information (`~astropy.io.fits.HDUList`)

        The HDU list contains:

        - ``EVENTS`` table HDU
        - ``GTI`` table HDU
        """
        filename = FermiVelaRegion.filenames()['events']
        return fits.open(filename)

    @staticmethod
    def exposure_cube():
        """Exposure cube (`~gammapy.data.SpectralCube`)."""
        from ..data import SpectralCube
        filename = FermiVelaRegion.filenames()['exposure_cube']
        return SpectralCube.read(filename)

    @staticmethod
    def livetime_cube():
        """Livetime cube (`~astropy.io.fits.HDUList`)."""
        filename = FermiVelaRegion.filenames()['livetime_cube']
        return fits.open(filename)


def load_lat_psf_performance(performance_file):
    """Loads Fermi-LAT TOTAL PSF performance data.

    These points are extracted by hand from:

    * `PSF_P7REP_SOURCE_V15 <http://www.slac.stanford.edu/exp/glast/groups/canda/archive/p7rep_v15/lat_Performance_files/cPsfEnergy_P7REP_SOURCE_V15.png>`_
    * `PSF_P7SOURCEV6 <http://www.slac.stanford.edu/exp/glast/groups/canda/archive/pass7v6/lat_Performance_files/cPsfEnergy_P7SOURCE_V6.png>`_

    As such, a 10% error in the values should be assumed.

    Parameters
    ----------
    performance_file : str
        Specify which PSF performance file to return.

        * ``P7REP_SOURCE_V15_68`` P7REP_SOURCE_V15, 68% containment
        * ``P7REP_SOURCE_V15_95`` P7REP_SOURCE_V15, 95% containment
        * ``P7SOURCEV6_68`` P7SOURCEV6, 68% containment
        * ``P7SOURCEV6_95`` P7SOURCEV6, 95% containment

    Returns
    -------
    table : `~astropy.table.Table`
        Table of psf size (deg) for selected containment radius and IRF at
        energies (MeV).
    """
    filename = gammapy_extra.filename('test_datasets/unbundled/fermi//fermi_irf_data.fits')
    hdus = fits.open(filename)

    perf_files = dict()
    perf_files['P7REP_SOURCE_V15_68'] = hdus[1]
    perf_files['P7REP_SOURCE_V15_95'] = hdus[4]
    perf_files['P7SOURCEV6_68'] = hdus[3]
    perf_files['P7SOURCEV6_95'] = hdus[2]
    hdu = perf_files[performance_file]
    table = Table(hdu.data)
    table.rename_column('col1', 'energy')
    table.rename_column('col2', 'containment_angle')

    table['energy'].unit = 'MeV'
    table['containment_angle'].unit = 'deg'

    return table
