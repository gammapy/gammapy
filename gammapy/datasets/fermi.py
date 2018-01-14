# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fermi datasets.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import warnings
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file
from astropy.utils import lazyproperty
from .core import gammapy_extra
from ..data import EventList
from ..cube import SkyCube
from ..maps import HpxNDMap
from ..irf import EnergyDependentTablePSF
from ..utils.scripts import make_path
from ..spectrum.models import TableModel
from ..catalog.gammacat import NoDataAvailableError

__all__ = [
    'FermiLATDataset',
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
        result = OrderedDict()
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
        """Diffuse model (`~gammapy.data.SkyCube`)"""
        from ..cube import SkyCube
        filename = FermiGalacticCenter.filenames()['diffuse_model']
        return SkyCube.read(filename, format='fermi-background')

    @staticmethod
    def exposure_cube():
        """Exposure cube (`~gammapy.data.SkyCube`)"""
        from ..cube import SkyCube
        filename = FermiGalacticCenter.filenames()['exposure_cube']
        return SkyCube.read(filename, format='fermi-exposure')


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

        result = OrderedDict()
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
        """Diffuse model (`~gammapy.data.SkyCube`)"""
        from ..cube import SkyCube
        filename = FermiVelaRegion.filenames()['diffuse_model']
        return SkyCube.read(filename, format='fermi-background')

    @staticmethod
    def background_image():
        """Predicted background counts image (`~gammapy.image.SkyImage`).

        Based on the Fermi Diffuse model (see class docstring).
        """
        from ..image import SkyImage
        filename = FermiVelaRegion.filenames()['background_image']
        return SkyImage.read(filename)

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
        """Exposure cube (`~gammapy.data.SkyCube`)."""
        from ..cube import SkyCube
        filename = FermiVelaRegion.filenames()['exposure_cube']
        return SkyCube.read(filename, format='fermi-exposure')

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

    perf_files = OrderedDict()
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


class FermiLATDataset(object):
    """Fermi dataset container class, with lazy data access.

    Parameters
    ----------
    filename : str
        Filename of the yaml file that specifies the data filenames.
    """

    def __init__(self, filename):
        import yaml
        path = make_path(filename)
        self._path = path.parents[0].resolve()
        with path.open() as fh:
            self.config = yaml.load(fh)

    @property
    def name(self):
        """Name of the dataset"""
        return self.config['name']

    def validate(self):
        raise NotImplementedError

    @lazyproperty
    def filenames(self):
        """Absolute path filenames."""
        filenames = OrderedDict()
        filenames_config = self.config['filenames']

        # merge with base path
        for _ in filenames_config:
            filenames[_] = str(self._path / filenames_config[_])

        filenames['galdiff'] = str(make_path('$FERMI_DIFFUSE_DIR/gll_iem_v06.fits'))
        return filenames

    @lazyproperty
    def exposure(self):
        """Exposure cube.

        Returns
        -------
        cube : `~gammapy.cube.SkyCube` or `~gammapy.maps.HpxNDMap`
            Exposure cube.
        """
        filename = self.filenames['exposure']
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cube = SkyCube.read(filename, format='fermi-exposure')
        except (ValueError, KeyError):
            cube = HpxNDMap.read(filename)

        return cube

    @lazyproperty
    def counts(self):
        """Counts cube.

        Returns
        -------
        cube : `~gammapy.cube.SkyCube` or `~gammapy.maps.HpxNDMap`
            Counts cube
        """
        try:
            filename = self.filenames['counts']
        except KeyError:
            raise NoDataAvailableError('Counts cube not available.')

        try:
            cube = SkyCube.read(filename, format='fermi-counts')
        except (ValueError, KeyError):
            cube = HpxNDMap.read(filename)

        return cube

    @lazyproperty
    def background(self):
        """Predicted total background counts (`~gammapy.cube.SkyCube`)."""
        try:
            filename = self.filenames['background']
        except KeyError:
            raise NoDataAvailableError('Predicted background counts cube not available.')

        cube = SkyCube.read(filename, format='fermi-counts')
        cube.name = 'background'
        return cube

    @lazyproperty
    def galactic_diffuse(self):
        """Diffuse galactic model (`~gammapy.cube.SkyCube`)."""
        try:
            filename = self.filenames['galdiff']
            cube = SkyCube.read(filename, format='fermi-background')
            cube.name = 'galactic diffuse'
            return cube
        except IOError:
            raise NoDataAvailableError('Fermi galactic diffuse model cube not available. '
                                       'Please set $FERMI_DIFFUSE_DIR environment variable')

    @lazyproperty
    def isotropic_diffuse(self):
        """Isotropic diffuse background model table.

        Returns
        -------
        spectral_model : `~gammapy.spectrum.models.TableModel`
            Isotropic diffuse background model.
        """
        table = self._read_iso_diffuse_table()

        background_isotropic = TableModel(table['Energy'].quantity,
                                          table['Flux'].quantity)

        return background_isotropic

    def _read_iso_diffuse_table(self):
        filename = self.filenames['isodiff']
        table = Table.read(filename, format='ascii')

        table.rename_column('col1', 'Energy')
        table['Energy'].unit = 'MeV'

        table.rename_column('col2', 'Flux')
        table['Flux'].unit = '1 / (cm2 MeV s sr)'

        table.rename_column('col3', 'Flux_Err')
        table['Flux_Err'].unit = '1 / (cm2 MeV s sr)'
        return table

    @property
    def events(self):
        """Event list (`~gammapy.data.EventList`)."""
        return EventList.read(self.filenames['events'])

    @property
    def psf(self):
        """PSF (`~gammapy.irf.EnergyDependentTablePSF`)."""
        return EnergyDependentTablePSF.read(self.filenames['psf'])

    def info(self):
        """Print summary info about the dataset."""
        print(self)

    def __str__(self):
        """Summary info string about the dataset."""
        info = self.__class__.__name__
        info += '\n\n\tname: {name} \n\n'.format(name=self.name)
        info += 'Filenames:\n\n'
        for name in sorted(self.config['filenames']):
            filename = self.config['filenames'][name]
            info += "\t{name:8s} : {filename}\n".format(name=name, filename=filename)

        return info
