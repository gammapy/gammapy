# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fermi datasets.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import warnings
from astropy.io import fits
from astropy.table import Table
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
]


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
