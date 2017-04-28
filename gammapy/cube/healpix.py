# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from ..image.healpix import SkyImageHealpix, WCSHealpix
from ..spectrum.utils import LogEnergyAxis
from ..utils.energy import EnergyBounds
from ..utils.scripts import make_path
from .core import SkyCube

__all__ = [
    'SkyCubeHealpix',
]


class SkyCubeHealpix(object):
    """
    Sky cube object with HEALPIX pixelisation.

    First axis is energy axis.
    """

    def __init__(self, name=None, data=None, wcs=None, energy_axis=None, meta=None):
        # TODO: check validity of inputs
        self.name = name
        self.data = data
        self.wcs = wcs
        self.meta = meta
        self.energy_axis = energy_axis

    @classmethod
    def read(cls, filename, format='fermi-exposure'):
        """Read sky cube healpix from FITS file.

        Parameters
        ----------
        filename : str
            File name
        format : {'fermi-exposure'}
            Fits file format.

        Returns
        -------
        sky_cube_healpix : `SkyCubeHealpix`
            Sky cube
        """
        filename = make_path(filename)
        hdulist = fits.open(str(filename))

        if format == 'fermi-exposure':
            energy = Table.read(hdulist['ENERGIES'])['Energy']
            energy_axis = LogEnergyAxis(u.Quantity(energy, 'MeV'), mode='center')
            data = hdulist['HPXEXPOSURES'].data
            data = np.vstack([data[energy] for energy in data.columns.names])
            data = u.Quantity(data, 'cm2 s')
            name = 'exposure'
            header = hdulist['HPXEXPOSURES'].header
            nside = header['NSIDE']
            scheme = header.get('ORDERING', 'ring').lower()
            wcs = WCSHealpix(nside, scheme=scheme)
        elif format == 'fermi-counts':
            energy = EnergyBounds.from_ebounds(hdulist['EBOUNDS'])
            energy_axis = LogEnergyAxis(energy, mode='edges')
            data = hdulist['SKYMAP'].data
            data = np.vstack([data[energy] for energy in data.columns.names])
            data = u.Quantity(data, 'count')
            name = 'counts'
            header = hdulist['SKYMAP'].header
            nside = header['NSIDE']
            scheme = header.get('ORDERING', 'ring').lower()
            wcs = WCSHealpix(nside, scheme=scheme)
        else:
            raise ValueError('Not a valid healpix cube fits format')

        return cls(name=name, data=data, wcs=wcs, energy_axis=energy_axis)

    def sky_image_healpix(self, energy):
        """
        Slice a 2-dim `~gammapy.image.healpix.SkyImageHealpix` from the cube
        at a given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy value
        """
        z = self.energy_axis.wcs_world2pix(energy)
        data = self.data[int(np.rint(z))].copy()
        wcs = self.wcs
        return SkyImageHealpix(name=self.name, data=data, wcs=wcs)

    def reproject(self, reference, **kwargs):
        """Spatially reprojects a `SkyCube` onto a reference.

        Parameters
        ----------
        reference : `~astropy.io.fits.Header`, `SkyImage` or `SkyCube`
            Reference wcs specification to reproject the data on.
        **kwargs : dict
            Keyword arguments passed to `~reproject.reproject_from_healpix`

        Returns
        -------
        reprojected_cube : `SkyCube`
            Cube spatially reprojected to the reference.
        """
        if isinstance(reference, SkyCube):
            reference = reference.sky_image_ref

        out = []
        for energy in self.energies():
            image_hpx = self.sky_image_healpix(energy)
            image_out = image_hpx.reproject(reference, **kwargs)
            out.append(image_out.data)

        data = u.Quantity(np.stack(out, axis=0), self.data.unit)
        wcs = image_out.wcs.copy()

        return SkyCube(name=self.name, data=data, wcs=wcs, meta=self.meta,
                       energy_axis=self.energy_axis)

    def energies(self, mode='center'):
        """
        Energy coordinate vector.

        Parameters
        ----------
        mode : {'center', 'edges'}
            Return coordinate values at the pixels edges or pixel centers.

        Returns
        -------
        coordinates : `~astropy.units.Quantity`
            Energy
        """
        if mode == 'center':
            z = np.arange(self.data.shape[0])
        elif mode == 'edges':
            z = np.arange(self.data.shape[0] + 1) - 0.5
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

        return self.energy_axis.wcs_pix2world(z)

    def __str__(self):
        # Copied from `spectral-cube` package
        info = "Healpix sky cube {} with shape={}".format(self.name, self.data.shape)
        if self.data.unit is u.dimensionless_unscaled:
            info += ":\n"
        else:
            info += " and unit={}:\n".format(self.data.unit)

        info += " n_pix:    {:5d}  coord_type:    {:15s}  coord_unit:    {}\n".format(
            self.data.shape[1], self.wcs.coordsys, 'deg')
        info += " n_energy: {:5d}  unit_energy: {}\n".format(
            self.data.shape[0], self.energy_axis._eunit)

        return info

    def info(self):
        """
        Print summary info about the cube.
        """
        print(str(self))
