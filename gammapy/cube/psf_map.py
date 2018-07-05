# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from gammapy.irf import EnergyDependentTablePSF, PSF3D
from gammapy.maps import WcsGeom, WcsNDMap, MapAxis, Map

__all__ = [
    'make_psf_map',
    'PSFMap'
]


def make_psf_map(psf, pointing, ref_geom, max_offset):
    """Make a psf map for a single observation

    Expected axes : rad and true energy in this specific order

    Parameters
    ----------
    psf : `~gammapy.irf.PSF3D`
        the PSF IRF
    pointing : `~astropy.coordinates.SkyCoord`
        the pointing direction
    ref_geom : `~gammapy.maps.MapGeom`
        the map geom to be used. It provides the target geometry.
        rad and true energy axes should be given in this specific order.
    max_offset : `~astropy.coordinates.Angle`
        maximum offset w.r.t. fov center

    Returns
    -------
    psfmap : `~gammapy.maps.Map`
        the resulting PSF map
    """
    # retrieve energies. This should be properly tested
    energy = ref_geom.axes[1].center * ref_geom.axes[1].unit

    # retrieve rad from psf IRF directly (this might be given as an argument instead)
    rad = ref_geom.axes[0].center * ref_geom.axes[0].unit

    # Compute separations with pointing position
    separations = pointing.separation(ref_geom.to_image().get_coord().skycoord)
    valid = np.where(separations < max_offset)

    # Compute PSF values
    psf_values = psf.evaluate(offset=separations[valid], energy=energy, rad=rad)
    # Re-order axes to be consistent with expected geometry
    psf_values = np.transpose(psf_values, axes=(2, 0, 1))
    # Create Map and fill relevant entries
    psfmap = Map.from_geom(ref_geom, unit='sr-1')
    psfmap.data[:, :, valid[0], valid[1]] += psf_values.to(psfmap.unit).value
    return psfmap


class PSFMap(object):
    """Class containing the Map of PSFs and allowing to interact with it.

    Parameters
    ----------
    psf_map : `~gammapy.maps.Map`
        the input PSF Map. Should be a Map with 2 non spatial axes.
        rad and true energy axes should be given in this specific order.

    """
    def __init__(self, psf_map):
        # Check the presence of an energy axis
        if psf_map.geom.axes[1].type is not 'energy':
            raise(ValueError,"Incorrect energy axis position in input Map")

        if not u.Unit(psf_map.geom.axes[0].unit).is_equivalent('deg'):
            raise(ValueError,"Incorrect rad axis position in input Map")

        self._psf_map = psf_map

    @property
    def psf_map(self):
        """the PSFMap itself (~gammapy.maps.Map)"""
        return self._psf_map

    @classmethod
    def read(cls, filename, **kwargs):
        """ Read a psf_map from file and create a PSFMap object"""

        psfmap = Map.read(filename, **kwargs)
        return cls(psfmap)

    def write(self, *args, **kwargs):
        """Write the Map object containing the PSF Library map."""
        self.psf_map.write(*args, **kwargs)

    def get_energy_dependent_table_psf(self, position):
        """ Returns EnergyDependentTable PSF at a given position

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            the target position. Should be a single coordinates

        Returns
        -------
        psf_table : `~gammapy.irf.EnergyDependentTablePSF`
            the table PSF
        """
        if position.size != 1:
            raise ValueError("EnergyDependentTablePSF can be extracted at one single position only.")

        # axes ordering fixed. Could be changed.
        pix_ener = np.arange(self.psf_map.geom.axes[1].nbin)
        pix_rad = np.arange(self.psf_map.geom.axes[0].nbin)

        # Convert position to pixels
        pix_lon, pix_lat = self.psf_map.geom.to_image().coord_to_pix(position)

        # Build the pixels tuple
        pix = np.meshgrid(pix_lon, pix_lat,pix_rad,pix_ener)

        # Interpolate in the PSF map. Squeeze to remove dimensions of length 1
        psf_values = np.squeeze(self.psf_map.interp_by_pix(pix)*u.Unit(self.psf_map.unit))

        energies = self.psf_map.geom.axes[1].center * self.psf_map.geom.axes[1].unit
        rad = self.psf_map.geom.axes[0].center * self.psf_map.geom.axes[0].unit

        # Beware. Need to revert rad and energies to follow the TablePSF scheme.
        return EnergyDependentTablePSF(energy=energies,rad=rad,psf_value=psf_values.T)

    def get_psf_kernel(self, position, geom, max_radius=None, factor=4):
        """Returns a PSF kernel at the given position.

        The PSF is returned in the form a WcsNDMap defined by the input MapGeom.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            the target position. Should be a single coordinate
        geom : `~gammapy.maps.MapGeom`
            the target geometry to use
        max_radius: `~astropy.coordinates.Angle`
            maximum angular size of the kernel map
        factor : int
            oversampling factor to compute the PSF

        Returns
        -------
        kernel : `~gammapy.cube.PSFKernel`
            the resulting kernel
        """
        table_psf = self.get_energy_dependent_table_psf(position)
        return PSFKernel.from_table_psf(table_psf, geom, factor)

    def containment_radius_map(self, fraction = 0.68):
        """Returns the containement radius map.

        Parameters
        ----------
        fraction : float
            the containment fraction (a positive number <=1). Default 0.68.

        Returns
        -------
        containment_radius_map : `~gammapy.maps.Map`
            a 3D map giving the containment radius at each energy and each position of the initial psf_map
        """

        raise NotImplementedError
