# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from ..irf import EnergyDispersion
from ..maps import Map

__all__ = ["make_edisp_map", "EDispMap"]

class EDispMap(object):
    """Class containing the Map of Energy Dispersions and allowing to interact with it.

    Parameters
    ----------
    edisp_map : `~gammapy.maps.Map`
        the input Energy Dispersion Map. Should be a Map with 2 non spatial axes.
        migra and true energy axes should be given in this specific order.

    Examples
    --------
    ::

        import numpy as np
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from gammapy.maps import Map, WcsGeom, MapAxis
        from gammapy.irf import EnergyDispersion2D
        from gammapy.cube import make_edisp_map, EDispMap

        # Define energy axis. Note that the name is fixed.
        energy_axis = MapAxis.from_edges(np.logspace(-1., 1., 4), unit='TeV', name='energy')
        # Define migra axis. Again note the axis name
        migras = np.linspace(0., 3.0, 100)
        migra_axis = MapAxis.from_edges(migras, unit='', name='migra')

        # Define parameters
        pointing = SkyCoord(0., 0., unit='deg')
        max_offset = 4 * u.deg

        # Create WcsGeom
        geom = WcsGeom.create(binsz=0.25*u.deg, width=10*u.deg, skydir=pointing, axes=[migra_axis, energy_axis])

        # Extract EnergyDependentTablePSF from CTA 1DC IRF
        filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
        edisp2D = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')

        # create the PSFMap for the specified pointing
        edisp_map = make_edisp_map(edisp2D, pointing, geom, max_offset)

        # Get an Energy Dispersion (1D) at any position in the image
        psf_table = edisp_map.get_energy_dispersion(SkyCoord(2., 2.5, unit='deg'))

        # Write map to disk
        edisp_map.write('edisp_map.fits')
    """

    def __init__(self, edisp_map):
        if edisp_map.geom.axes[1].name.upper() != "ENERGY":
            raise ValueError("Incorrect energy axis position in input Map")

        if edisp_map.geom.axes[0].name.upper() != "MIGRA":
            raise ValueError("Incorrect migra axis position in input Map")

        self._edisp_map = edisp_map

    @property
    def edisp_map(self):
        """the EDispMap itself (`~gammapy.maps.Map`)"""
        return self._edisp_map

    @property
    def data(self):
        """the EDispMap data"""
        return self._edisp_map.data

    @property
    def quantity(self):
        """the EDispMap data as a quantity"""
        return self._edisp_map.quantity

    @property
    def geom(self):
        """The EDispMap MapGeom object"""
        return self._edisp_map.geom

    @classmethod
    def read(cls, filename, **kwargs):
        """Read a edisp_map from file and create a EDispMap object"""
        edmap = Map.read(filename, **kwargs)
        return cls(edmap)

    def write(self, *args, **kwargs):
        """Write the Map object containing the EDisp Library map."""
        self._edisp_map.write(*args, **kwargs)

    def get_energy_dispersion(self, position, e_reco, migra_step=5e-3):
        """ Returns EnergyDispersion at a given position

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            the target position. Should be a single coordinates

        Returns
        -------
        psf_table : `~gammapy.irf.EnergyDependentTablePSF`
            the table PSF
        e_reco : `~astropy.units.Quantity`
            Reconstruced energy axis binning
        migra_step : float
            Integration step in migration
        """
        if position.size != 1:
            raise ValueError(
                "EnergyDispersion can be extracted at one single position only."
            )

        # axes ordering fixed. Could be changed.
        pix_ener = np.arange(self.geom.axes[1].nbin)

        # Define a vector of migration with mig_step step
        mrec_min = self.geom.axes[0].edges[0]
        mrec_max = self.geom.axis[0].edges[-1]
        mig_array = np.arange(mrec_min, mrec_max, migra_step)
        pix_migra = (mig_array - mrec_min)/mrec_max * self.geom.axis[0].nbin

        # Convert position to pixels
        pix_lon, pix_lat = self.edisp_map.geom.to_image().coord_to_pix(position)

        # Build the pixels tuple
        pix = np.meshgrid(pix_lon, pix_lat, pix_migra, pix_ener)

        # Interpolate in the PSF map. Squeeze to remove dimensions of length 1
        edisp_values = np.squeeze(
            self.edisp_map.interp_by_pix(pix) * u.Unit(self.edisp_map.unit)
        )

        e_true = self.edisp_map.geom.axes[1].center * self.edisp_map.geom.axes[1].unit

        # We now perform integration over migra
        # The code is adapted from `~gammapy.EnergyDispersion2D.get_response`

        # migration value of e_reco bounds
        migra_e_reco = e_reco / e_true

        # Compute normalized cumulative sum to prepare integration
        tmp = np.nan_to_num(np.cumsum(edisp_values))

        # Determine positions (bin indices) of e_reco bounds in migration array
        pos_mig = np.digitize(migra_e_reco, mig_array) - 1
        # We ensure that no negative values are found
        pos_mig = np.maximum(pos_mig, 0)

        # We compute the difference between 2 successive bounds in e_reco
        # to get integral over reco energy bin
        integral = np.diff(tmp[pos_mig])


        # Beware. Need to revert rad and energies to follow the TablePSF scheme.
        return EnergyDependentTablePSF(energy=energies, rad=rad, psf_value=psf_values.T)
