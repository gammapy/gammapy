# Licensed under a 3-clause BSD style license - see LICENSE.rst
from copy import deepcopy
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from astropy.coordinates import Angle
from gammapy.irf import EnergyDependentTablePSF
from gammapy.maps import Map, MapCoord
from gammapy.utils.random import InverseCDFSampler, get_random_state
from .psf_kernel import PSFKernel

__all__ = ["make_psf_map", "PSFMap"]


def make_psf_map(psf, pointing, geom, max_offset, exposure_map=None):
    """Make a psf map for a single observation

    Expected axes : rad and true energy in this specific order
    The name of the rad MapAxis is expected to be 'rad'

    Parameters
    ----------
    psf : `~gammapy.irf.PSF3D`
        the PSF IRF
    pointing : `~astropy.coordinates.SkyCoord`
        the pointing direction
    geom : `~gammapy.maps.Geom`
        the map geom to be used. It provides the target geometry.
        rad and true energy axes should be given in this specific order.
    max_offset : `~astropy.coordinates.Angle`
        maximum offset w.r.t. fov center
    exposure_map : `~gammapy.maps.Map`, optional
        the associated exposure map.
        default is None

    Returns
    -------
    psfmap : `~gammapy.cube.PSFMap`
        the resulting PSF map
    """
    energy_axis = geom.get_axis_by_name("energy")
    energy = energy_axis.center

    rad_axis = geom.get_axis_by_name("theta")
    rad = Angle(rad_axis.center, unit=rad_axis.unit)

    # Compute separations with pointing position
    separations = pointing.separation(geom.to_image().get_coord().skycoord)
    valid = np.where(separations < max_offset)

    # Compute PSF values
    psf_values = psf.evaluate(offset=separations[valid], energy=energy, rad=rad)

    # Re-order axes to be consistent with expected geometry
    psf_values = np.transpose(psf_values, axes=(2, 0, 1))

    # TODO: this probably does not ensure that probability is properly normalized in the PSFMap
    # Create Map and fill relevant entries
    psfmap = Map.from_geom(geom, unit="sr-1")
    psfmap.data[:, :, valid[0], valid[1]] += psf_values.to_value(psfmap.unit)

    return PSFMap(psfmap, exposure_map)


class PSFMap:
    """Class containing the Map of PSFs and allowing to interact with it.

    Parameters
    ----------
    psf_map : `~gammapy.maps.Map`
        the input PSF Map. Should be a Map with 2 non spatial axes.
        rad and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`
        Associated exposure map. Needs to have a consistent map geometry.

    Examples
    --------
    ::

        import numpy as np
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from gammapy.maps import Map, WcsGeom, MapAxis
        from gammapy.irf import EnergyDependentMultiGaussPSF, EffectiveAreaTable2D
        from gammapy.cube import make_psf_map, PSFMap, make_map_exposure_true_energy

        # Define energy axis. Note that the name is fixed.
        energy_axis = MapAxis.from_edges(np.logspace(-1., 1., 4), unit='TeV', name='energy')
        # Define rad axis. Again note the axis name
        rads = np.linspace(0., 0.5, 100) * u.deg
        rad_axis = MapAxis.from_edges(rads, unit='deg', name='theta')

        # Define parameters
        pointing = SkyCoord(0., 0., unit='deg')
        max_offset = 4 * u.deg

        # Create WcsGeom
        geom = WcsGeom.create(binsz=0.25*u.deg, width=10*u.deg, skydir=pointing, axes=[rad_axis, energy_axis])

        # Extract EnergyDependentTablePSF from CTA 1DC IRF
        filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
        psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
        psf3d = psf.to_psf3d(rads)
        aeff2d = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')

        # Create the exposure map
        exposure_geom = geom.to_image().to_cube([energy_axis])
        exposure_map = make_map_exposure_true_energy(pointing, "1 h", aeff2d, exposure_geom)

        # create the PSFMap for the specified pointing
        psf_map = make_psf_map(psf3d, pointing, geom, max_offset, exposure_map)

        # Get an EnergyDependentTablePSF at any position in the image
        psf_table = psf_map.get_energy_dependent_table_psf(SkyCoord(2., 2.5, unit='deg'))

        # Write map to disk
        psf_map.write('psf_map.fits')
    """

    def __init__(self, psf_map, exposure_map=None):
        if psf_map.geom.axes[1].name.upper() != "ENERGY":
            raise ValueError("Incorrect energy axis position in input Map")

        if psf_map.geom.axes[0].name.upper() != "THETA":
            raise ValueError("Incorrect theta axis position in input Map")

        self.psf_map = psf_map

        if exposure_map is not None and exposure_map.geom.ndim == 3:
            energy_axis = psf_map.geom.get_axis_by_name("energy")
            rad_axis = psf_map.geom.get_axis_by_name("theta")
            geom_image = exposure_map.geom.to_image()
            geom = geom_image.to_cube([rad_axis.squash(), energy_axis])
            data = exposure_map.data[:, np.newaxis, :, :]
            exposure_map = Map.from_geom(geom=geom, data=data, unit=exposure_map.unit)

        self.exposure_map = exposure_map

    @classmethod
    def from_hdulist(
        cls,
        hdulist,
        psf_hdu="PSFMAP",
        psf_hdubands="BANDSPSF",
        exposure_hdu="EXPMAP",
        exposure_hdubands="BANDSEXP",
    ):
        """Convert to `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        psf_hdu : str
            Name or index of the HDU with the psf_map data.
        psf_hdubands : str
            Name or index of the HDU with the psf_map BANDS table.
        exposure_hdu : str
            Name or index of the HDU with the exposure_map data.
        exposure_hdubands : str
            Name or index of the HDU with the exposure_map BANDS table.
        """
        psf_map = Map.from_hdulist(hdulist, psf_hdu, psf_hdubands, "auto")
        if exposure_hdu in hdulist:
            exposure_map = Map.from_hdulist(
                hdulist, exposure_hdu, exposure_hdubands, "auto"
            )
        else:
            exposure_map = None

        return cls(psf_map, exposure_map)

    @classmethod
    def read(cls, filename, **kwargs):
        """Read a psf_map from file and create a PSFMap object"""
        with fits.open(filename, memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, **kwargs)

    def to_hdulist(
        self,
        psf_hdu="PSFMAP",
        psf_hdubands="BANDSPSF",
        exposure_hdu="EXPMAP",
        exposure_hdubands="BANDSEXP",
    ):
        """Convert to `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        psf_hdu : str
            Name or index of the HDU with the psf_map data.
        psf_hdubands : str
            Name or index of the HDU with the psf_map BANDS table.
        exposure_hdu : str
            Name or index of the HDU with the exposure_map data.
        exposure_hdubands : str
            Name or index of the HDU with the exposure_map BANDS table.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
        """
        hdulist = self.psf_map.to_hdulist(hdu=psf_hdu, hdu_bands=psf_hdubands)
        if self.exposure_map is not None:
            new_hdulist = self.exposure_map.to_hdulist(
                hdu=exposure_hdu, hdu_bands=exposure_hdubands
            )
            hdulist.extend(new_hdulist[1:])
        return hdulist

    def write(self, filename, overwrite=False, **kwargs):
        """Write to fits"""
        hdulist = self.to_hdulist(**kwargs)
        hdulist.writeto(filename, overwrite=overwrite)

    def get_energy_dependent_table_psf(self, position):
        """Get energy-dependent PSF at a given position.

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
            raise ValueError(
                "EnergyDependentTablePSF can be extracted at one single position only."
            )

        # axes ordering fixed. Could be changed.
        pix_ener = np.arange(self.psf_map.geom.axes[1].nbin)
        pix_rad = np.arange(self.psf_map.geom.axes[0].nbin)

        # Convert position to pixels
        pix_lon, pix_lat = self.psf_map.geom.to_image().coord_to_pix(position)

        # Build the pixels tuple
        pix = np.meshgrid(pix_lon, pix_lat, pix_rad, pix_ener)

        # Interpolate in the PSF map. Squeeze to remove dimensions of length 1
        psf_values = np.squeeze(
            self.psf_map.interp_by_pix(pix) * u.Unit(self.psf_map.unit)
        )

        energies = self.psf_map.geom.axes[1].center
        rad = self.psf_map.geom.axes[0].center

        # Beware. Need to revert rad and energies to follow the TablePSF scheme.
        return EnergyDependentTablePSF(energy=energies, rad=rad, psf_value=psf_values.T)

    def get_psf_kernel(self, position, geom, max_radius=None, factor=4):
        """Returns a PSF kernel at the given position.

        The PSF is returned in the form a WcsNDMap defined by the input Geom.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            the target position. Should be a single coordinate
        geom : `~gammapy.maps.Geom`
            the target geometry to use
        max_radius : `~astropy.coordinates.Angle`
            maximum angular size of the kernel map
        factor : int
            oversampling factor to compute the PSF

        Returns
        -------
        kernel : `~gammapy.cube.PSFKernel`
            the resulting kernel
        """
        table_psf = self.get_energy_dependent_table_psf(position)
        return PSFKernel.from_table_psf(table_psf, geom, max_radius, factor)

    def containment_radius_map(self, energy, fraction=0.68):
        """Containment radius map.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Scalar energy at which to compute the containment radius
        fraction : float
            the containment fraction (range: 0 to 1)

        Returns
        -------
        containment_radius_map : `~gammapy.maps.Map`
            Containment radius map
        """
        coords = self.psf_map.geom.to_image().get_coord().skycoord.flatten()
        m = Map.from_geom(self.psf_map.geom.to_image(), unit="deg")

        for coord in coords:
            psf_table = self.get_energy_dependent_table_psf(coord)
            containment_radius = psf_table.containment_radius(energy, fraction)
            m.fill_by_coord(coord, containment_radius)

        return m

    def stack(self, other):
        """Stack PSFMap with another one.

        Parameters
        ----------
        other : `~gammapy.cube.PSFMap`
            the psfmap to be stacked with this one.

        Returns
        -------
        new : `~gammapy.cube.PSFMap`
            the stacked psfmap
        """
        if self.exposure_map is None or other.exposure_map is None:
            raise ValueError("Missing exposure map for PSFMap.stack")

        geom_image = other.psf_map.geom.to_image()
        coords = geom_image.get_coord()

        # compute indices in the map to stack in
        idx_x, idx_y = self.psf_map.geom.to_image().coord_to_idx(coords)
        slice_ = (Ellipsis, idx_y, idx_x)

        self.psf_map.data[slice_] *= self.exposure_map.data[slice_]
        self.psf_map.data[slice_] += other.psf_map.data * other.exposure_map.data
        self.exposure_map.data[slice_] += other.exposure_map.data
        with np.errstate(invalid="ignore"):
            self.psf_map.data[slice_] /= self.exposure_map.data[slice_]
            self.psf_map.data = np.nan_to_num(self.psf_map.data)

    def copy(self):
        """Copy PSFMap"""
        return deepcopy(self)

    @classmethod
    def from_geom(cls, geom):
        """Create psf map from geom.

        Parameters
        ----------
        geom : `Geom`
            PSF map geometry.

        Returns
        -------
        psf_map : `PSFMap`
            Point spread function map.
        """
        energy_true_axis = geom.get_axis_by_name("energy")
        geom_exposure_psf = geom.to_image().to_cube([energy_true_axis])
        exposure_psf = Map.from_geom(geom_exposure_psf, unit="m2 s")
        psf_map = Map.from_geom(geom, unit="sr-1")
        return cls(psf_map, exposure_psf)

    def sample_coord(self, map_coord, random_state=0):
        """Apply PSF corrections on the coordinates of a set of simulated events.

        Parameters
        ----------
        map_coord : `~gammapy.maps.MapCoord` object.
            Sequence of coordinates and energies of sampled events.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.

        Returns
        -------
        corr_coord : `~gammapy.maps.MapCoord` object.
            Sequence of PSF-corrected coordinates of the input map_coord map.
        """

        random_state = get_random_state(random_state)
        rad_axis = self.psf_map.geom.get_axis_by_name("theta")

        coord = {
            "skycoord": map_coord.skycoord.reshape(-1, 1),
            "energy": map_coord["energy"].reshape(-1, 1),
            "theta": rad_axis.center,
        }

        pdf = self.psf_map.interp_by_coord(coord)

        sample_pdf = InverseCDFSampler(pdf, axis=1, random_state=random_state)
        pix_coord = sample_pdf.sample_axis()
        separation = rad_axis.pix_to_coord(pix_coord)

        position_angle = random_state.uniform(360, size=len(map_coord.lon)) * u.deg

        event_positions = map_coord.skycoord.directional_offset_by(
            position_angle=position_angle, separation=separation
        )
        return MapCoord.create(
            {"skycoord": event_positions, "energy": map_coord["energy"]}
        )
