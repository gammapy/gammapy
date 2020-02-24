# Licensed under a 3-clause BSD style license - see LICENSE.rst
from copy import deepcopy
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from gammapy.maps import Map, MapAxis, MapCoord, WcsGeom
from gammapy.utils.random import InverseCDFSampler, get_random_state
from gammapy.modeling.models import PowerLawSpectralModel
from .psf_kernel import PSFKernel
from .psf_table import EnergyDependentTablePSF


__all__ = ["PSFMap"]


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
        if psf_map.geom.axes[1].name.upper() != "ENERGY_TRUE":
            raise ValueError("Incorrect energy axis position in input Map")

        if psf_map.geom.axes[0].name.upper() != "THETA":
            raise ValueError("Incorrect theta axis position in input Map")

        self.psf_map = psf_map
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

        energy = self.psf_map.geom.get_axis_by_name("energy_true").center
        rad = self.psf_map.geom.get_axis_by_name("theta").center

        coords = {
            "skycoord": position,
            "energy_true": energy.reshape((-1, 1, 1, 1)),
            "theta": rad.reshape((1, -1, 1, 1)),
        }

        data = self.psf_map.interp_by_coord(coords)
        psf_values = u.Quantity(data[:, :, 0, 0], unit=self.psf_map.unit, copy=False)

        if self.exposure_map is not None:
            coords = {
                "skycoord": position,
                "energy_true": energy.reshape((-1, 1, 1)),
                "theta": 0 * u.deg,
            }
            data = self.exposure_map.interp_by_coord(coords).squeeze()
            exposure = data * self.exposure_map.unit
        else:
            exposure = None

        # Beware. Need to revert rad and energies to follow the TablePSF scheme.
        return EnergyDependentTablePSF(
            energy=energy, rad=rad, psf_value=psf_values, exposure=exposure
        )

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
        if max_radius is None:
            max_radius = np.max(table_psf.rad)
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

    def stack(self, other, weights=None):
        """Stack PSFMap with another one in place.

        Parameters
        ----------
        other : `~gammapy.cube.PSFMap`
            the psfmap to be stacked with this one.

        """
        if self.exposure_map is None or other.exposure_map is None:
            raise ValueError("Missing exposure map for PSFMap.stack")

        cutout_info = other.psf_map.geom.cutout_info

        if cutout_info is not None:
            slices = cutout_info["parent-slices"]
            parent_slices = Ellipsis, slices[0], slices[1]
        else:
            parent_slices = None

        self.psf_map.data[parent_slices] *= self.exposure_map.data[parent_slices]
        self.psf_map.stack(other.psf_map * other.exposure_map.data, weights=weights)

        # stack exposure map
        self.exposure_map.stack(other.exposure_map, weights=weights)

        with np.errstate(invalid="ignore"):
            self.psf_map.data[parent_slices] /= self.exposure_map.data[parent_slices]
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
        geom_exposure_psf = geom.squash(axis="theta")
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
            "energy_true": map_coord["energy_true"].reshape(-1, 1),
            "theta": rad_axis.center,
        }

        pdf = (
            self.psf_map.interp_by_coord(coord)
            * rad_axis.center.value
            * rad_axis.bin_width.value
        )

        sample_pdf = InverseCDFSampler(pdf, axis=1, random_state=random_state)
        pix_coord = sample_pdf.sample_axis()
        separation = rad_axis.pix_to_coord(pix_coord)

        position_angle = random_state.uniform(360, size=len(map_coord.lon)) * u.deg

        event_positions = map_coord.skycoord.directional_offset_by(
            position_angle=position_angle, separation=separation
        )
        return MapCoord.create(
            {"skycoord": event_positions, "energy": map_coord["energy_true"]}
        )

    @classmethod
    def from_energy_dependent_table_psf(cls, table_psf):
        """Create PSF map from table PSF object.

        Helper function to create an allsky PSF map from
        table PSF, which does not depend on position.

        Parameters
        ----------
        table_psf : `EnergyDependentTablePSF`
            Table PSF

        Returns
        -------
        psf_map : `PSFMap`
            Point spread function map.
        """
        energy_axis = MapAxis.from_nodes(table_psf.energy, name="energy_true", interp="log")
        rad_axis = MapAxis.from_nodes(table_psf.rad, name="theta")

        geom = WcsGeom.create(
            npix=(2, 1), proj="CAR", binsz=180, axes=[rad_axis, energy_axis]
        )
        coords = geom.get_coord()

        # TODO: support broadcasting in .evaluate()
        data = table_psf._interpolate((coords["energy_true"], coords["theta"])).to_value(
            "sr-1"
        )
        psf_map = Map.from_geom(geom, data=data, unit="sr-1")

        geom_exposure = geom.squash(axis="theta")

        data = table_psf.exposure.reshape((-1, 1, 1, 1))

        exposure_map = Map.from_geom(geom_exposure, unit="cm2 s")
        exposure_map.quantity += data
        return cls(psf_map=psf_map, exposure_map=exposure_map)

    def cutout(self, position, width, mode="trim"):
        """Cutout map psf map.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Center position of the cutout region.
        width : tuple of `~astropy.coordinates.Angle`
            Angular sizes of the region in (lon, lat) in that specific order.
            If only one value is passed, a square region is extracted.
        mode : {'trim', 'partial', 'strict'}
            Mode option for Cutout2D, for details see `~astropy.nddata.utils.Cutout2D`.

        Returns
        -------
        cutout : `PSFMap`
            Cutout psf map.
        """
        psf_map = self.psf_map.cutout(position, width, mode)
        exposure_map = self.exposure_map.cutout(position, width, mode)
        return self.__class__(psf_map=psf_map, exposure_map=exposure_map)

    def to_image(self, spectrum=None, keepdims=True):
        """Reduce to a 2-D map after weighing
        with the associated exposure and a spectrum

        Parameters
        ----------
        spectrum : `~gammapy.modeling.models.SpectralModel`, optional
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.
        keepdims : bool, optional
            If True, the energy axis is kept with one bin.
            If False, the axis is removed


        Returns
        -------
        psf_out : `PSFMap`
            `PSFMap` with the energy axis summed over
        """
        from gammapy.cube.exposure import _map_spectrum_weight

        if spectrum is None:
            spectrum = PowerLawSpectralModel(index=2.0)

        exp_weighed = _map_spectrum_weight(self.exposure_map, spectrum)
        exposure = exp_weighed.sum_over_axes(axes=["energy_true"], keepdims=keepdims)

        psf_data = exp_weighed.data * self.psf_map.data / exposure.data
        psf_map = Map.from_geom(geom=self.psf_map.geom, data=psf_data, unit="sr-1")

        psf = psf_map.sum_over_axes(axes=["energy_true"], keepdims=keepdims)
        return self.__class__(psf_map=psf, exposure_map=exposure)
