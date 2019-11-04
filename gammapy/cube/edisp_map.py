# Licensed under a 3-clause BSD style license - see LICENSE.rst
from copy import deepcopy
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from gammapy.irf import EnergyDispersion
from gammapy.maps import Map, MapCoord
from gammapy.utils.random import InverseCDFSampler, get_random_state

__all__ = ["make_edisp_map", "EDispMap"]


def make_edisp_map(edisp, pointing, geom, max_offset, exposure_map=None):
    """Make a edisp map for a single observation

    Expected axes : migra and true energy in this specific order
    The name of the migra MapAxis is expected to be 'migra'

    Parameters
    ----------
    edisp : `~gammapy.irf.EnergyDispersion2D`
        the 2D Energy Dispersion IRF
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
    edispmap : `~gammapy.cube.EDispMap`
        the resulting EDisp map
    """
    energy_axis = geom.get_axis_by_name("energy")
    energy = energy_axis.center

    migra_axis = geom.get_axis_by_name("migra")
    migra = migra_axis.center

    # Compute separations with pointing position
    separations = pointing.separation(geom.to_image().get_coord().skycoord)
    valid = np.where(separations < max_offset)

    # Compute EDisp values
    edisp_values = edisp.data.evaluate(
        offset=separations[valid],
        e_true=energy[:, np.newaxis],
        migra=migra[:, np.newaxis, np.newaxis],
    )

    # Re-order axes to be consistent with expected geometry
    edisp_values = np.transpose(edisp_values, axes=(1, 0, 2))

    # Create Map and fill relevant entries
    edispmap = Map.from_geom(geom, unit="")
    edispmap.data[:, :, valid[0], valid[1]] += edisp_values.to_value(edispmap.unit)

    return EDispMap(edispmap, exposure_map)


class EDispMap:
    """Energy dispersion map.

    Parameters
    ----------
    edisp_map : `~gammapy.maps.Map`
        the input Energy Dispersion Map. Should be a Map with 2 non spatial axes.
        migra and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`, optional
        Associated exposure map. Needs to have a consistent map geometry.

    Examples
    --------
    ::

        import numpy as np
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from gammapy.maps import WcsGeom, MapAxis
        from gammapy.irf import EnergyDispersion2D, EffectiveAreaTable2D
        from gammapy.cube import make_edisp_map, make_map_exposure_true_energy

        # Define energy dispersion map geometry
        energy_axis = MapAxis.from_edges(np.logspace(-1, 1, 4), unit="TeV", name="energy")
        migra_axis = MapAxis.from_edges(np.linspace(0, 3, 100), name="migra")
        pointing = SkyCoord(0, 0, unit="deg")
        max_offset = 4 * u.deg
        geom = WcsGeom.create(
            binsz=0.25 * u.deg,
            width=10 * u.deg,
            skydir=pointing,
            axes=[migra_axis, energy_axis],
        )

        # Extract EnergyDispersion2D from CTA 1DC IRF
        filename = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
        edisp2D = EnergyDispersion2D.read(filename, hdu="ENERGY DISPERSION")
        aeff2d = EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")

        # Create the exposure map
        exposure_geom = geom.to_image().to_cube([energy_axis])
        exposure_map = make_map_exposure_true_energy(pointing, "1 h", aeff2d, exposure_geom)

        # create the EDispMap for the specified pointing
        edisp_map = make_edisp_map(edisp2D, pointing, geom, max_offset, exposure_map)

        # Get an Energy Dispersion (1D) at any position in the image
        pos = SkyCoord(2.0, 2.5, unit="deg")
        e_reco = np.logspace(-1.0, 1.0, 10) * u.TeV
        edisp = edisp_map.get_energy_dispersion(pos=pos, e_reco=e_reco)

        # Write map to disk
        edisp_map.write("edisp_map.fits")
    """

    def __init__(self, edisp_map, exposure_map):
        if edisp_map.geom.axes[1].name.upper() != "ENERGY":
            raise ValueError("Incorrect energy axis position in input Map")

        if edisp_map.geom.axes[0].name.upper() != "MIGRA":
            raise ValueError("Incorrect migra axis position in input Map")

        self.edisp_map = edisp_map

        if exposure_map is not None and exposure_map.geom.ndim == 3:
            energy_axis = edisp_map.geom.get_axis_by_name("energy")
            migra_axis = edisp_map.geom.get_axis_by_name("migra")
            geom_image = exposure_map.geom.to_image()
            geom = geom_image.to_cube([migra_axis.squash(), energy_axis])
            data = exposure_map.data[:, np.newaxis, :, :]
            exposure_map = Map.from_geom(geom=geom, data=data, unit=exposure_map.unit)

        self.exposure_map = exposure_map

    @classmethod
    def from_hdulist(
        cls,
        hdulist,
        edisp_hdu="EDISPMAP",
        edisp_hdubands="BANDSEDISP",
        exposure_hdu="EXPMAP",
        exposure_hdubands="BANDSEXP",
    ):
        """Convert to `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        edisp_hdu : str
            Name or index of the HDU with the edisp_map data.
        edisp_hdubands : str
            Name or index of the HDU with the edisp_map BANDS table.
        exposure_hdu : str
            Name or index of the HDU with the exposure_map data.
        exposure_hdubands : str
            Name or index of the HDU with the exposure_map BANDS table.
        """
        edisp_map = Map.from_hdulist(hdulist, edisp_hdu, edisp_hdubands, "auto")
        if exposure_hdu in hdulist:
            exposure_map = Map.from_hdulist(
                hdulist, exposure_hdu, exposure_hdubands, "auto"
            )
        else:
            exposure_map = None

        return cls(edisp_map, exposure_map)

    @classmethod
    def read(cls, filename, **kwargs):
        """Read an edisp_map from file and create an EDispMap object"""
        with fits.open(filename, memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, **kwargs)

    def to_hdulist(
        self,
        edisp_hdu="EDISPMAP",
        edisp_hdubands="BANDSEDISP",
        exposure_hdu="EXPMAP",
        exposure_hdubands="BANDSEXP",
    ):
        """Convert to `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        edisp_hdu : str
            Name or index of the HDU with the edisp_map data.
        edisp_hdubands : str
            Name or index of the HDU with the edisp_map BANDS table.
        exposure_hdu : str
            Name or index of the HDU with the exposure_map data.
        exposure_hdubands : str
            Name or index of the HDU with the exposure_map BANDS table.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
        """
        hdulist = self.edisp_map.to_hdulist(hdu=edisp_hdu, hdu_bands=edisp_hdubands)
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

    def get_energy_dispersion(self, position, e_reco, migra_step=5e-3):
        """Get energy dispersion at a given position.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            the target position. Should be a single coordinates
        e_reco : `~astropy.units.Quantity`
            Reconstructed energy axis binning
        migra_step : float
            Integration step in migration

        Returns
        -------
        edisp : `~gammapy.irf.EnergyDispersion`
            the energy dispersion (i.e. rmf object)
        """
        # TODO: reduce code duplication with EnergyDispersion2D.get_response
        if position.size != 1:
            raise ValueError(
                "EnergyDispersion can be extracted at one single position only."
            )

        # axes ordering fixed. Could be changed.
        pix_ener = np.arange(self.edisp_map.geom.axes[1].nbin)

        # Define a vector of migration with mig_step step
        mrec_min = self.edisp_map.geom.axes[0].edges[0]
        mrec_max = self.edisp_map.geom.axes[0].edges[-1]
        mig_array = np.arange(mrec_min, mrec_max, migra_step)
        pix_migra = (mig_array - mrec_min) / mrec_max * self.edisp_map.geom.axes[0].nbin

        # Convert position to pixels
        pix_lon, pix_lat = self.edisp_map.geom.to_image().coord_to_pix(position)

        # Build the pixels tuple
        pix = np.meshgrid(pix_lon, pix_lat, pix_migra, pix_ener)
        # Interpolate in the EDisp map. Squeeze to remove dimensions of length 1
        edisp_values = np.squeeze(
            self.edisp_map.interp_by_pix(pix)
            * u.Unit(self.edisp_map.unit)  # * migra_step
        )
        e_trues = self.edisp_map.geom.axes[1].center
        data = []

        for i, e_true in enumerate(e_trues):
            # We now perform integration over migra
            # The code is adapted from `~gammapy.EnergyDispersion2D.get_response`

            # migration value of e_reco bounds
            migra_e_reco = e_reco / e_true

            # Compute normalized cumulative sum to prepare integration
            tmp = np.nan_to_num(
                np.cumsum(edisp_values[:, i]) / np.sum(edisp_values[:, i])
            )

            # Determine positions (bin indices) of e_reco bounds in migration array
            pos_mig = np.digitize(migra_e_reco, mig_array) - 1
            # We ensure that no negative values are found
            pos_mig = np.maximum(pos_mig, 0)

            # We compute the difference between 2 successive bounds in e_reco
            # to get integral over reco energy bin
            integral = np.diff(tmp[pos_mig])

            data.append(integral)

        data = np.asarray(data)
        # EnergyDispersion uses edges of true energy bins
        e_true_edges = self.edisp_map.geom.axes[1].edges

        e_lo, e_hi = e_true_edges[:-1], e_true_edges[1:]
        ereco_lo, ereco_hi = (e_reco[:-1], e_reco[1:])

        return EnergyDispersion(
            e_true_lo=e_lo,
            e_true_hi=e_hi,
            e_reco_lo=ereco_lo,
            e_reco_hi=ereco_hi,
            data=data,
        )

    def stack(self, other):
        """Stack EDispMap with another one.

        Parameters
        ----------
        other : `~gammapy.cube.EDispMap`
            the edispmap to be stacked with this one.

        Returns
        -------
        new : `~gammapy.cube.EDispMap`
            the stacked edispmap
        """
        if self.exposure_map is None or other.exposure_map is None:
            raise ValueError("Missing exposure map for PSFMap.stack")

        geom_image = other.edisp_map.geom.to_image()
        coords = geom_image.get_coord()

        # compute indices in the map to stack in
        idx_x, idx_y = self.edisp_map.geom.to_image().coord_to_idx(coords)
        slice_ = (Ellipsis, idx_y, idx_x)

        self.edisp_map.data[slice_] *= self.exposure_map.data[slice_]
        self.edisp_map.data[slice_] += other.edisp_map.data * other.exposure_map.data
        self.exposure_map.data[slice_] += other.exposure_map.data
        with np.errstate(invalid="ignore"):
            self.edisp_map.data[slice_] /= self.exposure_map.data[slice_]
            self.edisp_map.data = np.nan_to_num(self.edisp_map.data)

    def copy(self):
        """Copy EDispMap"""
        return deepcopy(self)

    @classmethod
    def from_geom(cls, geom):
        """Create edisp map from geom.

        By default a diagonal edisp matrix is created.

        Parameters
        ----------
        geom : `Geom`
            Edisp map geometry.

        Returns
        -------
        edisp_map : `EDispMap`
            Energy dispersion map.
        """
        energy_true_axis = geom.get_axis_by_name("energy")

        geom_exposure_edisp = geom.to_image().to_cube([energy_true_axis])
        exposure_edisp = Map.from_geom(geom_exposure_edisp, unit="m2 s")

        migra_axis = geom.get_axis_by_name("migra")
        edisp_map = Map.from_geom(geom, unit="")
        loc = migra_axis.edges.searchsorted(1.0)
        edisp_map.data[:, loc, :, :] = 1.0
        return cls(edisp_map, exposure_edisp)

    def sample_coord(self, map_coord, random_state=0):
        """Apply the energy dispersion corrections on the coordinates of a set of simulated events.

        Parameters
        ----------
        map_coord : `~gammapy.maps.MapCoord` object.
            Sequence of coordinates and energies of sampled events.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.

        Returns
        -------
        `~gammapy.maps.MapCoord`.
            Sequence of Edisp-corrected coordinates of the input map_coord map.
        """
        random_state = get_random_state(random_state)
        migra_axis = self.edisp_map.geom.get_axis_by_name("migra")

        coord = {
            "skycoord": map_coord.skycoord.reshape(-1, 1),
            "energy": map_coord["energy"].reshape(-1, 1),
            "migra": migra_axis.center,
        }

        pdf_edisp = self.edisp_map.interp_by_coord(coord)

        sample_edisp = InverseCDFSampler(pdf_edisp, axis=1, random_state=random_state)
        pix_edisp = sample_edisp.sample_axis()
        energy_reco = migra_axis.pix_to_coord(pix_edisp)

        return MapCoord.create({"skycoord": map_coord.skycoord, "energy": energy_reco})
