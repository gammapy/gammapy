# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from gammapy.irf import EnergyDependentMultiGaussPSF, EDispMap, PSFMap
from gammapy.maps import Map
from .background import make_map_background_irf
from .exposure import make_map_exposure_true_energy

__all__ = ["MapDatasetMaker"]

log = logging.getLogger(__name__)


def make_psf_map(psf, pointing, geom, exposure_map=None):
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
    exposure_map : `~gammapy.maps.Map`, optional
        the associated exposure map.
        default is None

    Returns
    -------
    psfmap : `~gammapy.cube.PSFMap`
        the resulting PSF map
    """
    energy_axis = geom.get_axis_by_name("energy_true")
    energy = energy_axis.center

    rad_axis = geom.get_axis_by_name("theta")
    rad = rad_axis.center

    # Compute separations with pointing position
    offset = geom.separation(pointing)

    # Compute PSF values
    # TODO: allow broadcasting in PSF3D.evaluate()
    psf_values = psf._interpolate(
        (
            rad[:, np.newaxis, np.newaxis],
            offset,
            energy[:, np.newaxis, np.newaxis, np.newaxis],
        )
    )

    # TODO: this probably does not ensure that probability is properly normalized in the PSFMap
    # Create Map and fill relevant entries
    data = psf_values.to_value("sr-1")
    psfmap = Map.from_geom(geom, data=data, unit="sr-1")
    return PSFMap(psfmap, exposure_map)


def make_edisp_map(edisp, pointing, geom, exposure_map=None):
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
    exposure_map : `~gammapy.maps.Map`, optional
        the associated exposure map.
        default is None

    Returns
    -------
    edispmap : `~gammapy.cube.EDispMap`
        the resulting EDisp map
    """
    energy_axis = geom.get_axis_by_name("energy_true")
    energy = energy_axis.center

    migra_axis = geom.get_axis_by_name("migra")
    migra = migra_axis.center

    # Compute separations with pointing position
    offset = geom.separation(pointing)

    # Compute EDisp values
    edisp_values = edisp.data.evaluate(
        offset=offset,
        energy_true=energy[:, np.newaxis, np.newaxis, np.newaxis],
        migra=migra[:, np.newaxis, np.newaxis],
    )

    # Create Map and fill relevant entries
    data = edisp_values.to_value("")
    edispmap = Map.from_geom(geom, data=data, unit="")
    return EDispMap(edispmap, exposure_map)



class MapDatasetMaker:
    """Make maps for a single IACT observation.

    Parameters
    ----------
    background_oversampling : int
        Background evaluation oversampling factor in energy.
    selection : list
        List of str, selecting which maps to make.
        Available: 'counts', 'exposure', 'background', 'psf', 'edisp'
        By default, all maps are made.
    """

    available_selection = ["counts", "exposure", "background", "psf", "edisp"]

    def __init__(self, background_oversampling=None, selection=None):
        self.background_oversampling = background_oversampling

        if selection is None:
            selection = self.available_selection

        selection = set(selection)

        if not selection.issubset(self.available_selection):
            difference = selection.difference(self.available_selection)
            raise ValueError(f"{difference} is not a valid method.")

        self.selection = selection

    @staticmethod
    def make_counts(geom, observation):
        """Make counts map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference map geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        counts : `~gammapy.maps.Map`
            Counts map.
        """
        counts = Map.from_geom(geom)
        counts.fill_events(observation.events)
        return counts

    @staticmethod
    def make_exposure(geom, observation):
        """Make exposure map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference map geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        exposure : `~gammapy.maps.Map`
            Exposure map.
        """
        return make_map_exposure_true_energy(
            pointing=observation.pointing_radec,
            livetime=observation.observation_live_time_duration,
            aeff=observation.aeff,
            geom=geom,
        )

    @staticmethod
    def make_exposure_irf(geom, observation):
        """Make exposure map with irf geometry.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        exposure : `~gammapy.maps.Map`
            Exposure map.
        """
        return make_map_exposure_true_energy(
            pointing=observation.pointing_radec,
            livetime=observation.observation_live_time_duration,
            aeff=observation.aeff,
            geom=geom,
        )

    def make_background(self, geom, observation):
        """Make background map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        background : `~gammapy.maps.Map`
            Background map.
        """
        bkg_coordsys = observation.bkg.meta.get("FOVALIGN", "RADEC")

        if bkg_coordsys == "ALTAZ":
            pointing = observation.fixed_pointing_info
        elif bkg_coordsys == "RADEC":
            pointing = observation.pointing_radec
        else:
            raise ValueError(
                f"Invalid background coordinate system: {bkg_coordsys!r}\n"
                "Options: ALTAZ, RADEC"
            )

        return make_map_background_irf(
            pointing=pointing,
            ontime=observation.observation_time_duration,
            bkg=observation.bkg,
            geom=geom,
            oversampling=self.background_oversampling,
        )

    def make_edisp(self, geom, observation):
        """Make energy dispersion map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        edisp : `~gammapy.cube.EDispMap`
            Edisp map.
        """
        exposure = self.make_exposure_irf(geom.squash(axis="migra"), observation)

        return make_edisp_map(
            edisp=observation.edisp,
            pointing=observation.pointing_radec,
            geom=geom,
            exposure_map=exposure,
        )

    def make_psf(self, geom, observation):
        """Make psf map.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            Reference geom.
        observation : `~gammapy.data.Observation`
            Observation container.

        Returns
        -------
        psf : `~gammapy.cube.PSFMap`
            Psf map.
        """
        psf = observation.psf
        if isinstance(psf, EnergyDependentMultiGaussPSF):
            rad_axis = geom.get_axis_by_name("theta")
            psf = psf.to_psf3d(rad=rad_axis.center)

        exposure = self.make_exposure_irf(geom.squash(axis="theta"), observation)

        return make_psf_map(
            psf=psf,
            pointing=observation.pointing_radec,
            geom=geom,
            exposure_map=exposure,
        )

    def run(self, dataset, observation):
        """Make map dataset.

        Parameters
        ----------
        dataset : `~gammapy.cube.MapDataset`
            Reference dataset.
        observation : `~gammapy.data.Observation`
            Observation

        Returns
        -------
        dataset : `~gammapy.cube.MapDataset`
            Map dataset.
        """
        from gammapy.datasets import MapDataset
        from gammapy.modeling.models import BackgroundModel

        kwargs = {"gti": observation.gti}

        mask_safe = Map.from_geom(dataset.counts.geom, dtype=bool)
        mask_safe.data |= True

        kwargs["mask_safe"] = mask_safe

        if "counts" in self.selection:
            counts = self.make_counts(dataset.counts.geom, observation)
            kwargs["counts"] = counts

        if "exposure" in self.selection:
            exposure = self.make_exposure(dataset.exposure.geom, observation)
            kwargs["exposure"] = exposure

        if "background" in self.selection:
            background_map = self.make_background(dataset.counts.geom, observation)
            kwargs["models"] = BackgroundModel(
                background_map, name=dataset.name + "-bkg", datasets_names=[dataset.name]
            )

        if "psf" in self.selection:
            psf = self.make_psf(dataset.psf.psf_map.geom, observation)
            kwargs["psf"] = psf

        if "edisp" in self.selection:
            edisp = self.make_edisp(dataset.edisp.edisp_map.geom, observation)
            kwargs["edisp"] = edisp

        return MapDataset(name=dataset.name, **kwargs)


