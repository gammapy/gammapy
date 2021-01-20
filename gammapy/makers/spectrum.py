# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from regions import CircleSkyRegion
from .map import MapDatasetMaker


__all__ = ["SpectrumDatasetMaker"]

log = logging.getLogger(__name__)


class SpectrumDatasetMaker(MapDatasetMaker):
    """Make spectrum for a single IACT observation.

    The irfs and background are computed at a single fixed offset,
    which is recommend only for point-sources.

    Parameters
    ----------
    selection : list
        List of str, selecting which maps to make.
        Available: 'counts', 'exposure', 'background', 'edisp'
        By default, all spectra are made.
    containment_correction : bool
        Apply containment correction for point sources and circular on regions.
    background_oversampling : int
        Background evaluation oversampling factor in energy.
    """

    tag = "SpectrumDatasetMaker"
    available_selection = ["counts", "background", "exposure", "edisp"]

    def __init__(self, selection=None, containment_correction=False, background_oversampling=None):
        self.containment_correction = containment_correction
        super().__init__(
            selection=selection, background_oversampling=background_oversampling
        )

    def make_exposure(self, geom, observation):
        """Make exposure.

        Parameters
        ----------
        geom : `~gammapy.maps.RegionGeom`
            Reference map geom.
        observation: `~gammapy.data.Observation`
            Observation to compute effective area for.

        Returns
        -------
        exposure : `~gammapy.irf.EffectiveAreaTable`
            Exposure map.
        """
        exposure = super().make_exposure(geom, observation)

        if self.containment_correction:
            if not isinstance(geom.region, CircleSkyRegion):
                raise TypeError(
                    "Containment correction only supported for circular regions."
                )
            offset = geom.separation(observation.pointing_radec)
            containment = observation.psf.containment(
                rad=geom.region.radius, offset=offset, energy_true=geom.axes["energy_true"].center
            )
            exposure.quantity *= containment.reshape(geom.data_shape)

        return exposure
