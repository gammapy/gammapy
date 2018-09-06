# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from regions import CircleSkyRegion
from .ring import ring_area_factor

__all__ = ["BackgroundEstimate", "ring_background_estimate"]


class BackgroundEstimate(object):
    """Container class for background estimate.

    This container holds the result from a region based background estimation
    for one observation. Currently, it is filled by the functions
    :func:`~gammapy.background.ring_background_estimate` and
    the `~gammapy.background.ReflectedRegionsBackgroundEstimator`.

    Parameters
    ----------
    on_region : `~regions.SkyRegion`
        Signal extraction region
    on_events : `~gammapy.data.EventList`
        Signal events
    off_region : `~regions.SkyRegion`
        Background extraction region
    off_events : `~gammapy.data.EventList`
        Background events
    a_on : float
        Relative background exposure of the on region
    a_off : float
        Relative background exposure of the off region
    method : str
        Background estimation method
    """

    def __init__(
        self,
        on_region,
        on_events,
        off_region,
        off_events,
        a_on,
        a_off,
        method="default",
    ):
        self.on_region = on_region
        self.on_events = on_events
        self.off_region = off_region
        self.off_events = off_events
        self.a_on = a_on
        self.a_off = a_off
        self.method = method

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n Method: {}".format(self.method)
        ss += "\n on region"
        ss += "\n {}".format(self.on_region)
        ss += "\n {}".format(self.on_events)
        ss += "\n off region"
        ss += "\n {}".format(self.off_region)
        ss += "\n {}".format(self.off_events)
        return ss


def ring_background_estimate(pos, on_radius, inner_radius, outer_radius, events):
    """Simple ring background estimate.

    No acceptance correction is applied

    TODO : Replace with AnnulusSkyRegion

    Parameters
    ----------
    pos : `~astropy.coordinates.SkyCoord`
        On region radius
    on_radius : `~astropy.coordinates.Angle`
        On region radius
    inner_radius, outer_radius : `~astropy.coordinates.Angle`
        Inner and outer ring radius
    events : `~gammapy.data.EventList`
        Event list

    Returns
    -------
    bkg : `~gammapy.data.BackgroundEstimate`
        Background estimate
    """
    on_region = CircleSkyRegion(center=pos, radius=on_radius)
    on_events = events.select_circular_region(on_region)

    off_region = dict(inner=inner_radius, outer=outer_radius)
    off_events = events.select_sky_ring(pos, inner_radius, outer_radius)

    # TODO: change to region areas here (e.g. in steratian?)
    a_on = 1
    a_off = ring_area_factor(on_radius, inner_radius, outer_radius).value

    return BackgroundEstimate(
        on_region=on_region,
        on_events=on_events,
        off_region=off_region,
        off_events=off_events,
        a_on=a_on,
        a_off=a_off,
        method="ring",
    )
