# Licensed under a 3-clause BSD style license - see LICENSE.rst
from collections import OrderedDict
import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy.utils import lazyproperty

from .utils import get_random_state

__all__ = ["InverseCDFSampler", "MapEventSampler"]


class InverseCDFSampler:
    """Inverse CDF sampler.

   It determines a set of random numbers and calculate the cumulative
   distribution function.

   Parameters
   ----------
   pdf : `~gammapy.maps.Map`
        Map of the predicted source counts.
   axis : int
        Axis along which sampling the indexes.
   random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.
    """

    def __init__(self, pdf, axis=None, random_state=0):
        self.random_state = get_random_state(random_state)
        self.axis = axis

        if axis is not None:
            self.cdf = np.cumsum(pdf, axis=self.axis)
            self.cdf /= self.cdf[:, [-1]]
        else:
            self.pdf_shape = pdf.shape

            pdf = pdf.ravel() / pdf.sum()
            self.sortindex = np.argsort(pdf, axis=None)

            self.pdf = pdf[self.sortindex]
            self.cdf = np.cumsum(self.pdf)

    def sample_axis(self):
        """Sample along a given axis.

        Returns
        -------
        index : tuple of `~numpy.ndarray`
            Coordinates of the drawn sample.

        """
        choice = self.random_state.uniform(high=1, size=len(self.cdf))

        # find the indices corresponding to this point on the CDF
        index = np.argmin(np.abs(choice.reshape(-1, 1) - self.cdf), axis=self.axis)

        return index + self.random_state.uniform(low=-0.5, high=0.5, size=len(self.cdf))

    def sample(self, size):
        """Draw sample from the given PDF.

        Parameters
        ----------
        size : int
            Number of samples to draw.

        Returns
        -------
        index : tuple of `~numpy.ndarray`
            Coordinates of the drawn sample.
        """
        # pick numbers which are uniformly random over the cumulative distribution function
        choice = self.random_state.uniform(high=1, size=size)

        # find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        index = self.sortindex[index]

        # map back to multi-dimensional indexing
        index = np.unravel_index(index, self.pdf_shape)
        index = np.vstack(index)

        index = index + self.random_state.uniform(low=-0.5, high=0.5, size=index.shape)
        return index


class MapEventSampler:
    """Map event sampler.

    Parameters
    ----------
    npred_map : `~gammapy.maps.Map`
            Predicted number of counts map.
    t_min : `~astropy.time.Time`
            Start time of the sampling.
    t_max : `~astropy.time.Time`
            Stop time of the sampling.
    t_delta : `~astropy.units.Quantity`
            Time step used for sampling of the temporal model.
    temporal_model : `~gammapy.time.models.LightCurveTableModel` or `~gammapy.time.models.PhaseCurveTableModel`
            Input light (or phase)-curve model of the source, given with columns labelled
            as "time" (or "phase)" and "normalization" (arbitrary units).
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.
    """

    time_unit = u.second

    def __init__(
        self,
        npred_map,
        t_min,
        t_max,
        t_delta="1 s",
        temporal_model=None,
        random_state=0,
    ):
        self.npred_map = npred_map
        self.temporal_model = temporal_model
        self.t_min = Time(t_min)
        self.t_max = Time(t_max)
        self.t_delta = u.Quantity(t_delta)
        self.random_state = get_random_state(random_state)

    @lazyproperty
    def npred_total(self):
        """Sample the total number of predicted events.

        Returns
        -------
        npred_total : int
            Number of predicted events.
        """

        return self.random_state.poisson(np.sum(self.npred_map.data))

    def sample_position_energy(self, n_events=None):
        """Sample position and energy of events.

        Parameters
        ----------
        n_events : int
            Number of events to sample.

        Returns
        -------
        coords : `~gammapy.maps.MapCoord` object.
            Sequence of coordinates and energies of the sampled events.
        """
        from ...maps import MapCoord

        n_events = self.npred_total if n_events is None else n_events

        sampler = InverseCDFSampler(self.npred_map.data, random_state=self.random_state)

        coords_pix = sampler.sample(n_events)
        coords = self.npred_map.geom.pix_to_coord(coords_pix[::-1])

        # TODO: pix_to_coord should return a MapCoord object
        geom = self.npred_map.geom
        axes_names = ["lon", "lat"] + [ax.name for ax in geom.axes]
        cdict = OrderedDict(zip(axes_names, coords))
        cdict["energy"] *= geom.get_axis_by_name("energy").unit
        return MapCoord.create(cdict, coordsys=geom.coordsys)

    @property
    def ontime(self):
        """On time (`~astropy.units.Quantity`)"""
        ontime = self.t_max - self.t_min
        return u.Quantity(ontime.sec, "s")

    def _get_time_meta(self):
        """Time meta information (`OrderedDict`)"""
        # TODO: extend the meta information according to
        #  https://gamma-astro-data-formats.readthedocs.io/en/latest/general/time.html#time-formats
        meta = OrderedDict()
        meta["ONTIME"] = np.round(self.ontime.to_value("s"), 1)
        return meta

    def sample_time(self, n_events=None):
        """Sample arrival times of events.

        Parameters
        ----------
        n_events : int
            Number of events to sample.

        Returns
        -------
        time : `~astropy.units.Quantity`
            Array with times of the sampled events.
        """
        n_events = self.npred_total if n_events is None else n_events
        t_stop = self.ontime.to_value(self.time_unit)

        # TODO: the separate time unit handling is unfortunate, but the quantity support for np.arange and np.interp
        #  is still incomplete, refactor once we change to recent numpy and astropy versions
        if self.temporal_model is not None:
            t_step = self.t_delta.to_value(self.time_unit)
            t = np.arange(0, t_stop, t_step)

            pdf = self.temporal_model.evaluate_norm_at_time(t * self.time_unit)

            sampler = InverseCDFSampler(pdf=pdf, random_state=self.random_state)
            time_pix = sampler.sample(n_events)[0]
            time = np.interp(time_pix, np.arange(len(t)), t) * self.time_unit
        else:
            time = (
                self.random_state.uniform(high=t_stop, size=n_events) * self.time_unit
            )

        return time

    def sample_events(self, n_events=None):
        """It converts the given sampled event list into an astropy table.

        Parameters
        ----------
        n_events : int
            Number of events to sample.


        Returns
        -------
        events : `~astropy.table`
            Sampled event list in an astropy table format.
        """

        coords = self.sample_position_energy(n_events)
        skycoord = coords.skycoord

        events = Table()
        events["RA_TRUE"] = skycoord.icrs.ra
        events["DEC_TRUE"] = skycoord.icrs.dec
        events["ENERGY_TRUE"] = coords["energy"]
        events["TIME"] = self.sample_time(n_events)
        events.meta.update(self._get_time_meta())
        return events
