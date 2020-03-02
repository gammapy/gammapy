# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Time-dependent models."""
import numpy as np
import scipy.interpolate
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from astropy.utils import lazyproperty
from gammapy.modeling import Parameter
from gammapy.utils.random import InverseCDFSampler, get_random_state
from gammapy.utils.scripts import make_path
from gammapy.utils.time import time_ref_from_dict
from .core import Model


# TODO: make this a small ABC to define a uniform interface.
class TemporalModel(Model):
    """Temporal model base class."""

    def __call__(self, time, **kwargs):
        """Call evaluate method"""
        return self.evaluate(time.mjd, **kwargs)


# TODO: make this a small ABC to define a uniform interface.
class TemplateTemporalModel(TemporalModel):
    """Template temporal model base class."""

    @classmethod
    def read(cls, path):
        """Read lightcurve model table from FITS file.

        TODO: This doesn't read the XML part of the model yet.
        """
        filename = str(make_path(path))
        return cls(Table.read(filename), filename=filename)

    def write(self, path=None, overwrite=False):
        if path is None:
            path = self.filename
        if path is None:
            raise ValueError(f"filename is required for {self.tag}")
        else:
            self.filename = str(make_path(path))
            self.table.write(self.filename, overwrite=overwrite)


class ConstantTemporalModel(TemporalModel):
    """Constant temporal model.
    """

    tag = "ConstantTemporalModel"

    def evaluate(self, time):
        """Evaluate for a given time.

        Parameters
        ----------
        time : array_like
            Time since the ``reference`` time.

        """
        return np.ones_like(time)

    def integral(self, t_min, t_max):
        """Evaluate the integrated flux within the given time intervals

        Parameters
        ----------
        t_min: `~astropy.time.Time`
            Start times of observation
        t_max: `~astropy.time.Time`
            Stop times of observation
        Returns
        -------
        norm: The model integrated flux
        """
        sum_gti = np.sum(u.Quantity(t_max.mjd - t_min.mjd, "day"))
        return u.Quantity(t_max.mjd - t_min.mjd, "day")/sum_gti



    def sample_time(self, n_events, t_min, t_max, random_state=0):
        """Sample arrival times of events.

        Parameters
        ----------
        n_events : int
            Number of events to sample.
        t_min : `~astropy.time.Time`
            Start time of the sampling.
        t_max : `~astropy.time.Time`
            Stop time of the sampling.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.

        Returns
        -------
        time : `~astropy.units.Quantity`
            Array with times of the sampled events.
        """
        random_state = get_random_state(random_state)

        t_min = Time(t_min)
        t_max = Time(t_max)

        t_stop = (t_max - t_min).sec

        time_delta = random_state.uniform(high=t_stop, size=n_events) * u.s

        return t_min + time_delta


class ExpDecayTemporalModel(TemporalModel):
    """Temporal model with an exponential rise or fall.
        """

    tag = "ExponentialDecayTemporalModel"


    def evaluate(self, time, t0):
        return np.exp(time, t0)


    def integral(self,):




class LightCurveTemplateTemporalModel(TemplateTemporalModel):
    """Temporal light curve model.

    The lightcurve is given as a table with columns ``time`` and ``norm``.

    The ``norm`` is supposed to be a unit-less multiplicative factor in the model,
    to be multiplied with a spectral model.

    The model does linear interpolation for times between the given ``(time, norm)`` values.

    The implementation currently uses `scipy.interpolate.InterpolatedUnivariateSpline`,
    using degree ``k=1`` to get linear interpolation.
    This class also contains an ``integral`` method, making the computation of
    mean fluxes for a given time interval a one-liner.

    Parameters
    ----------
    table : `~astropy.table.Table`
        A table with 'TIME' vs 'NORM'

    Examples
    --------
    Read an example light curve object:

    >>> from gammapy.modeling.models import LightCurveTemplateTemporalModel
    >>> path = '$GAMMAPY_DATA/tests/models/light_curve/lightcrv_PKSB1222+216.fits'
    >>> light_curve = LightCurveTemplateTemporalModel.read(path)

    Show basic information about the lightcurve:

    >>> print(light_curve)
    LightCurve model summary:
    Start time: 59000.5 MJD
    End time: 61862.5 MJD
    Norm min: 0.01551196351647377
    Norm max: 1.0

    Compute ``norm`` at a given time:

    >>> light_curve.evaluate(46300)
    0.49059393580053845

    Compute mean ``norm`` in a given time interval:

    >>> light_curve.mean_norm_in_time_interval(46300, 46301)
    """

    tag = "LightCurveTemplateTemporalModel"

    def __init__(self, table, filename=None):
        self.table = table
        if filename is not None:
            filename = str(make_path(filename))
        self.filename = filename
        super().__init__()

    def __str__(self):
        norm = self.table["NORM"]
        return (
            f"{self.__class__.__name__} model summary:\n"
            f"Start time: {self._time[0].mjd} MJD\n"
            f"End time: {self._time[-1].mjd} MJD\n"
            f"Norm min: {norm.min()}\n"
            f"Norm max: {norm.max()}\n"
        )

    @lazyproperty
    def _interpolator(self, ext=0):
        x = self._time.value
        y = self.table["NORM"].data
        return scipy.interpolate.InterpolatedUnivariateSpline(x, y, k=1, ext=ext)

    @lazyproperty
    def _time_ref(self):
        return time_ref_from_dict(self.table.meta)

    @lazyproperty
    def _time(self):
        return self._time_ref + self.table["TIME"].data * getattr(
            u, self.table.meta["TIMEUNIT"]
        )

    def evaluate(self, time, ext=0):
        """Evaluate for a given time.

        Parameters
        ----------
        time : array_like
            Time since the ``reference`` time.
        ext : int or str, optional
            Parameter passed to ~scipy.interpolate.InterpolatedUnivariateSpline
            Controls the extrapolation mode for GTIs outside the range
            if ext=0 or ‘extrapolate’, return the extrapolated value.
            if ext=1 or ‘zeros’, return 0
            if ext=2 or ‘raise’, raise a ValueError
            if ext=3 of ‘const’, return the boundary value.
            The default value is 0.

        Returns
        -------
        norm : array_like
        """
        return self._interpolator(time, ext=ext)

    def integral(self, t_min, t_max):
        """Evaluate the integrated flux within the given time intervals

        Parameters
        ----------
        t_min: `~astropy.time.Time`
            Start times of observation
        t_max: `~astropy.time.Time`
            Stop times of observation
        Returns
        -------
        norm: The model integrated flux
        """

        n1 = self._interpolator.antiderivative()(t_max.mjd)
        n2 = self._interpolator.antiderivative()(t_min.mjd)
        sum_gti = np.sum(u.Quantity(t_max.mjd - t_min.mjd, "day"))
        return u.Quantity(n1 - n2, "day")/sum_gti

    def mean_norm_in_time_interval(self, time_min, time_max):
        """Compute mean ``norm`` in a given time interval.

        TODO: vectorise, i.e. allow arrays of time intervals in a single call.

        Parameters
        ----------
        time_min, time_max : float
            Time interval

        Returns
        -------
        norm : float
            Mean norm
        """
        dt = time_max - time_min
        integral = self._interpolator.integral(time_min, time_max)
        return integral / dt

    def sample_time(self, n_events, t_min, t_max, t_delta="1 s", random_state=0):
        """Sample arrival times of events.

        Parameters
        ----------
        n_events : int
            Number of events to sample.
        t_min : `~astropy.time.Time`
            Start time of the sampling.
        t_max : `~astropy.time.Time`
            Stop time of the sampling.
        t_delta : `~astropy.units.Quantity`
            Time step used for sampling of the temporal model.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.

        Returns
        -------
        time : `~astropy.units.Quantity`
            Array with times of the sampled events.
        """
        time_unit = getattr(u, self.table.meta["TIMEUNIT"])

        t_min = Time(t_min)
        t_max = Time(t_max)
        t_delta = u.Quantity(t_delta)
        random_state = get_random_state(random_state)

        ontime = u.Quantity((t_max - t_min).sec, "s")
        t_stop = ontime.to_value(time_unit)

        # TODO: the separate time unit handling is unfortunate, but the quantity support for np.arange and np.interp
        #  is still incomplete, refactor once we change to recent numpy and astropy versions
        t_step = t_delta.to_value(time_unit)
        t = np.arange(0, t_stop, t_step)

        pdf = self.evaluate(t)

        sampler = InverseCDFSampler(pdf=pdf, random_state=random_state)
        time_pix = sampler.sample(n_events)[0]
        time = np.interp(time_pix, np.arange(len(t)), t) * time_unit

        return t_min + time

    @classmethod
    def from_dict(cls, data):
        return cls.read(data["filename"])

    def to_dict(self, overwrite=False):
        """Create dict for YAML serilisation"""
        return {"type": self.tag, "filename": self.filename}
