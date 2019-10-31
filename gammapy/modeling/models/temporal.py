# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Time-dependent models."""
import numpy as np
import scipy.interpolate
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from astropy.utils import lazyproperty
from gammapy.modeling import Model, Parameter
from gammapy.utils.random import InverseCDFSampler, get_random_state
from gammapy.utils.scripts import make_path
from gammapy.utils.time import time_ref_from_dict


# TODO: make this a small ABC to define a uniform interface.
class TemporalModel(Model):
    """Temporal model base class."""


class ConstantTemporalModel(TemporalModel):
    """Constant temporal model.

    Parameters
    ----------
    norm : float
        The normalization of the constant temporal model
    """

    tag = "ConstantTemporalModel"
    norm = Parameter("norm", 1)

    def evaluate_norm_at_time(self, time):
        """Evaluate for a given time.

        Parameters
        ----------
        time : array_like
            Time since the ``reference`` time.

        Returns
        -------
        norm : float
            Mean norm
        """
        return np.ones_like(time) * self.norm.value

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


class PhaseCurveTemplateTemporalModel(TemporalModel):
    r"""Temporal phase curve model.

    Phase for a given time is computed as:

    .. math::
        \phi(t) = \phi_0 + f_0(t-t_0) + (1/2)f_1(t-t_0)^2 + (1/6)f_2(t-t_0)^3

    Strictly periodic sources such as gamma-ray binaries have ``f1=0`` and ``f2=0``.
    Sources like some pulsars where the period spins up or down have ``f1!=0``
    and / or ``f2 !=0``. For a binary, ``f0`` should be calculated as 1/T,
    where T is the period of the binary in unit of ``seconds``.

    The "phase curve", i.e. multiplicative flux factor for a given phase is given
    by a `~astropy.table.Table` of nodes ``(phase, norm)``, using linear interpolation
    and circular behaviour, where ``norm(phase=0) == norm(phase=1)``.

    Parameters
    ----------
    table : `~astropy.table.Table`
        A table of 'PHASE' vs 'NORM' should be given
    time_0 : float
        The MJD value where phase is considered as 0.
    phase_0 : float
        Phase at the reference MJD
    f0, f1, f2 : float
        Derivatives of the function phi with time of order 1, 2, 3
        in units of ``s^-1, s^-2 & s^-3``, respectively.

    Examples
    --------
    Create an example phase curve object::

        from astropy.table import Table
        from gammapy.utils.scripts import make_path
        from gammapy.modeling.models import PhaseCurveTemplateTemporalModel
        filename = make_path('$GAMMAPY_DATA/tests/phasecurve_LSI_DC.fits')
        table = Table.read(filename)
        phase_curve = PhaseCurveTemplateTemporalModel(table, time_0=43366.275, phase_0=0.0, f0=4.367575e-7, f1=0.0, f2=0.0)

    Use it to compute a phase and evaluate the phase curve model for a given time:

    >>> phase_curve.phase(time=46300.0)
    0.7066006737999402
    >>> phase_curve.evaluate_norm_at_time(46300)
    0.49059393580053845
    """

    tag = "PhaseCurveTemplateTemporalModel"
    # TODO: allow parameters without default?
    time_0 = Parameter("time_0", 0)
    phase_0 = Parameter("phase_0", 0)
    f0 = Parameter("f0", 0)
    f1 = Parameter("f1", 0)
    f2 = Parameter("f2", 0)

    def __init__(self, table, time_0, phase_0, f0, f1=0, f2=0):
        self.table = table
        super().__init__(time_0=time_0, phase_0=phase_0, f0=f0, f1=f1, f2=f2)

    def phase(self, time):
        """Evaluate phase for a given time.

        Parameters
        ----------
        time : array_like

        Returns
        -------
        phase : array_like
        """
        pars = self.parameters
        time_0 = pars["time_0"].value
        phase_0 = pars["phase_0"].value
        f0 = pars["f0"].value
        f1 = pars["f1"].value
        f2 = pars["f2"].value

        t = (time - time_0) * u.day.to(u.second)
        phase = self._evaluate_phase(t, phase_0, f0, f1, f2)
        return np.remainder(phase, 1)

    @staticmethod
    def _evaluate_phase(t, phase_0, f0, f1, f2):
        return phase_0 + t * (f0 + t * (f1 / 2 + f2 / 6 * t))

    def evaluate_norm_at_time(self, time):
        """Evaluate for a given time.

        Parameters
        ----------
        time : array_like
            Time since the ``reference`` time.

        Returns
        -------
        norm : array_like
        """
        phase = self.phase(time)
        return self.evaluate_norm_at_phase(phase)

    def evaluate_norm_at_phase(self, phase):
        xp = self.table["PHASE"]
        fp = self.table["NORM"]
        return np.interp(x=phase, xp=xp, fp=fp, period=1)

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
        time_unit = u.second

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

        pdf = self.evaluate_norm_at_time(t)

        sampler = InverseCDFSampler(pdf=pdf, random_state=random_state)
        time_pix = sampler.sample(n_events)[0]
        time = np.interp(time_pix, np.arange(len(t)), t) * time_unit

        return t_min + time


class LightCurveTemplateTemporalModel(TemporalModel):
    """Temporal light curve model.

    The lightcurve is given as a table with columns ``time`` and ``norm``.

    The ``norm`` is supposed to be a unite-less multiplicative factor in the model,
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

    >>> light_curve.evaluate_norm_at_time(46300)
    0.49059393580053845

    Compute mean ``norm`` in a given time interval:

    >>> light_curve.mean_norm_in_time_interval(46300, 46301)
    """

    tag = "LightCurveTemplateTemporalModel"

    def __init__(self, table):
        self.table = table
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

    @classmethod
    def read(cls, path):
        """Read lightcurve model table from FITS file.

        TODO: This doesn't read the XML part of the model yet.
        """
        return cls(Table.read(make_path(path)))

    @lazyproperty
    def _interpolator(self):
        x = self.table["TIME"].data
        y = self.table["NORM"].data
        return scipy.interpolate.InterpolatedUnivariateSpline(x, y, k=1)

    @lazyproperty
    def _time_ref(self):
        return time_ref_from_dict(self.table.meta)

    @lazyproperty
    def _time(self):
        return self._time_ref + self.table["TIME"].data * u.s

    def evaluate_norm_at_time(self, time, ext_mode=3):
        """Evaluate for a given time.

        Parameters
        ----------
        time : array_like
            Time since the ``reference`` time.
        ext_mode : int or str, optional
            Controls the extrapolation mode for elements not in the interval defined by the knot sequence.
            if ext=0 or ‘extrapolate’, return the extrapolated value.
            if ext=1 or ‘zeros’, return 0
            if ext=2 or ‘raise’, raise a ValueError
            if ext=3 of ‘const’, return the boundary value.
            The default value is 0.

        Returns
        -------
        norm : array_like
        """
        return self._interpolator(time, ext=ext_mode)

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
        time_unit = u.second

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

        pdf = self.evaluate_norm_at_time(t * time_unit)

        sampler = InverseCDFSampler(pdf=pdf, random_state=random_state)
        time_pix = sampler.sample(n_events)[0]
        time = np.interp(time_pix, np.arange(len(t)), t) * time_unit

        return t_min + time
