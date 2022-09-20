# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Time-dependent models."""
import numpy as np
import scipy.interpolate
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from astropy.utils import lazyproperty
from gammapy.maps import RegionNDMap, TimeMapAxis
from gammapy.modeling import Parameter
from gammapy.utils.random import InverseCDFSampler, get_random_state
from gammapy.utils.scripts import make_path
from gammapy.utils.time import time_ref_from_dict
from .core import ModelBase

__all__ = [
    "ConstantTemporalModel",
    "ExpDecayTemporalModel",
    "GaussianTemporalModel",
    "GeneralizedGaussianTemporalModel",
    "LightCurveTemplateTemporalModel",
    "LinearTemporalModel",
    "PowerLawTemporalModel",
    "SineTemporalModel",
    "TemporalModel",
]


# TODO: make this a small ABC to define a uniform interface.
class TemporalModel(ModelBase):
    """Temporal model base class.
    evaluates on  astropy.time.Time objects"""

    _type = "temporal"

    def __call__(self, time):
        """Evaluate model

        Parameters
        ----------
        time : `~astropy.time.Time`
            Time object
        """
        kwargs = {par.name: par.quantity for par in self.parameters}
        time = u.Quantity(time.mjd, "day")
        return self.evaluate(time, **kwargs)

    @property
    def type(self):
        return self._type

    @staticmethod
    def time_sum(t_min, t_max):
        """
        Total time between t_min and t_max

        Parameters
        ----------
        t_min, t_max : `~astropy.time.Time`
            Lower and upper bound of integration range

        Returns
        -------
        time_sum : `~astropy.time.TimeDelta`
            Summed time in the intervals.

        """
        diff = t_max - t_min
        # TODO: this is a work-around for https://github.com/astropy/astropy/issues/10501
        return u.Quantity(np.sum(diff.to_value("day")), "day")

    def plot(self, time_range, ax=None, **kwargs):
        """
        Plot Temporal Model.

        Parameters
        ----------
        time_range : `~astropy.time.Time`
            times to plot the model
        ax : `~matplotlib.axes.Axes`, optional
            Axis to plot on
        **kwargs : dict
            Keywords forwarded to `~matplotlib.pyplot.errorbar`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            axis
        """
        time_min, time_max = time_range
        time_axis = TimeMapAxis.from_time_bounds(
            time_min=time_min, time_max=time_max, nbin=100
        )

        m = RegionNDMap.create(region=None, axes=[time_axis])
        kwargs.setdefault("marker", "None")
        kwargs.setdefault("ls", "-")
        kwargs.setdefault("xerr", None)
        m.quantity = self(time_axis.time_mid)
        ax = m.plot(ax=ax, **kwargs)
        ax.set_ylabel("Norm / A.U.")
        return ax

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
        t_min = Time(t_min)
        t_max = Time(t_max)
        t_delta = u.Quantity(t_delta)
        random_state = get_random_state(random_state)

        ontime = u.Quantity((t_max - t_min).sec, "s")

        time_unit = (
            u.Unit(self.table.meta["TIMEUNIT"])
            if hasattr(self, "table")
            else ontime.unit
        )

        # TODO: the separate time unit handling is unfortunate, but the quantity support for np.arange and np.interp
        #  is still incomplete, refactor once we change to recent numpy and astropy versions
        t_step = t_delta.to_value(time_unit)
        t_step = (t_step * u.s).to("d")

        t = Time(np.arange(t_min.mjd, t_max.mjd, t_step.value), format="mjd")

        pdf = self(t)

        sampler = InverseCDFSampler(pdf=pdf, random_state=random_state)
        time_pix = sampler.sample(n_events)[0]
        time = (
                np.interp(time_pix, np.arange(len(t)), t.value - min(t.value))
                * t_step.unit
                ).to(time_unit)

        return t_min + time

    def integral(self, t_min, t_max, oversampling_factor=100, **kwargs):
        """Evaluate the integrated flux within the given time intervals

        Parameters
        ----------
        t_min: `~astropy.time.Time`
            Start times of observation
        t_max: `~astropy.time.Time`
            Stop times of observation
        oversampling_factor : int
            Oversampling factor to be used for numerical integration.

        Returns
        -------
        norm : float
            Integrated flux norm on the given time intervals
        """
        t_values = np.linspace(t_min.mjd, t_max.mjd, oversampling_factor, axis=-1)
        times = Time(t_values, format="mjd")
        values = self(times)
        integral = np.sum(values / oversampling_factor, axis=-1)
        return integral / self.time_sum(t_min, t_max).to_value("d")


class ConstantTemporalModel(TemporalModel):
    """Constant temporal model.

    For more information see :ref:`constant-temporal-model`.
    """

    tag = ["ConstantTemporalModel", "const"]

    @staticmethod
    def evaluate(time):
        """Evaluate at given times."""
        return np.ones(time.shape)

    def integral(self, t_min, t_max):
        """Evaluate the integrated flux within the given time intervals

        Parameters
        ----------
        t_min : `~astropy.time.Time`
            Start times of observation
        t_max : `~astropy.time.Time`
            Stop times of observation

        Returns
        -------
        norm : `~astropy.units.Quantity`
            Integrated flux norm on the given time intervals
        """
        return (t_max - t_min) / self.time_sum(t_min, t_max)


class LinearTemporalModel(TemporalModel):
    """Temporal model with a linear variation.

    For more information see :ref:`linear-temporal-model`.

    Parameters
    ----------
    alpha : float
        Constant term of the baseline flux
    beta : `~astropy.units.Quantity`
        Time variation coefficient of the flux
    t_ref : `~astropy.units.Quantity`
        The reference time in mjd. Frozen per default, at 2000-01-01.
    """

    tag = ["LinearTemporalModel", "linear"]

    alpha = Parameter("alpha", 1.0, frozen=False)
    beta = Parameter("beta", 0.0, unit="d-1", frozen=False)
    _t_ref_default = Time("2000-01-01")
    t_ref = Parameter("t_ref", _t_ref_default.mjd, unit="day", frozen=True)

    @staticmethod
    def evaluate(time, alpha, beta, t_ref):
        """Evaluate at given times"""
        return alpha + beta * (time - t_ref)

    def integral(self, t_min, t_max):
        """Evaluate the integrated flux within the given time intervals

        Parameters
        ----------
        t_min : `~astropy.time.Time`
            Start times of observation
        t_max : `~astropy.time.Time`
            Stop times of observation

        Returns
        -------
        norm : float
            Integrated flux norm on the given time intervals
        """
        pars = self.parameters
        alpha = pars["alpha"]
        beta = pars["beta"].quantity
        t_ref = Time(pars["t_ref"].quantity, format="mjd")
        value = alpha * (t_max - t_min) + beta / 2.0 * (
            (t_max - t_ref) * (t_max - t_ref) - (t_min - t_ref) * (t_min - t_ref)
        )
        return value / self.time_sum(t_min, t_max)


class ExpDecayTemporalModel(TemporalModel):
    r"""Temporal model with an exponential decay.

    For more information see :ref:`expdecay-temporal-model`.

    Parameters
    ----------
    t0 : `~astropy.units.Quantity`
        Decay time scale
    t_ref : `~astropy.units.Quantity`
        The reference time in mjd. Frozen per default, at 2000-01-01 .
    """

    tag = ["ExpDecayTemporalModel", "exp-decay"]

    t0 = Parameter("t0", "1 d", frozen=False)
    _t_ref_default = Time("2000-01-01")
    t_ref = Parameter("t_ref", _t_ref_default.mjd, unit="day", frozen=True)

    @staticmethod
    def evaluate(time, t0, t_ref):
        """Evaluate at given times"""
        return np.exp(-(time - t_ref) / t0)

    def integral(self, t_min, t_max):
        """Evaluate the integrated flux within the given time intervals

        Parameters
        ----------
        t_min : `~astropy.time.Time`
            Start times of observation
        t_max : `~astropy.time.Time`
            Stop times of observation

        Returns
        -------
        norm : float
            Integrated flux norm on the given time intervals
        """
        pars = self.parameters
        t0 = pars["t0"].quantity
        t_ref = Time(pars["t_ref"].quantity, format="mjd")
        value = self.evaluate(t_max, t0, t_ref) - self.evaluate(t_min, t0, t_ref)
        return -t0 * value / self.time_sum(t_min, t_max)


class GaussianTemporalModel(TemporalModel):
    r"""A Gaussian temporal profile

    For more information see :ref:`gaussian-temporal-model`.

    Parameters
    ----------
    t_ref : `~astropy.units.Quantity`
        The reference time in mjd at the peak.
    sigma : `~astropy.units.Quantity`
        Width of the gaussian profile.
    """

    tag = ["GaussianTemporalModel", "gauss"]

    _t_ref_default = Time("2000-01-01")
    t_ref = Parameter("t_ref", _t_ref_default.mjd, unit="day", frozen=False)
    sigma = Parameter("sigma", "1 d", frozen=False)

    @staticmethod
    def evaluate(time, t_ref, sigma):
        return np.exp(-((time - t_ref) ** 2) / (2 * sigma**2))

    def integral(self, t_min, t_max, **kwargs):
        """Evaluate the integrated flux within the given time intervals

        Parameters
        ----------
        t_min : `~astropy.time.Time`
            Start times of observation
        t_max : `~astropy.time.Time`
            Stop times of observation

        Returns
        -------
        norm : float
            Integrated flux norm on the given time intervals
        """
        pars = self.parameters
        sigma = pars["sigma"].quantity
        t_ref = Time(pars["t_ref"].quantity, format="mjd")
        norm = np.sqrt(np.pi / 2) * sigma

        u_min = (t_min - t_ref) / (np.sqrt(2) * sigma)
        u_max = (t_max - t_ref) / (np.sqrt(2) * sigma)

        integral = norm * (scipy.special.erf(u_max) - scipy.special.erf(u_min))
        return integral / self.time_sum(t_min, t_max)


class GeneralizedGaussianTemporalModel(TemporalModel):
    r"""A generalized Gaussian temporal profile

    For more information see :ref:`generalized-gaussian-temporal-model`.

    Parameters
    ----------
    t_ref : `~astropy.units.Quantity`
        The time of the pulse's maximum intensity.
    t_rise : `~astropy.units.Quantity`
        Rise time constant.
    t_decay : `~astropy.units.Quantity`
        Decay time constant.
    eta : `~astropy.units.Quantity`
        Inverse pulse sharpness -> higher values implies a more peaked pulse

    """

    tag = ["GeneralizedGaussianTemporalModel", "gengauss"]

    _t_ref_default = Time("2000-01-01")
    t_ref = Parameter("t_ref", _t_ref_default.mjd, unit="day", frozen=False)
    t_rise = Parameter("t_rise", "1d", frozen=False)
    t_decay = Parameter("t_decay", "1d", frozen=False)
    eta = Parameter("eta", 1/2, unit="", frozen=False)

    @staticmethod
    def evaluate(time, t_ref, t_rise, t_decay, eta):
        val_rise = np.exp(
            -0.5
            * (np.abs(u.Quantity(time - t_ref, "d")) ** (1 / eta))
            / (t_rise ** (1 / eta))
        )
        val_decay = np.exp(
            -0.5
            * (np.abs(u.Quantity(time - t_ref, "d")) ** (1 / eta))
            / (t_decay ** (1 / eta))
        )
        val = np.where(time < t_ref, val_rise, val_decay)
        return val


class LightCurveTemplateTemporalModel(TemporalModel):
    """Temporal light curve model.

    The lightcurve is given as a table with columns ``time`` and ``norm``.

    The ``norm`` is supposed to be a unit-less multiplicative factor in the model,
    to be multiplied with a spectral model.

    The model does linear interpolation for times between the given ``(time, norm)`` values.

    The implementation currently uses `scipy.interpolate. InterpolatedUnivariateSpline`,
    using degree ``k=1`` to get linear interpolation.
    This class also contains an ``integral`` method, making the computation of
    mean fluxes for a given time interval a one-liner.

    For more information see :ref:`LightCurve-temporal-model`.

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
    LightCurveTemplateTemporalModel model summary:
    Start time: 59000.5 MJD
    End time: 61862.5 MJD
    Norm min: 0.01551196351647377
    Norm max: 1.0
    <BLANKLINE>

    Compute ``norm`` at a given time:

    >>> light_curve.evaluate(60000)
    array(0.01551196)

    Compute mean ``norm`` in a given time interval:

    >>> from astropy.time import Time
    >>> times = Time([60000, 61000], format='mjd')
    >>> light_curve.integral(times[0], times[1])
    <Quantity 0.01721725>
    """

    tag = ["LightCurveTemplateTemporalModel", "template"]

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
        ext : int or str, optional, default: 0
            Parameter passed to ~scipy.interpolate.InterpolatedUnivariateSpline
            Controls the extrapolation mode for GTIs outside the range
            0 or "extrapolate", return the extrapolated value.
            1 or "zeros", return 0
            2 or "raise", raise a ValueError
            3 or "const", return the boundary value.


        Returns
        -------
        norm : array_like
            Norm at the given times.
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
        return u.Quantity(n1 - n2, "day") / self.time_sum(t_min, t_max)

    @classmethod
    def from_dict(cls, data):
        return cls.read(data["temporal"]["filename"])

    def to_dict(self, full_output=False):
        """Create dict for YAML serialisation"""
        return {self._type: {"type": self.tag[0], "filename": self.filename}}


class PowerLawTemporalModel(TemporalModel):
    """Temporal model with a Power Law decay.

    For more information see :ref:`powerlaw-temporal-model`.

    Parameters
    ----------
    alpha : float
        Decay time power
    t_ref: `~astropy.units.Quantity`
        The reference time in mjd. Frozen by default, at 2000-01-01.
    t0: `~astropy.units.Quantity`
        The scaling time in mjd. Fixed by default, at 1 day.
    """

    tag = ["PowerLawTemporalModel", "powerlaw"]

    alpha = Parameter("alpha", 1.0, frozen=False)
    _t_ref_default = Time("2000-01-01")
    t_ref = Parameter("t_ref", _t_ref_default.mjd, unit="day", frozen=True)
    t0 = Parameter("t0", "1 d", frozen=True)

    @staticmethod
    def evaluate(time, alpha, t_ref, t0=1 * u.day):
        """Evaluate at given times"""
        return np.power((time - t_ref) / t0, alpha)

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
        norm : float
            Integrated flux norm on the given time intervals
        """
        pars = self.parameters
        alpha = pars["alpha"].quantity
        t0 = pars["t0"].quantity
        t_ref = Time(pars["t_ref"].quantity, format="mjd")
        if alpha != -1:
            value = self.evaluate(t_max, alpha + 1.0, t_ref, t0) - self.evaluate(
                t_min, alpha + 1.0, t_ref, t0
            )
            return t0 / (alpha + 1.0) * value / self.time_sum(t_min, t_max)
        else:
            value = np.log((t_max - t_ref) / (t_min - t_ref))
            return t0 * value / self.time_sum(t_min, t_max)


class SineTemporalModel(TemporalModel):
    """Temporal model with a sinusoidal modulation.

    For more information see :ref:`sine-temporal-model`.

    Parameters
    ----------
    amp : float
        Amplitude of the sinusoidal function
    t_ref: `~astropy.units.Quantity`
        The reference time in mjd.
    omega: `~astropy.units.Quantity`
        Pulsation of the signal.
    """

    tag = ["SineTemporalModel", "sinus"]

    amp = Parameter("amp", 1.0, frozen=False)
    omega = Parameter("omega", "1. rad/day", frozen=False)
    _t_ref_default = Time("2000-01-01")
    t_ref = Parameter("t_ref", _t_ref_default.mjd, unit="day", frozen=False)

    @staticmethod
    def evaluate(time, amp, omega, t_ref):
        """Evaluate at given times"""
        return 1.0 + amp * np.sin(omega * (time - t_ref))

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
        norm : float
            Integrated flux norm on the given time intervals
        """
        pars = self.parameters
        omega = pars["omega"].quantity.to_value("rad/day")
        amp = pars["amp"].value
        t_ref = Time(pars["t_ref"].quantity, format="mjd")
        value = (t_max - t_min) - amp / omega * (
            np.sin(omega * (t_max - t_ref).to_value("day"))
            - np.sin(omega * (t_min - t_ref).to_value("day"))
        )
        return value / self.time_sum(t_min, t_max)
