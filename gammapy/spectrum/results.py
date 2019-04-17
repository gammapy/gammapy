# Licensed under a 3-clause BSD style license - see LICENSE.rst
import yaml
import numpy as np
from astropy.table import Table, Column
import astropy.units as u
from ..spectrum import CountsSpectrum, models
from ..utils.scripts import read_yaml, make_path
from ..utils.energy import EnergyBounds

__all__ = ["SpectrumFitResult"]


class SpectrumFitResult:
    """Result of a `~gammapy.spectrum.SpectrumFit`.

    All fit results should be accessed via this class.

    Parameters
    ----------
    model : `~gammapy.spectrum.models.SpectralModel`
        Best-fit model
    fit_range : `~astropy.units.Quantity`
        Energy range of the spectral fit
    statname : str, optional
        Statistic used for the fit
    statval : float, optional
        Final fit statistic
    stat_per_bin : float, optional
        Fit statistic value per bin
    npred : array-like, optional
        Counts predicted by the fit
    obs : `~gammapy.spectrum.SpectrumObservation`
        Input data used for the fit
    """

    __slots__ = [
        "model",
        "fit_range",
        "statname",
        "statval",
        "stat_per_bin",
        "npred",
        "obs",
    ]

    def __init__(
        self,
        model,
        fit_range=None,
        statname=None,
        statval=None,
        stat_per_bin=None,
        npred=None,
        obs=None,
    ):
        self.model = model
        self.fit_range = fit_range
        self.statname = statname
        self.statval = statval
        self.stat_per_bin = stat_per_bin
        self.npred = npred
        self.obs = obs

    @classmethod
    def from_yaml(cls, filename):
        """Create from YAML file.

        Parameters
        ----------
        filename : str, Path
            File to read
        """
        filename = make_path(filename)
        val = read_yaml(str(filename))
        return cls.from_dict(val)

    def to_yaml(self, filename, mode="w"):
        """Write to YAML file.

        Parameters
        ----------
        filename : str
            File to write
        mode : str
            Write mode
        """
        d = self.to_dict()
        val = yaml.safe_dump(d, default_flow_style=False)

        with open(str(filename), mode) as outfile:
            outfile.write(val)

    def to_dict(self):
        """Convert to dict."""
        val = dict()
        val["model"] = self.model.to_dict()
        if self.fit_range is not None:
            val["fit_range"] = dict(
                min=self.fit_range[0].value,
                max=self.fit_range[1].value,
                unit=self.fit_range.unit.to_string("fits"),
            )
        if self.statval is not None:
            val["statval"] = float(self.statval)
        if self.statname is not None:
            val["statname"] = self.statname

        return val

    @classmethod
    def from_dict(cls, val):
        """Create from dict."""
        modeldict = val["model"]
        model = models.SpectralModel.from_dict(modeldict)
        try:
            erange = val["fit_range"]
            energy_range = u.Quantity([erange["min"], erange["max"]], erange["unit"])
        except KeyError:
            energy_range = None

        return cls(model=model, fit_range=energy_range)


    def __str__(self):
        s = "\nFit result info \n"
        s += "--------------- \n"
        s += "Model: {} \n".format(self.model)
        if self.statval is not None:
            s += "\nStatistic: {:.3f} ({})".format(self.statval, self.statname)
        if self.fit_range is not None:
            s += "\nFit Range: {}".format(self.fit_range)
        s += "\n"
        return s

    @property
    def expected_source_counts(self):
        """Predicted source counts (`~gammapy.spectrum.CountsSpectrum`)."""
        energy = self.obs.on_vector.energy
        data = self.npred
        return CountsSpectrum(data=data, energy_lo=energy.lo, energy_hi=energy.hi)




def get_plot_axis(**kwargs):
    """Axis setup used for standard plots.

    kwargs are forwarded to plt.figure()

    Returns
    -------
    ax0 : `~matplotlib.axes.Axes`
        Main plot
    ax1 : `~matplotlib.axes.Axes`
        Residuals
    """
    from matplotlib import gridspec
    import matplotlib.pyplot as plt

    fig = plt.figure(**kwargs)

    gs = gridspec.GridSpec(5, 1)

    ax0 = plt.subplot(gs[:-2, :])
    ax1 = plt.subplot(gs[3, :], sharex=ax0)

    gs.update(hspace=0.1)
    plt.setp(ax0.get_xticklabels(), visible=False)

    ax0.set_xscale("log")
    ax1.set_xscale("log")

    return ax0, ax1
