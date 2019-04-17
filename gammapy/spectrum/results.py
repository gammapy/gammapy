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

    # TODO: rather add this to Parameters?
    def to_table(self, energy_unit="TeV", flux_unit="cm-2 s-1 TeV-1", **kwargs):
        """Convert to `~astropy.table.Table`.

        Produce overview table containing the most important parameters
        """
        t = Table()
        t["model"] = [self.model.__class__.__name__]

        for par_name, value in self.model.parameters._ufloats.items():
            val = value.n
            err = value.s

            # Apply correction factor for units
            # TODO: Refactor
            current_unit = u.Unit(self.model.parameters[par_name].unit)
            if current_unit.is_equivalent(energy_unit):
                factor = current_unit.to(energy_unit)
                col_unit = energy_unit
            elif current_unit.is_equivalent(1 / u.Unit(energy_unit)):
                factor = current_unit.to(1 / u.Unit(energy_unit))
                col_unit = 1 / u.Unit(energy_unit)
            elif current_unit.is_equivalent(flux_unit):
                factor = current_unit.to(flux_unit)
                col_unit = flux_unit
            elif current_unit.is_equivalent(u.dimensionless_unscaled):
                factor = 1
                col_unit = current_unit
            else:
                raise ValueError(current_unit)

            t[par_name] = Column(
                data=np.atleast_1d(val * factor), unit=col_unit, **kwargs
            )
            t["{}_err".format(par_name)] = Column(
                data=np.atleast_1d(err * factor), unit=col_unit, **kwargs
            )

        t["fit_range"] = Column(
            data=[self.fit_range.to(energy_unit)], unit=energy_unit, **kwargs
        )

        return t

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

    def butterfly(self, energy=None, flux_unit="TeV-1 cm-2 s-1"):
        """Compute butterfly table.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`, optional
            Energies at which to evaluate the butterfly.
        flux_unit : str
            Flux unit for the butterfly.

        Returns
        -------
        table : `~astropy.table.Table`
            Butterfly info in table (cols: 'energy', 'flux', 'flux_lo', 'flux_hi')
        """
        if energy is None:
            energy = EnergyBounds.equal_log_spacing(
                self.fit_range[0], self.fit_range[1], 100
            )

        flux, flux_err = self.model.evaluate_error(energy)

        table = Table()
        table["energy"] = energy
        table["flux"] = flux.to(flux_unit)
        table["flux_lo"] = flux - flux_err.to(flux_unit)
        table["flux_hi"] = flux + flux_err.to(flux_unit)
        return table

    @property
    def expected_source_counts(self):
        """Predicted source counts (`~gammapy.spectrum.CountsSpectrum`)."""
        energy = self.obs.on_vector.energy
        data = self.npred
        return CountsSpectrum(data=data, energy_lo=energy.lo, energy_hi=energy.hi)

    # TODO: is this the quantity, and sign, we want for residuals?
    @property
    def residuals(self):
        """Residuals (predicted source - excess).
        """
        resspec = self.expected_source_counts.copy()
        resspec.data.data -= self.obs.excess_vector.data.data
        return resspec

    def plot(self, **kwargs):
        """Plot counts and residuals in two panels.

        Calls ``plot_counts`` and ``plot_residuals``.
        """
        ax0, ax1 = get_plot_axis(**kwargs)

        self.plot_counts(ax0)
        self.plot_residuals(ax1)

        return ax0, ax1

    def plot_counts(self, ax):
        """Plot predicted and detected counts."""
        self.expected_source_counts.plot(ax=ax, label="mu_src")

        self.obs.excess_vector.plot(ax=ax, label="excess", fmt=".", energy_unit="TeV")

        ax.axvline(
            self.fit_range.to_value("TeV")[0],
            color="black",
            linestyle="dashed",
            label="fit range",
        )

        ax.axvline(self.fit_range.to_value("TeV")[1], color="black", linestyle="dashed")

        ax.legend(numpoints=1)
        ax.set_title("")

    def plot_residuals(self, ax):
        """Plot residuals."""
        self.residuals.plot(ax=ax, ecolor="black", fmt="none")
        ax.axhline(color="black")

        ymax = 1.4 * max(self.residuals.data.data.value)
        ax.set_ylim(-ymax, ymax)

        ax.set_xlabel("Energy [{}]".format("TeV"))
        ax.set_ylabel("ON (Predicted - Detected)")



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
