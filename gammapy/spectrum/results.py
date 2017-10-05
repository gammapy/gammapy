# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.table import Table, Column
import astropy.units as u
from ..spectrum import CountsSpectrum, models
from ..utils.scripts import read_yaml, make_path
from ..utils.energy import EnergyBounds

__all__ = [
    'SpectrumFitResult',
    'SpectrumResult',
]


class SpectrumFitResult(object):
    """Class representing the result of a spectral fit

    TODO: This only supports WStat fits at the moment. Find a solution to
    display also Cash fits (flag, subclass).

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
    npred_src : array-like, optional
        Source counts predicted by the fit
    npred_bkg : array-like, optional
        Background counts predicted by the fit
    background_model : `~gammapy.spectrum.SpectralModel`
        Best-fit background model
    flux_at_1TeV : dict, optional
        Flux for the fitted model at 1 TeV
    flux_at_1TeV_err : dict, optional
        Error on the flux for the fitted model at 1 TeV
    obs : `~gammapy.spectrum.SpectrumObservation`
        Input data used for the fit
    """

    def __init__(self, model, fit_range=None, statname=None, statval=None,
                 stat_per_bin=None, npred_src=None, npred_bkg=None,
                 background_model=None, fluxes=None, flux_errors=None,
                 obs=None):

        self.model = model
        self.fit_range = fit_range
        self.statname = statname
        self.statval = statval
        self.stat_per_bin = stat_per_bin
        self.npred_src = npred_src
        self.npred_bkg = npred_bkg
        self.background_model = background_model
        self.fluxes = fluxes
        self.flux_errors = flux_errors
        self.obs = obs

    @classmethod
    def from_yaml(cls, filename):
        """Create cls from YAML file

        Parameters
        ----------
        filename : str, Path
            File to read
        """
        filename = make_path(filename)
        val = read_yaml(str(filename))
        return cls.from_dict(val)

    def to_yaml(self, filename, mode='w'):
        """Write YAML file

        Parameters
        ----------
        filename : str
            File to write
        mode : str
            Write mode
        """
        import yaml

        d = self.to_dict()
        val = yaml.safe_dump(d, default_flow_style=False)

        with open(str(filename), mode) as outfile:
            outfile.write(val)

    def to_dict(self):
        """Convert to dict"""
        val = dict()
        val['model'] = self.model.to_dict()
        if self.fit_range is not None:
            val['fit_range'] = dict(min=self.fit_range[0].value,
                                    max=self.fit_range[1].value,
                                    unit=str(self.fit_range.unit))
        if self.statval is not None:
            val['statval'] = float(self.statval)
        if self.statname is not None:
            val['statname'] = self.statname

        return val

    @classmethod
    def from_dict(cls, val):
        modeldict = val['model']
        model = models.SpectralModel.from_dict(modeldict)
        try:
            erange = val['fit_range']
            energy_range = (erange['min'], erange['max']) * u.Unit(erange['unit'])
        except KeyError:
            energy_range = None

        try:
            fl = val['fluxes']
        except KeyError:
            fluxes = None
            flux_errors = None
        else:
            fluxes = dict()
            flux_errors = dict()
            for flu in fl:
                fluxes[flu] = fl[flu]['value'] * u.Unit(fl[flu]['unit'])
                flux_errors[flu] = fl[flu]['error'] * u.Unit(fl[flu]['unit'])

        return cls(model=model,
                   fit_range=energy_range,
                   fluxes=fluxes,
                   flux_errors=flux_errors)

    # TODO: rather add this to ParameterList?
    def to_table(self, energy_unit='TeV', flux_unit='cm-2 s-1 TeV-1', **kwargs):
        """Convert to `~astropy.table.Table`

        Produce overview table containing the most important parameters
        """
        t = Table()
        t['model'] = [self.model.__class__.__name__]
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
                data=np.atleast_1d(val * factor),
                unit=col_unit,
                **kwargs)
            t['{}_err'.format(par_name)] = Column(
                data=np.atleast_1d(err * factor),
                unit=col_unit,
                **kwargs)

        t['fit_range'] = Column(
            data=[self.fit_range.to(energy_unit)],
            unit=energy_unit,
            **kwargs)

        return t

    def __str__(self):
        """
        Summary info string.
        """
        info = '\nFit result info \n'
        info += '--------------- \n'
        info += 'Model: {} \n'.format(self.model)
        if self.statval is not None:
            info += '\nStatistic: {0:.3f} ({1})'.format(self.statval, self.statname)
        if self.fit_range is not None:
            info += '\nFit Range: {}'.format(self.fit_range)
        info += '\n'
        return info

    def info(self):
        """
        Print summary info.
        """
        print(str(self))

    def butterfly(self, energy=None, flux_unit='TeV-1 cm-2 s-1'):
        """
        Compute butterfly table.

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
            energy = EnergyBounds.equal_log_spacing(self.fit_range[0],
                                                    self.fit_range[1], 100)

        flux, flux_err = self.model.evaluate_error(energy)

        table = Table()
        table['energy'] = energy
        table['flux'] = flux.to(flux_unit)
        table['flux_lo'] = flux - flux_err.to(flux_unit)
        table['flux_hi'] = flux + flux_err.to(flux_unit)
        return table

    @property
    def expected_source_counts(self):
        """`~gammapy.spectrum.CountsSpectrum` of predicted source counts
        """
        energy = self.obs.on_vector.energy
        data = self.npred_src * u.ct
        return CountsSpectrum(data=data, energy_lo=energy.lo,
                              energy_hi=energy.hi)

    @property
    def expected_background_counts(self):
        """`~gammapy.spectrum.CountsSpectrum` of predicted background counts
        """
        try:
            energy = self.obs.e_reco
            data = self.npred_bkg * u.ct
            return CountsSpectrum(data=data, energy_hi=energy.upper_bounds,
                                  energy_lo=energy.lower_bounds)
        except TypeError:
            return None

    @property
    def expected_on_counts(self):
        """`~gammapy.spectrum.CountsSpectrum` of predicted on counts
        """
        mu_on = self.expected_source_counts.copy()
        mu_on.data.data += self.expected_background_counts.data.data
        return mu_on

    @property
    def residuals(self):
        """Residuals

        Prediced on counts - expected on counts
        """
        resspec = self.expected_on_counts.copy()
        resspec.data.data -= self.obs.on_vector.data.data
        return resspec

    def plot(self, **kwargs):
        """Standard debug plot.

        Plot ON counts in comparison to model.
        """
        ax0, ax1 = get_plot_axis(**kwargs)

        self.plot_counts(ax0)
        self.plot_residuals(ax1)

        return ax0, ax1

    def plot_counts(self, ax):
        """Plot predicted and detected counts."""
        self.expected_source_counts.plot(ax=ax,
                                         label='mu_source')

        self.expected_background_counts.plot(ax=ax,
                                             label='mu_background',
                                             energy_unit='TeV')

        self.expected_on_counts.plot(ax=ax, label='mu_on', energy_unit='TeV')

        self.obs.on_vector.plot(ax=ax,
                                label='n_on',
                                show_poisson_errors=True,
                                fmt='.',
                                energy_unit='TeV')

        ax.legend(numpoints=1)
        ax.set_title('')

    def plot_residuals(self, ax):
        """Plot residuals."""
        self.residuals.plot(ax=ax, ecolor='black', fmt='none')
        xx = ax.get_xlim()
        yy = [0, 0]
        ax.plot(xx, yy, color='black')

        ymax = 1.4 * max(self.residuals.data.data.value)
        ax.set_ylim(-ymax, ymax)

        xmin = self.fit_range.to('TeV').value[0] * 0.8
        xmax = self.fit_range.to('TeV').value[1] * 1.2
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel('Energy [{}]'.format('TeV'))
        ax.set_ylabel('ON (Predicted - Detected)')


class SpectrumResult(object):
    """Class holding all results of a spectral analysis

    Best fit model, flux points

    Parameters
    ----------
    model : `~gammapy.spectrum.models.SpectralModel`
        Best Fit model
    points : `~gammapy.spectrum.FluxPoints`, optional
        Flux points
    """

    def __init__(self, model, points):
        self.model = model
        self.points = points

    @property
    def flux_point_residuals(self):
        """Residuals

        Defined as ``(points - model) / model``

        Returns
        -------
        residuals : np.array
            Residuals
        residuals_err : np.array
            Residuals error
        """
        e_ref = self.points.table['e_ref'].quantity
        points = self.points.table['dnde'].quantity
        points_err = self.points.get_flux_err()

        # Deal with asymetric errors
        if type(points_err) == tuple:
            points_err = np.sqrt(points_err[0] * points_err[1])

        model_val = self.model(e_ref)
        residuals = ((points - model_val) / model_val).to('').value
        residuals_err = (points_err / model_val).to('').value

        return residuals, residuals_err

    def plot(self, energy_range, energy_unit='TeV', flux_unit='cm-2 s-1 TeV-1',
             energy_power=0, fit_kwargs=dict(),
             butterfly_kwargs=dict(), point_kwargs=dict(), fig_kwargs=dict()):
        """Plot spectrum

        Plot best fit model, flux points and residuals

        Parameters
        ----------
        energy_range : `~astropy.units.Quantity`
            Energy range for the plot
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        flux_unit : str, `~astropy.units.Unit`, optional
            Unit of the flux axis
        energy_power : int
            Power of energy to multiply flux axis with
        fit_kwargs : dict, optional
            forwarded to :func:`gammapy.spectrum.models.SpectralModel.plot`
        butterfly_kwargs : dict, optional
            forwarded to :func:`gammapy.spectrum.SpectralModel.plot_error`
        point_kwargs : dict, optional
            forwarded to :func:`gammapy.spectrum.FluxPoints.plot`
        fig_kwargs : dict, optional
            forwarded to :func:`matplotlib.pyplot.figure`

        Returns
        -------
        ax0 : `~matplotlib.axes.Axes`
            Spectrum plot axis
        ax1 : `~matplotlib.axes.Axes`
            Residuals plot axis
        """
        import matplotlib.pyplot as plt

        ax0, ax1 = get_plot_axis(**fig_kwargs)
        ax0.set_yscale('log')

        common_kwargs = dict(
            energy_unit=energy_unit,
            flux_unit=flux_unit,
            energy_power=energy_power)
        fit_kwargs.update(common_kwargs)
        point_kwargs.update(common_kwargs)
        butterfly_kwargs.update(common_kwargs)

        self.model.plot(energy_range=energy_range,
                        ax=ax0,
                        **fit_kwargs)
        self.model.plot_error(energy_range=energy_range,
                              ax=ax0,
                              **butterfly_kwargs)
        self.points.plot(ax=ax0,
                         **point_kwargs)
        point_kwargs.pop('flux_unit')
        point_kwargs.pop('energy_power')
        ax0.set_xlabel('')
        self._plot_residuals(ax=ax1,
                             **point_kwargs)

        return ax0, ax1

    def _plot_residuals(self, ax=None, energy_unit='TeV', **kwargs):
        """Plot residuals

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault('fmt', '.')

        y, y_err = self.flux_point_residuals
        x = self.points.e_ref
        x = x.to(energy_unit).value
        ax.errorbar(x, y, yerr=y_err, **kwargs)

        ax.axhline(0, color='black')

        ax.set_xlabel('Energy [{}]'.format(energy_unit))
        ax.set_ylabel('Residuals')

        return ax


def get_plot_axis(**kwargs):
    """Axis setup used for standard plots

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

    ax0.set_xscale('log')
    ax1.set_xscale('log')

    return ax0, ax1
