# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.utils.decorators import lazyproperty
from astropy.table import Table, Column
import astropy.units as u
from .butterfly import SpectrumButterfly
from ..spectrum import CountsSpectrum, models
from ..extern.bunch import Bunch
from ..utils.scripts import read_yaml, make_path
from ..utils.energy import EnergyBounds

__all__ = [
    'SpectrumFitResult',
    'SpectrumResult',
]


class SpectrumFitResult(object):
    """Class representing the result of a spectral fit

    Parameters
    ----------
    model : `~gammapy.spectrum.models.SpectralModel`
        Best-fit model
    covariance : array-like, optional
        Covariance matrix
    covar_axis : list, optional
        List of strings defining the parameter order in covariance
    fit_range : `~astropy.units.Quantity`
        Energy range of the spectral fit
    statname : str, optional
        Statistic used for the fit
    statval : float, optional
        Final fit statistic
    npred : array-like, optional
        On counts predicted by the fit
    flux_at_1TeV : dict, optional
        Flux for the fitted model at 1 TeV
    flux_at_1TeV_err : dict, optional
        Error on the flux for the fitted model at 1 TeV
    obs : `~gammapy.spectrum.SpectrumObservation`
        Input data used for the fit
    """

    def __init__(self, model, covariance=None, covar_axis=None, fit_range=None,
                 statname=None, statval=None, npred=None, fluxes=None,
                 flux_errors=None, obs=None):

        self.model = model
        self.covariance = covariance
        self.covar_axis = covar_axis
        self.fit_range = fit_range
        self.statname = statname
        self.statval = statval
        self.npred = npred
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
            val['statval'] = self.statval
        if self.statname is not None:
            val['statname'] = self.statname
        if self.covariance is not None:
            val['covariance'] = dict(matrix=self.covariance.tolist(),
                                     axis=self.covar_axis)
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
            fluxes = Bunch()
            flux_errors = Bunch()
            for flu in fl:
                fluxes[flu] = fl[flu]['value'] * u.Unit(fl[flu]['unit'])
                flux_errors[flu] = fl[flu]['error'] * u.Unit(fl[flu]['unit'])
        try:
            covar = val['covariance']
            covar_axis = covar['axis']
            covariance = np.array(covar['matrix'])
        except KeyError:
            covar_axis = None
            covariance = None

        return cls(model=model,
                   fit_range=energy_range,
                   fluxes=fluxes,
                   flux_errors=flux_errors,
                   covar_axis=covar_axis,
                   covariance=covariance)

    def to_table(self, energy_unit='TeV', flux_unit='cm-2 s-1 TeV-1', **kwargs):
        """Convert to `~astropy.table.Table`

        Produce overview table containing the most important parameters
        """
        t = Table()
        t['model'] = [self.model.__class__.__name__]
        for parname, par in self.model_with_uncertainties.parameters.items():
            try:
                val = par.n
                err = par.s
            except AttributeError:
                val = par
                err = 0

            # Apply correction factor for units
            # TODO: Refactor
            current_unit = self.model.parameters[parname].unit
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

            t[parname] = Column(
                data=np.atleast_1d(val * factor),
                unit=col_unit,
                **kwargs)
            t['{}_err'.format(parname)] = Column(
                data=np.atleast_1d(err * factor),
                unit=col_unit,
                **kwargs)

        t['fit_range'] = Column(
            data=[self.fit_range.to(energy_unit)],
            unit=energy_unit,
            **kwargs)

        return t

    @lazyproperty
    def model_with_uncertainties(self):
        """Best fit model with uncertainties

        The parameters on the model will have the units of the model attribute.
        The covariance matrix passed on initialization must also have these
        units.

        TODO: Add to gammapy.spectrum.models

        This function uses the uncertainties packages as explained here
        https://pythonhosted.org/uncertainties/user_guide.html#use-of-a-covariance-matrix

        Examples
        --------
        TODO
        """
        import uncertainties
        pars = self.model.parameters

        # convert existing parameters to ufloats
        values = [pars[_].value for _ in self.covar_axis]
        ufloats = uncertainties.correlated_values(values, self.covariance)
        upars = dict(zip(self.covar_axis, ufloats))

        # add parameters missing in covariance
        for name in pars:
            upars.setdefault(name, pars[name].value)

        return self.model.__class__(**upars)

    def __str__(self):
        """
        Summary info string.
        """
        info = '\nFit result info \n'
        info += '--------------- \n'
        info += 'Model: {} \n'.format(self.model_with_uncertainties)
        if self.statval is not None:
            info += '\nStatistic: {0:.3f} ({1})'.format(self.statval, self.statname)
        if self.covariance is not None:
            info += '\nCovariance:\n{}\n{}'.format(self.covar_axis, self.covariance)
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
        Compute butterfly.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`, optional
            Energies at which to evaluate the butterfly.
        flux_unit : str
            Flux unit for the butterfly.

        Returns
        -------
        butterfly : `~gammapy.spectrum.SpectrumButterfly`
            Butterfly object.
        """
        from uncertainties import unumpy

        if energy is None:
            energy = EnergyBounds.equal_log_spacing(self.fit_range[0],
                                                    self.fit_range[1],
                                                    100)
        flux = self.model(energy)

        butterfly = SpectrumButterfly()
        butterfly['energy'] = energy
        butterfly['flux'] = flux.to(flux_unit)

        # compute uncertainties
        umodel = self.model_with_uncertainties

        if self.model.__class__.__name__ == 'PowerLaw2':
            energy_unit = self.model.parameters.emin.unit
        else:
            energy_unit = self.model.parameters.reference.unit

        values = umodel(energy.to(energy_unit).value)

        # unit conversion factor, in case it doesn't match
        conversion_factor = flux.to(flux_unit).value / unumpy.nominal_values(values)
        flux_err = u.Quantity(unumpy.std_devs(values), flux_unit) * conversion_factor

        butterfly['flux_lo'] = flux - flux_err
        butterfly['flux_hi'] = flux + flux_err
        return butterfly

    @property
    def expected_source_counts(self):
        """`~gammapy.spectrum.CountsSpectrum` of predicted counts
        """
        energy = self.obs.on_vector.energy
        data = self.npred * u.ct
        idx = np.isnan(data)
        data[idx] = 0
        return CountsSpectrum(data=data, energy=energy)

    def plot(self, mode='wstat'):
        """Standard debug plot.

        Plot ON counts in comparison to model. The model can contain predicted
        source and background counts (CStat) or prediced source counts plus a
        background estimate from off regions (WStat). The ``mode`` parameter
        controls this.
        """
        if mode != 'wstat':
            raise NotImplementedError('Mode {}'.format(mode))

        ax0, ax1 = get_plot_axis()

        self.expected_source_counts.plot(ax=ax0,
                                         fmt='none',
                                         label='mu_source')

        self.obs.background_vector.plot(ax=ax0,
                                        label='mu_background',
                                        fmt='none',
                                        energy_unit='TeV')

        mu_on = self.expected_source_counts.copy()
        mu_on.data = self.expected_source_counts.data + self.obs.background_vector.data
        mu_on.plot(ax=ax0, label='mu_on', energy_unit='TeV')

        self.obs.on_vector.plot(ax=ax0,
                                label='n_on',
                                show_poisson_errors=True,
                                fmt='none',
                                energy_unit='TeV')

        ax0.legend(numpoints=1)
        ax0.set_title('')

        resspec = mu_on.copy()
        resspec.data = mu_on.data - self.obs.on_vector.data
        resspec.plot(ax=ax1, ecolor='black', fmt='none')
        xx = ax1.get_xlim()
        yy = [0, 0]
        ax1.plot(xx, yy, color='black')

        ymax = 1.4 * max(resspec.data.value)
        ax1.set_ylim(-ymax, ymax)

        xmin = self.fit_range.to('TeV').value[0] * 0.8
        xmax = self.fit_range.to('TeV').value[1] * 1.2
        ax1.set_xlim(xmin, xmax)
        ax1.set_xlabel('Energy [{}]'.format('TeV'))
        ax1.set_ylabel('ON (Predicted - Detected)')

        return ax0, ax1


class SpectrumResult(object):
    """Class holding all results of a spectral analysis

    Best fit model, flux points

    TODO: Rewrite once `~gammapy.spectrum.models.SpectralModel` can hold
    covariance matrix

    Parameters
    ----------
    fit : `~SpectrumFitResult`
        Spectrum fit result holding best fit model
    points : `~gammapy.spectrum.DifferentialFluxPoints`, optional
        Flux points
    """

    def __init__(self, fit=None, obs=None, points=None):
        self.fit = fit
        self.points = points

    @property
    def flux_point_residuals(self):
        """Residuals

        Based on best fit model and fluxpoints.
        Defined as ``(points - model)/model``

        Returns
        -------
        residuals : `~uncertainties.ufloat`
            Residuals
        """
        from uncertainties import ufloat
        # Get units right
        pars = self.fit.model.parameters
        if self.fit.model.__class__.__name__ == 'PowerLaw2':
            energy_unit = pars.emin.unit
            flux_unit = pars.amplitude.unit / energy_unit
        else:
            energy_unit = pars.reference.unit
            flux_unit = pars.amplitude.unit

        x = self.points['ENERGY'].quantity.to(energy_unit)
        y = self.points['DIFF_FLUX'].quantity.to(flux_unit)
        y_err = self.points['DIFF_FLUX_ERR_HI'].quantity.to(flux_unit)

        points = list()
        for val, err in zip(y.value, y_err.value):
            points.append(ufloat(val, err))

        func = self.fit.model_with_uncertainties(x.value)
        residuals = (points - func) / func

        return residuals

    def plot(self, energy_range, energy_unit='TeV', flux_unit='cm-2 s-1 TeV-1',
             energy_power=0, fit_kwargs=dict(),
             butterfly_kwargs=dict(), point_kwargs=dict()):
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
            forwarded to :func:`gammapy.spectrum.SpectrumButterfly.plot`
        point_kwargs : dict, optional
            forwarded to :func:`gammapy.spectrum.DifferentialFluxPoints.plot`

        Returns
        -------
        ax0 : `~matplotlib.axes.Axes`
            Spectrum plot axis
        ax1 : `~matplotlib.axes.Axes`
            Residuals plot axis
        """
        import matplotlib.pyplot as plt

        ax0, ax1 = get_plot_axis()
        ax0.set_yscale('log')

        fit_kwargs.setdefault('lw', '2')
        fit_kwargs.setdefault('color', 'navy')
        point_kwargs.setdefault('marker', '.')
        point_kwargs.setdefault('color', 'navy')
        butterfly_kwargs.setdefault('facecolor', 'darkblue')
        butterfly_kwargs.setdefault('alpha', '0.5')
        common_kwargs = dict(
            energy_unit=energy_unit,
            flux_unit=flux_unit,
            energy_power=energy_power)
        fit_kwargs.update(common_kwargs)
        point_kwargs.update(common_kwargs)
        butterfly_kwargs.update(common_kwargs)

        self.fit.model.plot(energy_range=energy_range,
                            ax=ax0,
                            **fit_kwargs)

        energy = EnergyBounds.equal_log_spacing(energy_range[0],
                                                energy_range[1],
                                                100)
        self.fit.butterfly(energy=energy).plot(ax=ax0, **butterfly_kwargs)
        self.points.plot(ax=ax0,
                         **point_kwargs)
        point_kwargs.pop('flux_unit')
        point_kwargs.pop('energy_power')
        ax0.set_xlabel('')
        self._plot_residuals(ax=ax1,
                             **point_kwargs)

        plt.xlim(energy_range[0].to(energy_unit).value * 0.9,
                 energy_range[1].to(energy_unit).value * 1.1)

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

        kwargs.setdefault('fmt', 'o')

        res = self.flux_point_residuals
        y = [_.n for _ in res]
        y_err = [_.s for _ in res]
        x = self.points['ENERGY'].quantity
        x = x.to(energy_unit).value
        ax.errorbar(x, y, yerr=y_err, **kwargs)

        xx = ax.get_xlim()
        yy = [0, 0]
        ax.plot(xx, yy, color='black')

        ax.set_xlabel('Energy [{}]'.format(energy_unit))
        ax.set_ylabel('(Points - Model) / Model')

        return ax


def get_plot_axis(figsize=(15, 10)):
    """Axis setup used for standard plots

    Returns
    -------
    ax0 : `~matplotlib.axes.Axes`
        Main plot
    ax1 : `~matplotlib.axes.Axes`
        Residuals
    """
    from matplotlib import gridspec
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(5, 1)

    ax0 = plt.subplot(gs[:-2, :])
    ax1 = plt.subplot(gs[3, :], sharex=ax0)

    gs.update(hspace=0.1)
    plt.setp(ax0.get_xticklabels(), visible=False)

    ax0.set_xscale('log')
    ax1.set_xscale('log')

    gs.tight_layout(fig)

    return ax0, ax1
