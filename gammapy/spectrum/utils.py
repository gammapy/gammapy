# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from gammapy.data import ObservationStats

__all__ = [
    'LogEnergyAxis',
    'calculate_predicted_counts',
    'calculate_flux_point_binning',
    'integrate_spectrum',
]


class LogEnergyAxis(object):
    """Log10 energy axis.

    Defines a transformation between:

    * ``energy = 10 ** x``
    * ``x = log10(energy)``
    * ``pix`` in the range [0, ..., len(x)] via linear interpolation of the ``x`` array,
      e.g. ``pix=0`` corresponds to ``x[0]``
      and ``pix=0.3`` is ``0.5 * (0.3 * x[0] + 0.7 * x[1])``

    .. note::
        The `specutils.Spectrum1DLookupWCS <http://specutils.readthedocs.io/en/latest/api/specutils.wcs.specwcs.Spectrum1DLookupWCS.html>`__
        class is similar (only that it doesn't include the ``log`` transformation and the API is different.
        Also see this Astropy feature request: https://github.com/astropy/astropy/issues/2362

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy array
    """

    def __init__(self, energy):
        self.energy = energy
        self.x = np.log10(energy.value)
        self.pix = np.arange(len(self.x))

    def world2pix(self, energy):
        """TODO: document.
        """
        # Convert `energy` to `x = log10(energy)`
        x = np.log10(energy.to(self.energy.unit).value)

        # Interpolate in `x`
        pix = np.interp(x, self.x, self.pix)

        return pix

    def pix2world(self, pix):
        """TODO: document.
        """
        # Interpolate in `x = log10(energy)`
        x = np.interp(pix, self.pix, self.x)

        # Convert `x` to `energy`
        energy = Quantity(10 ** x, self.energy.unit)

        return energy

    def closest_point(self, energy):
        """TODO: document
        """
        x = np.log10(energy.value)
        # TODO: I'm not sure which is faster / better here?
        index = np.argmin(np.abs(self.x - x))
        # np.searchsorted(self.x, x)
        return index

    def bin_edges(self, energy):
        """TODO: document.

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """
        try:
            pix = np.where(energy >= self.energy)[0][-1]
        except ValueError:
            # Loop over es by hand
            pix1 = np.empty_like(energy, dtype=int)
            for ii in range(energy.size):
                # print ii, e[ii], np.where(e[ii] >= self.e)
                pix1[ii] = np.where(energy[ii] >= self.energy)[0][-1]
        pix2 = pix1 + 1
        energy1 = self.energy[pix1]
        energy2 = self.energy[pix2]

        return pix1, pix2, energy1, energy2


def calculate_predicted_counts(model, aeff, edisp, livetime, e_reco=None):
    """Get npred 

    The true energy binning is inferred from the provided
    `~gammapy.irf.EffectiveAreaTable`. The reco energy binning can be inferred
    from the `~gammapy.irf.EnergyDispersion` or be given as a parameter.

    Parameters
    ----------
    model : `~gammapy.spectrum.models.SpectralModel`
        Spectral model
    livetime : `~astropy.units.Quantity`
        Observation duration
    aeff : `~gammapy.irf.EffectiveAreaTable`
        EffectiveArea
    edisp : `~gammapy.irf.EnergyDispersion`
        EnergyDispersion
    e_reco : `~astropy.units.Quantity`, optional
        Desired energy axis of the prediced counts vector By default, the reco
        energy axis of the energy dispersion matrix is used.

    Returns
    -------
    counts : `~gammapy.spectrum.CountsSpectrum`
        Predicted counts

    Examples
    --------
    Calculate prediced counts in a desired reconstruced energy binning 

    .. plot::
        :include-source:

        from gammapy.irf import EnergyDispersion, EffectiveAreaTable
        from gammapy.spectrum import models, calculate_predicted_counts
        import numpy as np
        import astropy.units as u
        import matplotlib.pyplot as plt

        e_true = np.logspace(-2,2.5,109) * u.TeV
        e_reco = np.logspace(-2,2,73) * u.TeV

        aeff = EffectiveAreaTable.from_parametrization(energy=e_true)
        edisp = EnergyDispersion.from_gauss(e_true=e_true, e_reco=e_reco,
                                            sigma=0.3)

        model = models.PowerLaw(index=2.3,
                                amplitude=2.5 * 1e-12 * u.Unit('cm-2 s-1 TeV-1'),
                                reference=1*u.TeV)

        livetime = 1 * u.h
        e_reco_desired = np.logspace(-1, 1, 15) * u.TeV

        npred = calculate_predicted_counts(model=model,
                                           aeff=aeff,
                                           edisp=edisp,
                                           livetime=livetime,
                                           e_reco=e_reco_desired)

        npred.plot_hist()
        plt.show()
    """
    from . import CountsSpectrum

    true_energy = aeff.energy.data.to('TeV')
    flux = model.integral(emin=true_energy[:-1], emax=true_energy[1:])

    # Need to fill nan values in aeff due to matrix multiplication with RMF
    counts = flux * livetime * aeff.evaluate(fill_nan=True)
    counts = counts.to('')
    reco_counts = edisp.apply(counts, e_reco=e_reco)
    e_reco = e_reco or edisp.e_reco.data
    return CountsSpectrum(data=reco_counts, energy=e_reco)


def calculate_flux_point_binning(obs_list, min_signif):
    """Compute energy binning

    This is usefule to get an energy binning to use with
    :func:`~gammapy.spectrum.DifferentialFluxPoints.compute` Each bin in the
    resulting energy binning will include a ``min_signif`` source detection. 

    TODO: It is required that at least two fine bins be included in one
    flux point interval, otherwise the sherpa covariance method breaks
    down.

    Parameters
    ----------
    obs_list : `~gammapy.spectrum.SpectrumObservationList`
        Observations
    min_signif : float
        Required significance for each bin

    Returns
    -------
    binning : `~astropy.units.Quantity` 
        Energy binning
    """
    from . import SpectrumObservation
    # NOTE: Results may vary from FitSpectrum since there the rebin
    # parameter can only have fixed values, here it grows linearly. Also it
    # has to start at 2 here (see docstring)

    # rebin_factor = 1
    rebin_factor = 2
    
    obs = SpectrumObservation.stack(obs_list)

    # First first bin above low threshold and last bin below high threshold
    current_ebins = obs.on_vector.energy
    current_bin = (current_ebins.find_node(obs.lo_threshold) + 1)[0]
    max_bin = (current_ebins.find_node(obs.hi_threshold))[0]

    # List holding final energy binning
    binning = [current_ebins.data[current_bin]]

    # Precompute ObservationStats for each bin
    obs_stats = [obs.stats(i) for i in range(current_ebins.nbins)]
    while(current_bin + rebin_factor <= max_bin):
        # Merge bins together until min_signif is reached
        stats_list = obs_stats[current_bin:current_bin + rebin_factor:1]
        stats = ObservationStats.stack(stats_list)
        sigma = stats.sigma
        if (sigma < min_signif or np.isnan(sigma)):
            rebin_factor += 1
            continue

        # Append upper bin edge of good energy bin to final binning
        binning.append(current_ebins.data[current_bin + rebin_factor])
        current_bin += rebin_factor

    binning = Quantity(binning)
    # Replace highest bin edge by high threshold
    binning[-1] = obs.hi_threshold

    return binning


def integrate_spectrum(func, xmin, xmax, ndecade=100, intervals=False):
    """
    Integrate 1d function using the log-log trapezoidal rule. If scalar values
    for xmin and xmax are passed an oversampled grid is generated using the
    ``ndecade`` keyword argument. If xmin and xmax arrays are passed, no
    oversampling is performed and the integral is computed in the provided
    grid.

    Parameters
    ----------
    func : callable
        Function to integrate.
    xmin : `~astropy.units.Quantity` or array-like
        Integration range minimum
    xmax : `~astropy.units.Quantity` or array-like 
        Integration range minimum
    ndecade : int, optional
        Number of grid points per decade used for the integration.
        Default : 100.
    intervals : bool, optional
        Return integrals in the grid not the sum, default: False
    """
    is_quantity = False
    if isinstance(xmin, Quantity):
        unit = xmin.unit
        xmin = xmin.value
        xmax = xmax.value
        is_quantity = True

    if np.isscalar(xmin):
        logmin = np.log10(xmin)
        logmax = np.log10(xmax)
        n = (logmax - logmin) * ndecade
        x = np.logspace(logmin, logmax, n)
    else:
        x = np.append(xmin, xmax[-1])

    if is_quantity:
        x = x * unit

    y = func(x)

    val = _trapz_loglog(y, x, ulog10=True, intervals=intervals)

    return val


# This function is copied over from https://github.com/zblz/naima/blob/master/naima/utils.py#L261
# and slightly modified to allow use with the uncertainties package

def _trapz_loglog(y, x, axis=-1, intervals=False, ulog10=False):
    """
    Integrate along the given axis using the composite trapezoidal rule in
    loglog space.

    Integrate `y` (`x`) along given axis in loglog space.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis.
    intervals : bool, optional
        Return array of shape x not the total integral, default: False
    ulog10 : bool, optional
        Use `~uncertainties.unumpy.log10` to allow uarrays for y and do error
        propagation for the integral value.

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    log10 = np.log10

    if ulog10:
        from uncertainties.unumpy import log10

    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.

    y = np.asanyarray(y)
    x = np.asanyarray(x)

    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)

    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    with np.errstate(invalid='ignore', divide='ignore'):
        # Compute the power law indices in each integration bin
        b = log10(y[slice2] / y[slice1]) / log10(x[slice2] / x[slice1])

        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use normal
        # powerlaw integration
        trapzs = np.where(
            np.abs(b + 1.) > 1e-10, (y[slice1] * (
                x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1])) / (b + 1),
            x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]))

    tozero = (y[slice1] == 0.) + (y[slice2] == 0.) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.

    if intervals:
        return trapzs * x_unit * y_unit

    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit

    return ret
