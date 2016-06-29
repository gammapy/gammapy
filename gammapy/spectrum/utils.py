# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from astropy.units import Quantity

from ..utils.scripts import read_yaml

__all__ = [
    'LogEnergyAxis',
    'plot_npred_vs_excess',
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


# Todo: find a better place for these functions (Spectrum analysis class?)
def plot_exclusion_mask(**kwargs):
    """Plot exclusion mask

    The plot will be centered at the pointing position

    Parameters
    ----------
    size : `~astropy.coordinates.Angle`
    Edge length of the plot
    """
    from gammapy.image import ExclusionMask
    from gammapy.spectrum import SpectrumExtraction
    # Todo: plot exclusion mask as contours with skymap class

    exclusion = ExclusionMask.from_fits(SpectrumExtraction.EXCLUDEDREGIONS_FILE)
    ax = exclusion.plot(**kwargs)
    return ax


def plot_npred_vs_excess(ogip_dir='ogip_data', npred_dir='n_pred', ax=None):
    """Plot predicted and measured excess counts

    Parameters
    ----------
    npred_dir : str, Path
        Directory holding npred fits files
    ogip_dir : str, Path
        Directory holding OGIP data
    """
    from ..spectrum.spectrum_extraction import SpectrumObservationList
    from ..spectrum import CountsSpectrum
    from ..utils.scripts import make_path

    import matplotlib.pyplot as plt
    ax = plt.gca() if ax is None else ax

    ogip_dir = make_path(ogip_dir)
    n_pred_dir = make_path(npred_dir)

    obs = SpectrumObservationList.read_ogip(ogip_dir)
    excess = np.sum([o.excess_vector for o in obs])

    # Need to give RMF file for reco energy binning
    id = obs[0].meta.obs_id
    rmf = str(ogip_dir/ 'rmf_run{}.fits'.format(id))
    val = [CountsSpectrum.read_bkg(_, rmf) for _ in n_pred_dir.glob('*.fits')]
    npred = np.sum(val)

    npred.plot(ax=ax, color='red', alpha=0.7, label='Predicted counts')
    excess.plot(ax=ax, color='green', alpha=0.7, label='Excess counts')
    ax.legend(numpoints=1)
    plt.xscale('log')

    return ax


def integrate_spectrum(func, xmin, xmax, ndecade=100, **kwargs):
    """
    Integrate 1d function using the log-log trapezoidal rule. 
    
    Parameters
    ----------
    func : callable
        Function to integrate.
    xmin : `~astropy.units.Quantity` or float
        Integration range minimum
    xmax : `~astropy.units.Quantity` or float
        Integration range minimum
    ndecade : int
        Number of grid points per decade used for the integration.
        Default ndecade = 100.
    kwargs : dict
        Keyword arguments passed to `trapz_loglog`
    """   
    try:
        logmin = np.log10(xmin.value)
        logmax = np.log10(xmax.to(xmin.unit).value)
        n = (logmax - logmin) * ndecade
        x = Quantity(np.logspace(logmin, logmax, n), xmin.unit)
        y = func(x)
        val = _trapz_loglog(y, x, **kwargs)
    except AttributeError:
        logmin = np.log10(xmin)
        logmax = np.log10(xmax)
        n = (logmax - logmin) * ndecade
        x = np.logspace(logmin, logmax, n)
        y = func(x)
        val = _trapz_loglog(y, x, ulog10=True, **kwargs)
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
    ulog10 : bool
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
                x[slice2] * (x[slice2] / x[slice1])**b - x[slice1])) / (b + 1),
            x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]))

    tozero = (y[slice1] == 0.) + (y[slice2] == 0.) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.

    if intervals:
        return trapzs * x_unit * y_unit

    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit

    return ret