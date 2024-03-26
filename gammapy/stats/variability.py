# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.stats as stats
from gammapy.utils.random import get_random_state

__all__ = [
    "compute_fvar",
    "compute_fpp",
    "compute_chisq",
    "compute_flux_doubling",
    "structure_function",
    "TimmerAlgorithm",
]


def compute_fvar(flux, flux_err, axis=0):
    r"""Calculate the fractional excess variance.

    This method accesses the ``FLUX`` and ``FLUX_ERR`` columns
    from the lightcurve data.

    The fractional excess variance :math:`F_{var}`, an intrinsic
    variability estimator, is given by:

    .. math::
        F_{var} = \sqrt{ \frac{S^{2} - \bar{ \sigma^{2}}}{ \bar{x}^{2}}}

    It is the excess variance after accounting for the measurement errors
    on the light curve :math:`\sigma`. :math:`S` is the variance.

    It is important to note that the errors on the flux must be gaussian.

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        The measured fluxes.
    flux_err : `~astropy.units.Quantity`
        The error on measured fluxes.
    axis : int, optional
        Axis along which the excess variance is computed.
        Default is 0.

    Returns
    -------
    fvar, fvar_err : `~numpy.ndarray`
        Fractional excess variance.

    References
    ----------
    .. [Vaughan2003] "On characterizing the variability properties of X-ray light
       curves from active galaxies", Vaughan et al. (2003)
       https://ui.adsabs.harvard.edu/abs/2003MNRAS.345.1271V
    """

    flux_mean = np.nanmean(flux, axis=axis)
    n_points = np.count_nonzero(~np.isnan(flux), axis=axis)

    s_square = np.nansum((flux - flux_mean) ** 2, axis=axis) / (n_points - 1)
    sig_square = np.nansum(flux_err**2, axis=axis) / n_points
    fvar = np.sqrt(np.abs(s_square - sig_square)) / flux_mean

    sigxserr_a = np.sqrt(2 / n_points) * sig_square / flux_mean**2
    sigxserr_b = np.sqrt(sig_square / n_points) * (2 * fvar / flux_mean)
    sigxserr = np.sqrt(sigxserr_a**2 + sigxserr_b**2)
    fvar_err = sigxserr / (2 * fvar)

    return fvar, fvar_err


def compute_fpp(flux, flux_err, axis=0):
    r"""Calculate the point-to-point excess variance.

    F_pp is a quantity strongly related to the fractional excess variance F_var
    implemented in `~gammapy.stats.compute_fvar`; F_pp probes the variability
    on a shorter timescale.

    For white noise, F_pp and F_var give the same value.
    However, for red noise, F_var will be larger
    than F_pp, as the variations will be larger on longer timescales.

    It is important to note that the errors on the flux must be Gaussian.

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        The measured fluxes.
    flux_err : `~astropy.units.Quantity`
        The error on measured fluxes.
    axis : int, optional
        Axis along which the excess variance is computed.
        Default is 0.

    Returns
    -------
    fpp, fpp_err : `~numpy.ndarray`
        Point-to-point excess variance.

    References
    ----------
    .. [Edelson2002] "X-Ray Spectral Variability and Rapid Variability
       of the Soft X-Ray Spectrum Seyfert 1 Galaxies
       Arakelian 564 and Ton S180", Edelson et al. (2002), equation 3,
       https://iopscience.iop.org/article/10.1086/323779
    """

    flux_mean = np.nanmean(flux, axis=axis)
    n_points = np.count_nonzero(~np.isnan(flux), axis=axis)
    flux = flux.swapaxes(0, axis).T

    s_square = np.nansum((flux[..., 1:] - flux[..., :-1]) ** 2, axis=-1) / (
        n_points.T - 1
    )
    sig_square = np.nansum(flux_err**2, axis=axis) / n_points
    fpp = np.sqrt(np.abs(s_square.T - sig_square)) / flux_mean

    sigxserr_a = np.sqrt(2 / n_points) * sig_square / flux_mean**2
    sigxserr_b = np.sqrt(sig_square / n_points) * (2 * fpp / flux_mean)
    sigxserr = np.sqrt(sigxserr_a**2 + sigxserr_b**2)
    fpp_err = sigxserr / (2 * fpp)

    return fpp, fpp_err


def compute_chisq(flux):
    r"""Calculate the chi-square test for `LightCurve`.

    Chisquare test is a variability estimator. It computes
    deviations from the expected value here mean value.

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        The measured fluxes.

    Returns
    -------
    ChiSq, P-value : tuple of float or `~numpy.ndarray`
        Tuple of Chi-square and P-value.
    """
    yexp = np.mean(flux)
    yobs = flux.data
    chi2, pval = stats.chisquare(yobs, yexp)
    return chi2, pval


def compute_flux_doubling(flux, flux_err, coords, axis=0):
    r"""Compute the minimum characteristic flux doubling and halving
    over a certain coordinate axis for a series of measurements.

    Computing the flux doubling can give the doubling time in a lightcurve
    displaying significant temporal variability, e.g. an AGN flare.

    The variable is computed as:

     .. math::
        doubling = min(\frac{t_(i+1)-t_i}{log_2{f_(i+1)/f_i}})

    where f_i and f_(i+1) are the fluxes measured at subsequent coordinates t_i and t_(i+1).
    The error is obtained by propagating the relative errors on the flux measures.

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        The measured fluxes.
    flux_err : `~astropy.units.Quantity`
        The error on measured fluxes.
    coords : `~astropy.units.Quantity`
        The coordinates at which the fluxes are measured.
    axis : int, optional
        Axis along which the value is computed.

    Returns
    -------
    doubling_dict : dict
        Dictionary containing the characteristic flux doubling, halving and errors,
        with coordinates at which they were found.
    """

    flux = np.atleast_2d(flux).swapaxes(0, axis).T
    flux_err = np.atleast_2d(flux_err).swapaxes(0, axis).T

    axes = np.diff(coords) / np.log2(flux[..., 1:] / flux[..., :-1])
    axes_err_1 = (
        np.diff(coords)
        * np.log(2)
        / flux[..., 1:]
        * np.log(flux[..., 1:] / flux[..., :-1]) ** 2
    )
    axes_err_2 = (
        np.diff(coords)
        * np.log(2)
        / flux[..., :-1]
        * np.log(flux[..., 1:] / flux[..., :-1]) ** 2
    )
    axes_err = np.sqrt(
        (flux_err[..., 1:] * axes_err_1) ** 2 + (flux_err[..., :-1] * axes_err_2) ** 2
    )

    imin = np.expand_dims(
        np.argmin(
            np.where(
                np.logical_and(np.isfinite(axes), axes > 0), axes, np.inf * coords.unit
            ),
            axis=-1,
        ),
        axis=-1,
    )
    imax = np.expand_dims(
        np.argmax(
            np.where(
                np.logical_and(np.isfinite(axes), axes < 0), axes, -np.inf * coords.unit
            ),
            axis=-1,
        ),
        axis=-1,
    )

    index = np.concatenate([imin, imax], axis=-1)
    coord = np.take_along_axis(coords, index.flatten(), axis=0).reshape(index.shape)

    doubling = np.take_along_axis(axes, index, axis=-1)
    doubling_err = np.take_along_axis(axes_err, index, axis=-1)

    doubling_dict = {
        "doubling": doubling.T[0],
        "doubling_err": doubling_err.T[0],
        "doubling_coord": coord.T[0],
        "halving": np.abs(doubling.T[1]),
        "halving_err": doubling_err.T[1],
        "halving_coord": coord.T[1],
    }

    return doubling_dict


def structure_function(flux, flux_err, time, tdelta_precision=5):
    """Compute the discrete structure function for a variable source.

    Parameters
    ----------
    flux : `~astropy.units.Quantity`
        The measured fluxes.
    flux_err : `~astropy.units.Quantity`
        The error on measured fluxes.
    time : `~astropy.units.Quantity`
        The time coordinates at which the fluxes are measured.
    tdelta_precision : int, optional
        The number of decimal places to check to separate the time deltas. Default is 5.

    Returns
    -------
    sf, distances : `~numpy.ndarray`, `~astropy.units.Quantity`
        Discrete structure function and array of time distances.

    References
    ----------
    .. [Emmanoulopoulos2010] "On the use of structure functions to study blazar variability:
    caveats and problems", Emmanoulopoulos et al. (2010)
    https://academic.oup.com/mnras/article/404/2/931/968488
    """

    dist_matrix = (time[np.newaxis, :] - time[:, np.newaxis]).round(
        decimals=tdelta_precision
    )
    distances = np.unique(dist_matrix)
    distances = distances[distances > 0]
    shape = distances.shape + flux.shape[1:]
    factor = np.zeros(shape)
    norm = np.zeros(shape)

    for i, distance in enumerate(distances):
        indexes = np.array(np.where(dist_matrix == distance))
        for index in indexes.T:
            f = (flux[index[1], ...] - flux[index[0], ...]) ** 2
            w = (flux[index[1], ...] / flux_err[index[1], ...]) * (
                flux[index[0], ...] / flux_err[index[0], ...]
            )

            f = np.nan_to_num(f)
            w = np.nan_to_num(w)
            factor[i] = factor[i] + f * w
            norm[i] = norm[i] + w

    sf = factor / norm
    return sf, distances


def TimmerAlgorithm(
    power_spectrum,
    npoints,
    spacing,
    type="discrete",
    random_state="random-seed",
    normalization=1.0,
):
    """Implementation of the Timmer-Koenig algorithm to simulate a time series from a power spectrum.

    Parameters
    ----------
    power_spectrum : float, `~numpy.ndarray`
        Power spectrum used to generate the time series. It can either be presented analytically, as a parameter or set
        of parameters of a function, or as a `~numpy.ndarray` describing the spectrum in a discrete manner.
    npoints : float
        Number of points in the output time series.
    spacing : float
        Sample spacing, inverse of the sampling rate.
    type : {"discrete", "powerlaw", "white"}
        Type of input periodogram. For `~numpy.ndarray` inputs, the type is "discrete". Default is "discrete".
        If 'type' = "discrete", 'powerspectrum' is expected to be  a `~numpy.ndarray` containing the discrete spectrum.
        If 'type' = "powerlaw", 'powerspectrum' is expected to be  a float representing the power index.
        If 'type' = "white", 'powerspectrum' is expected to be  a float representing the standard deviation.
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`. Default is "random-seed".
    normalization : float
        Normalization factor to be applied to the final time series

    Returns
    -------
    time_series: `~numpy.ndarray`
        Simulated time series.

    References
    ----------
    ..[Timmer1995]"On generating power law noise", J. Timmer and M, Konig, section 3.
    """
    if isinstance(power_spectrum, np.ndarray) and type != "discrete":
        raise ValueError(
            "Power spectrum input is an array but the requested type is not 'discrete'."
        )

    random_state = get_random_state(random_state)

    frequencies = np.fft.fftfreq(npoints, spacing)

    # To obtain real data only the positive or negative part of the frequency is necessary.
    real_frequencies = np.sort(np.abs(frequencies[frequencies < 0]))

    if type == "discrete":
        periodogram = power_spectrum
    elif type == "powerlaw":
        periodogram = (1 / real_frequencies) ** power_spectrum
    elif type == "white":
        periodogram = np.full_like(real_frequencies, power_spectrum)
    else:
        raise ValueError("Power spectrum type not accepted!")

    random_numbers = random_state.normal(
        0, 1, len(periodogram) - 1
    ) + 1j * random_state.normal(0, 1, len(periodogram) - 1)
    fourier_coeffs = np.sqrt(0.5 * periodogram[:-1]) * random_numbers

    # Nyquist frequency component handling
    if npoints % 2 == 0:
        fourier_coeffs = np.concatenate(
            [
                fourier_coeffs,
                np.sqrt(0.5 * periodogram[-1:]) * random_state.normal(0, 1),
            ]
        )
    else:
        fourier_coeffs = np.concatenate(
            [
                fourier_coeffs,
                np.sqrt(0.5 * periodogram[-1:])
                * (random_state.normal(0, 1) + 1j * random_state.normal(0, 1)),
            ]
        )

    # Complex conjugate for negative frequencies
    fourier_coeffs = np.concatenate(
        [fourier_coeffs, np.conjugate(fourier_coeffs[-2::-1])]
    )
    fourier_coeffs = np.insert(fourier_coeffs, 0, 0)
    time_series = np.fft.ifft(fourier_coeffs).real

    if time_series.max() < np.abs(time_series.min()):
        time_series = -time_series

    time_series = (0.5 + 0.5 * time_series / time_series.max()) * normalization

    return time_series
