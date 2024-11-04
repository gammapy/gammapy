# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.stats as stats
import astropy.units as u
from gammapy.utils.random import get_random_state

__all__ = [
    "compute_fvar",
    "compute_fpp",
    "compute_chisq",
    "compute_flux_doubling",
    "structure_function",
    "TimmerKonig_lightcurve_simulator",
    "discrete_correlation",
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


def discrete_correlation(flux1, flux_err1, flux2, flux_err2, time1, time2, tau, axis=0):
    """Compute the discrete correlation function for a variable source.

    Parameters
    ----------
    flux1, flux_err1: `~astropy.units.Quantity`
        The first set of measured fluxes and associated error.
    flux2, flux_err2 : `~astropy.units.Quantity`
        The second set of measured fluxes and associated error.
    time1, time2 : `~astropy.units.Quantity`
        The time coordinates at which the fluxes are measured.
    tau : `~astropy.units.Quantity`
        Size of the bins to compute the discrete correlation.
    axis : int, optional
        Axis along which the correlation is computed.
        Default is 0.

    Returns
    -------
    bincenters: `~astropy.units.Quantity`
        Array of discrete time bins.
    discrete_correlation: `~numpy.ndarray`
        Array of discrete correlation function values for each bin.
    discrete_correlation_err : `~numpy.ndarray`
        Error associated to the discrete correlation values.

    References
    ----------
    .. [Edelson1988] "THE DISCRETE CORRELATION FUNCTION: A NEW METHOD FOR ANALYZING
       UNEVENLY SAMPLED VARIABILITY DATA", Edelson et al. (1988)
       https://ui.adsabs.harvard.edu/abs/1988ApJ...333..646E/abstract
    """
    flux1 = np.rollaxis(flux1, axis, 0)
    flux2 = np.rollaxis(flux2, axis, 0)

    if np.squeeze(flux1).shape[1:] != np.squeeze(flux2).shape[1:]:
        raise ValueError(
            "flux1 and flux2 must have the same squeezed shape, apart from the chosen axis."
        )

    tau = tau.to(time1.unit)
    time2 = time2.to(time1.unit)

    mean1, mean2 = np.nanmean(flux1, axis=0), np.nanmean(flux2, axis=0)
    sigma1, sigma2 = np.nanstd(flux1, axis=0), np.nanstd(flux2, axis=0)

    udcf1 = (flux1 - mean1) / np.sqrt((sigma1**2 - np.nanmean(flux_err1, axis=0) ** 2))
    udcf2 = (flux2 - mean2) / np.sqrt((sigma2**2 - np.nanmean(flux_err2, axis=0) ** 2))

    udcf = np.empty(((flux1.shape[0],) + flux2.shape))
    dist = u.Quantity(np.empty(((flux1.shape[0], flux2.shape[0]))), unit=time1.unit)

    for i, x1 in enumerate(udcf1):
        for j, x2 in enumerate(udcf2):
            udcf[i, j, ...] = x1 * x2
            dist[i, j] = time1[i] - time2[j]

    maxfactor = np.floor(np.amax(dist) / tau).value + 1
    minfactor = np.floor(np.amin(dist) / tau).value

    bins = (
        np.linspace(
            minfactor, maxfactor, int(np.abs(maxfactor) + np.abs(minfactor) + 1)
        )
        * tau
    )

    bin_indices = np.digitize(dist, bins).flatten()

    udcf = np.reshape(udcf, (udcf.shape[0] * udcf.shape[1], -1))
    discrete_correlation = np.array(
        [np.nanmean(udcf[bin_indices == i], axis=0) for i in range(1, len(bins))]
    )

    discrete_correlation_err = []
    for i in range(1, len(bins)):
        terms = (discrete_correlation[i - 1] - udcf[bin_indices == i]) ** 2
        num = np.sqrt(np.nansum(terms, axis=0))
        den = len(udcf[bin_indices == i]) - 1
        discrete_correlation_err.append(num / den)

    bincenters = (bins[1:] + bins[:-1]) / 2

    return bincenters, discrete_correlation, np.array(discrete_correlation_err)


def TimmerKonig_lightcurve_simulator(
    power_spectrum,
    npoints,
    spacing,
    nchunks=10,
    random_state="random-seed",
    power_spectrum_params=None,
    mean=0.0,
    std=1.0,
    poisson=False,
):
    """Implementation of the Timmer-Koenig algorithm to simulate a time series from a power spectrum.

    Parameters
    ----------
    power_spectrum : function
        Power spectrum used to generate the time series. It is expected to be
        a function mapping the input frequencies to the periodogram.
    npoints : int
        Number of points in the output time series.
    spacing : `~astropy.units.Quantity`
        Sample spacing, inverse of the sampling rate. The units are inherited by the resulting time axis.
    nchunks : int, optional
        Factor by which to multiply the length of the time series to avoid red noise leakage. Default is 10.
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}, optional
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`. Default is "random-seed".
    power_spectrum_params : dict, optional
        Dictionary of parameters to be provided to the power spectrum function.
    mean : float, `~astropy.units.Quantity`, optional
        Desired mean of the final series. Default is 0.
    std : float, `~astropy.units.Quantity`, optional
        Desired standard deviation of the final series. Default is 1.
    poisson : bool, optional
        Whether to apply poissonian noise to the final time series. Default is False.


    Returns
    -------
    time_series : `~numpy.ndarray`
        Simulated time series.
    time_axis : `~astropy.units.Quantity`
        Time axis of the series in the same units as 'spacing'. It will be defined with length 'npoints', from 0 to
        'npoints'*'spacing'.

    Examples
    --------
    To pass the function to be used in the simlation one can use either the 'lambda' keyword or an extended definition.
    Parameters of the function can be passed using the 'power_spectrum_params' keyword.
    For example, these are three ways to pass a power law (red noise) with index 2:

    >>> from gammapy.stats import TimmerKonig_lightcurve_simulator
    >>> import astropy.units as u
    >>> def powerlaw(x):
    ...     return x**(-2)
    >>> def powerlaw_with_parameters(x, i):
    ...     return x**(-i)
    >>> ts, ta = TimmerKonig_lightcurve_simulator(lambda x: x**(-2), 20, 1*u.h)
    >>> ts2, ta2 = TimmerKonig_lightcurve_simulator(powerlaw, 20, 1*u.h)
    >>> ts3, ta3 = TimmerKonig_lightcurve_simulator(powerlaw_with_parameters,
    ...                                            20, 1*u.h, power_spectrum_params={"i":2})

    References
    ----------
    .. [Timmer1995] "On generating power law noise", J. Timmer and M, Konig, section 3
       https://ui.adsabs.harvard.edu/abs/1995A%26A...300..707T/abstract
    """
    if not callable(power_spectrum):
        raise ValueError(
            "The power spectrum has to be provided as a callable function."
        )

    if not isinstance(npoints * nchunks, int):
        raise TypeError("npoints and nchunks must be integers")

    if poisson:
        if isinstance(mean, u.Quantity):
            wmean = mean.value * spacing.value
        else:
            wmean = mean * spacing.value
        if wmean < 1.0:
            raise Warning(
                "Poisson noise was requested but the target mean is too low - resulting counts will likely be 0."
            )

    random_state = get_random_state(random_state)

    npoints_ext = npoints * nchunks

    frequencies = np.fft.fftfreq(npoints_ext, spacing.value)

    # To obtain real data only the positive or negative part of the frequency is necessary.
    real_frequencies = np.sort(np.abs(frequencies[frequencies < 0]))

    if power_spectrum_params:
        periodogram = power_spectrum(real_frequencies, **power_spectrum_params)
    else:
        periodogram = power_spectrum(real_frequencies)

    real_part = random_state.normal(0, 1, len(periodogram) - 1)
    imaginary_part = random_state.normal(0, 1, len(periodogram) - 1)

    # Nyquist frequency component handling
    if npoints_ext % 2 == 0:
        idx0 = -2
        random_factor = random_state.normal(0, 1)
    else:
        idx0 = -1
        random_factor = random_state.normal(0, 1) + 1j * random_state.normal(0, 1)

    fourier_coeffs = np.concatenate(
        [
            np.sqrt(0.5 * periodogram[:-1]) * (real_part + 1j * imaginary_part),
            np.sqrt(0.5 * periodogram[-1:]) * random_factor,
        ]
    )
    fourier_coeffs = np.concatenate(
        [fourier_coeffs, np.conjugate(fourier_coeffs[idx0::-1])]
    )

    fourier_coeffs = np.insert(fourier_coeffs, 0, 0)
    time_series = np.fft.ifft(fourier_coeffs).real

    ndiv = npoints_ext // (2 * nchunks)
    setstart = npoints_ext // 2 - ndiv
    setend = npoints_ext // 2 + ndiv
    if npoints % 2 != 0:
        setend = setend + 1
    time_series = time_series[setstart:setend]

    time_series = (time_series - time_series.mean()) / time_series.std()
    time_series = time_series * std + mean

    if poisson:
        time_series = (
            random_state.poisson(
                np.where(time_series >= 0, time_series, 0) * spacing.value
            )
            / spacing.value
        )

    time_axis = np.linspace(0, npoints * spacing.value, npoints) * spacing.unit

    return time_series, time_axis
