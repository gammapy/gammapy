# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.ndimage
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.datasets.map import MapEvaluator
from gammapy.maps import Map, MapAxis, TimeMapAxis, WcsNDMap
from gammapy.modeling import Parameter
from gammapy.modeling.models import (
    ConstantFluxSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.stats import (
    compute_flux_doubling,
    compute_fpp,
    compute_fvar,
    discrete_correlation,
)
from gammapy.stats.utils import ts_to_sigma
from .map.core import FluxMaps

__all__ = [
    "get_combined_significance_maps",
    "estimate_exposure_reco_energy",
    "find_peaks",
    "find_peaks_in_flux_map",
    "resample_energy_edges",
    "get_rebinned_axis",
    "compute_lightcurve_fvar",
    "compute_lightcurve_fpp",
    "compute_lightcurve_doublingtime",
    "compute_lightcurve_discrete_correlation",
]


def find_peaks(image, threshold, min_distance=1):
    """Find local peaks in an image.

    This is a very simple peak finder, that finds local peaks
    (i.e. maxima) in images above a given ``threshold`` within
    a given ``min_distance`` around each given pixel.

    If you get multiple spurious detections near a peak, usually
    it's best to smooth the image a bit, or to compute it using
    a different method in the first place to result in a smooth image.
    You can also increase the ``min_distance`` parameter.

    The output table contains one row per peak and the following columns:

    - ``x`` and ``y`` are the pixel coordinates (first pixel at zero).
    - ``ra`` and ``dec`` are the RA / DEC sky coordinates (ICRS frame).
    - ``value`` is the pixel value.

    It is sorted by peak value, starting with the highest value.

    If there are no pixel values above the threshold, an empty table is returned.

    There are more featureful peak finding and source detection methods
    e.g. in the ``photutils`` or ``scikit-image`` Python packages.

    Parameters
    ----------
    image : `~gammapy.maps.WcsNDMap`
        Image like Map.
    threshold : float or array-like
        The data value or pixel-wise data values to be used for the
        detection threshold.  A 2D ``threshold`` must have the same
        shape as the map ``data``.
    min_distance : int or `~astropy.units.Quantity`
        Minimum distance between peaks. An integer value is interpreted
        as pixels. Default is 1.

    Returns
    -------
    output : `~astropy.table.Table`
        Table with parameters of detected peaks.

    Examples
    --------
    >>> import astropy.units as u
    >>> from gammapy.datasets import MapDataset
    >>> from gammapy.estimators import ExcessMapEstimator
    >>> from gammapy.estimators.utils import find_peaks
    >>>
    >>> dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
    >>> estimator = ExcessMapEstimator(
    ...     correlation_radius="0.1 deg", energy_edges=[0.1, 10] * u.TeV
    ... )
    >>> maps = estimator.run(dataset)
    >>> # Find the peaks which are above 5 sigma
    >>> sources = find_peaks(maps["sqrt_ts"], threshold=5, min_distance="0.25 deg")
    >>> print(sources)
    value   x   y      ra       dec
                      deg       deg
    ------ --- --- --------- ---------
    32.191 161 118 266.41924 -28.98772
      18.7 125 124 266.80571 -28.14079
    9.4498 257 122 264.86178 -30.97529
    9.3784 204 103 266.14201 -30.10041
    5.3493 282 150 263.78083 -31.12704
    """
    # Input validation

    if not isinstance(image, WcsNDMap):
        raise TypeError("find_peaks only supports WcsNDMap")

    if not image.geom.is_flat:
        raise ValueError(
            "find_peaks only supports flat Maps, with no spatial axes of length 1."
        )

    if isinstance(min_distance, (str, u.Quantity)):
        min_distance = np.mean(u.Quantity(min_distance) / image.geom.pixel_scales)
        min_distance = np.round(min_distance).to_value("")

    size = 2 * min_distance + 1

    # Remove non-finite values to avoid warnings or spurious detection
    data = image.sum_over_axes(keepdims=False).data
    data[~np.isfinite(data)] = np.nanmin(data)

    # Handle edge case of constant data; treat as no peak
    if np.all(data == data.flat[0]):
        return Table()

    # Run peak finder
    data_max = scipy.ndimage.maximum_filter(data, size=size, mode="constant")
    mask = (data == data_max) & (data > threshold)
    y, x = mask.nonzero()
    value = data[y, x]

    # Make and return results table

    if len(value) == 0:
        return Table()

    coord = SkyCoord.from_pixel(x, y, wcs=image.geom.wcs).icrs

    table = Table()
    table["value"] = value * image.unit
    table["x"] = x
    table["y"] = y
    table["ra"] = coord.ra
    table["dec"] = coord.dec

    table["ra"].format = ".5f"
    table["dec"].format = ".5f"
    table["value"].format = ".5g"

    table.sort("value")
    table.reverse()

    return table


def find_peaks_in_flux_map(maps, threshold, min_distance=1):
    """Find local test statistic peaks for a given Map.

    Utilises the `~gammapy.estimators.utils.find_peaks` function to find various parameters from FluxMaps.

    Parameters
    ----------
    maps : `~gammapy.estimators.FluxMaps`
        Input flux map object.
    threshold : float or array-like
        The test statistic data value or pixel-wise test statistic data values to be used for the
        detection threshold.  A 2D ``threshold`` must have the same.
        shape as the map ``data``.
    min_distance : int or `~astropy.units.Quantity`
        Minimum distance between peaks. An integer value is interpreted
        as pixels. Default is 1.

    Returns
    -------
    output : `~astropy.table.Table`
        Table with parameters of detected peaks.

    Examples
    --------
    >>> import astropy.units as u
    >>> from gammapy.datasets import MapDataset
    >>> from gammapy.estimators import ExcessMapEstimator
    >>> from gammapy.estimators.utils import find_peaks_in_flux_map
    >>>
    >>> dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
    >>> estimator = ExcessMapEstimator(
    ...     correlation_radius="0.1 deg", energy_edges=[0.1, 10]*u.TeV
    ... )
    >>> maps = estimator.run(dataset)
    >>> # Find the peaks which are above 5 sigma
    >>> sources = find_peaks_in_flux_map(maps, threshold=5, min_distance=0.1*u.deg)
    >>> print(sources[:4])
    x   y      ra       dec      npred   npred_excess   counts     ts    sqrt_ts   norm  norm_err     flux      flux_err
               deg       deg                                                                       1 / (s cm2) 1 / (s cm2)
    --- --- --------- --------- --------- ------------ --------- -------- ------- ------- -------- ----------- -----------
    158 135 266.05019 -28.70181 192.00000     61.33788 192.00000 25.11839 5.01183 0.28551  0.06450   2.827e-12   6.385e-13
     92 133 267.07022 -27.31834 137.00000     51.99467 137.00000 26.78181 5.17511 0.37058  0.08342   3.669e-12   8.259e-13
    176 134 265.80492 -29.09805 195.00000     65.15990 195.00000 28.29158 5.31898 0.30561  0.06549   3.025e-12   6.484e-13
    282 150 263.78083 -31.12704  84.00000     39.99004  84.00000 28.61526 5.34932 0.55027  0.12611   5.448e-12   1.249e-12
    """
    quantity_for_peaks = maps["sqrt_ts"]

    if not isinstance(maps, FluxMaps):
        raise TypeError(
            f"find_peaks_in_flux_map expects FluxMaps input. Got {type(maps)} instead."
        )

    if not quantity_for_peaks.geom.is_flat:
        raise ValueError(
            "find_peaks_in_flux_map only supports flat Maps, with energy axis of length 1."
        )

    table = find_peaks(quantity_for_peaks, threshold, min_distance)

    if len(table) == 0:
        return Table()

    x = np.array(table["x"])
    y = np.array(table["y"])

    table.remove_column("value")

    for name in maps.available_quantities:
        values = maps[name].quantity
        peaks = values[0, y, x]
        table[name] = peaks

    flux_data = maps["flux"].quantity
    table["flux"] = flux_data[0, y, x]
    flux_err_data = maps["flux_err"].quantity
    table["flux_err"] = flux_err_data[0, y, x]

    for column in table.colnames:
        if column.startswith(("flux", "flux_err")):
            table[column].format = ".3e"
        elif column.startswith(
            (
                "npred",
                "npred_excess",
                "counts",
                "sqrt_ts",
                "norm",
                "ts",
                "norm_err",
                "stat",
                "stat_null",
            )
        ):
            table[column].format = ".5f"

    table.reverse()

    return table


def estimate_exposure_reco_energy(dataset, spectral_model=None, normalize=True):
    """Estimate an exposure map in reconstructed energy.

    Parameters
    ----------
    dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
        The input dataset.
    spectral_model : `~gammapy.modeling.models.SpectralModel`, optional
        Assumed spectral shape. If None, a Power Law of index 2 is assumed. Default is None.
    normalize : bool
        Normalize the exposure to the total integrated flux of the spectral model.
        When not normalized it directly gives the predicted counts from the spectral
        model. Default is True.

    Returns
    -------
    exposure : `~gammapy.maps.Map`
        Exposure map in reconstructed energy.
    """
    if spectral_model is None:
        spectral_model = PowerLawSpectralModel()

    model = SkyModel(
        spatial_model=ConstantFluxSpatialModel(), spectral_model=spectral_model
    )

    energy_axis = dataset._geom.axes["energy"]

    if dataset.edisp is not None:
        edisp = dataset.edisp.get_edisp_kernel(position=None, energy_axis=energy_axis)
    else:
        edisp = None

    eval = MapEvaluator(model=model, exposure=dataset.exposure, edisp=edisp)
    reco_exposure = eval.compute_npred()

    if normalize:
        ref_flux = spectral_model.integral(
            energy_axis.edges[:-1], energy_axis.edges[1:]
        )
        reco_exposure = reco_exposure / ref_flux[:, np.newaxis, np.newaxis]

    return reco_exposure


def _satisfies_conditions(info_dict, conditions):
    satisfies = True
    for key in conditions.keys():
        satisfies &= info_dict[key.strip("_min")] > conditions[key]
    return satisfies


def resample_energy_edges(dataset, conditions={}):
    """Return energy edges that satisfy given condition on the per bin statistics.

    Parameters
    ----------
    dataset : `~gammapy.datasets.SpectrumDataset` or `~gammapy.datasets.SpectrumDatasetOnOff`
        The input dataset.
    conditions : dict
        Keyword arguments containing the per-bin conditions used to resample the axis.
        Available options are: 'counts_min', 'background_min', 'excess_min', 'sqrt_ts_min',
        'npred_min', 'npred_background_min', 'npred_signal_min'. Default is {}.

    Returns
    -------
    energy_edges : list of `~astropy.units.Quantity`
        Energy edges for the resampled energy axis.

    Examples
    --------
    >>> from gammapy.datasets import Datasets, SpectrumDatasetOnOff
    >>> from gammapy.estimators.utils import resample_energy_edges
    >>>
    >>> datasets = Datasets()
    >>>
    >>> for obs_id in [23523, 23526]:
    ...     dataset = SpectrumDatasetOnOff.read(
    ...         f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obs_id}.fits"
    ...     )
    ...     datasets.append(dataset)
    >>>
    >>> spectrum_dataset = Datasets(datasets).stack_reduce()
    >>> # Resample the energy edges so the minimum sqrt_ts is 2
    >>> resampled_energy_edges = resample_energy_edges(
    ...     spectrum_dataset,
    ...     conditions={"sqrt_ts_min": 2}
    ... )
    """
    if not isinstance(dataset, (SpectrumDataset, SpectrumDatasetOnOff)):
        raise NotImplementedError(
            "This method is currently supported for spectral datasets only."
        )

    available_conditions = [
        "counts_min",
        "background_min",
        "excess_min",
        "sqrt_ts_min",
        "npred_min",
        "npred_background_min",
        "npred_signal_min",
    ]
    for key in conditions.keys():
        if key not in available_conditions:
            raise ValueError(
                f"Unrecognized option {key}. The available methods are: {available_conditions}."
            )

    axis = dataset.counts.geom.axes["energy"]
    energy_min_all, energy_max_all = dataset.energy_range_total
    energy_edges = [energy_max_all]

    while energy_edges[-1] > energy_min_all:
        for energy_min in reversed(axis.edges_min):
            if energy_min >= energy_edges[-1]:
                continue
            elif len(energy_edges) == 1 and energy_min == energy_min_all:
                raise ValueError("The given conditions cannot be met.")

            sliced = dataset.slice_by_energy(
                energy_min=energy_min, energy_max=energy_edges[-1]
            )

            with np.errstate(invalid="ignore"):
                info = sliced.info_dict()

            if _satisfies_conditions(info, conditions):
                energy_edges.append(energy_min)
                break
    return u.Quantity(energy_edges[::-1])


def compute_lightcurve_fvar(lightcurve, flux_quantity="flux"):
    """
    Compute the fractional excess variance of the input lightcurve.

    Internally calls the `~gammapy.stats.compute_fvar` function.


    Parameters
    ----------
    lightcurve : `~gammapy.estimators.FluxPoints`
        The lightcurve object.
    flux_quantity : str
        Flux quantity to use for calculation. Should be 'dnde', 'flux', 'e2dnde' or 'eflux'. Default is 'flux'.

    Returns
    -------
    fvar : `~astropy.table.Table`
        Table of fractional excess variance and associated error for each energy bin of the lightcurve.
    """
    flux = getattr(lightcurve, flux_quantity)
    flux_err = getattr(lightcurve, flux_quantity + "_err")

    time_id = flux.geom.axes.index_data("time")

    fvar, fvar_err = compute_fvar(flux.data, flux_err.data, axis=time_id)

    significance = fvar / fvar_err

    energies = lightcurve.geom.axes["energy"].edges
    table = Table(
        [energies[:-1], energies[1:], fvar, fvar_err, significance],
        names=("min_energy", "max_energy", "fvar", "fvar_err", "significance"),
        meta=lightcurve.meta,
    )

    return table


def compute_lightcurve_fpp(lightcurve, flux_quantity="flux"):
    """
    Compute the point-to-point excess variance of the input lightcurve.

    Internally calls the `~gammapy.stats.compute_fpp` function

    Parameters
    ----------
    lightcurve : `~gammapy.estimators.FluxPoints`
        The lightcurve object.
    flux_quantity : str
        Flux quantity to use for calculation. Should be 'dnde', 'flux', 'e2dnde' or 'eflux'. Default is 'flux'.

    Returns
    -------
    table : `~astropy.table.Table`
        Table of point-to-point excess variance and associated error for each energy bin of the lightcurve.
    """
    flux = getattr(lightcurve, flux_quantity)
    flux_err = getattr(lightcurve, flux_quantity + "_err")

    time_id = flux.geom.axes.index_data("time")

    fpp, fpp_err = compute_fpp(flux.data, flux_err.data, axis=time_id)

    significance = fpp / fpp_err

    energies = lightcurve.geom.axes["energy"].edges
    table = Table(
        [energies[:-1], energies[1:], fpp, fpp_err, significance],
        names=("min_energy", "max_energy", "fpp", "fpp_err", "significance"),
        meta=dict(quantity=flux_quantity),
    )

    return table


def compute_lightcurve_doublingtime(lightcurve, flux_quantity="flux"):
    """
    Compute the minimum characteristic flux doubling and halving time for the input lightcurve.

    Internally calls the `~gammapy.stats.compute_flux_doubling` function.

    The characteristic doubling time  is estimated to obtain the
    minimum variability timescale for the light curves in which
    rapid variations are clearly evident: for example it is useful in AGN flaring episodes.

    This quantity, especially for AGN flares, is often expressed
    as the pair of doubling time and halving time, or the minimum characteristic time
    for the rising and falling components respectively.

    Parameters
    ----------
    lightcurve : `~gammapy.estimators.FluxPoints`
        The lightcurve object.
    axis_name : str
        Name of the axis over which to compute the flux doubling.
    flux_quantity : str
        Flux quantity to use for calculation. Should be 'dnde', 'flux', 'e2dnde' or 'eflux'.
        Default is 'flux'.

    Returns
    -------
    table : `~astropy.table.Table`
        Table of flux doubling/halving and associated error for each energy bin of the lightcurve
        with axis coordinates at which they were found.


    References
    ----------
    .. [Brown2013] "Locating the γ-ray emission region
       of the flat spectrum radio quasar PKS 1510−089", Brown et al. (2013)
       https://academic.oup.com/mnras/article/431/1/824/1054498
    """
    flux = getattr(lightcurve, flux_quantity)
    flux_err = getattr(lightcurve, flux_quantity + "_err")
    coords = lightcurve.geom.axes["time"].center

    axis = flux.geom.axes.index_data("time")

    doubling_dict = compute_flux_doubling(flux.data, flux_err.data, coords, axis=axis)

    energies = lightcurve.geom.axes["energy"].edges
    table = Table(
        [
            energies[:-1],
            energies[1:],
            doubling_dict["doubling"],
            doubling_dict["doubling_err"],
            lightcurve.geom.axes["time"].reference_time
            + doubling_dict["doubling_coord"],
            doubling_dict["halving"],
            doubling_dict["halving_err"],
            lightcurve.geom.axes["time"].reference_time
            + doubling_dict["halving_coord"],
        ],
        names=(
            "min_energy",
            "max_energy",
            "doublingtime",
            "doubling_err",
            "doubling_coord",
            "halvingtime",
            "halving_err",
            "halving_coord",
        ),
        meta=dict(flux_quantity=flux_quantity),
    )

    return table


def compute_lightcurve_discrete_correlation(
    lightcurve1, lightcurve2=None, flux_quantity="flux", tau=None
):
    """Compute the discrete correlation function for two lightcurves, or the discrete autocorrelation if only one lightcurve is provided.

    NaN values will be ignored in the computation in order to account for possible gaps in the data.

    Internally calls the `~gammapy.stats.discrete_correlation` function.

    Parameters
    ----------
    lightcurve1 : `~gammapy.estimators.FluxPoints`
        The first lightcurve object.
    lightcurve2 : `~gammapy.estimators.FluxPoints`, optional
        The second lightcurve object. If not provided, the autocorrelation for the first lightcurve will be computed.
        Default is None.
    flux_quantity : str
        Flux quantity to use for calculation. Should be 'dnde', 'flux', 'e2dnde' or 'eflux'.
        The choice does not affect the computation. Default is 'flux'.
    tau : `~astropy.units.Quantity`, optional
        Size of the bins to compute the discrete correlation.
        If None, the bin size will be double the bins of the first lightcurve. Default is None.

    Returns
    -------
    discrete_correlation_dict : dict
        Dictionary containing the discrete correlation results. Entries are:

                * "bins" : the array of discrete time bins
                * "discrete_correlation" : discrete correlation function values
                * "discrete_correlation_err" : associated error

    References
    ----------
    .. [Edelson1988] "THE DISCRETE CORRELATION FUNCTION: A NEW METHOD FOR ANALYZING
       UNEVENLY SAMPLED VARIABILITY DATA", Edelson et al. (1988)
       https://ui.adsabs.harvard.edu/abs/1988ApJ...333..646E/abstract
    """
    flux1 = getattr(lightcurve1, flux_quantity)
    flux_err1 = getattr(lightcurve1, flux_quantity + "_err")
    coords1 = lightcurve1.geom.axes["time"].center
    axis = flux1.geom.axes.index_data("time")

    if tau is None:
        tau = (coords1[-1] - coords1[0]) / (0.5 * len(coords1))

    if lightcurve2:
        flux2 = getattr(lightcurve2, flux_quantity)
        flux_err2 = getattr(lightcurve2, flux_quantity + "_err")
        coords2 = lightcurve2.geom.axes["time"].center
        bins, dcf, dcf_err = discrete_correlation(
            flux1.data,
            flux_err1.data,
            flux2.data,
            flux_err2.data,
            coords1,
            coords2,
            tau,
            axis,
        )

    else:
        bins, dcf, dcf_err = discrete_correlation(
            flux1.data,
            flux_err1.data,
            flux1.data,
            flux_err1.data,
            coords1,
            coords1,
            tau,
            axis,
        )

    discrete_correlation_dict = {
        "bins": bins,
        "discrete_correlation": dcf,
        "discrete_correlation_err": dcf_err,
    }

    return discrete_correlation_dict


def get_edges_fixed_bins(fluxpoint, group_size, axis_name="energy"):
    """Rebin the flux point to combine value adjacent bins.

    Parameters
    ----------
    fluxpoint : `~gammapy.estimators.FluxPoints`
        The flux points object to rebin.
    group_size : int
        Number of bins to combine.
    axis_name : str, optional
        The axis name to combine along. Default is 'energy'.

    Returns
    -------
    edges_min : `~astropy.units.Quantity` or `~astropy.time.Time`
        Minimum bin edge for the new axis.
    edges_max : `~astropy.units.Quantity` or `~astropy.time.Time`
        Maximum bin edge for the new axis.
    """
    ax = fluxpoint.geom.axes[axis_name]
    nbin = ax.nbin
    if not isinstance(group_size, int):
        raise ValueError("Only integer number of bins can be combined")
    idx = np.arange(0, nbin, group_size)
    if idx[-1] < nbin:
        idx = np.append(idx, nbin)
    edges_min = ax.edges_min[idx[:-1]]
    edges_max = ax.edges_max[idx[1:] - 1]
    return edges_min, edges_max


def get_edges_min_ts(fluxpoint, ts_threshold, axis_name="energy"):
    """Rebin the flux point to combine adjacent bins until a minimum TS is obtained.

    Note that to convert TS to significance, it is necessary to take the number
    of degrees of freedom into account.


    Parameters
    ----------
    fluxpoint : `~gammapy.estimators.FluxPoints`
        The flux points object to rebin.
    ts_threshold : float
        The minimum significance desired.
    axis_name : str, optional
        The axis name to combine along. Default is 'energy'.

    Returns
    -------
    edges_min : `~astropy.units.Quantity` or `~astropy.time.Time`
        Minimum bin edge for the new axis.
    edges_max : `~astropy.units.Quantity` or `~astropy.time.Time`
        Maximum bin edge for the new axis.
    """
    ax = fluxpoint.geom.axes[axis_name]
    nbin = ax.nbin

    e_min, e_max = ax.edges_min[0], ax.edges_max[0]
    edges_min = np.zeros(nbin) * e_min.unit
    edges_max = np.zeros(nbin) * e_max.unit
    i, i1 = 0, 0
    while e_max < ax.edges_max[-1]:
        ts = fluxpoint.ts.data[i]
        e_min = ax.edges_min[i]
        while ts < ts_threshold and i < ax.nbin - 1:
            i = i + 1
            ts = ts + fluxpoint.ts.data[i]
        e_max = ax.edges_max[i]
        i = i + 1
        edges_min[i1] = e_min
        edges_max[i1] = e_max
        i1 = i1 + 1
    edges_max = edges_max[:i1]
    edges_min = edges_min[:i1]

    return edges_min, edges_max


RESAMPLE_METHODS = {
    "fixed-bins": get_edges_fixed_bins,
    "min-ts": get_edges_min_ts,
}


def get_rebinned_axis(fluxpoint, axis_name="energy", method=None, **kwargs):
    """Get the rebinned axis for resampling the flux point object along the mentioned axis.

    Parameters
    ----------
    fluxpoint : `~gammapy.estimators.FluxPoints`
        The flux point object to rebin.
    axis_name : str, optional
        The axis name to combine along. Default is 'energy'.
    method : str
        The method to resample the axis. Supported options are
        'fixed_bins' and 'min-ts'.
    kwargs : dict
        Keywords passed to `get_edges_fixed_bins` or `get_edges_min_ts`.
        If method is 'fixed-bins', keyword should be `group_size`.
        If method is 'min-ts', keyword should be `ts_threshold`.

    Returns
    -------
    axis_new : `~gammapy.maps.MapAxis` or `~gammapy.maps.TimeMapAxis`
        The new axis.

    Examples
    --------
    >>> from gammapy.estimators.utils import get_rebinned_axis
    >>> from gammapy.estimators import FluxPoints
    >>>
    >>> # Rebin lightcurve axis
    >>> lc_1d = FluxPoints.read(
    ...         "$GAMMAPY_DATA/estimators/pks2155_hess_lc/pks2155_hess_lc.fits",
    ...         format="lightcurve",
    ...     )
    >>> # Rebin axis by combining adjacent bins as per the group_size
    >>> new_axis = get_rebinned_axis(
    ...     lc_1d, method="fixed-bins", group_size=2, axis_name="time"
    ... )
    >>>
    >>> # Rebin HESS flux points axis
    >>> fp = FluxPoints.read(
    ...         "$GAMMAPY_DATA/estimators/crab_hess_fp/crab_hess_fp.fits"
    ... )
    >>> # Rebin according to a minimum significance
    >>> axis_new = get_rebinned_axis(
    ...     fp, method='min-ts', ts_threshold=4, axis_name='energy'
    ... )
    """
    # TODO: Make fixed_bins and fixed_edges work for multidimensions
    if not fluxpoint.geom.axes.is_unidimensional:
        raise ValueError(
            "Rebinning is supported only for Unidimensional FluxPoints \n "
            "Please use `iter_by_axis` to create Unidimensional FluxPoints"
        )

    if method not in RESAMPLE_METHODS.keys():
        raise ValueError("Incorrect option. Choose from", RESAMPLE_METHODS.keys())

    edges_min, edges_max = RESAMPLE_METHODS[method](
        fluxpoint=fluxpoint, axis_name=axis_name, **kwargs
    )
    ax = fluxpoint.geom.axes[axis_name]

    if isinstance(ax, TimeMapAxis):
        axis_new = TimeMapAxis.from_time_edges(
            time_min=edges_min + ax.reference_time,
            time_max=edges_max + ax.reference_time,
        )
    else:
        edges = np.append(edges_min, edges_max[-1])
        axis_new = MapAxis.from_edges(edges, name=axis_name, interp=ax.interp)
    return axis_new


def get_combined_significance_maps(estimator, datasets):
    """Compute excess and significance for a set of datasets.

    The significance computation assumes that the model contains
    one degree of freedom per valid energy bin in each dataset.
    This method implemented here is valid under the assumption
    that the TS in each independent bin follows a Chi2 distribution,
    then the sum of the TS also follows a Chi2 distribution (with the sum of degree of freedom).

    See, Zhen (2014): https://www.sciencedirect.com/science/article/abs/pii/S0167947313003204,
    Lancaster (1961): https://onlinelibrary.wiley.com/doi/10.1111/j.1467-842X.1961.tb00058.x


    Parameters
    ----------
    estimator : `~gammapy.estimators.ExcessMapEstimator` or `~gammapy.estimators.TSMapEstimator`
        Excess Map Estimator or TS Map Estimator
    dataset : `~gammapy.datasets.Datasets`
        Datasets containing only `~gammapy.datasets.MapDataset`.

    Returns
    -------
    results : dict
        Dictionary with entries:

                * "significance" : joint significance map.
                * "df" : degree of freedom map (one norm per valid bin).
                * "npred_excess" : summed excess map.
                * "estimator_results" : dictionary containing the estimator results for each dataset.
    """
    from .map.excess import ExcessMapEstimator
    from .map.ts import TSMapEstimator

    if not isinstance(estimator, (ExcessMapEstimator, TSMapEstimator)):
        raise TypeError(
            f"estimator type should be ExcessMapEstimator or TSMapEstimator), got {type(estimator)} instead."
        )

    geom = datasets[0].counts.geom.to_image()
    ts_sum = Map.from_geom(geom)
    ts_sum_sign = Map.from_geom(geom)
    npred_excess_sum = Map.from_geom(geom)
    df = Map.from_geom(geom)

    results = dict()
    for kd, d in enumerate(datasets):
        result = estimator.run(d)
        results[d.name] = result

        df += np.sum(result["ts"].data > 0, axis=0)  # one dof (norm) per valid bin
        ts_sum += result["ts"].reduce_over_axes()
        ts_sum_sign += (
            result["ts"] * np.sign(result["npred_excess"])
        ).reduce_over_axes()
        npred_excess_sum += result["npred_excess"].reduce_over_axes()

    significance = Map.from_geom(geom)
    significance.data = ts_to_sigma(ts_sum.data, df.data) * np.sign(ts_sum_sign)
    return dict(
        significance=significance,
        df=df,
        npred_excess=npred_excess_sum,
        estimator_results=results,
    )


def combine_flux_maps(
    maps, method="gaussian_errors", reference_model=None, dnde_scan_axis=None
):
    """Create a FluxMaps by combining a list of flux maps with the same geometry.

     This assumes the flux maps are independent measurements of the same true value.
     The GTI is stacked in the process.

    Parameters
    ----------
    maps : list of `~gammapy.estimators.FluxMaps`
        List of maps with the same geometry.
    method : {"gaussian_errors"}
        * gaussian_errors :
            Under the gaussian error approximation the likelihood is given by the gaussian distibution.
            The product of gaussians is also a gaussian so can derive dnde, dnde_err, and ts.
        * distrib :
            Likelihood profile approximation assuming that probabilities distributions for
            flux points correspond to asymmetric gaussians and for upper limits to complementary error functions.
            Use available quantities among dnde, dnde_err, dnde_errp, dnde_errn, dnde_ul, and ts.
        * profile :
            Sum the likelihood profile maps.
            The flux maps must contains the `stat_scan` maps.

        Default is "gaussian_errors" which is the faster but least accurate solution,
        "distrib"  will be more accurate if dnde_errp and dnde_errn are available,
        "profile"  will be even more accurate if "stat_scan" is available.

    reference_model : `~gammapy.modeling.models.SkyModel`, optional
        Reference model to use for conversions.
        Default is None and is will use the reference_model of the first FluxMaps in the list.

    dnde_scan_axis : `~gammapy.maps.MapAxis`
        Map axis providing the dnde values used to compute the profile.
        Default is None and it will be derived from the first FluxMaps in the list.
        Used only if `method` is distrib or profile.

    Returns
    -------
    flux_maps : `~gammapy.estimators.FluxMaps`
        Joint flux map.
    """
    gtis = [map_.gti for map_ in maps if map_.gti is not None]
    if np.any(gtis):
        gti = gtis[0].copy()
        for k in range(1, len(gtis)):
            gti.stack(gtis[k])
    else:
        gti = None
    # TODO : change this once we have stackable metadata objets
    metas = [map_.meta for map_ in maps if map_.meta is not None]
    meta = {}
    if np.any(metas):
        for data in metas:
            meta.update(data)
    if reference_model is None:
        reference_model = maps[0].reference_model

    if method == "gaussian_errors":
        means = [map_.dnde.copy() for map_ in maps]
        sigmas = [map_.dnde_err.copy() for map_ in maps]
        # compensate for the ts deviation from gaussian approximation expectation in each map
        ts_diff = np.nansum(
            [
                map_.ts.data - (map_.dnde.data / map_.dnde_err.data) ** 2
                for map_ in maps
            ],
            axis=0,
        )

        mean = means[0]
        sigma = sigmas[0]

        for k in range(1, len(means)):
            mean_k = means[k].quantity.to_value(mean.unit)
            sigma_k = sigmas[k].quantity.to_value(sigma.unit)

            mask_valid = np.isfinite(mean) & np.isfinite(sigma) & (sigma.data != 0)
            mask_valid_k = np.isfinite(mean_k) & np.isfinite(sigma_k) & (sigma_k != 0)
            mask = mask_valid & mask_valid_k
            mask_k = ~mask_valid & mask_valid_k

            mean.data[mask] = (
                (mean.data * sigma_k**2 + mean_k * sigma.data**2)
                / (sigma.data**2 + sigma_k**2)
            )[mask]
            sigma.data[mask] = (
                sigma.data * sigma_k / np.sqrt(sigma.data**2 + sigma_k**2)
            )[mask]

            mean.data[mask_k] = mean_k[mask_k]
            sigma.data[mask_k] = sigma_k[mask_k]

        ts = mean * mean / sigma / sigma + ts_diff
        ts.data[~np.isfinite(ts.data)] = np.nan

        kwargs = dict(
            sed_type="dnde", reference_model=reference_model, meta=meta, gti=gti
        )
        return FluxMaps.from_maps(dict(dnde=mean, dnde_err=sigma, ts=ts), **kwargs)

    elif method in ["distrib", "profile"]:
        if dnde_scan_axis is None:
            dnde_scan_axis = _default_scan_map(maps[0]).geom.axes["dnde"]
        for k, map_ in enumerate(maps):
            if method == "profile":
                map_stat_scan = interpolate_profile_map(map_, dnde_scan_axis)
            else:
                map_stat_scan = approximate_profile_map(map_, dnde_scan_axis)
            map_stat_scan.data[np.isnan(map_stat_scan.data)] = 0.0
            if k == 0:
                stat_scan = map_stat_scan
            else:
                stat_scan.data += map_stat_scan.data

        return get_flux_map_from_profile(
            {"stat_scan": stat_scan},
            reference_model=reference_model,
            meta=meta,
            gti=gti,
        )

    else:
        raise ValueError(
            f'Invalid method provided : {method}. Available methods are : "gaussian_errors", "distrib", "profile"'
        )


def _default_scan_map(flux_map, dnde_scan_axis=None):
    if dnde_scan_axis is None:
        dnde_scan_axis = MapAxis(
            _generate_scan_values() * flux_map.dnde_ref.squeeze(),
            interp="lin",
            node_type="center",
            name="dnde",
            unit=flux_map.dnde_ref.unit,
        )

    geom = flux_map.dnde.geom
    geom_scan = geom.to_image().to_cube([dnde_scan_axis] + list(geom.axes))
    return Map.from_geom(geom_scan, data=np.nan, unit="")


def interpolate_profile_map(flux_map, dnde_scan_axis=None):
    """Interpolate sparse likelihood profile to regular grid.

    Parameters
    ----------
    flux_map : `~gammapy.estimators.FluxMaps`
        Flux map.
    dnde_scan_axis : `~gammapy.maps.MapAxis`
        Map axis providing the dnde values used to compute the profile.
        Default is None and it will be derived from the flux_map.

    Returns
    -------
    scan_map: `~gammapy.estimators.Maps`
        Likelihood profile map.

    """
    stat_scan = _default_scan_map(flux_map, dnde_scan_axis)
    dnde_scan_axis = stat_scan.geom.axes["dnde"]

    mask_valid = ~np.isnan(flux_map.dnde.data)
    dnde_scan_values = flux_map.dnde_scan_values.quantity.to_value(dnde_scan_axis.unit)

    for ij, il, ik in zip(*np.where(mask_valid)):
        spline = InterpolatedUnivariateSpline(
            dnde_scan_values[ij, :, il, ik],
            flux_map.stat_scan.data[ij, :, il, ik],
            k=1,
            ext="raise",
            check_finite=True,
        )
        stat_scan.data[ij, :, il, ik] = spline(dnde_scan_axis.center)
    return stat_scan


def approximate_profile_map(
    flux_map, dnde_scan_axis=None, sqrt_ts_threshold_ul="ignore"
):
    """Likelihood profile approximation assuming that probabilities distributions for
    flux points correspond to asymmetric gaussians and for upper limits to complementary error functions.
    Use available quantities among dnde, dnde_err, dnde_errp, dnde_errn, dnde_ul and ts.

    Parameters
    ----------
    flux_map : `~gammapy.estimators.FluxMaps`
        Flux map.
    dnde_scan_axis : `~gammapy.maps.MapAxis`
        Map axis providing the dnde values used to compute the profile.
        Default is None and it will be derived from the flux_map.
    sqrt_ts_threshold_ul : int
        Threshold value in sqrt(TS) for upper limits.
        Default is `ignore` and no threshold is applied.
        Setting to `None` will use the one of `flux_map`.

    Returns
    -------
    scan_map: `~gammapy.estimators.Maps`
        Likelihood profile map.
    """
    stat_approx = _default_scan_map(flux_map, dnde_scan_axis)
    dnde_coord = stat_approx.geom.get_coord()["dnde"].value

    if sqrt_ts_threshold_ul is None:
        sqrt_ts_threshold_ul = flux_map.sqrt_ts_threshold_ul

    mask_valid = ~np.isnan(flux_map.dnde.data)
    ij, il, ik = np.where(mask_valid)
    loc = flux_map.dnde.data[mask_valid][:, None]
    value = dnde_coord[ij, :, il, ik]
    try:
        mask_p = dnde_coord >= flux_map.dnde.data
        mask_p2d = mask_p[ij, :, il, ik]
        new_axis = np.ones(mask_p2d.shape[1], dtype=bool)[None, :]
        scale = np.zeros(mask_p2d.shape)
        scale[mask_p2d] = (flux_map.dnde_errp.data[mask_valid][:, None] * new_axis)[
            mask_p2d
        ]
        scale[~mask_p2d] = (flux_map.dnde_errn.data[mask_valid][:, None] * new_axis)[
            ~mask_p2d
        ]
    except AttributeError:
        scale = flux_map.dnde_err.data[mask_valid]
        scale = scale[:, None]
    stat_approx.data[ij, :, il, ik] = ((value - loc) / scale) ** 2

    try:
        invalid_value = 999
        stat_min_p = (stat_approx.data + invalid_value * (~mask_p)).min(
            axis=1, keepdims=True
        )
        stat_min_m = (stat_approx.data + invalid_value * mask_p).min(
            axis=1, keepdims=True
        )

        mask_minp = mask_p & (stat_min_p > stat_min_m)
        stat_approx.data[mask_minp] = (stat_approx.data + stat_min_m - stat_min_p)[
            mask_minp
        ]
        mask_minn = ~mask_p & (stat_min_m >= stat_min_p)
        stat_approx.data[mask_minn] = (stat_approx.data + stat_min_p - stat_min_m)[
            mask_minn
        ]
    except NameError:
        pass

    if not sqrt_ts_threshold_ul == "ignore" and sqrt_ts_threshold_ul is not None:
        mask_ul = (flux_map.sqrt_ts.data < sqrt_ts_threshold_ul) & ~np.isnan(
            flux_map.dnde_ul.data
        )
        ij, il, ik = np.where(mask_ul)
        value = dnde_coord[ij, :, il, ik]
        loc_ul = flux_map.dnde_ul.data[mask_ul][:, None]
        scale_ul = flux_map.dnde_ul.data[mask_ul][:, None]
        stat_approx.data[ij, :, il, ik] = -2 * np.log(
            (special.erfc((-loc_ul + value) / scale_ul) / 2)
            / (special.erfc((-loc_ul + 0) / scale_ul) / 2)
        )

    stat_approx.data[np.isnan(stat_approx.data)] = np.inf
    stat_approx.data += -flux_map.ts.data - stat_approx.data.min(axis=1)
    return stat_approx


def get_flux_map_from_profile(
    flux_map, n_sigma=1, n_sigma_ul=2, reference_model=None, meta=None, gti=None
):
    """Create a new flux map using the likehood profile (stat_scan)
    to get ts, dnde, dnde_err, dnde_errp, dnde_errn, and dnde_ul.

    Parameters
    ----------
    flux_maps : `~gammapy.estimators.FluxMaps` or dict of `~gammapy.maps.WcsNDMap`
        Flux map or dict containing  a `stat_scan` entry
    n_sigma : int
        Number of sigma for flux error. Default is 1.
    n_sigma_ul : int
        Number of sigma for flux upper limits. Default is 2.
    reference_model : `~gammapy.modeling.models.SkyModel`, optional
        The reference model to use for conversions. If None, a model consisting
        of a point source with a power law spectrum of index 2 is assumed.
        Default is None and the one of `flux_map` will be used if available
    meta : dict, optional
        Dict of metadata.
        Default is None and the one of `flux_map` will be used if available
    gti : `~gammapy.data.GTI`, optional
        Maps GTI information.
        Default is None and the one of `flux_map` will be used if available

    Returns
    -------
    flux_maps : `~gammapy.estimators.FluxMaps`
        Flux map.

    """
    if isinstance(flux_map, dict):
        output_maps = flux_map
    else:
        if reference_model is None:
            reference_model = flux_map.reference_model
        if gti is None:
            gti = flux_map.gti
        if meta is None:
            meta = flux_map.meta

        output_maps = dict(
            stat_scan=flux_map.stat_scan, dnde_scan_values=flux_map.dnde_scan_values
        )

    if getattr(flux_map, "dnde_scan_values", False):
        dnde_coord = flux_map["dnde_scan_values"].quantity
    else:
        dnde_coord = flux_map["stat_scan"].geom.get_coord()["dnde"]

    geom = (
        flux_map["stat_scan"]
        .geom.to_image()
        .to_cube([flux_map["stat_scan"].geom.axes["energy"]])
    )

    ts = -flux_map["stat_scan"].data.min(axis=1) * u.Unit("")
    ind = flux_map["stat_scan"].data.argmin(axis=1)
    ij, ik, il = np.indices(ind.shape)
    dnde = dnde_coord[ij, ind, ik, il]

    maskp = dnde_coord > dnde
    stat_diff = flux_map["stat_scan"].data - flux_map["stat_scan"].data.min(axis=1)

    invalid_value = 999
    ind = np.abs(stat_diff + invalid_value * maskp - n_sigma**2).argmin(axis=1)
    dnde_errn = dnde - dnde_coord[ij, ind, ik, il]

    ind = np.abs(stat_diff + invalid_value * (~maskp) - n_sigma**2).argmin(axis=1)
    dnde_errp = dnde_coord[ij, ind, ik, il] - dnde

    ind = np.abs(stat_diff + invalid_value * (~maskp) - n_sigma_ul**2).argmin(axis=1)
    dnde_ul = dnde_coord[ij, ind, ik, il]

    dnde_err = (dnde_errn + dnde_errp) / 2

    maps = dict(
        ts=ts,
        dnde=dnde,
        dnde_err=dnde_err,
        dnde_errn=dnde_errn,
        dnde_errp=dnde_errp,
        dnde_ul=dnde_ul,
    )
    for key in maps.keys():
        maps[key] = Map.from_geom(geom, data=maps[key].value, unit=maps[key].unit)
    kwargs = dict(sed_type="dnde", gti=gti, reference_model=reference_model, meta=meta)
    output_maps.update(maps)
    return FluxMaps.from_maps(output_maps, **kwargs)


def _generate_scan_values(power_min=-6, power_max=2, relative_error=1e-2):
    """Values sampled such as we can probe a given `relative_error` on the norm
    between 10**`power_min` and 10**`power_max`.

    """
    arrays = []
    for power in range(power_min, power_max):
        vmin = 10**power
        vmax = 10 ** (power + 1)
        bin_per_decade = int((vmax - vmin) / (vmin * relative_error))
        arrays.append(np.linspace(vmin, vmax, bin_per_decade + 1, dtype=np.float32))
    scan_1side = np.unique(np.concatenate(arrays))
    return np.concatenate((-scan_1side[::-1], [0], scan_1side))


def _get_default_norm(
    norm,
    scan_min=0.2,
    scan_max=5,
    scan_n_values=11,
    scan_values=None,
    interp="lin",
):
    """Create default norm parameter."""
    if norm is None or isinstance(norm, dict):
        norm_kwargs = dict(
            name="norm",
            value=1,
            unit="",
            interp=interp,
            frozen=False,
            scan_min=scan_min,
            scan_max=scan_max,
            scan_n_values=scan_n_values,
            scan_values=scan_values,
        )
        if isinstance(norm, dict):
            norm_kwargs.update(norm)
        try:
            norm = Parameter(**norm_kwargs)
        except TypeError as error:
            raise TypeError(f"Invalid dict key for norm init : {error}")
    if norm.name != "norm":
        raise ValueError("norm.name is not 'norm'")
    return norm


def _get_norm_scan_values(norm, result):
    """Compute norms based on the fit result to sample the stat profile at different scales."""
    norm_err = result["norm_err"]
    norm_value = result["norm"]
    if ~np.isfinite(norm_err) or norm_err == 0:
        norm_err = 0.1
    if ~np.isfinite(norm_value) or norm_value == 0:
        norm_value = 1.0
    sparse_norms = np.concatenate(
        (
            norm_value + np.linspace(-2.5, 2.5, 51) * norm_err,
            norm_value + np.linspace(-10, 10, 21) * norm_err,
            np.abs(norm_value) * np.linspace(-10, 10, 21),
            np.linspace(-10, 10, 21),
            np.linspace(norm.scan_values[0], norm.scan_values[-1], 2),
        )
    )
    sparse_norms = np.unique(sparse_norms)
    if len(sparse_norms) != 109:
        rand_norms = 20 * np.random.rand(109 - len(sparse_norms)) - 10
        sparse_norms = np.concatenate((sparse_norms, rand_norms))
    return np.sort(sparse_norms)
