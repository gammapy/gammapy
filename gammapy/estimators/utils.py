# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.ndimage
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.datasets.map import MapEvaluator
from gammapy.maps import WcsNDMap
from gammapy.modeling.models import (
    ConstantFluxSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.stats import compute_fpp, compute_fvar
from .map.core import FluxMaps

__all__ = [
    "estimate_exposure_reco_energy",
    "find_peaks",
    "resample_energy_edges",
    "compute_lightcurve_fvar",
    "compute_lightcurve_fpp",
    "find_peaks_in_flux_map",
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

    - ``x`` and ``y`` are the pixel coordinates (first pixel at zero)
    - ``ra`` and ``dec`` are the RA / DEC sky coordinates (ICRS frame)
    - ``value`` is the pixel value

    It is sorted by peak value, starting with the highest value.

    If there are no pixel values above the threshold, an empty table is returned.

    There are more featureful peak finding and source detection methods
    e.g. in the ``photutils`` or ``scikit-image`` Python packages.

    Parameters
    ----------
    image : `~gammapy.maps.WcsNDMap`
        Image like Map
    threshold : float or array-like
        The data value or pixel-wise data values to be used for the
        detection threshold.  A 2D ``threshold`` must have the same
        shape as the map ``data``.
    min_distance : int or `~astropy.units.Quantity`
        Minimum distance between peaks. An integer value is interpreted
        as pixels.

    Returns
    -------
    output : `~astropy.table.Table`
        Table with parameters of detected peaks
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
    """Find local TS peaks for a given Map.

    Utilises the find_peaks function to find various parameters from FluxMaps.

    Parameters
    ----------
    maps : `~gammapy.estimators.map.FluxMaps`
        Input flux map object
    threshold : float or array-like
        The TS data value or pixel-wise TS data values to be used for the
        detection threshold.  A 2D ``threshold`` must have the same
        shape as the map ``data``.
    min_distance : int or `~astropy.units.Quantity`
        Minimum distance between peaks. An integer value is interpreted
        as pixels.

    Returns
    -------
    output : `~astropy.table.Table`
        Table with parameters of detected peaks
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
            the input dataset
    spectral_model : `~gammapy.modeling.models.SpectralModel`
            assumed spectral shape. If none, a Power Law of index 2 is assumed
    normalize : bool
        Normalize the exposure to the total integrated flux of the spectral model.
        When not normalized it directly gives the predicted counts from the spectral
        model.

    Returns
    -------
    exposure : `Map`
        Exposure map in reconstructed energy
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
    dataset:`~gammapy.datasets.SpectrumDataset` or `~gammapy.datasets.SpectrumDatasetOnOff`
            the input dataset
    conditions : dict
        Keyword arguments containing the per-bin conditions used to resample the axis.
        Available options are: 'counts_min', 'background_min', 'excess_min', 'sqrt_ts_min',
        'npred_min', 'npred_background_min', 'npred_signal_min'.
    Returns
    -------
    energy_edges : list of `~astropy.units.Quantity`
        Energy edges for the resampled energy axis.
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
    r"""Compute the fractional excess variance of the input lightcurve.

    Internally calls the `~gammapy.stats.compute_fvar` function


    Parameters
    ----------
    lightcurve : '~gammapy.estimators.FluxPoints'
        the lightcurve object
    flux_quantity : str
        flux quantity to use for calculation. Should be 'dnde', 'flux', 'e2dnde' or 'eflux'. Default is 'flux'.
        Useful in case of custom lightcurves computed outside gammapy

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
    r"""Compute the point-to-point excess variance of the input lightcurve.

    Internally calls the `~gammapy.stats.compute_fpp` function


    Parameters
    ----------
    lightcurve : `~gammapy.estimators.FluxPoints`
        the lightcurve object
    flux_quantity : str
        flux quantity to use for calculation. Should be 'dnde', 'flux', 'e2dnde' or 'eflux'. Default is 'flux'.
        Useful in case of custom lightcurves computed outside gammapy

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
        meta=lightcurve.meta,
    )

    table.meta["flux_quantity"] = flux_quantity

    return table
