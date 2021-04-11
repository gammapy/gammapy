# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.ndimage
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from gammapy.datasets.map import MapEvaluator
from gammapy.maps import WcsNDMap
from gammapy.modeling.models import (
    ConstantFluxSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)

__all__ = ["find_roots", "find_peaks", "estimate_exposure_reco_energy"]

from scipy.optimize import root_scalar


def find_roots(
    f,
    x_bounds,
    nbin=1000,
    args=(),
    method="brentq",
    fprime=None,
    fprime2=None,
    xtol=None,
    rtol=None,
    maxiter=None,
    options=None,
):
    """ Find roots of a scalar function within a given range.
    
    Parameters
    ----------
    f : callable
        A function to find roots of.
    x_bounds : A sequence of 2 floats
        The search range to find roots.
    nbin : int
        Number of bins to sample the search range
    args : tuple, optional
        Extra arguments passed to the objective function and its derivative(s).
    method : str, optional
        Solver available in `~scipy.optimize.root_scalar`.  Should be one of :
            - 'brentq' (default),
            - 'brenth',
            - 'bisect', 
            - 'ridder',
            - 'toms748',
            - 'newton',
            - 'secant',
            - 'halley',
    fprime : bool or callable, optional
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of the objective function and of the derivative.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    fprime2 : bool or callable, optional
        If `fprime2` is a boolean and is True, `f` is assumed to return the
        value of the objective function and of the
        first and second derivatives.
        `fprime2` can also be a callable returning the second derivative of `f`.
        In this case, it must accept the same arguments as `f`.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options : dict, optional
        A dictionary of solver options.
        See `~scipy.optimize.root_scalar` for details.

    Returns
    -------
    roots : `~numpy.array` or None
        The function roots in the search range.
        If no roots are not found, it returns None.
        If the solver failed to converge in a bracketing range array element is NaN.
    """

    kwargs = dict(
        args=(),
        method=method,
        fprime=fprime,
        fprime2=fprime2,
        xtol=xtol,
        rtol=rtol,
        maxiter=maxiter,
        options=options,
    )

    x = np.linspace(x_bounds[0], x_bounds[1], nbin)
    signs = np.sign(f(x))
    ind = np.where(signs[:-1] != signs[1:])[0]
    nroots = len(ind)
    if nroots > 0:
        roots = np.ones(nroots) * np.nan
    else:
        return None

    for k, idx in enumerate(ind):
        bracket = [x[idx], x[idx + 1]]
        if method in ["bisection", "brentq", "brenth", "ridder", "toms748"]:
            kwargs["bracket"] = bracket
        elif method in ["secant", "newton", "halley"]:
            kwargs["x0"] = bracket[0]
            kwargs["x1"] = bracket[1]
        else:
            raise ValueError(f'Unknown solver "{method}"')
        sol = root_scalar(f, **kwargs)
        if sol.converged:
            roots[k] = sol.root
    return roots


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
        shape as tha map ``data``.
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


def estimate_exposure_reco_energy(dataset, spectral_model=None):
    """Estimate an exposure map in reconstructed energy.

    Parameters
    ----------
    dataset:`~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            the input dataset
    spectral_model: `~gammapy.modeling.models.SpectralModel`
            assumed spectral shape. If none, a Power Law of index 2 is assumed

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

    edisp = None

    if dataset.edisp is not None:
        edisp = dataset.edisp.get_edisp_kernel(position=None, energy_axis=energy_axis)

    meval = MapEvaluator(model=model, exposure=dataset.exposure, edisp=edisp)
    npred = meval.compute_npred()
    ref_flux = spectral_model.integral(energy_axis.edges[:-1], energy_axis.edges[1:])
    reco_exposure = npred / ref_flux[:, np.newaxis, np.newaxis]
    return reco_exposure
