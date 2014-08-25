# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools to create profiles (i.e. 1D "slices" from 2D images)"""
from __future__ import print_function, division
import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from ..image import coordinates

__all__ = ['compute_binning',
           'FluxProfile',
           'image_profile',
           ]


def compute_binning(data, n_bins, method='equal width', eps=1e-10):
    """Computes 1D array of bin edges.

    The range of the bin_edges is always [min(data), max(data)]

    Note that bin_edges has n_bins bins, i.e. length n_bins + 1.

    Parameters
    ----------
    data : array_like
        Data to be binned (any dimension)
    n_bins : int
        Number of bins
    method : str
        One of: 'equal width', 'equal entries'
    eps : float
        added to range so that the max data point is inside the
        last bin. If eps=0 it falls on the right edge of the last
        data point and thus would be not cointained.

    Returns
    -------
    bin_edges : 1D ndarray
        Array of bin edges.
    """
    data = np.asanyarray(data)

    if method == 'equal width':
        bin_edges = np.linspace(np.nanmin(data), np.nanmax(data), n_bins + 1)
    elif method == 'equal entries':
        # We use np.percentile to achieve equal number of entries per bin
        # It takes a list of quantiles in the range [0, 100] as input 
        quantiles = list(np.linspace(0, 100, n_bins + 1))
        bin_edges = np.percentile(data, quantiles)
    else:
        raise ValueError('Invalid option: method = {0}'.format(method))
    bin_edges[-1] += eps
    return bin_edges


class FluxProfile(object):
    """Flux profile.

    Note: over- and underflow is ignored and not stored in the profile

    Note: this is implemented by creating bin labels and storing the
    input 2D data in 1D `pandas.DataFrame` tables.
    The 1D profile is also stored as a `pandas.DataFrame` and computed
    using the fast and flexible pandas groupby and apply functions.

    * TODO: take mask into account everywhere
    * TODO: separate FluxProfile.profile into a separate ProfileStack or HistogramStack class?
    * TODO: add ``solid_angle`` to input arrays.

    Parameters
    ----------
    x_image : array_like
        Label image (2-dimensional)
    x_edges : array_like
        Defines binning in ``x`` (could be GLON, GLAT, DIST, ...)
    counts, background, exposure : array_like
        Input images (2-dimensional)
    mask : array_like
        possibility to mask pixels (i.e. ignore in computations)
    """

    def __init__(self, x_image, x_edges, counts, background, exposure, mask=None):
        import pandas as pd
        # Make sure inputs are numpy arrays
        x_edges = np.asanyarray(x_edges)
        x_image = np.asanyarray(x_image)
        counts = np.asanyarray(counts)
        background = np.asanyarray(background)
        exposure = np.asanyarray(exposure)
        if mask:
            mask = np.asanyarray(mask)

        assert (x_image.shape == counts.shape == background.shape ==
                exposure.shape == mask.shape)

        # Remember the shape of the 2D input arrays
        self.shape = x_image.shape

        # Store all input data as 1D vectors in a pandas.DataFrame
        d = pd.DataFrame(index=np.arange(x_image.size))
        d['x'] = x_image.flat
        # By default np.digitize uses 0 as the underflow bin.
        # Here we ignore under- and overflow, thus the -1
        d['label'] = np.digitize(d['x'], x_edges) - 1
        d['counts'] = counts.flat
        d['background'] = background.flat
        d['exposure'] = exposure.flat
        if mask:
            d['mask'] = mask.flat
        else:
            d['mask'] = np.ones_like(d['x'])
        self.data = d

        self.bins = np.arange(len(x_edges) + 1)

        # Store all per-profile bin info in a pandas.DataFrame
        p = pd.DataFrame(index=np.arange(x_edges.size - 1))
        p['x_lo'] = x_edges[:-1]
        p['x_hi'] = x_edges[1:]
        p['x_center'] = 0.5 * (p['x_hi'] + p['x_lo'])
        p['x_width'] = p['x_hi'] - p['x_lo']
        self.profile = p

        # The x_edges array is longer by one than the profile arrays,
        # so we store it separately
        self.x_edges = x_edges

    def compute(self):
        """Compute the flux profile.

        TODO: call `~gammapy.stats.compute_total_stats` instead.

        Note: the current implementation is very inefficienct in speed and memory.
        There are various fast implementations, but none is flexible enough to
        allow combining many input quantities (counts, background, exposure) in a
        flexlible way:
        - `numpy.histogram`
        - `scipy.ndimage.measurements.labeled_comprehension` and special cases

        pandas DataFrame groupby followed by apply is flexible enough, I think:

        http://pandas.pydata.org/pandas-docs/dev/groupby.html

        Returns
        -------
        results : dict
            Dictionary of profile measurements, also stored in ``self.profile``.

        See also
        --------
        gammapy.stats.compute_total_stats
        """
        # Shortcuts to access class info needed in this method
        d = self.data
        # Here the pandas magic happens: we group pixels by label
        g = d.groupby('label')
        p = self.profile

        # Compute number of entries in each profile bin
        p['n_entries'] = g['x'].aggregate(len)
        for name in ['counts', 'background', 'exposure']:
            p['{0}'.format(name)] = g[name].sum()
            #p['{0}_mean'.format(name)] = p['{0}_sum'.format(name)] / p['n_entries']
        p['excess'] = p['counts'] - p['background']
        p['flux'] = p['excess'] / p['exposure']

        return p

    def plot(self, which='n_entries', xlabel='Distance (deg)', ylabel=None):
        """Plot flux profile.

        Parameters
        ----------
        TODO
        """
        import matplotlib.pyplot as plt
        if ylabel == None:
            ylabel = which
        p = self.profile
        x = p['x_center']
        xerr = 0.5 * p['x_width']
        y = p[which]
        plt.errorbar(x, y, xerr=xerr, fmt='o')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        #plt.ylim(-10, 20)

    def save(self, filename):
        """Save all profiles to a FITS file.

        Parameters
        ----------
        """
        raise NotImplementedError


def image_profile(profile_axis, image, lats, lons, binsz, counts=None,
                  mask=None, errors=False, standard_error=0.1):
    """Creates a latitude or longitude profile from input flux image HDU.

    Parameters
    ----------
    profile_axis : String, {'lat', 'lon'}
        Specified whether galactic latitude ('lat') or longitude ('lon')
        profile is to be returned.
    image : `~astropy.io.fits.ImageHDU`
        Image HDU object to produce GLAT or GLON profile.
    lats : array_like
        Specified as [GLAT_min, GLAT_max], with GLAT_min and GLAT_max
        as floats. A 1x2 array specifying the maximum and minimum latitudes to
        include in the region of the image for which the profile is formed, 
        which should be within the spatial bounds of the image.
    lons : array_like
        Specified as [GLON_min, GLON_max], with GLON_min and GLON_max
        as floats. A 1x2 array specifying the maximum and minimum longitudes to
        include in the region of the image for which the profile is formed, 
        which should be within the spatial bounds of the image.
    binsz : float
        Latitude bin size of the resulting latitude profile. This should be
        no less than 5 times the pixel resolution.
    counts : `~astropy.io.fits.ImageHDU`
        Counts image to allow Poisson errors to be calculated. If not provided,
        a standard_error should be provided, or zero errors will be returned.
        (Optional).
    mask : array_like
        2D mask array, matching spatial dimensions of input image. (Optional).
        A mask value of True indicates a value that should be ignored, while a
        mask value of False indicates a valid value.
    errors : bool
        If True, computes errors, if possible, according to provided inputs.
        If False (default), returns all errors as zero.
    standard_error : float
        If counts image is not provided, but error values required, this
        specifies a standard fractional error to be applied to values.
        Default = 0.1.

    Returns
    -------
    table : `~astropy.table.Table`
        Galactic latitude or longitude profile as table, with latitude bin
        boundaries, profile values and errors.
    """
    if profile_axis == 'lat':
        res_bound = 5 * np.abs(image.header['CDELT2'])
    elif profile_axis == 'lon':
        res_bound = 5 * np.abs(image.header['CDELT1'])
    # Assert that the called binsz is at least 5 times
    # GLAT image resolution to avoid binning bug
    if binsz < res_bound:
        raise ValueError('Minimum bin size is 5 times resolution.')

    lon, lat = coordinates(image)
    mask_init = (lats[0] <= lat) & (lat < lats[1])
    mask_bounds = mask_init & (lons[0] <= lon) & (lon < lons[1])
    if mask != None:
        mask = mask_bounds & mask
    else:
        mask = mask_bounds

    # Need to preserve shape here so use multiply
    cut_image = image.data * mask
    if counts != None:
        cut_counts = counts.data * mask
    values = []
    count_vals = []

    if profile_axis == 'lat':

        bins = np.arange((lats[1] - lats[0]) / binsz)
        glats_min = lats[0] + bins[:-1] * binsz
        glats_max = lats[0] + bins[1:] * binsz

        # Table is required here to avoid rounding problems with floats
        bounds = Table([glats_min, glats_max], names=('GLAT_MIN', 'GLAT_MAX'))
        for bin in bins[:-1]:
            lat_mask = (bounds['GLAT_MIN'][bin] <= lat) & (lat < bounds['GLAT_MAX'][bin])
            lat_band = cut_image[lat_mask]
            values.append(lat_band.sum())
            if counts != None:
                count_band = cut_counts[lat_mask]
                count_vals.append(count_band.sum())
            else:
                count_vals.append(0)

    elif profile_axis == 'lon':

        bins = np.arange((lons[1] - lons[0]) / binsz)
        glons_min = lons[0] + bins[:-1] * binsz
        glons_max = lons[0] + bins[1:] * binsz

        # Table is required here to avoid rounding problems with floats
        bounds = Table([glons_min, glons_max], names=('GLON_MIN', 'GLON_MAX'))
        for bin in bins[:-1]:
            lon_mask = (bounds['GLON_MIN'][bin] <= lon) & (lon < bounds['GLON_MAX'][bin])
            lon_band = cut_image[lon_mask]
            values.append(lon_band.sum())
            if counts != None:
                count_band = cut_counts[lon_mask]
                count_vals.append(count_band.sum())
            else:
                count_vals.append(0)

    if errors == True:
        if counts != None:
            rel_errors = 1. / np.sqrt(count_vals)
            error_vals = values * rel_errors
        else:
            error_vals = values * (np.ones(len(values)) * standard_error)
    else:
        error_vals = np.zeros_like(values)

    if profile_axis == 'lat':
        table = Table([Quantity(glats_min, 'deg'),
                       Quantity(glats_max, 'deg'),
                       values,
                       error_vals],
                      names=('GLAT_MIN', 'GLAT_MAX', 'BIN_VALUE', 'BIN_ERR'))

    elif profile_axis == 'lon':
        table = Table([Quantity(glons_min, 'deg'),
                       Quantity(glons_max, 'deg'),
                       values,
                       error_vals],
                      names=('GLON_MIN', 'GLON_MAX', 'BIN_VALUE', 'BIN_ERR'))

    return table
