# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools to create profiles (i.e. 1D "slices" from 2D images)"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy.table import Table
from astropy import units as u
from .core import SkyImage

__all__ = [
    'ImageProfile',
    'ImageProfileEstimator'
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
        raise ValueError('Invalid option: method = {}'.format(method))

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
            p['{}_sum'.format(name)] = g[name].sum()
            p['{}_mean'.format(name)] = p['{}_sum'.format(name)] / p['n_entries']
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
        if ylabel is None:
            ylabel = which
        p = self.profile
        x = p['x_center']
        xerr = 0.5 * p['x_width']
        y = p[which]
        plt.errorbar(x, y, xerr=xerr, fmt='o')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        # plt.ylim(-10, 20)

    def save(self, filename):
        """Save all profiles to a FITS file.

        Parameters
        ----------
        """
        raise NotImplementedError


# TODO: implement measuring profile along arbitrary directions
# TODO: think better about error handling. e.g. MC based methods
class ImageProfileEstimator(object):
    """
    Estimate profile from image.

    Parameters
    ----------
    x_edges : `~astropy.coordinates.Angle`
        Coordinate edges to define a custom measument grid (optional).
    method : ['sum', 'mean']
        Compute sum or mean within profile bins.
    axis : ['lon', 'lat']
        Along which axis to estimate the profile.

    Examples
    --------
    This example shows how to compute a counts profile for the Fermi galactic
    center region:

    .. code:: python

        import matplotlib.pyplot as plt
        from gammapy.datasets import FermiGalacticCenter
        from gammapy.image import ImageProfile, ImageProfileEstimator
        from gammapy.image import SkyImage
        from astropy import units as u

        # load example data
        fermi_cts = SkyImage.from_image_hdu(FermiGalacticCenter.counts())
        fermi_cts.unit = u.count

        # set up profile estimator and run
        p = ImageProfileEstimator(axis='lon', method='sum')
        profile = p.run(fermi_cts)

        # smooth profile and plot
        smoothed = profile.smooth(kernel='gauss')
        smoothed.peek()

        plt.show()

    """

    def __init__(self, x_edges=None, method='sum', axis='lon'):

        self._x_edges = x_edges

        if method not in ['sum', 'mean']:
            raise ValueError("Not a valid method, choose either 'sum' or 'mean'")

        if axis not in ['lon', 'lat']:
            raise ValueError("Not a valid axis, choose either 'lon' or 'lat'")

        self.parameters = OrderedDict(method=method, axis=axis)

    def _get_x_edges(self, image):
        """
        Get x_ref coordinate array.
        """
        if self._x_edges is not None:
            return self._x_edges

        p = self.parameters
        coordinates = image.coordinates(mode='edges')

        if p['axis'] == 'lat':
            x_edges = coordinates[:, 0].data.lat
        elif p['axis'] == 'lon':
            lon = coordinates[0, :].data.lon
            x_edges = lon.wrap_at('180d')
        return x_edges

    def _estimate_profile(self, image, image_err, mask):
        """
        Estimate image profile.
        """
        from scipy import ndimage

        p = self.parameters
        labels = self._label_image(image, mask)

        profile_err = None

        index = np.arange(1, len(self._get_x_edges(image)))

        if p['method'] == 'sum':
            profile = ndimage.sum(image.data, labels.data, index)

            if image.unit.is_equivalent('counts'):
                profile_err = np.sqrt(profile)
            elif image_err:
                # gaussian error propagation
                err_sum = ndimage.sum(image_err.data ** 2, labels.data, index)
                profile_err = np.sqrt(err_sum)

        elif p['method'] == 'mean':
            # gaussian error propagation
            profile = ndimage.mean(image.data, labels.data, index)
            if image_err:
                N = ndimage.sum(~np.isnan(image_err.data), labels.data, index)
                err_sum = ndimage.sum(image_err.data ** 2, labels.data, index)
                profile_err = np.sqrt(err_sum) / N

        return profile, profile_err

    def _label_image(self, image, mask=None):
        """
        Compute label image.
        """
        p = self.parameters

        label_image = SkyImage.empty_like(image)
        coordinates = image.coordinates()
        x_edges = self._get_x_edges(image)

        if p['axis'] == 'lon':
            lon = coordinates.data.lon.wrap_at('180d')
            data = np.digitize(lon.degree, x_edges.deg)

        elif p['axis'] == 'lat':
            lat = coordinates.data.lat
            data = np.digitize(lat.degree, x_edges.deg)

        if mask is not None:
            # assign masked values to background
            data[mask.data] = 0

        label_image.data = data
        return label_image

    def run(self, image, image_err=None, mask=None):
        """
        Run image profile estimator.

        Parameters
        ----------
        image : `~gammapy.image.SkyImage`
            Input image to run profile estimator on.
        image_err : `~gammapy.image.SkyImage`
            Input error image to run profile estimator on.
        mask : `~gammapy.image.SkyImage`
            Optional mask to exclude regions from the measurement.

        Returns
        -------
        profile : `ImageProfile`
            Result image profile object.
        """
        p = self.parameters
        image = image.copy()

        if image.unit.is_equivalent('count'):
            image_err = SkyImage.empty_like(image)
            image_err.data = np.sqrt(image.data)

        if image_err:
            image_err = image_err.copy()

        profile, profile_err = self._estimate_profile(image, image_err, mask)

        result = Table()
        x_edges = self._get_x_edges(image)
        result['x_min'] = x_edges[:-1]
        result['x_max'] = x_edges[1:]
        result['x_ref'] = (x_edges[:-1] + x_edges[1:]) / 2
        result['profile'] = profile * image.unit

        if profile_err is not None:
            result['profile_err'] = profile_err * image.unit

        result.meta['PROFILE_TYPE'] = p['axis']
        return ImageProfile(result)


class ImageProfile(object):
    """
    Image profile class.

    The image profile data is stored in `~astropy.table.Table` object, with the
    following columns:

        * `x_ref` Coordinate bin center (required).
        * `x_min` Coordinate bin minimum (optional).
        * `x_max` Coordinate bin maximum (optional).
        * `profile` Image profile data (required).
        * `profile_err` Image profile data error (optional).

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table instance with the columns specified as above.

    """

    def __init__(self, table):
        self.table = table

    def smooth(self, kernel='box', radius=0.1 * u.deg, **kwargs):
        """
        Smooth profile with error propagation.

        Smoothing is described by a convolution:

        .. math::

                x_j = \sum_i x_{(j - i)} h_i

        Where :math:`h_i` are the coefficients of the convolution kernel.

        The corresponding error on :math:`x_j` is then estimated using Gaussian
        error propagation, neglecting correlations between the individual
        :math:`x_{(j - i)}`:

        .. math::

                \Delta x_j = \sqrt{\sum_i \Delta x^{2}_{(j - i)} h^{2}_i}


        Parameters
        ----------
        kernel : {'gauss', 'box'}
            Kernel shape
        radius : `~astropy.units.Quantity` or float
            Smoothing width given as quantity or float. If a float is given it
            is interpreted as smoothing width in pixels. If an (angular) quantity
            is given it is converted to pixels using `xref[1] - x_ref[0]`.
        kwargs : dict
            Keyword arguments passed to `~scipy.ndimage.uniform_filter`
            ('box') and `~scipy.ndimage.gaussian_filter` ('gauss').

        Returns
        -------
        profile : `ImageProfile`
            Smoothed image profile.
        """
        from scipy.ndimage.filters import uniform_filter, gaussian_filter
        from scipy.ndimage import convolve
        from astropy.convolution import Gaussian1DKernel, Box1DKernel

        table = self.table.copy()
        profile = table['profile']

        radius = np.abs(radius / np.diff(self.x_ref))[0]
        width = 2 * radius.value + 1

        if kernel == 'box':
            smoothed = uniform_filter(profile.astype('float'), width, **kwargs)
            # renormalize data
            if table['profile'].unit.is_equivalent('count'):
                smoothed *= int(width)
                smoothed_err = np.sqrt(smoothed)
            elif 'profile_err' in table.colnames:
                profile_err = table['profile_err']
                # use gaussian error propagation
                box = Box1DKernel(width)
                err_sum = convolve(profile_err ** 2, box.array ** 2)
                smoothed_err = np.sqrt(err_sum)
        elif kernel == 'gauss':
            smoothed = gaussian_filter(profile.astype('float'), width, **kwargs)
            # use gaussian error propagation
            if 'profile_err' in table.colnames:
                profile_err = table['profile_err']
                gauss = Gaussian1DKernel(width)
                err_sum = convolve(profile_err ** 2, gauss.array ** 2)
                smoothed_err = np.sqrt(err_sum)
        else:
            raise ValueError("Not valid kernel choose either 'box' or 'gauss'")

        table['profile'] = smoothed * self.table['profile'].unit
        if 'profile_err' in table.colnames:
            table['profile_err'] = smoothed_err * self.table['profile'].unit
        return self.__class__(table)

    def plot(self, ax=None, **kwargs):
        """
        Plot image profile.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes object
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes.plot`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        y = self.table['profile'].data
        x = self.x_ref.value
        ax.plot(x, y, **kwargs)
        ax.set_xlabel('lon')
        ax.set_ylabel('profile')
        ax.set_xlim(x.max(), x.min())
        return ax

    def plot_err(self, ax=None, **kwargs):
        """
        Plot image profile error as band.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes object
        **kwargs : dict
            Keyword arguments passed to plt.fill_between()

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        y = self.table['profile'].data
        ymin = y - self.table['profile_err'].data
        ymax = y + self.table['profile_err'].data
        x = self.x_ref.value

        # plotting defaults
        kwargs.setdefault('alpha', 0.5)

        ax.fill_between(x, ymin, ymax, **kwargs)
        ax.set_xlabel('x (deg)')
        ax.set_ylabel('profile')
        return ax

    @property
    def x_ref(self):
        """
        Reference x coordinates.
        """
        return self.table['x_ref'].quantity

    @property
    def x_min(self):
        """
        Min. x coordinates.
        """
        return self.table['x_min'].quantity

    @property
    def x_max(self):
        """
        Max. x coordinates.
        """
        return self.table['x_max'].quantity

    @property
    def profile(self):
        """
        Image profile quantity.
        """
        return self.table['profile'].quantity

    @property
    def profile_err(self):
        """
        Image profile error quantity.
        """
        try:
            return self.table['profile_err'].quantity
        except KeyError:
            return None

    def peek(self, figsize=(8, 4.5), **kwargs):
        """
        Show image profile and error.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to `ImageProfile.plot_profile()`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax = self.plot(ax, **kwargs)

        if 'profile_err' in self.table.colnames:
            opts = {}
            opts['color'] = kwargs.get('c')
            ax = self.plot_err(ax, **opts)
        return ax

    def normalize(self, mode='peak'):
        """
        Normalize profile to peak value or integral.

        Parameters
        ----------
        mode : ['integral', 'peak']
            Normalize image profile so that it integrates to unity ('integral')
            or the maximum value corresponds to one ('peak').

        Returns
        -------
        profile : `ImageProfile`
            Normalized image profile.
        """
        table = self.table.copy()
        profile = self.table['profile']
        if mode == 'peak':
            norm = np.nanmax(profile)
        elif mode == 'integral':
            norm = np.nansum(profile)
        else:
            raise ValueError("Not a valid normalization mode. Choose either"
                             " 'peak' or 'integral'")

        table['profile'] /= norm

        if 'profile_err' in table.colnames:
            table['profile_err'] /= norm

        return self.__class__(table)
