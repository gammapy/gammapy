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
    'ImageProfileEstimator',
    'radial_profile',
    'radial_profile_label_image',
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

def radial_profile(image, center, radius):
    """
    Image radial profile.

    TODO: show example and explain handling of "overflow"
    and "underflow" bins (see ``radial_profile_label_image`` docstring).

    Calls `numpy.digitize` to compute a label image and then
    `scipy.ndimage.sum` to do measurements.

    Parameters
    ----------
    image : `~gammapy.image.SkyImage`
        Image
    center : `~astropy.coordinates.SkyCoord`
        Center position
    radius : `~astropy.coordinates.Angle`
        Offset bin edge array.

    Returns
    -------
    table : `~astropy.table.Table`

        Table with the following fields that define the binning:

        * ``RADIUS_BIN_ID`` : Integer bin ID (starts at ``1``).
        * ``RADIUS_MEAN`` : Radial bin center
        * ``RADIUS_MIN`` : Radial bin minimum edge
        * ``RADIUS_MAX`` : Radial bin maximum edge

        And the following measurements from the pixels in each bin:

        * ``N_PIX`` : Number of pixels
        * ``SUM`` : Sum of pixel values
        * ``MEAN`` : Mean of pixel values, computed as ``SUM / N_PIX``


    Examples
    --------

    Make some example data::

        from astropy.coordinates import Angle
        from gammapy.image import SkyImage
        image = SkyImage.empty()
        image.fill(value=1)
        center = image.center
        radius = Angle([0.1, 0.2, 0.4, 0.5, 1.0], 'deg')

    Compute and print a radial profile::

        from gammapy.image import radial_profile
        table = radial_profile(image, center, radius)
        table.pprint()

    If your measurement represents counts, you could e.g. use this
    method to compute errors::

        import numpy as np
        table['SUM_ERR'] = np.sqrt(table['SUM'])
        table['MEAN_ERR'] = table['SUM_ERR'] / table['N_PIX']

    If you need to do special measurements or error computation
    in each bin with access to the pixel values,
    you could get the label image and then do the measurements yourself::

        labels = radial_profile_label_image(image, center, radius)
        labels.show()
    """
    labels = radial_profile_label_image(image, center, radius)

    # Note: here we could decide to also measure overflow and underflow bins.
    index = np.arange(1, len(radius))
    table = _radial_profile_measure(image, labels, index)

    table['RADIUS_BIN_ID'] = index
    table['RADIUS_MIN'] = radius[:-1]
    table['RADIUS_MAX'] = radius[1:]
    table['RADIUS_MEAN'] = 0.5 * (table['RADIUS_MAX'] + table['RADIUS_MIN'])

    meta = dict(
        type='radial profile',
        center=center,
    )

    table.meta.update(meta)
    return table


def radial_profile_label_image(image, center, radius):
    """
    Image radial profile label image.

    The ``radius`` array defines ``n_bins = len(radius) - 1`` bins.

    The label image has the following values:
    * Value ``1`` to ``n_bins`` for pixels in ``(radius[0], radius[-1])``
    * Value ``0`` for pixels with ``r < radius[0]``
    * Value ``n_bins`` for pixels with ``r >= radius[-1]``

    Parameters
    ----------
    image : `~gammapy.image.SkyImage`
        Image
    center : `~astropy.coordinates.SkyCoord`
        Center position
    radius : `~astropy.coordinates.Angle`
        Offset bin edge array.

    Returns
    -------
    labels : `~gammapy.image.SkyImage`
        Label image (1 to max_label; outside pixels have value 0)
    """
    radius_image = image.coordinates().separation(center)
    labels = np.digitize(radius_image.deg, radius.deg)

    return SkyImage(name='labels', data=labels, wcs=image.wcs.copy())


def _radial_profile_measure(image, labels, index):
    """
    Measurements for radial profile.

    This is a helper function to do measurements.

    TODO: this should call the generic function,
    nothing radial profile-specific here.
    """
    from scipy import ndimage

    # This function takes `SkyImage` objects as inputs
    # but only operates on their `data`
    image = image.data
    labels = labels.data

    table = Table()
    table['N_PIX'] = ndimage.sum(np.ones_like(image), labels, index=index)
    table['SUM'] = ndimage.sum(image, labels, index=index)
    table['MEAN'] = table['SUM'] / table['N_PIX']

    return table
