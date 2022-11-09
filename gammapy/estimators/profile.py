# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools to create profiles (i.e. 1D "slices" from 2D images)."""
import numpy as np
import scipy.ndimage
from astropy import units as u
from astropy.convolution import Box1DKernel, Gaussian1DKernel
from astropy.coordinates import Angle
from astropy.table import Table
import matplotlib.pyplot as plt
from .core import Estimator

__all__ = ["ImageProfile", "ImageProfileEstimator"]


# TODO: implement measuring profile along arbitrary directions
# TODO: think better about error handling. e.g. MC based methods
class ImageProfileEstimator(Estimator):
    """Estimate profile from image.

    Parameters
    ----------
    x_edges : `~astropy.coordinates.Angle`
        Coordinate edges to define a custom measument grid (optional).
    method : ['sum', 'mean']
        Compute sum or mean within profile bins.
    axis : ['lon', 'lat', 'radial']
        Along which axis to estimate the profile.
    center : `~astropy.coordinates.SkyCoord`
        Center coordinate for the radial profile option.

    Examples
    --------
    This example shows how to compute a counts profile for the Fermi galactic
    center region::

        import matplotlib.pyplot as plt
        from gammapy.estimators import ImageProfileEstimator
        from gammapy.maps import Map
        from astropy import units as u

        # load example data
        filename = '$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz'
        fermi_cts = Map.read(filename)

        # set up profile estimator and run
        p = ImageProfileEstimator(axis='lon', method='sum')
        profile = p.run(fermi_cts)

        # smooth profile and plot
        smoothed = profile.smooth(kernel='gauss')
        smoothed.peek()
        plt.show()
    """

    tag = "ImageProfileEstimator"

    def __init__(self, x_edges=None, method="sum", axis="lon", center=None):
        if method not in ["sum", "mean"]:
            raise ValueError("Not a valid method, choose either 'sum' or 'mean'")

        if axis not in ["lon", "lat", "radial"]:
            raise ValueError("Not a valid axis, choose either 'lon', 'lat' or 'radial'")

        if axis == "radial" and center is None:
            raise ValueError("Please provide center coordinate for radial profiles")

        self._x_edges = x_edges
        self.parameters = {"method": method, "axis": axis, "center": center}

    def _get_x_edges(self, image):
        if self._x_edges is not None:
            return self._x_edges

        p = self.parameters
        coordinates = image.geom.get_coord(mode="edges").skycoord

        if p["axis"] == "lat":
            x_edges = coordinates[:, 0].data.lat
        elif p["axis"] == "lon":
            lon = coordinates[0, :].data.lon
            x_edges = lon.wrap_at("180d")
        elif p["axis"] == "radial":
            rad_step = image.geom.pixel_scales.mean()
            corners = [0, 0, -1, -1], [0, -1, 0, -1]
            rad_max = coordinates[corners].separation(p["center"]).max()
            x_edges = Angle(np.arange(0, rad_max.deg, rad_step.deg), unit="deg")

        return x_edges

    def _estimate_profile(self, image, image_err, mask):
        p = self.parameters
        labels = self._label_image(image, mask)

        profile_err = None

        index = np.arange(1, len(self._get_x_edges(image)))

        if p["method"] == "sum":
            profile = scipy.ndimage.sum(image.data, labels.data, index)

            if image.unit.is_equivalent("counts"):
                profile_err = np.sqrt(profile)
            elif image_err:
                # gaussian error propagation
                err_sum = scipy.ndimage.sum(image_err.data**2, labels.data, index)
                profile_err = np.sqrt(err_sum)

        elif p["method"] == "mean":
            # gaussian error propagation
            profile = scipy.ndimage.mean(image.data, labels.data, index)
            if image_err:
                N = scipy.ndimage.sum(~np.isnan(image_err.data), labels.data, index)
                err_sum = scipy.ndimage.sum(image_err.data**2, labels.data, index)
                profile_err = np.sqrt(err_sum) / N

        return profile, profile_err

    def _label_image(self, image, mask=None):
        p = self.parameters

        coordinates = image.geom.get_coord().skycoord
        x_edges = self._get_x_edges(image)

        if p["axis"] == "lon":
            lon = coordinates.data.lon.wrap_at("180d")
            data = np.digitize(lon.degree, x_edges.deg)

        elif p["axis"] == "lat":
            lat = coordinates.data.lat
            data = np.digitize(lat.degree, x_edges.deg)

        elif p["axis"] == "radial":
            separation = coordinates.separation(p["center"])
            data = np.digitize(separation.degree, x_edges.deg)

        if mask is not None:
            # assign masked values to background
            data[mask.data] = 0

        return image.copy(data=data)

    def run(self, image, image_err=None, mask=None):
        """Run image profile estimator.

        Parameters
        ----------
        image : `~gammapy.maps.Map`
            Input image to run profile estimator on.
        image_err : `~gammapy.maps.Map`
            Input error image to run profile estimator on.
        mask : `~gammapy.maps.Map`
            Optional mask to exclude regions from the measurement.

        Returns
        -------
        profile : `ImageProfile`
            Result image profile object.
        """
        p = self.parameters

        if image.unit.is_equivalent("count"):
            image_err = image.copy(data=np.sqrt(image.data))

        profile, profile_err = self._estimate_profile(image, image_err, mask)

        result = Table()
        x_edges = self._get_x_edges(image)
        result["x_min"] = x_edges[:-1]
        result["x_max"] = x_edges[1:]
        result["x_ref"] = (x_edges[:-1] + x_edges[1:]) / 2
        result["profile"] = profile * image.unit

        if profile_err is not None:
            result["profile_err"] = profile_err * image.unit

        result.meta["PROFILE_TYPE"] = p["axis"]

        return ImageProfile(result)


class ImageProfile:
    """Image profile class.

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

    def smooth(self, kernel="box", radius="0.1 deg", **kwargs):
        r"""Smooth profile with error propagation.

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
        radius : `~astropy.units.Quantity`, str or float
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
        table = self.table.copy()
        profile = table["profile"]

        radius = u.Quantity(radius)
        radius = np.abs(radius / np.diff(self.x_ref))[0]
        width = 2 * radius.value + 1

        if kernel == "box":
            smoothed = scipy.ndimage.uniform_filter(
                profile.astype("float"), width, **kwargs
            )
            # renormalize data
            if table["profile"].unit.is_equivalent("count"):
                smoothed *= int(width)
                smoothed_err = np.sqrt(smoothed)
            elif "profile_err" in table.colnames:
                profile_err = table["profile_err"]
                # use gaussian error propagation
                box = Box1DKernel(width)
                err_sum = scipy.ndimage.convolve(profile_err**2, box.array**2)
                smoothed_err = np.sqrt(err_sum)
        elif kernel == "gauss":
            smoothed = scipy.ndimage.gaussian_filter(
                profile.astype("float"), width, **kwargs
            )
            # use gaussian error propagation
            if "profile_err" in table.colnames:
                profile_err = table["profile_err"]
                gauss = Gaussian1DKernel(width)
                err_sum = scipy.ndimage.convolve(profile_err**2, gauss.array**2)
                smoothed_err = np.sqrt(err_sum)
        else:
            raise ValueError("Not valid kernel choose either 'box' or 'gauss'")

        table["profile"] = smoothed * self.table["profile"].unit
        if "profile_err" in table.colnames:
            table["profile_err"] = smoothed_err * self.table["profile"].unit
        return self.__class__(table)

    def plot(self, ax=None, **kwargs):
        """Plot image profile.

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
        if ax is None:
            ax = plt.gca()

        y = self.table["profile"].data
        x = self.x_ref.value
        ax.plot(x, y, **kwargs)
        ax.set_xlabel(self.table.meta.get("PROFILE_TYPE", "axis"))
        ax.set_ylabel("profile")
        ax.set_xlim(x.max(), x.min())
        return ax

    def plot_err(self, ax=None, **kwargs):
        """Plot image profile error as band.

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
        if ax is None:
            ax = plt.gca()

        y = self.table["profile"].data
        ymin = y - self.table["profile_err"].data
        ymax = y + self.table["profile_err"].data
        x = self.x_ref.value

        # plotting defaults
        kwargs.setdefault("alpha", 0.5)

        ax.fill_between(x, ymin, ymax, **kwargs)
        ax.set_xlabel("x (deg)")
        ax.set_ylabel("profile")
        return ax

    @property
    def x_ref(self):
        """Reference x coordinates."""
        return self.table["x_ref"].quantity

    @property
    def x_min(self):
        """Min. x coordinates."""
        return self.table["x_min"].quantity

    @property
    def x_max(self):
        """Max. x coordinates."""
        return self.table["x_max"].quantity

    @property
    def profile(self):
        """Image profile quantity."""
        return self.table["profile"].quantity

    @property
    def profile_err(self):
        """Image profile error quantity."""
        try:
            return self.table["profile_err"].quantity
        except KeyError:
            return None

    def peek(self, figsize=(8, 4.5), **kwargs):
        """Show image profile and error.

        Parameters
        ----------
        figsize : tuple
            Size of the figure.
        **kwargs : dict
            Keyword arguments passed to `ImageProfile.plot_profile()`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax = self.plot(ax, **kwargs)

        if "profile_err" in self.table.colnames:
            ax = self.plot_err(ax, color=kwargs.get("c"))

        return ax

    def normalize(self, mode="peak"):
        """Normalize profile to peak value or integral.

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
        profile = self.table["profile"]
        if mode == "peak":
            norm = np.nanmax(profile)
        elif mode == "integral":
            norm = np.nansum(profile)
        else:
            raise ValueError(f"Invalid normalization mode: {mode!r}")

        table["profile"] /= norm

        if "profile_err" in table.colnames:
            table["profile_err"] /= norm

        return self.__class__(table)
