# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.table import Table
from .core import SkyImage

__all__ = [
    'radial_profile',
    'radial_profile_label_image',
]


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
