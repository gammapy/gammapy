# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Measure source properties"""
from __future__ import print_function, division
import numpy as np

__all__ = ['aperphot', 'find_max',
           'lookup', 'lookup_max',
           'measure_labeled_regions'
           ]


def _split_xys(pos):
    """Helper function to work with `scipy.ndimage`."""
    x = np.array(pos)[:, 1]
    y = np.array(pos)[:, 0]
    return x, y


def _split_slices(slices):
    """Helper function to work with `scipy.ndimage`."""
    # scipy.ndimage.find_objects returns a list of
    # tuples of slices, which is not what we want.
    # The following list comprehensions extract
    # the format we need.
    xmin = np.asarray([t[1].start for t in slices])
    xmax = np.asarray([t[1].stop for t in slices])
    ymin = np.asarray([t[0].start for t in slices])
    ymax = np.asarray([t[0].stop for t in slices])
    return xmin, xmax, ymin, ymax


def _measure_area(labels):
    """Measure the area in pix of each segment."""
    nsegments = labels.max()
    area = np.zeros(nsegments)
    for i in range(nsegments):
        area[i] = (labels == i + 1).sum()
    return area


def measure_labeled_regions(data, labels, tag='IMAGE',
                            measure_positions=True, measure_values=True,
                            fits_offset=True, bbox_offset=True):
    """Measure source properties in image.
    
    Sources are defined by a label image.
    
    Parameters
    ----------
    TODO
    
    Returns
    -------
    TODO
    """
    import scipy.ndimage as nd
    from astropy.table import Table, Column
    # Measure all segments
    nsegments = labels.max()
    index = np.arange(1, nsegments + 1)  # Measure all sources
    # Measure stuff
    sum = nd.sum(data, labels, index)
    max = nd.maximum(data, labels, index)
    mean = nd.mean(data, labels, index)
    x, y = _split_xys(nd.center_of_mass(data, labels, index))
    xpeak, ypeak = _split_xys(nd.maximum_position(data, labels, index))
    xmin, xmax, ymin, ymax = _split_slices(nd.find_objects(labels))
    area = _measure_area(labels)
    # Use FITS convention, i.e. start counting at 1
    FITS_OFFSET = 1 if fits_offset else 0
    # Use SExtractor convention, i.e. slice max is inside
    BBOX_OFFSET = -1 if bbox_offset else 0
    # Create a table
    table = Table()
    table.add_column(Column(data=index, name='NUMBER'))

    if measure_positions:
        table.add_column(Column(data=x + FITS_OFFSET, name='X_IMAGE'))
        table.add_column(Column(data=y + FITS_OFFSET, name='Y_IMAGE'))
        table.add_column(Column(data=xpeak + FITS_OFFSET, name='XPEAK_IMAGE'))
        table.add_column(Column(data=ypeak + FITS_OFFSET, name='YPEAK_IMAGE'))
        table.add_column(Column(data=xmin + FITS_OFFSET, name='XMIN_IMAGE'))
        table.add_column(Column(data=xmax + FITS_OFFSET + BBOX_OFFSET, name='XMAX_IMAGE'))
        table.add_column(Column(data=ymin + FITS_OFFSET, name='YMIN_IMAGE'))
        table.add_column(Column(data=ymax + FITS_OFFSET + BBOX_OFFSET, name='YMAX_IMAGE'))
        table.add_column(Column(data=area, name='AREA'))

    if measure_values:
        table.add_column(Column(data=max, name=tag + '_MAX'))
        table.add_column(Column(data=sum, name=tag + '_SUM'))
        table.add_column(Column(data=mean, name=tag + '_MEAN'))

    return table


def find_max(image):
    """Find position of maximum in an image.
    
    Parameters
    ----------
    image : `astropy.io.fits.ImageHDU`
        Input image
    
    Returns
    -------
    lon, lat, value : float
        Maximum value and its position 
    """
    from scipy.ndimage import maximum_position
    from astropy.wcs import WCS
    proj = WCS(image.header)
    data = image.data
    data[np.isnan(data)] = -np.inf
    y, x = maximum_position(data)
    GLON, GLAT = proj.wcs_pix2world(x, y, 0)
    val = data[int(y), int(x)]
    return GLON, GLAT, val


def aperphot(img, x, y, aper, sky, sky_type='median', verbose=False):
    """Perform circular aperture photometry on a set of images.

    Performs the aperture photometry of a given x,y point in an image.
    Note: Doesn't handle subpixel integration.

    Parameters
    ----------
    img : numpy 2D array of the image to perform the photometry on.
        y_dimension, x_dimension = img.shape
        # Note: x is the second axis as opposed to the first in IDL!
    x, y : array_like
        Aperture center pixel coordinates
    aper : aperture size in pixel.
    sky : 2-element list/tuple/array providing the inner and outer radii
        to calculate the sky from.
    sky_type : if 'median' will use a median for the sky level
        determination within the sky annulus, otherwise will use a mean.

    Examples
    --------
    >>> flux, flux_err = aperphot(img, 32.2, 35.6, 5., [10.,15.], sky_type='median')
    """
    dimy, dimx = img.shape
    indy, indx = np.mgrid[0:dimy, 0:dimx]
    r = np.sqrt((indy - y) ** 2 + (indx - x) ** 2)
    ind_aper = r < aper
    n_counts = ind_aper.sum()
    tot_counts = img[ind_aper].sum()
    ind_bkg = (sky[0] < r) * (r < sky[1])
    n_bkg = ind_bkg.sum()
    if sky_type.lower() == 'median':
        bkg = np.median(img[ind_bkg])
        std_bkg = img[ind_bkg].std()
    else:
        tot_bkg = img[ind_bkg].sum()
        bkg = tot_bkg / n_bkg
        std_bkg = img[ind_bkg].std()
    flux = tot_counts - n_counts * bkg
    if verbose:
        print('%8.2f %8.2f %10.2f %10.2f %10.2f %10.2f %10.2f' %
              (x, y, tot_counts, n_counts, bkg, flux, std_bkg) * np.sqrt(n_counts))
    return flux, std_bkg * np.sqrt(n_counts)


def _lookup_pix(image, x, y):
    """Look up values in an image for given pixel coordinates.
    
    image = numpy array
    x, y = array_like of pixel coordinates (floats OK)
    """
    # Find the right pixel
    x_int = np.round(x).astype(int)
    y_int = np.round(y).astype(int)

    # Return it's value
    # Note that numpy has index order (y, x)
    values = image[y_int, x_int]
    return values


def _lookup_world(image, lon, lat):
    """Look up values in an image for given world coordinates.
    
    image = astropy.io.fits.HDU
    lon, lat = world coordinates (float OK)
    """
    from astropy.wcs import WCS
    wcs = WCS(image.header)
    x, y = wcs.wcs_world2pix(lon, lat, 0)
    return _lookup_pix(image.data, x, y)


def lookup(image, x, y, world=True):
    """Look up values in an image.

    Parameters
    ----------
    image : array_like if world=False, astropy.io.fits.ImageHDU if world=True
        Array or image to look up the value
    x : array_like
        Array of X lookup positions
    y : array_like
        Array of Y lookup positions
    world : bool
        Are (x, y) WCS coordinates?
    """
    if world:
        return _lookup_world(image, x, y)
    else:
        return _lookup_pix(image, x, y)


def lookup_max(image, GLON, GLAT, theta):
    """Look up the max image values within a circle of radius theta
    around lists of given positions (nan if outside)"""
    from .utils import coordinates
    GLON = np.asarray(GLON)
    GLON = np.where(GLON > 180, GLON - 360, GLON)
    GLAT = np.asarray(GLAT)
    n_pos = len(GLON)
    theta = np.asarray(theta) * np.ones(n_pos, dtype='float32')

    ll, bb = coordinates(image)

    val = np.nan * np.ones(n_pos, dtype='float32')
    for ii in range(n_pos):
        mask = ((GLON[ii] - ll) ** 2 +
                (GLAT[ii] - bb) ** 2 <=
                theta[ii] ** 2)
        try:
            val[ii] = image.dat[mask].max()
        except ValueError:
            pass
    return val
