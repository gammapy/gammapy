# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Image utility functions"""
from __future__ import print_function, division
import logging
import numpy as np
from astropy.units import Quantity
from astropy.io import fits
from astropy.wcs import WCS


__all__ = ['atrous_hdu', 'atrous_image',
           'bin_events_in_cube', 'bin_events_in_image',
           'binary_dilation_circle', 'binary_disk',
           'binary_opening_circle', 'binary_ring',
           'contains', 'coordinates',
           'cube_to_image', 'cube_to_spec',
           'cut_out',
           'disk_correlate', 'exclusion_distance',
           'image_groupby', 'images_to_cube',
           'make_empty_image', 'make_header',
           'paste_cutout_into_image', 'process_image_pixels', 'ring_correlate',
           'separation', 'solid_angle', 'threshold',
           ]


def _get_structure_indices(radius):
    """Get arrays of indices for a symmetric structure.
    
    Always generate an odd number of pixels and 0 at the center.
    
    Parameters
    ----------
    TODO
    
    Returns
    -------
    TODO
    """
    radius = int(radius)
    y, x = np.mgrid[-radius: radius + 1, -radius: radius + 1]
    return x, y


def binary_disk(radius):
    """Generate a binary disk mask.
    
    Value 1 inside and 0 outside.

    Useful as a structure element for morphological transformations.

    Note that the returned structure always has an odd number
    of pixels so that shifts during correlation are avoided.

    Parameters
    ----------
    radius : float
        Disk radius in pixels
    
    Returns
    -------
    structure : `numpy.array`
        Structure element (bool array)
    """
    x, y = _get_structure_indices(radius)
    structure = x ** 2 + y ** 2 <= radius ** 2
    return structure


def binary_ring(r_in, r_out):
    """Generate a binary ring mask.
    
    Value 1 inside and 0 outside.

    Useful as a structure element for morphological transformations.

    Note that the returned structure always has an odd number
    of pixels so that shifts during correlation are avoided.

    Parameters
    ----------
    r_in : float
        Ring inner radius in pixels

    r_out : float
        Ring outer radius in pixels
    
    Returns
    -------
    structure : `numpy.array`
        Structure element (bool array)
    """
    x, y = _get_structure_indices(r_out)
    mask1 = r_in ** 2 <= x ** 2 + y ** 2
    mask2 = x ** 2 + y ** 2 <= r_out ** 2
    return mask1 & mask2


def disk_correlate(image, radius, mode='constant'):
    """Correlate image with binary disk kernel.
    
    Parameters
    ----------
    TODO
    
    Returns
    -------
    TODO
    """
    from scipy.ndimage import convolve
    structure = binary_disk(radius)
    return convolve(image, structure, mode=mode)


def ring_correlate(image, r_in, r_out, mode='constant'):
    """Correlate image with binary ring kernel.

    Parameters
    ----------
    TODO
    
    Returns
    -------
    TODO
    """
    from scipy.ndimage import convolve
    structure = binary_ring(r_in, r_out)
    return convolve(image, structure, mode=mode)


def exclusion_distance(exclusion):
    """Distance to nearest exclusion region.
    
    Compute distance map, i.e. the Euclidean (=Cartesian 2D)
    distance (in pixels) to the nearest exclusion region.

    We need to call distance_transform_edt twice because it only computes
    dist for pixels outside exclusion regions, so to get the
    distances for pixels inside we call it on the inverted mask
    and then combine both distance images into one, using negative
    distances (note the minus sign) for pixels inside exclusion regions.
    
    Parameters
    ----------
    TODO
    
    Returns
    -------
    TODO
    """
    from scipy.ndimage import distance_transform_edt
    exclusion = np.asanyarray(exclusion, bool)
    distance_outside = distance_transform_edt(exclusion)
    distance_inside = distance_transform_edt(np.invert(exclusion))
    distance = np.where(exclusion, distance_outside, -distance_inside)
    return distance


def atrous_image(image, n_levels):
    """Compute a trous transform for a given image.

    Parameters
    ----------
    image : 2D array
        Input image
    n_levels : integer
        Number of wavelet scales.
    
    Returns
    -------
    images : list of 2D arrays
        Wavelet transformed images.
    """
    # https://code.google.com/p/image-funcut/
    from imfun import atrous
    return atrous.decompose2d(image, level=n_levels)


def atrous_hdu(hdu, n_levels):
    """Compute a trous transform for a given FITS HDU.

    Parameters
    ----------
    hdu : 2D image HDU
        Input image
    n_levels : integer
        Number of wavelet scales.

    Returns
    -------
    images : HDUList
        Wavelet transformed images.
    """
    image = hdu.data
    logging.info('Computing a trous transform for {0} levels ...'.format(n_levels))
    images = atrous_image(image, n_levels)
    hdus = fits.HDUList()

    for level, image in enumerate(images):
        if level < len(images) - 1:
            name = 'level_{0}'.format(level)
        else:
            name = 'residual'
        scale_pix = 2 ** level
        scale_deg = hdu.header['CDELT2'] * scale_pix
        logging.info('HDU name = {0:10s}: scale = {1:5d} pix = {2:10.5f} deg'
                     ''.format(name, scale_pix, scale_deg))
        hdus.append(fits.ImageHDU(data=image, header=hdu.header, name=name))

    return hdus


def coordinates(image, world=True, lon_sym=True, radians=False, system=None):
    """Get coordinate images for a given image.

    This function is useful if you want to compute
    an image with values that are a function of position.

    Parameters
    ----------
    image : `astropy.io.fits.ImageHDU`
        Input image
    world : bool
        Use world coordinates (or pixel coordinates)?
    lon_sym : bool
        Use symmetric longitude range ``(-180, 180)`` (or ``(0, 360)``)?

    Returns
    -------
    (lon, lat) : tuple of arrays
        Images as numpy arrays with values
        containing the position of the given pixel.

    Examples
    --------
    >>> import numpy as np
    >>> from gammapy.data import FermiGalacticCenter
    >>> lon, lat = coordinates(FermiGalacticCenter.counts())
    >>> dist = np.sqrt(lon ** 2 + lat ** 2)
    """
    # Create arrays of pixel coordinates
    y, x = np.indices(image.data.shape, dtype='int32') + 1

    if not world:
        return x, y

    wcs = WCS(image.header)
    lon, lat = wcs.wcs_pix2world(x, y, 1)
    
    if lon_sym:
        lon = np.where(lon > 180, lon - 360, lon)
    
    if radians:
        lon = np.radians(lon)
        lat = np.radians(lat)

    return lon, lat


def separation(image, center, world=True, radians=False):
    """Compute distance image from a given center point.
    
    Parameters
    ----------
    image : `~astropy.io.fits.ImageHDU`
        Input image
    center : (x, y) tuple
        Center position
    world : bool
        Use world coordinates (or pixel coordinates)?
    
    Returns
    -------
    separation : array
        Image of pixel separation to ``center``.
    """
    x_center, y_center = center
    x, y = coordinates(image, world=world, radians=radians)
    
    if world:
        from ..utils.coordinates import separation as sky_dist
        d = sky_dist(x, y, x_center, y_center)
        if radians:
            d = np.radians(d)
    else:
        d = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

    return d


def process_image_pixels(images, kernel, out, pixel_function):
    """Process images for a given kernel and per-pixel function.
    
    This is a helper function for the following common task:
    For a given set of same-shaped images and a smaller-shaped kernel,
    process each image pixel by moving the kernel at that position,
    cut out kernel-shaped parts from the images and call a function
    to compute output values for that position.

    This function loops over image pixels and takes care of bounding
    box computations, including image boundary handling.

    Parameters
    ----------
    images : dict of arrays
        Images needed to compute out
    kernel : array (shape must be odd-valued)
        kernel shape must be odd-valued
    out : single array or dict of arrays
        These arrays must have been pre-created by the caller
    pixel_function : function to process a part of the images

    Examples
    --------

    As an example, here is how to implement convolution as a special
    case of process_image_pixels with one input and output image::
    
        def convolve(image, kernel):
            '''Convolve image with kernel'''
            from gammapy.image.utils import process_image_pixels
            images = dict(image=np.asanyarray(image))
            kernel = np.asanyarray(kernel)
            out = dict(image=np.empty_like(image))
            def convolve_function(images, kernel):
                value = np.sum(images['image'] * kernel)
                return dict(image=value)
            process_image_pixels(images, kernel, out, convolve_function)
            return out['image']

    * TODO: add different options to treat the edges
    * TODO: implement multiprocessing version
    * TODO: this function is similar to view_as_windows in scikit-image:
            http://scikit-image.org/docs/dev/api/skimage.util.html#view-as-windows
            Is this function needed or can everything be done with view_as_windows?  
    """
    if isinstance(out, dict):
        n0, n1 = out.values()[0].shape
    else:
        n0, n1 = out.shape

    # Check kernel shape
    k0, k1 = kernel.shape
    if (k0 % 2 == 0) or (k1 % 2 == 0):
        raise ValueError('Kernel shape must have odd dimensions')
    k0, k1 = (k0 - 1) / 2, (k1 - 1) / 2

    # Loop over all pixels
    for i0 in range(0, n0):
        for i1 in range(0, n1):
            # Compute low and high extension
            # (# pixels, not counting central pixel) 
            i0_lo = min(k0, i0)
            i1_lo = min(k1, i1)
            i0_hi = min(k0, n0 - i0 - 1)
            i1_hi = min(k1, n1 - i1 - 1)
            
            # Cut out relevant parts of the image arrays
            # This creates views, i.e. is fast and memory efficient
            image_parts = dict()
            for name, image in images.items():
                # hi + 1 because with Python slicing the hi edge is not included
                part = image[i0 - i0_lo: i0 + i0_hi + 1,
                             i1 - i1_lo: i1 + i1_hi + 1]
                image_parts[name] = part

            # Cut out relevant part of the kernel array
            # This only applies when close to the edge
            # hi + 1 because with Python slicing the hi edge is not included
            kernel_part = kernel[k0 - i0_lo: k0 + i0_hi + 1,
                                 k1 - i1_lo: k1 + i1_hi + 1]

            # Call pixel_function for this one part
            out_part = pixel_function(image_parts, kernel_part)

            if isinstance(out_part, dict):
                # Store output
                for name, image in out.items():
                    out[name][i0, i1] = out_part[name]
            else:
                out[i0, i1] = out_part

def image_groupby(images, labels):
    """Group pixel by labels.
    
    This function is similar to `scipy.ndimage.measurements.labeled_comprehension`,
    but more general because it supports multiple input and output images.
    
    Parameters
    ----------
    TODO    
    
    Returns
    -------
    TODO
    """
    for image in images:
        assert image.shape == labels.shape
    
    # Store data in 1D data frame (i.e. as pixel lists)
    # TODO: should we use array.flat or array.ravel() here?
    # It's not clear to me what the difference is and which is more efficient here.
    data = dict()
    data['labels'] = labels.flat
    for name, values in images.items():
        data[name] = values.flat

    # Group pixels by labels
    groups = data.groupby('labels')

    return groups
    # out = groups.aggregate(function)
    # return out


def images_to_cube(hdu_list):
    """Convert a list of image HDUs into one cube.
    
    Parameters
    ----------
    hdu_list : `astropy.io.fits.HDUList`
        List of 2-dimensional image HDUs

    Returns
    -------
    cube : `astropy.io.fits.ImageHDU`
        3-dimensional cube HDU
    """
    shape = list(hdu_list[0].data.shape)
    shape.insert(0, len(hdu_list))
    data = np.empty(shape=shape, dtype=hdu_list[0].data.dtype)
    for ii, hdu in enumerate(hdu_list):
        data[ii] = hdu.data
    header = hdu_list[0].header
    header['NAXIS'] = 3
    header['NAXIS3'] = len(hdu_list)
    # header['CRVAL3']
    # header['CDELT3']
    # header['CTYPE3']
    # header['CRPIX3']
    # header['CUNIT3']
    return fits.ImageHDU(data=data, header=header)


def bin_events_in_image(events, reference_image):
    """Bin events into an image.
    
    Parameters
    ----------
    events : `astropy.table.table.Table`
        Event list table
    reference_image : `astropy.io.fits.ImageHDU`
        An image defining the spatial bins.
    
    Returns
    -------
    count_image : `astropy.io.fits.ImageHDU`
        Count image
    """
    if 'GLON' in reference_image.header['CTYPE1']:
        lon = events['GLON']
        lat = events['GLAT']
    else:
        lon = events['RA']
        lat = events['DEC']

    # Get pixel coordinates    
    wcs = WCS(reference_image.header)
    xx, yy = wcs.wcs_world2pix(lon, lat, 0)

    # Histogram pixel coordinates with appropriate binning.
    # This was checked against the `ctskymap` ctool
    # http://cta.irap.omp.eu/ctools/
    shape = reference_image.data.shape
    bins = np.arange(shape[0] + 1) - 0.5, np.arange(shape[1] + 1) - 0.5
    data = np.histogramdd([yy, xx], bins)[0]

    hdu = fits.ImageHDU(data, reference_image.header)
    return hdu


def bin_events_in_cube(events, reference_cube, energies):
    """Bin events in LON-LAT-Energy cube.
    
    Parameters
    ----------
    events : `astropy.table.table.Table`
        Event list table
    cube : `astropy.io.fits.ImageHDU`
        A cube defining the spatial bins.
    energies : `astropy.table.table.Table`
        Table defining the energy bins.
    
    Returns
    -------
    count_cube : `astropy.io.fits.ImageHDU`
        Count cube
    """
    if 'GLON' in reference_cube.header['CTYPE1']:
        lon = events['GLON']
        lat = events['GLAT']
    else:
        lon = events['RA']
        lat = events['DEC']

    # Get pixel coordinates    
    wcs = WCS(reference_cube.header)
    # We're not interested in the energy axis, so we give a dummy value of 1
    xx, yy = wcs.wcs_world2pix(lon, lat, 1, 0)[:-1]

    event_energies = events['Energy']
    zz = np.searchsorted(event_energies, energies)

    # Histogram pixel coordinats with appropriate binning.
    # This was checked against the `ctskymap` ctool
    # http://cta.irap.omp.eu/ctools/
    shape = reference_cube.data.shape
    bins = np.arange(shape[0]), np.arange(shape[1] + 1) - 0.5, np.arange(shape[2] + 1) - 0.5
    data = np.histogramdd([zz, yy, xx], bins)[0]

    hdu = fits.ImageHDU(data, reference_cube.header)
    return hdu


def threshold(array, threshold=5):
    """Set all pixels below threshold to zero.
    
    Parameters
    ----------
    array : array_like
        Input array
    threshold : float
        Minimum threshold
    
    Returns
    -------
    TODO
    """
    # TODO: np.clip is simpler, no?
    from scipy import stats
    # NaNs are set to 1 by thresholding, which is not
    # what we want for detection, so we replace them with 0 here.
    data = np.nan_to_num(array)

    data = stats.threshold(data, threshold, None, 0)
    # Note that scipy.stats.threshold doesn't binarize,
    # it only sets values below the threshold to 0,
    # which is not what we want here.
    return data.astype(np.bool).astype(np.uint8)


def binary_dilation_circle(input, radius):
    """Dilate with disk of given radius.
    
    Parameters
    ----------
    input : array_like
        Input array
    radius : float
        Dilation radius (pix)
    
    Returns
    -------
    TODO    
    """
    from scipy.ndimage import binary_dilation
    structure = binary_disk(radius)
    return binary_dilation(input, structure)


def binary_opening_circle(input, radius):
    """Binary opening with circle as structuring element.
    
    This calls `scipy.ndimage.morphology.binary_opening` with a `binary_disk`
    as structuring element.
    
    Parameters
    ----------
    TODO
    
    Returns
    -------
    TODO
    """
    from scipy.ndimage import binary_opening
    structure = binary_disk(radius)
    return binary_opening(input, structure)


def solid_angle(image):
    """Compute the solid angle of each pixel.

    This will only give correct results for CAR maps!

    Parameters
    ----------
    image : `~astropy.io.fits.ImageHDU`
        Input image

    Returns
    -------
    area_image : `~astropy.units.quantity.Quantity`
        Solid angle image (matching the input image) in steradians.
    """
    # Area of one pixel at the equator
    cdelt0 = image.header['CDELT1']
    cdelt1 = image.header['CDELT2']
    equator_area = Quantity(abs(cdelt0 * cdelt1), 'sr')

    # Compute image with fraction of pixel area at equator
    glat = coordinates(image)[1]
    area_fraction = np.cos(np.radians(glat))

    result = area_fraction * equator_area

    return result


def make_header(nxpix=100, nypix=100, binsz=0.1, xref=0, yref=0,
                proj='CAR', coordsys='GAL',
                xrefpix=None, yrefpix=None):
    """Generate a FITS header from scratch.

    Uses the same parameter names as the Fermi tool gtbin.

    If no reference pixel position is given it is assumed ot be
    at the center of the image.
    
    Parameters
    ----------
    TODO
    
    Returns
    -------
    header : `astropy.io.fits.Header`
        Header
    """
    nxpix = int(nxpix)
    nypix = int(nypix)
    if not xrefpix:
        xrefpix = (nxpix + 1) / 2.
    if not yrefpix:
        yrefpix = (nypix + 1) / 2.

    if coordsys == 'CEL':
        ctype1, ctype2 = 'RA---', 'DEC--'
    elif coordsys == 'GAL':
        ctype1, ctype2 = 'GLON-', 'GLAT-'
    else:
        raise Exception('Unsupported coordsys: %s' % proj)

    pars = {'NAXIS': 2, 'NAXIS1': nxpix, 'NAXIS2': nypix,
            'CTYPE1': ctype1 + proj,
            'CRVAL1': xref, 'CRPIX1': xrefpix, 'CUNIT1': 'deg', 'CDELT1':-binsz,
            'CTYPE2': ctype2 + proj,
            'CRVAL2': yref, 'CRPIX2': yrefpix, 'CUNIT2': 'deg', 'CDELT2': binsz,
            }

    header = fits.Header()
    header.update(pars)

    return header


def make_empty_image(nxpix=100, nypix=100, binsz=0.1, xref=0, yref=0,
                     proj='CAR', coordsys='GAL',
                     xrefpix=None, yrefpix=None, dtype='float32'):
    """Make an empty (i.e. values 0) image.
    
    Uses the same parameter names as the Fermi tool gtbin.

    If no reference pixel position is given it is assumed ot be
    at the center of the image.

    Parameters
    ----------
    TODO

    Returns
    -------
    image : `astropy.io.fits.ImageHDU`
        Empty image
    """
    header = make_header(nxpix, nypix, binsz, xref, yref,
                         proj, coordsys, xrefpix, yrefpix)

    # Note that FITS and NumPy axis order are reversed
    shape = (header['NAXIS2'], header['NAXIS1'])
    data = np.zeros(shape, dtype=dtype)

    return fits.ImageHDU(data, header)


def cut_out(image, center, fov=[2, 2]):
    """Cut out a part of an image.

    Parameters
    ----------
    image : `astropy.io.fits.ImageHDU`
        Input image
    center : [glon, glat]
        Cutout center position
    fov : [glon_fov, glat_fov]
        Cutout field of view dimensions
    
    Returns
    -------
    TODO
    """
    raise NotImplementedError
    
    # Unpack center and fov
    glon, glat = center
    glon_fov, glat_fov = fov

    # Calculate image limits
    glon_lim = [glon - glon_fov, glon + glon_fov]
    glat_lim = [glat - glat_fov, glat + glat_fov]

    # Cut out the requested part
    xlim = image.proj.topixel((glon_lim, [0, 0]))[0]
    xlim = [xlim[1], xlim[0]]  # longitude axis runs backwards
    ylim = image.proj.topixel(([0, 0], glat_lim))[1]
    image.set_limits(pxlim=xlim, pylim=ylim)


def cube_to_image(cube, slicepos=None):
    """Slice or project 3-dim cube into a 2-dim image.
    
    Parameters
    ----------
    cube : `~astropy.io.fits.ImageHDU`
        3-dim FITS cube
    slicepos : int or None
        Slice position (None means to sum along the spectral axis)
    
    Returns
    -------
    image : `~astropy.io.fits.ImageHDU`
        2-dim FITS image
    """
    from astropy.io.fits import ImageHDU
    header = cube.header.copy()
    header['NAXIS'] = 2
    del header['NAXIS3']
    del header['CRVAL3']
    del header['CDELT3']
    del header['CTYPE3']
    del header['CRPIX3']
    del header['CUNIT3']
    if slicepos == None:
        data = cube.data.sum()
    else:
        data = cube.data[slicepos]
    return ImageHDU(data, header)


def cube_to_spec(cube, mask, weighting='none'):
    """Integrate spatial dimensions of a FITS cube to give a spectrum.
 
    TODO: give formulas.
 
    Parameters
    ----------
    cube : `~astropy.io.fits.ImageHDU`
        3-dim FITS cube
    mask : numpy.array
        2-dim mask array.
    weighting : {'none', 'solid_angle'}
        Weighting factor to use.
    
    Returns
    -------
    spectrum : numpy.array
        Summed spectrum of pixels in the mask.
    """
    value = cube.dat
    A = solid_angle(cube)
    # Note that this is the correct way to get an average flux:

    spec = (value * A).sum(-1).sum(-1)
    return spec


def contains(image, x, y, world=True):
    """Check if given pixel or world positions are in an image.
    
    Parameters
    ----------
    image : `~astropy.io.fits.ImageHDU`
        2-dim FITS image
    
    Returns
    -------
    containment : array
        Bool array
    """
    header = image.header
    
    if world:
        wcs = WCS(header)
        x, y = wcs.wcs_world2pix(x, y, 0)

    nx, ny = header['NAXIS2'], header['NAXIS1']
    return (x >= 0.5) & (x <= nx + 0.5) & (y >= 0.5) & (y <= ny + 0.5)


def paste_cutout_into_image(total, cutout, method='sum'):
    """Paste cutout into a total image.
    
    Parameters
    ----------
    total, cutout : `astropy.io.fits.ImageHDU`
        Total and cutout image.
    method : {'sum', 'replace'}
        Sum or replace total values with cutout values.

    Returns
    -------
    total : `astropy.io.fits.ImageHDU`
        A reference to the total input HDU that was modified in-place.
    """
    # find offset
    lon, lat = WCS(cutout.header).wcs_pix2world(0, 0, 0)
    x, y = WCS(total.header).wcs_world2pix(lon, lat, 0)
    x, y = int(np.round(x)), int(np.round(y))
    dy, dx = cutout.shape
    # import IPython; IPython.embed(); 1/0

    if method == 'sum':
        total.data[y : y + dy, x : x + dx] += cutout.data
    elif method == 'replace':
        total.data[y : y + dy, x : x + dx] = cutout.data
    else:
        raise ValueError('Invalid method: {0}'.format(method))
