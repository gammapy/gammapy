"""Helper functions for working with maps.

@todo: Make it a package that contains modules
scipy / astropy / FITSimage part separately?"""

import logging
import numpy as np

try:
    import scipy.stats
    import scipy.ndimage
except:
    print 'scipy import failed.'

try:
    from astropy.io import fits
    import astropy.wcs as wcs
except:
    print 'astropy import failed.'

try:
    from kapteyn.maputils import FITSimage
    from kapteyn.wcs import WCSinvalid
except:
    print 'kapteyn import failed.'

DEFAULT_MODE = 'constant'


def get_stats(a, mask_nan=False, mask_zero=False):
    """
    Returns the following stats on an array:
    Min, Max, Mean, Median, Sigma, Sum

    NPix, NaNs, Infs, Zeros, NonInts
    """
    # Build boolean masks of special pix values
    nan_mask = np.isnan(a)
    inf_mask = np.isinf(a)
    zero_mask = (a == 0)
    nonint_mask = (a == a.astype(int))  # don't know if this works, maybe round?

    # Count special pix values
    NPix = a.size
    NaNs = nan_mask.sum()
    Infs = inf_mask.sum()
    Zeros = zero_mask.sum()
    NonInts = nonint_mask.sum()
    count_stats = [NPix, NaNs, Infs, Zeros, NonInts]

    # Build a mask and apply it
    mask = np.invert((mask_nan * nan_mask) | (mask_zero * zero_mask))
    a = a[mask]

    value_stats = [a.min(), a.max(), a.mean(), np.median(a), a.std(), a.sum()]

    return count_stats + value_stats


def print_stats_header_line(title='Array statistics:'):
    """ Print header line that explains the columns of print_stats() """
    format = '%20s %20s %12s %12s %12s %12s %12s'
    values = ('label', 'shape', 'min', 'max', 'mean',
              'std', 'median')
    print title
    print format % values
    print


def print_stats(a, label='array', oneline=True):
    """ Print basic stats for a numpy.ndarray. """
    if oneline:
        format = '%20s %20s %12g %12g %12g %12g %12g'
        values = (label, a.shape, a.min(), a.max(), a.mean(),
                  a.std(), np.median(a))
        print format % values
    else:
        print 'shape:', a.shape
        print 'min:', a.min()
        print 'max:', a.max()
        print 'mean:', a.mean()
        print 'std:', a.std()
        print 'median:', np.median(a)


def make_header(nxpix=100, nypix=100, binsz=0.1, xref=0, yref=0,
           proj='CAR', coordsys='GAL',
           xrefpix=None, yrefpix=None, txt=False):
    """
    Generate a new FITS header dictionary.
    Uses the same parameter names as the Fermi tool gtbin.

    If no reference pixel position is given it is assumed ot be
    at the center of the image.
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

    header = {'NAXIS': 2, 'NAXIS1': nxpix, 'NAXIS2': nypix,
          'CTYPE1': ctype1 + proj,
          'CRVAL1': xref, 'CRPIX1': xrefpix, 'CUNIT1': 'deg', 'CDELT1': -binsz,
          'CTYPE2': ctype2 + proj,
          'CRVAL2': yref, 'CRPIX2': yrefpix, 'CUNIT2': 'deg', 'CDELT2': binsz,
              }

    if txt:
        header = """
SIMPLE  = T
BITPIX  = -64
NAXIS   = 2
NAXIS1  = {NAXIS1}
NAXIS2  = 1800
CTYPE1  = ''
CTYPE2  = 'GLAT-AIT'
CRVAL1  =    0.0
CRVAL2  =    0.0
CRPIX1  = 1800.5
CRPIX2  =  900.5
CDELT1  =   -0.1
CDELT2  =    0.1
EQUINOX = 2000.0
END""".format(header)

    logging.debug('Generating header:')
    logging.debug(header)
    return header


def empty_image(nxpix=100, nypix=100, binsz=0.1, xref=0, yref=0,
                proj='CAR', coordsys='GAL',
                xrefpix=None, yrefpix=None, dtype='float32'):
    """
    Generate a maputils.FITSimage object.
    Uses the same parameter names as the Fermi tool gtbin.

    If no reference pixel position is given it is assumed ot be
    at the center of the image.
    """
    logging.debug('Generating empty image now.')
    header = make_header(nxpix, nypix, binsz, xref, yref,
                         proj, coordsys, xrefpix, yrefpix)
    # Note that FITS and NumPy axis order are reversed
    shape = (header['NAXIS2'], header['NAXIS1'])
    data = np.zeros(shape, dtype=dtype)
    return FITSimage(externalheader=header, externaldata=data)


def get_copy(image, newval=None):
    header = image.hdr.copy()
    data = image.dat.copy()
    if newval:
        data = np.ones_like(data) * newval
    return FITSimage(externalheader=header,
                     externaldata=data)


def cut_out(image, center, fov=[2, 2]):
    """
    Cut out a part of an image.
    image: maputils.FITSimage
    center: [glon, glat]
    fov : [glon_fov, glat_fov]
    """
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


def save_with_limits(image, infile, outfile, clobber=False):
    """image should be a kapteyn.FITSimage.
    Set limits first by hand or by using image.utils.cut_out"""
    xmin, xmax = np.array(image.pxlim, dtype=int)
    ymin, ymax = np.array(image.pylim, dtype=int)
    string = ('[{xmin}:{xmax},{ymin}:{ymax}]'
              ''.format(**locals()))
    from subprocess import call
    # Use ftcopy because this way we are sure to
    # have correct world coordinates in the output image.
    # ftcopy 'image.fits[101:200, 101:200]' out.fits
    cmd = 'ftcopy "%s%s" %s' % (infile, string, outfile)
    if clobber:
        cmd += ' clobber=yes'
    print cmd
    call(cmd, shell=True)


def apply_limits(f, x, y, width, height, world=True):
    # logging.debug
    if world:
        cdelt = f.axisinfo[2].cdelt
        x, y = f.proj.topixel((x, y))
        width, height = width / cdelt, height / cdelt

    pxlim = max(x - width / 2, 0), min(x + width / 2, f.proj.naxis[0])
    pylim = max(y - height / 2, 0), min(y + height / 2, f.proj.naxis[1])
    f.set_limits(pxlim, pylim)


def cutout_box(x, y, radius, nx, ny, format='string'):
    x, y, radius = int(x), int(y), int(radius)
    xmin = max(x - radius, 0)
    xmax = min(x + radius, nx)
    ymin = max(y - radius, 0)
    ymax = min(y + radius, ny)
    box_coords = (xmin, xmax, ymin, ymax)
    box_string = '[{xmin}:{xmax},{ymin}:{ymax}]'.format(**locals())
    if format == 'coords':
        return box_coords
    elif format == 'string':
        return box_string
    elif format == 'both':
        return box_coords, box_string


def bbox(mask, margin, binsz):
    """Determine the bounding box of a mask"""
    from scipy.ndimage.measurements import find_objects
    box = find_objects(mask.astype(int))[0]
    ny, nx = mask.shape
    xmin = max(0, int(box[1].start - margin / binsz)) + 1
    xmax = min(nx - 1, int(box[1].stop + margin / binsz)) + 1
    ymin = max(0, int(box[0].start - margin / binsz)) + 1
    ymax = min(ny - 1, int(box[0].stop + margin / binsz)) + 1
    box_string = '[{xmin}:{xmax},{ymin}:{ymax}]'.format(**locals())
    logging.info('box = {box}, box_string = {box_string}'
                 ''.format(**locals()))
    box = xmin, xmax, ymin, ymax
    return box, box_string


def _convert_coordinates(convert, x, y):
    """Helper function that flattens the x and y arrays, as required by kapteyn"""
    x_shape, y_shape = x.shape, y.shape
    x.shape, y.shape = (x.size,), (y.size,)

    # A hack so that cubes can be processed, too:
    try:
        try:
            x, y = convert((x, y))
        except WCSinvalid:
            logging.debug('Invalid pixel coordinates have been set to value NaN.')
            # This occurs e.g. for the corners of all-sky AIT images, which
            # don't have valid world coordinates.
            # If this exception occurs, x and y will not contain the coordinates.
            # Calling toworld again makes the copy. See kapteyn docu.
            x, y = convert()
    except:  # TODO: WCSerror import how?
        dummy = np.zeros_like(x)
        x, y, dummy = convert((x, y, dummy))
        del dummy
    x.shape, y.shape = x_shape, y_shape
    return x.astype('float32'), y.astype('float32')
    """
    Maybe the function above is overly complicated and this would work as well!

    from numpy import indices, zeros
    from kapteyn.wcs import WCSinvalid
    y, x = indices(shape) + 1
    lon = zeros(shape)
    lat = zeros(shape)
    try:
        lon, lat = image.proj.toworld((x.flatten(), y.flatten()))
    except WCSinvalid:
        logging.info('Invalid pixel coordinates have been set to value NaN.')
        # Ignore if some pixels don't have a world coordinate,
        # which happens e.g. in the corners of all-sky AIT images.
        lon, lat = image.proj.toworld()
    lon = lon.reshape(shape).astype('float32')
    lat = lat.reshape(shape).astype('float32')
    """


def _to_world(image, x, y):
    return _convert_coordinates(image.proj.toworld, x, y)


def _to_pixel(image, x, y):
    return _convert_coordinates(image.proj.topixel, x, y)


def _prepare_arrays(image, x, y, world=True):
    """Helper function"""
    x = np.asarray(x)
    y = np.asarray(y)
    if world:
        x, y = _to_pixel(image, x, y)
    return x, y


def cube_to_image(cube, slicepos=None):
    """ Make an image out of a cube.
    Both in- and output should by fits.HDUs"""
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
    return fits.ImageHDU(data, header)


def cube_to_spec(cube):
    """ Integrate spatial dimensions of a FITS cube to give a spectrum """
    from astropy.units import Unit
    value = cube.dat
    A = area(cube) * Unit('deg**2').to(Unit('sr'))
    logging.debug('cube.shape: {0}'.format(value.shape))
    logging.debug('A.shape: {0}'.format(A.shape))
    # Note that this is the correct way to get an average flux:

    spec = (value * A).sum(-1).sum(-1)
    return spec


def axis_coordinates(fitsimage, return_world=True):
    """Compute vector of world or pixel positions
    for a kapteyn.maputils.FITSimage"""
    pos = {}
    for axis in fitsimage.axisinfo.values():
        # Make an array of pixel coordinates
        pix = np.arange(axis.axlen) + 1
        world = axis.crval + (pix - axis.crpix) * axis.cdelt

        if return_world:
            pos[axis.axname] = world
        else:
            pos[axis.axname] = pix

    return pos


def coordinates(image, world=True, lon_sym=True, radians=False):
    """Get coordinate images for a given image.

    This function is useful if you want to compute
    an image with values that are a function of position.

    Parameters
    ----------
    image : `astropy.io.fits.ImageHDU`
    world : bool
        Use world coordinates (or pixel coordinates)?
    lon_sym : bool
        Use symmetric longitude range `(-180, 180)` (or `(0, 360)`)?

    Returns
    -------
    (lon, lat) : tuple of arrays
        Images as numpy arrays with values
        containing the position of the given pixel.

    Examples
    --------
    >>> l, b = coordinates(image)
    >>> dist = sqrt( (l-42)**2 + (b-43)**2)
    """
    # Create arrays of pixel coordinates
    y, x = np.indices(image.data.shape, dtype='int32') + 1

    if not world:
        return x, y

    from astropy.wcs import WCS
    wcs = WCS(image.header)
    lon, lat = wcs.wcs_pix2world(x, y, 1)
    
    if lon_sym:
        lon = np.where(lon > 180, lon - 360, lon)
    
    if radians:
        lon = np.radians(lon)
        lat = np.radians(lat)

    return lon, lat


def coordinates_outside(image, x_pix, y_pix, boxsize, glon_sym=True):
    """Returns coordinate map of size 2*boxsize, centered at x_pix, y_pix.

    @todo: is this function really necessary? it seems it duplicates coordinates() code!"""
    # Set up pixels
    y, x = np.indices((boxsize, boxsize), np.int) + 1

    # Slide to right position
    x += x_pix - boxsize / 2
    y += y_pix - boxsize / 2

    # Compute coordinates
    l, b = image.proj.toworld((x.flatten(), y.flatten()))

    if glon_sym:
        # Correct longitude range
        l = np.where(l > 180, l - 360, l)

    # Reshape l and b arrays
    l.shape, b.shape = x.shape, y.shape
    return l, b


def separation(image, ref_pos=None):
    """ Returns an image where each pixel contains
    the angular separation on the sky to a reference position.

    If no reference position is given the separation to
    the FITS header reference position is computed."""
    if ref_pos:
        ref_l, ref_b = ref_pos
    else:
        ref_l, ref_b = image.hdr['CRVAL1'], image.hdr['CRVAL2']

    logging.info('Computing coordinate maps')
    l, b = coordinates(image)

    logging.info('Computing separation map')
    from kapteyn.maputils import dist_on_sphere
    return dist_on_sphere(l, b, ref_l, ref_b)


def area(image, deg=True):
    """Compute the area of each pixel.
    This will only work for cartesian maps!

    deg = True: output in deg **2
    deg = False: output in sr

    @todo Something more general should be implemented, maybe this:

    /***********************************************************************//**
 * @brief Returns solid angle of pixel in units of steradians
 *
 * @param[in] pix Pixel index (x,y)
 *
 * Estimate solid angles of pixel by compuing the coordinates in the 4 pixel
 * corners. The surface is computed using a cartesian approximation:
 *           a
 *     1-----------2                 a+b
 * h  /             \    where A = h ---
 *   4---------------3                2
 *           b
 * This is a brute force technique that works sufficiently well for non-
 * rotated sky maps. Something more intelligent should be implemented in
 * the future.
 *
 * @todo Implement accurate solid angle computation (so far only brute force
 *       estimation)
 ***************************************************************************/
double GWcsCAR::omega(const GSkyPixel& pix) const
{
    // Bypass const correctness
    GWcsCAR* ptr = (GWcsCAR*)this;

    // Get the sky directions of the 4 corners
    GSkyDir dir1 = ptr->xy2dir(GSkyPixel(pix.x()-0.5, pix.y()-0.5));
    GSkyDir dir2 = ptr->xy2dir(GSkyPixel(pix.x()+0.5, pix.y()-0.5));
    GSkyDir dir3 = ptr->xy2dir(GSkyPixel(pix.x()+0.5, pix.y()+0.5));
    GSkyDir dir4 = ptr->xy2dir(GSkyPixel(pix.x()-0.5, pix.y()+0.5));
    GSkyDir dir5 = ptr->xy2dir(GSkyPixel(pix.x(), pix.y()-0.5));
    GSkyDir dir6 = ptr->xy2dir(GSkyPixel(pix.x(), pix.y()+0.5));

    // Compute distances between sky directions
    double a = dir1.dist(dir2);
    double b = dir3.dist(dir4);
    double h = dir5.dist(dir6);

    // Compute solid angle
    double omega = 0.5*(h*(a+b));

    // Return solid angle
    return omega;
}
    """
    from astropy.units import Unit
    from astropy.io import fits
    # Area of one pixel at the equator
    cdelt0 = image.header['CDELT1']
    cdelt1 = image.header['CDELT2']
    equator_area = abs(cdelt0 * cdelt1)
    if not deg:
        equator_area = equator_area / Unit('deg2').to(Unit('sr'))

    # Compute image with fraction of pixel area at equator
    glat = coordinates(image)[1]
    area_fraction = np.cos(np.radians(glat))

    return equator_area * area_fraction


def contains(image, x, y, world=True):
    """
    Check if given pixel or world positions are in an image.
    """
    x, y = _prepare_arrays(image, x, y, world)

    nx, ny = image.proj.naxis
    return (x >= 0.5) & (x <= nx + 0.5) & (y >= 0.5) & (y <= ny + 0.5)


def lookup(image, x, y, world=True, outside_value=np.nan):
    """
    Look up image values at given pixel or world positions.

    Works with lists x, y for now since I couldn't figure out
    how to do it with numpy arrays
    """
    x, y = _prepare_arrays(image, x, y, world)

    shape = x.shape
    x = np.round(x).astype(int).reshape((x.size,))
    y = np.round(y).astype(int).reshape((y.size,))

    in_image = contains(image, x, y, False)

    """
    # FIXME: why doesn't this work?
    print image.dat.shape
    val = np.where(in_image, image.dat[y - 1, x - 1], np.nan)
    return val
    """

    """
    val = [image.dat[y[i] - 1, x[i] - 1]
           for i in range(val.size)
           if in_image[i]
           else outside_value]
    return np.array(val).reshape(shape)
    """

    val = np.zeros(x.shape, dtype='float32')
    for ii in range(val.size):
        if in_image[ii]:
            logging.debug('Looking up position {0} of {1}: '
                          'x = {2}, y = {3}'
                          ''.format(ii, len(x), x[ii], y[ii]))
            # The -1 converts FITS pixel coordinates to Numpy indices
            val[ii] = image.dat[y[ii] - 1, x[ii] - 1]
        else:
            logging.debug('Position {0} of {1} is not in the image: '
                          'x = {2}, y = {3}'
                          ''.format(ii, len(x), x[ii], y[ii]))
            val[ii] = outside_value
    return val.reshape(shape)


def lookup_max(image, GLON, GLAT, theta):
    """Look up the max image values within a circle of radius theta
    around lists of given positions (nan if outside)"""
    GLON = np.asarray(GLON)
    GLON = np.where(GLON > 180, GLON - 360, GLON)
    GLAT = np.asarray(GLAT)
    n_pos = len(GLON)
    theta = np.asarray(theta) * np.ones(n_pos, dtype='float32')

    ll, bb = coordinates(image)

    val = np.nan * np.ones(n_pos, dtype='float32')
    for ii in range(n_pos):
        logging.debug('Looking up position {0} of {1}: '
                      'GLON = {2}, GLAT = {3}, theta={4}'
                      ''.format(ii, n_pos, GLON[ii],
                                GLAT[ii], theta[ii]))
        mask = ((GLON[ii] - ll) ** 2 +
                (GLAT[ii] - bb) ** 2 <=
                theta[ii] ** 2)
        try:
            val[ii] = image.dat[mask].max()
        except ValueError:
            logging.debug('Position {0} of {1} is not in the image: '
                          'GLON = {2}, GLAT = {3}, theta={4}'
                          ''.format(ii, n_pos, GLON[ii],
                                    GLAT[ii], theta[ii]))
    return val


def print_projection_info(proj, print_header=False):
    """ Print basic info about projection object """

    if print_header:
        print 'Header:'
        print proj.source

    print 'skysys:   ', proj.skysys
    print 'skyout:   ', proj.skyout
    print 'naxis:    ', proj.naxis
    print 'lonaxnum: ', proj.lonaxnum
    print 'lataxnum: ', proj.lataxnum

    print 'ctype:    ', proj.ctype
    print 'cunit:    ', proj.cunit
    print 'crval:    ', proj.crval
    print 'cdelt:    ', proj.cdelt
    print 'crpix:    ', proj.crpix

    nx, ny = proj.naxis

    print 'Coordinates of the most central image pixel:'
    pixel = [(nx / 2, ny / 2)]
    print_pixel_info(proj, pixel)

    print 'Coordinates of the image corners:'
    pixel = [(1, 1), (1, ny), (nx, 1), (nx, ny)]
    print_pixel_info(proj, pixel)

    print 'Coordinates of the image edge middle points:'
    pixel = [(1, ny / 2), (nx, ny / 2), (nx / 2, 1), (nx / 2, ny)]
    print_pixel_info(proj, pixel)


def print_pixel_info(proj, pixel):
    """
    Convert pixel to world coordinates print a nice table
    pixel should be an iterable (i.e. list or array) of tuples
    """
    world = proj.toworld(pixel)
    lon_name = proj.ctype[0].split('-')[0]
    lat_name = proj.ctype[1].split('-')[0]
    print '%10s %10s %10s %10s' % ('x', 'y', lon_name, lat_name)
    for p, w in zip(pixel, world):
        print '%10g %10g %10g %10g' \
            % (p[0], p[1], w[0], w[1])

#------------------------------------------------------------
# scipy wrappers and helper functions
#------------------------------------------------------------


def _get_structure_indices(radius):
    """
    Get arrays of indices for a symmetric structure,
    i.e. with an odd number of pixels and 0 at the center
    """
    radius = int(radius)
    y, x = np.mgrid[-radius: radius + 1, -radius: radius + 1]
    return x, y


def binary_disk(radius):
    """
    Generate a binary disk.
    Value 1 inside and 0 outside.

    Useful as a structure element for morphological transformations.

    Note that the returned structure always has an odd number
    of pixels so that shifts during correlation are avoided.
    """
    x, y = _get_structure_indices(radius)
    structure = x ** 2 + y ** 2 <= radius ** 2
    return structure


def binary_ring(r_in, r_out):
    """
    Generate a binary ring.
    Value 1 inside and 0 outside.

    Useful as a structure element for morphological transformations.

    Note that the returned structure always has an odd number
    of pixels so that shifts during correlation are avoided.
    """
    x, y = _get_structure_indices(r_out)
    mask1 = r_in ** 2 <= x ** 2 + y ** 2
    mask2 = x ** 2 + y ** 2 <= r_out ** 2
    return mask1 & mask2


def opening(data, radius):
    """ Dilate with disk of given radius """
    logging.info('radius: {0}'.format(radius))
    structure = binary_disk(radius)
    data = scipy.ndimage.binary_opening(data, structure)
    return data.astype(np.uint8)


def segment(data):
    """ Segment the image, i.e. label connected pixel components. """
    logging.info('Segmenting image.')
    segments, nsources = scipy.ndimage.label(data)
    logging.info('Found {0} segments'.format(nsources))
    return segments, nsources


def distance_transform_edt(segments):
    """  Exact Euclidean distance transform.
    Computes distances to background for a segmented image."""
    logging.info('Computing exact Euclidean distance transform.')
    return scipy.ndimage.distance_transform_edt(segments)


def tophat_correlate(data, radius, mode=DEFAULT_MODE):
    """ Correlate with disk of given radius """
    logging.debug('tophat_correlate shape = {0}, radius = {1}'
                  ''.format(data.shape, radius))
    structure = binary_disk(radius)
    return scipy.ndimage.convolve(data, structure, mode=mode)


def maximum_filter(data, radius, mode=DEFAULT_MODE):
    """ Maximum filter with disk of given radius """
    logging.debug('maximum_filter shape = {0}, radius = {1}'
                  ''.format(data.shape, radius))
    structure = binary_disk(radius)
    return scipy.ndimage.maximum_filter(data, footprint=structure, mode=mode)


def tophat_pixel_correction(radius):
    """ Return area correction factor due to pixelation """
    actual = binary_disk(radius).sum()
    desired = np.pi * radius ** 2
    pixel_correction = actual / desired
    logging.debug('pixel_correction = {0}'.format(pixel_correction))
    return pixel_correction


def ring_correlate(data, r_in, r_out, mode=DEFAULT_MODE):
    """ Correlate with ring of given radii """
    logging.debug('ring_correlate shape = {0}, r_in = {1}, r_out = {2}'
                  ''.format(data.shape, r_in, r_out))
    structure = binary_ring(r_in, r_out)
    return scipy.ndimage.convolve(data, structure, mode=mode)


def ring_pixel_correction(r_in, r_out):
    """ Return area correction factor due to pixelation """
    actual = binary_ring(r_in, r_out).sum()
    desired = np.pi * (r_out ** 2 - r_in ** 2)
    pixel_correction = actual / desired
    logging.debug('pixel_correction = {0}'.format(pixel_correction))
    return pixel_correction


def report_mask_stats(mask, invert=False):
    """
    Report summary statistics on a mask
    mask: kapteyn.maputils.FITSimage
    invert: set to True for exclusion regions
    """
    def print_info(label, parameter, scale):
        """ Little helper function to avoid typing """
        print '%s_pix %g' % (label, parameter)
        print '%s_deg %g' % (label, parameter * scale)

    # Compute summary statistics
    deg_per_pix = mask.hdr['CDELT2']
    mask = mask.dat.astype(bool)
    if invert:
        mask = np.invert(mask)
    labels, nregions = scipy.ndimage.label(mask)
    distances = scipy.ndimage.distance_transform_edt(mask)
    distances *= deg_per_pix

    # Print summary statistics
    print 'nregions', nregions
    print_info('image_area', mask.size, deg_per_pix ** 2)
    print_info('mask_area', labels.astype(bool).sum(), deg_per_pix ** 2)


def gaussian_psf(psf_width):
    """ Returns a Gaussian kernel. psf_width in pix """
    def psf(image):
        return scipy.ndimage.gaussian_filter(image, psf_width)
    return psf


def gaussian_filter(image, sigma, mode=DEFAULT_MODE):
    """Kernel size is automatically chosen how?

    Note that this function is consistent with fgauss.
    You just have to use a kernel size of many sigmas with fgauss:
    fgauss nsigma=6
    """
    return scipy.ndimage.gaussian_filter(image, sigma, mode=mode)


def aperphot(img, x, y, aper, sky, sky_type='median', verbose=False):
    """ aperphot(img, x, y, aper, sky, sky_type='median', verbose=False)

    Performs the aperture photometry of a given x,y point in an image.
    Note: Doesn't handle subpixel integration.

    img: numpy 2D array of the image to perform the photometry on.
        y_dimension, x_dimension = img.shape
        # Note: x is the second axis as opposed to the first in IDL!
    x: x coordinate of the position to perform the photometry on.
        # Note: the first pixel as a coordinate 0, not 1!
    y: y coordinate of the position to perform the photometry on.
    aper: aperture size in pixel.
    sky: 2-element list/tuple/array providing the inner and outer radii
        to calculate the sky from.
    sky_type ('median'): if 'median' will use a median for the sky level
        determination within the sky annulus, otherwise will use a mean.
    verbose (False): If true, will print some information.

    >>> flux, err_flux = aperphot(img, 32.2, 35.6, 5., [10.,15.], sky_type='median')
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


def pyfits_find_index(hdulist, key, value):
    """Find a FITS extension by header key/value.

    Returns -1 if none found
    f: astropy.io.fits.HDUList
    """
    for i, hdu in enumerate(hdulist):
        value_found = hdu.header[key]
        if value.lower() == value_found.lower():
            logging.debug(i, hdu, key, value.lower(), value_found.lower())
            return i
    return -1


def find_max(image):
    """image = kapteyn.maputils.FITSimage"""
    from scipy.ndimage import maximum_position
    if isinstance(image, str):
        image = fits.open(image)
    proj = wcs.WCS(image.header)
    data = image.data
    data[np.isnan(data)] = -np.inf
    y, x = maximum_position(data)
    GLON, GLAT = proj.wcs_pix2world(x, y, 0)
    val = data[int(y), int(x)]
    return GLON, GLAT, val
