# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""PyFACT compatibility module.

This module contains a few functions and classes from PyFACT:

- http://pyfact.readthedocs.org/
- https://github.com/gammapy/gammapy/pull/68

TODO: This is a short-term solution until we find time to refactor
this functionality into gammapy.
"""
from __future__ import print_function, division
import logging
import numpy as np
from astropy.io import fits
from ..stats import significance_on_off

__all__ = ['ChisquareFitter',
           'SkyCircle',
           'SkyCoord',
           'circle_circle_intersection_array',
           'circle_circle_intersection_float',
           'create_sky_map',
           'get_cam_acc',
           'get_exclusion_region_map',
           'get_sky_mask_circle',
           'get_sky_mask_ring',
           'oversample_sky_map',
           'plot_skymaps',
           'skycircle_from_str',
           ]


class ChisquareFitter :
    """
    Convenience class to perform Chi^2 fits.

    Attributes
    ----------

    fitfunc : function
        Fitfunction
    results : array
        Fit results from scipy.optimize.leastsq()
    chi_arr : float array
        Array with the final chi values
    chi2 : float
        Summed Chi^2
    dof : float
        Degrees of freedom
    prob : float
        Probability of the fit

    Parameters
    ----------

    fitfunc : function
        Fit function.
    """

    def __init__(self, fitfunc) :
        self.fitfunc = fitfunc
        self.results = None


    def fit_data(self, p0, x, y, y_err) :
        """
        Perform actual fit.

        Parameters
        ----------

        p0 : float array
            Start parameters
        x, y, y_err : float arrays
            Data to be fitted.
        """
        from scipy.optimize import leastsq
        from scipy.special import gammainc

        self.results = leastsq(self.chi_func, p0, args=(x, y, y_err), full_output=True)
        if self.results[4] :
            self.chi_arr = self.chi_func(self.results[0], x, y, y_err)
            self.chi2 = np.sum(np.power(self.chi_arr, 2.))
            self.dof = len(x) - len(p0)
            # self.prob = scipy.special.gammainc(.5 * self.dof, .5 * self.chi2) / scipy.special.gamma(.5 * self.dof)
            self.prob = 1. - gammainc(.5 * self.dof, .5 * self.chi2)
        return self.results[4]

    def chi_func(self, p, x, y, err):
        """Returns Chi"""
        return (self.fitfunc(p, x) - y) / err  # Distance to the target function

    def print_results(self) :
        """Prints out results to the command line using the logging module."""
        if self.results == None :
            logging.warning('No fit results to report since no fit has been performed yet')
            return
        if self.results[4] < 5 :
            logging.info('Fit was successful!')
        else :
            logging.warning('Fitting failed!')
            logging.warning('Message: {0}'.format(self.results[3]))
        logging.info('Chi^2  : {0:f}'.format(self.chi2))
        logging.info('d.o.f. : {0:d}'.format(self.dof))
        logging.info('Prob.  : {0:.4e}'.format(self.prob))
        for i, v in enumerate(self.results[0]) :
            if self.results[1] != None :
                logging.info('P{0}     : {1:.4e} +/- {2:.4e}'.format(i, v,
                                                                     np.sqrt(self.results[1][i][i])))
            else :
                logging.info('P{0}     : {1:.4e}'.format(i, v))


class SkyCircle:
    """A circle on the sky."""
    
    def __init__(self, c, r) :
        """
        A circle on the sky.

        Parameters
        ----------
        coord : SkyCoord
            Coordinates of the circle center (RA, Dec)
        r : float
            Radius of the circle (deg).
        """
        self.c, self.r = c, r

    def contains(self, c) :
        """
        Checks if the coordinate lies inside the circle.

        Parameters
        ----------
        c : SkyCoord

        Returns
        -------
        contains : bool
            True if c lies in the SkyCircle.
        """
        return self.c.dist(c) <= self.r

    def intersects(self, sc) :
        """
        Checks if two sky circles overlap.

        Parameters
        ----------
        sc : SkyCircle
        """
        return self.c.dist(sc.c) <= self.r + sc.r


class SkyCoord:
    """Sky coordinate in RA and Dec. All units should be degree."""
    
    def __init__(self, ra, dec) :
        """
        Sky coordinate in RA and Dec. All units should be degree.
        
        In the current implementation it should also work with arrays, though one has to be careful in dist.

        Parameters
        ----------
        ra : float/array
            Right ascension of the coordinate.
        dec : float/array
            Declination of the coordinate.
        """
        self.ra, self.dec = ra, dec

    def dist(self, c) :
        """
        Return the distance of the coordinates in degree following the haversine formula,
        see e.g. http://en.wikipedia.org/wiki/Great-circle_distance.

        Parameters
        ----------
        c : SkyCoord

        Returns
        -------
        distance : float
            Return the distance of the coordinates in degree following the haversine formula.

        Notes
        -----
        http://en.wikipedia.org/wiki/Great-circle_distance
        """
        return 2. * np.arcsin(np.sqrt(np.sin((self.dec - c.dec) / 360. * np.pi) ** 2.
                                      + np.cos(self.dec / 180. * np.pi) * np.cos(c.dec / 180. * np.pi)\
                                          * np.sin((self.ra - c.ra) / 360. * np.pi) ** 2.)) / np.pi * 180.


def circle_circle_intersection_array(R, r, d) :
    """Calculate overlap area between two circles on the sphere.

    Works _only_ with numpy arrays of equal length.

    Parameters
    ----------
    R : array
        Radius of the first circle.
    r : array
        Radius of the second circle.
    d : array
        Distance of the two circle (center to center).

    Returns
    -------
    area : array
        Overlap area between the two circles (steradian).
    """

    # Define a few useful functions
    X = lambda R, r, d: (d * d + r * r - R * R) / (2. * d * r)
    Y = lambda R, r, d: (d * d + R * R - r * r) / (2. * d * R)
    Z = lambda R, r, d: (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)

    result = np.zeros(len(R))
    mask1 = R >= d + r
    if mask1.any() :
        result[mask1] = np.pi * r[mask1] ** 2.
    mask2 = r >= d + R
    if mask2.any() :
        result[mask2] = np.pi * R[mask2] ** 2.
    mask = (R + r > d) * np.invert(mask1) * np.invert(mask2)
    if mask.any() :
        r, R, d = r[mask], R[mask], d[mask]
        result[mask] = (r ** 2.) * np.arccos(X(R, r, d)) + (R ** 2.) * np.arccos(Y(R, r, d)) - .5 * np.sqrt(Z(R, r, d));
    return result


def circle_circle_intersection_float(R, r, d) :
    """Calculate overlap area between two circles on the sphere.

    Works _only_ with floats.

    Parameters
    ----------
    R : float
        Radius of the first circle.
    r : float
        Radius of the second circle.
    d : float
        Distance of the two circle (center to center).

    Returns
    -------
    area : float
        Overlap area between the two circles (steradian).
    """

    # Define a few useful functions
    X = lambda R, r, d: (d * d + r * r - R * R) / (2. * d * r)
    Y = lambda R, r, d: (d * d + R * R - r * r) / (2. * d * R)
    Z = lambda R, r, d: (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)

    if R >= d + r :
        return np.pi * r ** 2.
    elif r >= d + R :
        return np.pi * R ** 2.
    elif R + r > d :
        return (r ** 2.) * np.arccos(X(R, r, d)) + (R ** 2.) * np.arccos(Y(R, r, d)) - .5 * np.sqrt(Z(R, r, d))
    else :
        return 0.


def skycircle_from_str(cstr) :
    """Creates SkyCircle from circle region string."""
    x, y, r = eval(cstr.upper().replace('CIRCLE', ''))
    return SkyCircle(SkyCoord(x, y), r)


def get_cam_acc(camdist, rmax=4., nbins=None, exreg=None, fit=False, fitfunc=None, p0=None) :
    """
    Calculates the camera acceptance histogram from a given list with camera distances (event list).

    Parameters
    ----------
    camdist : array
        Numpy array of camera distances (event list).
    rmax : float, optional
        Maximum radius for the acceptance histogram.
    nbins : int, optional
        Number of bins for the acceptance histogram (default = 0.1 deg).
    exreg : array, optional
        Array of exclusion regions. Exclusion regions are given by an aray of size 2
        [r, d] with r = radius, d = distance to camera center
    fit : bool, optional
        Fit acceptance histogram (default=False).
    """
    if not nbins :
        nbins = int(rmax / .1)
    # Create camera distance histogram
    n, bins = np.histogram(camdist, bins=nbins, range=[0., rmax])
    nerr = np.sqrt(n)
    # Bin center array
    r = (bins[1:] + bins[:-1]) / 2.
    # Bin area (ring) array
    r_a = (bins[1:] ** 2. - bins[:-1] ** 2.) * np.pi
    # Deal with exclusion regions
    ex_a = None
    if exreg :
        ex_a = np.zeros(len(r))
        t = np.ones(len(r))
        for reg in exreg :
            ex_a += (circle_circle_intersection_array(bins[1:], t * reg[0], t * reg[1])
                     - circle_circle_intersection_array(bins[:-1], t * reg[0], t * reg[1]))
        ex_a /= r_a
    # Fit the data
    fitter = None
    if fit :
        # fitfunc = lambda p, x: p[0] * x ** p[1] * (1. + (x / p[2]) ** p[3]) ** ((p[1] + p[4]) / p[3])
        if not fitfunc :
            fitfunc = lambda p, x: p[0] * x ** 0. * (1. + (x / p[1]) ** p[2]) ** ((0. + p[3]) / p[2])
            # fitfunc = lambda p, x: p[0] * x ** 0. * (1. + (x / p[1]) ** p[2]) ** ((0. + p[3]) / p[2]) + p[4] / (np.exp(p[5] * (x - p[6])) + 1.)            
        if not p0 :
            p0 = [n[0] / r_a[0], 1.5, 3., -5.]  # Initial guess for the parameters
            # p0 = [.5 * n[0] / r_a[0], 1.5, 3., -5., .5 * n[0] / r_a[0], 100., .5] # Initial guess for the parameters            
        fitter = ChisquareFitter(fitfunc)
        m = (n > 0.) * (nerr > 0.) * (r_a != 0.) * ((1. - ex_a) != 0.)
        if np.sum(m) <= len(p0) :
            logging.error('Could not fit camera acceptance (dof={0}, bins={1})'.format(len(p0), np.sum(m)))
        else :
            # ok, this _should_ be improved !!!
            x, y, yerr = r[m], n[m] / r_a[m] / (1. - ex_a[m]) , nerr[m] / r_a[m] / (1. - ex_a[m])
            m = np.isfinite(x) * np.isfinite(y) * np.isfinite(yerr) * (yerr != 0.)
            if np.sum(m) <= len(p0) :
                logging.error('Could not fit camera acceptance (dof={0}, bins={1})'.format(len(p0), np.sum(m)))
            else :
                fitter.fit_data(p0, x[m], y[m], yerr[m])
    return (n, bins, nerr, r, r_a, ex_a, fitter)


def get_sky_mask_circle(r, bin_size) :
    """
    Returns a 2d numpy histogram with (2. * r / bin_size) bins per axis
    where a circle of radius has bins filled 1.s, all other bins are 0.

    Parameters
    ----------
    r : float
        Radius of the circle.
    bin_size : float
        Physical size of the bin, same units as rmin, rmax.

    Returns
    -------
    sky_mask : 2d numpy array
        Returns a 2d numpy histogram with (2. * r / bin_size) bins per axis
        where a circle of radius has bins filled 1.s, all other bins are 0.
    """
    nbins = int(np.ceil(2. * r / bin_size))
    sky_x = np.ones((nbins, nbins)) * np.linspace(bin_size / 2., 2. * r - bin_size / 2., nbins)
    sky_y = np.transpose(sky_x)
    sky_mask = np.where(np.sqrt((sky_x - r) ** 2. + (sky_y - r) ** 2.) < r, 1., 0.)
    return sky_mask


def get_sky_mask_ring(rmin, rmax, bin_size) :
    """
    Returns a 2d numpy histogram with (2. * rmax / bin_size) bins per axis
    filled with a ring with inner radius rmin and outer radius rmax of 1.,
    all other bins are 0..

    Parameters
    ----------
    rmin : float
        Inner radius of the ring.
    rmax : float
        Outer radius of the ring.
    bin_size : float
        Physical size of the bin, same units as rmin, rmax.

    Returns
    -------
    sky_mask : 2d numpy array
        Returns a 2d numpy histogram with (2. * rmax / bin_size) bins per axis
        filled with a ring with inner radius rmin and outer radius rmax of 1.,
        all other bins are 0..
    """
    nbins = int(np.ceil(2. * rmax / bin_size))
    sky_x = np.ones((nbins, nbins)) * np.linspace(bin_size / 2., 2. * rmax - bin_size / 2., nbins)
    sky_y = np.transpose(sky_x)
    sky_mask = np.where((np.sqrt((sky_x - rmax) ** 2. + (sky_y - rmax) ** 2.) < rmax) * (np.sqrt((sky_x - rmax) ** 2. + (sky_y - rmax) ** 2.) > rmin), 1., 0.)
    return sky_mask


def get_exclusion_region_map(map, rarange, decrange, exreg) :
    """
    Creates a map (2d numpy histogram) with all bins inside of exclusion regions set to 0. (others 1.).

    Dec is on the 1st axis (x), RA is on the 2nd (y).

    Parameters
    ----------
    map : 2d array
    rarange : array
    decrange : array
    exreg : array-type of SkyCircle
    """
    xnbins, ynbins = map.shape
    xstep, ystep = (decrange[1] - decrange[0]) / float(xnbins), (rarange[1] - rarange[0]) / float(ynbins)
    sky_mask = np.ones((xnbins, ynbins))
    for x, xval in enumerate(np.linspace(decrange[0] + xstep / 2., decrange[1] - xstep / 2., xnbins)) :
        for y, yval in enumerate(np.linspace(rarange[0] + ystep / 2., rarange[1] - ystep / 2., ynbins)) :
            for reg in exreg :
                if reg.contains(SkyCoord(yval, xval)) :
                    sky_mask[x, y] = 0.
    return sky_mask


def oversample_sky_map(sky, mask, exmap=None) :
    """Oversamples a 2d numpy histogram with a given mask.

    Parameters
    ----------
    sky : 2d array 
    mask : 2d array
    exmap : 2d array
    """
    from scipy.ndimage import convolve
    sky = np.copy(sky)

    sky_nx, sky_ny = sky.shape[0], sky.shape[1]
    mask_nx, mask_ny = mask.shape[0], mask.shape[1]
    mask_centerx, mask_centery = (mask_nx - 1) / 2, (mask_ny - 1) / 2

    # new oversampled sky plot
    sky_overs = np.zeros((sky_nx, sky_ny))

    # 2d hist keeping the number of bins used (alpha)
    sky_alpha = np.ones((sky_nx, sky_ny))

    sky_base = np.ones((sky_nx, sky_ny))
    if exmap != None :
        sky *= exmap
        sky_base *= exmap

    convolve(sky, mask, sky_overs, mode='constant')
    convolve(sky_base, mask, sky_alpha, mode='constant')

    return (sky_overs, sky_alpha)


def create_sky_map(input_file_name,
                   skymap_size=5.,
                   skymap_bin_size=0.05,
                   r_overs=.125,
                   ring_bg_radii=None,
                   template_background=None,
                   skymap_center=None,
                   write_output=False,
                   fov_acceptance=False,
                   do_graphical_output=True,
                   loglevel='INFO'):
    """Create sky map.
    
    TODO: describe
    
    Parameters
    ----------
    TODO
    
    Returns
    -------
    TODO
    """
    import os
    from scipy.interpolate import UnivariateSpline
    import matplotlib.pyplot as plt

    #---------------------------------------------------------------------------
    # Loop over the file list, calculate quantities, & fill histograms

    # Skymap definition
    # skymap_size, skymap_bin_size = 6., 0.05
    rexdeg = .3

    # Intialize some variables
    skycenra, skycendec, pntra, pntdec = None, None, None, None
    if skymap_center :
        skycenra, skycendec = eval(skymap_center)
        logging.info('Skymap center: RA {0}, Dec {1}'.format(skycenra, skycendec))

    ring_bg_r_min, ring_bg_r_max = .3, .7
    if ring_bg_radii :
        ring_bg_r_min, ring_bg_r_max = eval(ring_bg_radii)

    if r_overs > ring_bg_r_min :
        logging.warning('Oversampling radius is larger than the inner radius chosen for the ring BG: {0} > {1}'.format(r_overs, ring_bg_r_min))

    logging.info('Skymap size         : {0} deg'.format(skymap_size))
    logging.info('Skymap bin size     : {0} deg'.format(skymap_bin_size))
    logging.info('Oversampling radius : {0} deg'.format(r_overs))
    logging.info('Ring BG radius      : {0} - {1} deg'.format(ring_bg_r_min, ring_bg_r_max))

    skymap_nbins, sky_dec_min, sky_dec_max, objcosdec, sky_ra_min, sky_ra_max = 0, 0., 0., 0., 0., 0.
    sky_hist, acc_hist, extent = None, None, None
    tpl_had_hist, tpl_acc_hist = None, None
    sky_ex_reg, sky_ex_reg_map = None, None

    telescope, object_ = 'NONE', 'NONE'

    firstloop = True

    exposure = 0.

    # Read in input file, can be individual fits or bankfile
    logging.info('Opening input file ..')

    def get_filelist(input_file_name) :
        # Check if we are dealing with a single file or a bankfile
        # and create/read in the file list accordingly
        try :
            f = fits.open(input_file_name)
            f.close()
            file_list = [input_file_name]
        except :
            # We are dealing with a bankfile
            logging.info('Reading files from bankfile {0}'.format(input_file_name))
            file_list = np.loadtxt(input_file_name, dtype='S', usecols=[0])
        return file_list

    file_list = get_filelist(input_file_name)

    tpl_file_list = None

    # Read in template files
    if template_background :
        tpl_file_list = get_filelist(template_background)
        if len(file_list) != len(tpl_file_list) :
            logging.warning('Different number of signal and template background files. Switching off template background analysis.')
            template_background = None

    # Main loop over input files
    for i, file_name in enumerate(file_list) :

        logging.info('Processing file {0}'.format(file_name))

        def get_evl(file_name) :
            # Open fits file
            hdulist = fits.open(file_name)
            # Access header of second extension
            hdr = hdulist['EVENTS'].header
            # Access data of first extension
            tbdata = hdulist['EVENTS'].data
            # Calculate some useful quantities and add them to the table
            # Distance from the camera (FOV) center
            camdist = np.sqrt(tbdata.field('DETX') ** 2. + tbdata.field('DETY') ** 2.)
            camdist_col = fits.Column(name='XCAMDIST', format='1E', unit='deg', array=camdist)
            # Add new columns to the table
            coldefs_new = fits.ColDefs([camdist_col])
            # coldefs_new = fits.ColDefs([camdist_col, detx_col, dety_col])
            newtable = fits.new_table(hdulist[1].columns + coldefs_new)
            # New table data
            return hdulist, hdr, newtable.data

        hdulist, ex1hdr, tbdata = get_evl(file_name)

        # Read in eventlist for template background
        tpl_hdulist, tpl_tbdata = None, None
        if template_background :
            tpl_hdulist, tpl_hdr, tpl_tbdata = get_evl(tpl_file_list[i])

        #---------------------------------------------------------------------------
        # Intialize & GTI

        objra, objdec = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ']
        if firstloop :
            # If skymap center is not set, set it to the object position of the first run
            if skycenra == None or skycendec == None :
                skycenra, skycendec = objra, objdec
                logging.debug('Setting skymap center to skycenra, skycendec = {0}, {1}'.format(skycenra, skycendec))
            if 'TELESCOP' in ex1hdr :
                telescope = ex1hdr['TELESCOP']
                logging.debug('Setting TELESCOP to {0}'.format(telescope))
            if 'OBJECT' in ex1hdr :
                object_ = ex1hdr['OBJECT']
                logging.debug('Setting OBJECT to {0}'.format(object_))

        mgit, tpl_mgit = np.ones(len(tbdata), dtype=np.bool), None
        if template_background :
            tpl_mgit = np.ones(len(tpl_tbdata), dtype=np.bool)
        try :
            # Note: according to the eventlist format document v1.0.0 Section 10
            # "The times are expressed in the same units as in the EVENTS
            # table (seconds since mission start in terresterial time)."
            for gti in hdulist['GTI'].data :
                mgit *= (tbdata.field('TIME') >= gti[0]) * (tbdata.field('TIME') <= gti[1])
                if template_background :
                    tpl_mgit *= (tpl_tbdata.field('TIME') >= gti[0]) * (tpl_tbdata.field('TIME') <= gti[1])
        except :
            logging.warning('File does not contain a GTI extension')
        
        #---------------------------------------------------------------------------
        #  Handle exclusion region

        # If no exclusion regions are given, use the object position from the first run
        if sky_ex_reg == None :
            sky_ex_reg = [SkyCircle(SkyCoord(objra, objdec), rexdeg)]
            logging.info('Setting exclusion region to object position (ra={0}, dec={1}, r={2}'.format(objra, objdec, rexdeg))

        pntra, pntdec = ex1hdr['RA_PNT'], ex1hdr['DEC_PNT']
        obj_cam_dist = SkyCoord(skycenra, skycendec).dist(SkyCoord(pntra, pntdec))

        exposure_run = ex1hdr['LIVETIME']
        exposure += exposure_run

        logging.info('RUN Start date/time : {0} {1}'.format(ex1hdr['DATE_OBS'], ex1hdr['TIME_OBS']))
        logging.info('RUN Stop date/time  : {0} {1}'.format(ex1hdr['DATE_END'], ex1hdr['TIME_END']))
        logging.info('RUN Exposure        : {0:.2f} [s]'.format(exposure_run))
        logging.info('RUN Pointing pos.   : RA {0:.4f} [deg], Dec {1:.4f} [deg]'.format(pntra, pntdec))
        logging.info('RUN Obj. cam. dist. : {0:.4f} [deg]'.format(obj_cam_dist))
        
        # Cut out source region for acceptance fitting
        exmask = None
        for sc in sky_ex_reg :
            if exmask :
                exmask *= sc.c.dist(SkyCoord(tbdata.field('RA'), tbdata.field('DEC'))) < sc.r
            else :
                exmask = sc.c.dist(SkyCoord(tbdata.field('RA'), tbdata.field('DEC'))) < sc.r
        exmask = np.invert(exmask)

        photbdata = tbdata[exmask * mgit]
        if len(photbdata) < 10 :
            logging.warning('Less then 10 events found in file {0} after exclusion region cuts'.format(file_name))

        hadtbdata = None
        if template_background :
            exmask = None
            for sc in sky_ex_reg :
                if exmask :
                    exmask *= sc.c.dist(SkyCoord(tpl_tbdata.field('RA'), tpl_tbdata.field('DEC'))) < sc.r
                else :
                    exmask = sc.c.dist(SkyCoord(tpl_tbdata.field('RA'), tpl_tbdata.field('DEC'))) < sc.r
            exmask = np.invert(exmask)
            hadtbdata = tpl_tbdata[exmask * tpl_mgit]

        #---------------------------------------------------------------------------
        # Calculate camera acceptance
        
        n, bins, nerr, r, r_a, ex_a, fitter = get_cam_acc(
            photbdata.field('XCAMDIST'),
            exreg=[(sc.r, sc.c.dist(SkyCoord(pntra, pntdec))) for sc in sky_ex_reg],
            fit=True,
            )

        # DEBUG
        if logging.root.level is logging.DEBUG :
            fitter.print_results()
            # DEBUG plot
            plt.errorbar(r, n / r_a / (1. - ex_a), nerr / r_a / (1. - ex_a))
            plt.plot(r, fitter.fitfunc(fitter.results[0], r))
            plt.show()

        had_acc, had_n, had_fit = None, None, None
        if template_background :
            had_acc = get_cam_acc(
                hadtbdata.field('XCAMDIST'),
                exreg=[(sc.r, sc.c.dist(SkyCoord(pntra, pntdec))) for sc in sky_ex_reg],
                fit=True
                )
            had_n, had_fit = had_acc[0], had_acc[6]
            logging.debug('Camera acceptance hadrons fit probability: {0}'.format(had_fit.prob))

        # !!! All photons including the exclusion regions
        photbdata = tbdata[mgit]
        if template_background :
            hadtbdata = tpl_tbdata[tpl_mgit]
        if len(photbdata) < 10 :
            logging.warning('Less then 10 events found in file {0} after GTI cut.'.format(file_name))

        tpl_acc_cor_use_interp = True
        tpl_acc_f, tpl_acc = None, None
        if template_background :
            if tpl_acc_cor_use_interp :
                tpl_acc_f = UnivariateSpline(r, n.astype(float) / had_n.astype(float), s=0, k=1)
            else :
                tpl_acc_f = lambda r: fitter.fitfunc(p1, r) / had_fit.fitfunc(had_fit.results[0], r)
            tpl_acc = tpl_acc_f(hadtbdata.field('XCAMDIST'))
            m = hadtbdata.field('XCAMDIST') > r[-1]
            tpl_acc[m] = tpl_acc_f(r[-1])
            m = hadtbdata.field('XCAMDIST') < r[0]
            tpl_acc[m] = tpl_acc_f(r[0])

        #---------------------------------------------------------------------------
        # Skymap - definitions/calculation

        # Object position in the sky
        if firstloop :
            # skycenra, objdec, skymap_size = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ'], 6.
            # if skycenra == None or objdec == None :
            #    skycenra, objdec = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ']

            # Calculate skymap limits
            skymap_nbins = int(skymap_size / skymap_bin_size)
            sky_dec_min, sky_dec_max = skycendec - skymap_size / 2., skycendec + skymap_size / 2.
            objcosdec = np.cos(skycendec * np.pi / 180.)
            sky_ra_min, sky_ra_max = skycenra - skymap_size / 2. / objcosdec, skycenra + skymap_size / 2. / objcosdec

            logging.debug('skymap_nbins = {0}'.format(skymap_nbins))
            logging.debug('sky_dec_min, sky_dec_max = {0}, {1}'.format(sky_dec_min, sky_dec_max))
            logging.debug('sky_ra_min, sky_ra_max = {0}, {1}'.format(sky_ra_min, sky_ra_max))

        # Create sky map (i.e. bin events)
        # NOTE: In histogram2d the first axes is the vertical (y, DEC) the 2nd the horizontal axes (x, RA)
        sky = np.histogram2d(x=photbdata.field('DEC     '), y=photbdata.field('RA      '),
                             bins=[skymap_nbins, skymap_nbins],
                             range=[[sky_dec_min, sky_dec_max], [sky_ra_min, sky_ra_max]])

        if firstloop :
            # Just used to have the x-min/max, y-min/max saved
            H, xedges, yedges = sky
            # NOTE: The zero point of the histogram 2d is at the lower left corner while
            #       the pyplot routine imshow takes [0,0] at the upper left corner (i.e.
            #       we have to invert the 1st axis before plotting, see below).
            #       Also, imshow uses the traditional 1st axes = x = RA, 2nd axes = y = DEC
            #       notation for the extend keyword
            # extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
            extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
            sky_hist = sky[0]

            sky_ex_reg_map = get_exclusion_region_map(sky_hist,
                                                         (sky_ra_min, sky_ra_max),
                                                         (sky_dec_min, sky_dec_max),
                                                         sky_ex_reg)
        else :
            sky_hist += sky[0]

        # Calculate camera acceptance
        dec_a = np.linspace(sky_dec_min, sky_dec_max, skymap_nbins + 1)
        ra_a = np.linspace(sky_ra_min, sky_ra_max, skymap_nbins + 1)
        xx, yy = np.meshgrid((ra_a[:-1] + ra_a[1:]) / 2. - pntra, (dec_a[:-1] + dec_a[1:]) / 2. - pntdec)
        rr = np.sqrt(xx ** 2. + yy ** 2.)
        p1 = fitter.results[0]
        acc = fitter.fitfunc(p1, rr) / fitter.fitfunc(p1, .01)
        m = rr > 4.
        acc[m] = fitter.fitfunc(p1, 4.) / fitter.fitfunc(p1, .01)
        if not fov_acceptance :
            logging.debug('Do _not_ apply FoV acceptance correction')
            acc = acc * 0. + 1.

        # DEBUG plot
        # plt.imshow(acc[::-1], extent=extent, interpolation='nearest')
        # plt.colorbar()
        # plt.title('acc_bg_overs')
        # plt.show()
        
        if firstloop :
            acc_hist = acc * exposure_run  # acc[0] before
        else :
            acc_hist += acc * exposure_run  # acc[0] before

        if template_background :
            # Create hadron event like map for template background
            tpl_had = np.histogram2d(x=hadtbdata.field('DEC     '), y=hadtbdata.field('RA      '),
                                     bins=[skymap_nbins, skymap_nbins],
                                     # weights=1./accept,
                                     range=[[sky_dec_min, sky_dec_max], [sky_ra_min, sky_ra_max]])
            if firstloop :
                tpl_had_hist = tpl_had[0]
            else :
                tpl_had_hist += tpl_had[0]

            # Create acceptance map for template background
            tpl_acc = np.histogram2d(x=hadtbdata.field('DEC     '), y=hadtbdata.field('RA      '),
                                     bins=[skymap_nbins, skymap_nbins],
                                     weights=tpl_acc,
                                     range=[[sky_dec_min, sky_dec_max], [sky_ra_min, sky_ra_max]])
            if firstloop :
                tpl_acc_hist = tpl_acc[0]
            else :
                tpl_acc_hist += tpl_acc[0]

        # Close fits file
        hdulist.close()
        if tpl_hdulist :
            tpl_hdulist.close()

        # Clean up memory
        newtable = None
        import gc
        gc.collect()

        firstloop = False

    #---------------------------------------------------------------------------
    # Calculate final skymaps

    logging.info('Processing final sky maps')

    # Calculate oversampled skymap, ring background, excess, and significances
    sc = get_sky_mask_circle(r_overs, skymap_bin_size)
    sr = get_sky_mask_ring(ring_bg_r_min, ring_bg_r_max, skymap_bin_size)
    acc_hist /= exposure

    logging.info('Calculating oversampled event map ..')
    sky_overs, sky_overs_alpha = oversample_sky_map(sky_hist, sc)

    logging.info('Calculating oversampled ring background map ..')
    sky_bg_ring, sky_bg_ring_alpha = oversample_sky_map(sky_hist, sr, sky_ex_reg_map)

    logging.info('Calculating oversampled event acceptance map ..')
    acc_overs, acc_overs_alpha = oversample_sky_map(acc_hist, sc)

    logging.info('Calculating oversampled ring background acceptance map ..')
    acc_bg_overs, acc_bg_overs_alpha = oversample_sky_map(acc_hist, sr, sky_ex_reg_map)


    rng_alpha = acc_hist / acc_bg_overs  # camera acceptance
    rng_exc = sky_hist - sky_bg_ring * rng_alpha
    rng_sig = significance_on_off(sky_hist, sky_bg_ring, rng_alpha)

    rng_alpha_overs = acc_overs / acc_bg_overs  # camera acceptance
    rng_exc_overs = sky_overs - sky_bg_ring * rng_alpha_overs
    rng_sig_overs = significance_on_off(sky_overs, sky_bg_ring, rng_alpha_overs)

    tpl_had_overs, tpl_sig_overs, tpl_exc_overs, tpl_alpha_overs = None, None, None, None

    if template_background :

        logging.info('Calculating oversampled template background map ..')
        tpl_had_overs, tpl_had_overs_alpha = oversample_sky_map(tpl_had_hist, sc)

        logging.info('Calculating oversampled template acceptance map ..')
        tpl_acc_overs, tpl_acc_overs_alpha = oversample_sky_map(tpl_acc_hist, sc)
        
        tpl_exc_overs = sky_overs - tpl_acc_overs
        tpl_alpha_overs = tpl_acc_overs / tpl_had_overs
        tpl_sig_overs = significance_on_off(sky_overs, tpl_had_overs, tpl_alpha_overs)

    #---------------------------------------------------------------------------
    # Write results to file

    if write_output :

        logging.info('Writing result to file ..')

        rarange, decrange = (sky_ra_min, sky_ra_max), (sky_dec_min, sky_dec_max)

        outfile_base_name = 'skymap_ring'
        outfile_data = {
            '_ev.fits': sky_hist,
            '_ac.fits': acc_hist,
            '_ev_overs.fits': sky_overs,
            '_bg_overs.fits': sky_bg_ring,
            '_si_overs.fits': rng_sig_overs,
            '_ex_overs.fits': rng_exc_overs,
            '_al_overs.fits': rng_alpha_overs,
            '_si.fits': rng_sig,
            '_ex.fits': rng_exc,
            '_al.fits': rng_alpha
            }
        outfile_base_name = unique_base_file_name(outfile_base_name, outfile_data.keys())

        for ext, data in outfile_data.iteritems() :
            image_to_primaryhdu(data, rarange, decrange, author='PyFACT pfmap',
                                 object_=object_, telescope=telescope).writeto(outfile_base_name + ext)

        if template_background :
            outfile_base_name = 'skymap_template'
            outfile_data = {
                '_bg.fits': tpl_had_hist,
                '_ac.fits': tpl_acc_hist,
                '_bg_overs.fits': tpl_had_overs,
                '_si_overs.fits': tpl_sig_overs,
                '_ex_overs.fits': tpl_exc_overs,
                '_al_overs.fits': tpl_alpha_overs,
                # '_si.fits': rng_sig,
                # '_ex.fits': rng_exc,
                # '_al.fits': rng_alpha
                }
            outfile_base_name = unique_base_file_name(outfile_base_name, outfile_data.keys())

            for ext, data in outfile_data.iteritems() :
                image_to_primaryhdu(data, rarange, decrange, author='PyFACT pfmap',
                                     object_=object_, telescope=telescope).writeto(outfile_base_name + ext)

        logging.info('The output files can be found in {0}'.format(os.getcwd()))

    #---------------------------------------------------------------------------
    # Plot results

    try:
        import matplotlib
        has_matplotlib = True
    except:
        has_matplotlib = False

    if has_matplotlib and do_graphical_output :

        logging.info('Plotting results (matplotlib v{0})'.format(matplotlib.__version__))

        plot_skymaps(sky_overs, rng_exc_overs, rng_sig_overs, sky_bg_ring, rng_alpha_overs, 'Ring BG overs.',
                     sky_ra_min, sky_ra_max, sky_dec_min, sky_dec_max, objcosdec, r_overs, extent,
                     skycenra, skycendec, ring_bg_r_min, ring_bg_r_max, sign_hist_r_max=2.)

        if template_background :
            plot_skymaps(sky_overs, tpl_exc_overs, tpl_sig_overs, tpl_had_overs, tpl_alpha_overs,
                         'Template BG overs.',
                         sky_ra_min, sky_ra_max, sky_dec_min, sky_dec_max, objcosdec, r_overs, extent,
                         skycenra, skycendec, sign_hist_r_max=2.)

    plt.show()


def plot_skymaps(event_map, excess_map, sign_map, bg_map, alpha_map, titlestr,
                 sky_ra_min, sky_ra_max, sky_dec_min, sky_dec_max, objcosdec,
                 r_overs, extent, skycenra, skycendec,
                 ring_bg_r_min=None, ring_bg_r_max=None, sign_hist_r_max=2.) :
    """Plot sky map function (using matplotlib only).
    
    TODO: document
    
    Parameters
    ----------
    TODO
    
    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    def set_title_and_axlabel(label) :
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.title(label, fontsize='medium')

    plt_r_ra = sky_ra_min + .08 * (sky_ra_max - sky_ra_min) + r_overs / objcosdec
    plt_r_dec = sky_dec_min + .08 * (sky_dec_max - sky_dec_min) + r_overs / objcosdec

    cir_overs = Circle((plt_r_ra, plt_r_dec), radius=r_overs / objcosdec,
                       fill=True, edgecolor='1.', facecolor='1.', alpha=.6)

    gauss_func = lambda p, x: p[0] * np.exp(-(x - p[1]) ** 2. / 2. / p[2] ** 2.)

    plt.figure(figsize=(13, 7))
    plt.subplots_adjust(wspace=.4, left=.07, right=.96, hspace=.25, top=.93)

    ax = plt.subplot(231) 
    plt.imshow(event_map[::-1], extent=extent, interpolation='nearest')    
    cb = plt.colorbar()
    cb.set_label('Events')
    set_title_and_axlabel('Events')
    ax.add_patch(cir_overs)

    ax = plt.subplot(232) 
    plt.imshow(excess_map[::-1], extent=extent, interpolation='nearest')
    cb = plt.colorbar()
    cb.set_label('Excess events')
    set_title_and_axlabel(titlestr + ' - Excess')

    ax = plt.subplot(233) 
    plt.imshow(sign_map[::-1], extent=extent, interpolation='nearest')
    cb = plt.colorbar()
    cb.set_label('Significance')
    set_title_and_axlabel(titlestr + '- Significance')

    ax = plt.subplot(234) 
    plt.imshow(bg_map[::-1], extent=extent, interpolation='nearest')
    cb = plt.colorbar()
    cb.set_label('Background events')
    set_title_and_axlabel(titlestr + ' - Background')

    if ring_bg_r_min and ring_bg_r_max :
        plt_r_ra = sky_ra_min + .03 * (sky_ra_max - sky_ra_min) + ring_bg_r_max / objcosdec
        plt_r_dec = sky_dec_min + .03 * (sky_dec_max - sky_dec_min) + ring_bg_r_max / objcosdec

        # Plot ring background region as two circles
        circle = Circle((plt_r_ra, plt_r_dec), radius=ring_bg_r_max / objcosdec,
                        fill=False, edgecolor='1.', facecolor='0.', linestyle='solid', linewidth=1.)
        ax.add_patch(circle)
        circle = Circle((plt_r_ra, plt_r_dec), radius=ring_bg_r_min / objcosdec,
                        fill=False, edgecolor='1.', facecolor='0.', linestyle='solid', linewidth=1.)
        ax.add_patch(circle)

    circle = Circle((plt_r_ra, plt_r_dec), radius=r_overs / objcosdec,
                    fill=True, edgecolor='1.', facecolor='1.', alpha=.6)
    ax.add_patch(circle)

    ax = plt.subplot(235) 
    plt.imshow(alpha_map[::-1], extent=extent, interpolation='nearest')
    cb = plt.colorbar()
    cb.set_label('Alpha')
    set_title_and_axlabel(titlestr + ' - Alpha')

    ax = plt.subplot(236)
    sky_ex_reg_map = get_exclusion_region_map(event_map, (sky_ra_min, sky_ra_max), (sky_dec_min, sky_dec_max),
                                                 [SkyCircle(SkyCoord(skycenra, skycendec), sign_hist_r_max)])
    n, bins, patches = plt.hist(sign_map[sky_ex_reg_map == 0.].flatten(), bins=100, range=(-8., 8.),
                                histtype='stepfilled', color='SkyBlue', log=True)

    x = np.linspace(-5., 8., 100)
    plt.plot(x, gauss_func([float(n.max()), 0., 1.], x), label='Gauss ($\sigma=1.$, mean=0.)')

    plt.xlabel('Significance R < {0}'.format(sign_hist_r_max))
    plt.title(titlestr, fontsize='medium')

    plt.ylim(1., n.max() * 5.)
    plt.legend(loc='upper left', prop={'size': 'small'})


def unique_base_file_name(name, extension=None) :
    """
    Checks if a given file already exists. If yes, creates a new unique filename.

    Parameters
    ----------
    name : str
        Base file name.
    extension : str/array, optional
        File extension(s).

    Returns
    -------
    filename : str
        Unique filename.
    """
    import os
    import datetime
    def filename_exists(name, extension) :
        exists = False
        if extension :
            try :
                len(extension)
                for ext in extension :
                    if os.path.exists(name + ext) :
                        exists = True
                        break
            except :
                if os.path.exists(name + extension) :
                    exists = True
                else :
                    if os.path.exists(name) :
                        exists = True
        return exists

    if filename_exists(name, extension) :
        name += datetime.datetime.now().strftime('_%Y%m%d-%H%M%S')
        if filename_exists(name, extension) :
            import random
            name += '_' + str(int(random.random() * 10000))
    return name


def image_to_primaryhdu(map, rarange, decrange, telescope='DUMMY', object_='DUMMY', author='DUMMY') :
    """Converts a 2d numpy array into a FITS primary HDU (image).

    Parameters
    ----------
    map : 2d array
        Skymap.
    rarange : array/tupel
        Tupel/Array with two entries giving the RA range of the map i.e. (ramin, ramax).
    decrange : array/tupel
        Tupel/Array with two entries giving the DEC range of the map i.e (decmin, decmax).

    Returns
    -------
    hdu : pyfits.PrimaryHDU
      FITS primary HDU containing the skymap.
    """
    return image_to_hdu(map, rarange, decrange, primary=True, telescope=telescope, object_=object_, author=author)


def image_to_hdu(image, rarange, decrange, primary=False,
                 telescope='DUMMY', object='DUMMY', author='DUMMY') :
    """Converts a 2d numpy array into a FITS primary HDU (image).

    Parameters
    ----------
    map : 2d array
        Skymap.
    rarange : array/tupel
        Tupel/Array with two entries giving the RA range of the map i.e. (ramin, ramax).
    decrange : array/tupel
        Tupel/Array with two entries giving the DEC range of the map i.e (decmin, decmax).

    Returns
    -------
    hdu : `astropy.io.fits.PrimaryHDU`
        FITS primary HDU containing the skymap.
    """
    from astropy.io import fits
    decnbins, ranbins = map.shape

    decstep = (decrange[1] - decrange[0]) / float(decnbins)
    rastep = (rarange[1] - rarange[0]) / float(ranbins)

    hdu = None
    if primary :
        hdu = fits.PrimaryHDU(image)
    else :
        hdu = fits.ImageHDU(image)
    hdr = hdu.header

    # Image definition
    hdr['CTYPE1'] = 'RA---CAR'
    hdr['CTYPE2'] = 'DEC--CAR'
    hdr['CUNIT1'] = 'deg'
    hdr['CUNIT2'] = 'deg'
    hdr['CRVAL1'] = rarange[0]
    hdr['CRVAL2'] = 0. # Must be zero for the lines to be rectilinear according to Calabretta (2002)
    hdr['CRPIX1'] = .5
    hdr['CRPIX2'] = - decrange[0] / decstep + .5 # Pixel outside of the image at DEC = 0.
    hdr['CDELT1'] = rastep
    hdr['CDELT2'] = decstep
    hdr['RADESYS'] = 'FK5'
    hdr['BUNIT'] = 'count'

    # Extra data
    hdr['TELESCOP'] = telescope
    hdr['OBJECT'] = object
    hdr['AUTHOR'] = author

    return hdu
