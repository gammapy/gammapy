# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""PyFACT compatibility module.

This module contains a few functions and classes from PyFACT:

- http://pyfact.readthedocs.org/
- https://github.com/gammapy/gammapy/pull/68

TODO: This is a short-term solution until we find time to refactor
this functionality into gammapy.
"""
from __future__ import print_function, division
import gc
import os
import logging
import datetime
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from astropy.time import Time, TimeDelta
from ..stats import significance_on_off
from ..irf import (np_to_rmf,
                   EnergyDispersion,
                   EffectiveAreaTable,
                   )
from ..spectrum import np_to_pha
from ..version import version


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
           'plot_th1',
           'fit_th1',
           'root_1dhist_to_array',
           'root_2dhist_to_array',
           'root_axis_to_array',
           'root_th1_to_fitstable',
           'cta_irf_root_to_fits',
           ]


class ChisquareFitter(object):
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

    def __init__(self, fitfunc):
        self.fitfunc = fitfunc
        self.results = None

    def fit_data(self, p0, x, y, y_err):
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
        if self.results[4]:
            self.chi_arr = self.chi_func(self.results[0], x, y, y_err)
            self.chi2 = np.sum(np.power(self.chi_arr, 2.))
            self.dof = len(x) - len(p0)
            # self.prob = scipy.special.gammainc(.5 * self.dof, .5 * self.chi2) / scipy.special.gamma(.5 * self.dof)
            self.prob = 1. - gammainc(.5 * self.dof, .5 * self.chi2)
        return self.results[4]

    def chi_func(self, p, x, y, err):
        """Returns Chi"""
        return (self.fitfunc(p, x) - y) / err  # Distance to the target function

    def print_results(self):
        """Prints out results to the command line using the logging module."""
        if self.results == None:
            logging.warning('No fit results to report since no fit has been performed yet')
            return
        if self.results[4] < 5:
            logging.info('Fit was successful!')
        else:
            logging.warning('Fitting failed!')
            logging.warning('Message: {0}'.format(self.results[3]))
        logging.info('Chi^2  : {0:f}'.format(self.chi2))
        logging.info('d.o.f. : {0:d}'.format(self.dof))
        logging.info('Prob.  : {0:.4e}'.format(self.prob))
        for i, v in enumerate(self.results[0]):
            if self.results[1] != None:
                logging.info('P{0}     : {1:.4e} +/- {2:.4e}'.format(i, v,
                                                                     np.sqrt(self.results[1][i][i])))
            else:
                logging.info('P{0}     : {1:.4e}'.format(i, v))


class SkyCircle:
    """A circle on the sky."""

    def __init__(self, c, r):
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

    def contains(self, c):
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

    def intersects(self, sc):
        """
        Checks if two sky circles overlap.

        Parameters
        ----------
        sc : SkyCircle
        """
        return self.c.dist(sc.c) <= self.r + sc.r


class SkyCoord:
    """Sky coordinate in RA and Dec. All units should be degree."""

    def __init__(self, ra, dec):
        """
        Sky coordinate in RA and Dec. All units should be degree.

        In the current implementation it should also work with arrays,
        though one has to be careful in dist.

        Parameters
        ----------
        ra : float/array
            Right ascension of the coordinate.
        dec : float/array
            Declination of the coordinate.
        """
        self.ra, self.dec = ra, dec

    def dist(self, c):
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


def circle_circle_intersection_array(R, r, d):
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
    if mask1.any():
        result[mask1] = np.pi * r[mask1] ** 2.
    mask2 = r >= d + R
    if mask2.any():
        result[mask2] = np.pi * R[mask2] ** 2.
    mask = (R + r > d) * np.invert(mask1) * np.invert(mask2)
    if mask.any():
        r, R, d = r[mask], R[mask], d[mask]
        result[mask] = (r ** 2.) * np.arccos(X(R, r, d)) + (R ** 2.) * np.arccos(Y(R, r, d)) - .5 * np.sqrt(Z(R, r, d));
    return result


def circle_circle_intersection_float(R, r, d):
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

    if R >= d + r:
        return np.pi * r ** 2.
    elif r >= d + R:
        return np.pi * R ** 2.
    elif R + r > d:
        return (r ** 2.) * np.arccos(X(R, r, d)) + (R ** 2.) * np.arccos(Y(R, r, d)) - .5 * np.sqrt(Z(R, r, d))
    else:
        return 0.


def sim_evlist(flux=.1,
               obstime=.5,
               arf=None,
               rmf=None,
               extra=None,
               output_filename_base=None,
               write_pha=False,
               do_graphical_output=True,
               loglevel='INFO'):
    """Simulate IACT eventlist using an ARF file.

    TODO: describe.

    Paramters
    ---------
    TODO

    Returns
    -------
    TODO
    """
    from scipy.interpolate import UnivariateSpline
    from scipy.integrate import quad
    import matplotlib.pyplot as plt
    #---------------------------------------------------------------------------

    logging.info('Exposure: {0} h'.format(obstime))
    obstime *= 3600.  # observation time in seconds

    obj_ra, obj_dec = 0., .5
    pnt_ra, pnt_dec = 0., 0.
    t_min = 24600.

    objcosdec = np.cos(obj_dec * np.pi / 180.)

    #---------------------------------------------------------------------------
    # Read ARF, RMF, and extra file

    logging.info('ARF: {0}'.format(arf))
    arf_obj = EffectiveAreaTable.read(arf)
    ea = arf_obj.effective_area.value
    ea_erange = np.hstack(arf_obj.energy_lo.value, arf_obj.energy_hi.value[-1])
    del arf_obj

    # DEBUG
    # ea /= irf_data[:,4]

    if rmf:
        logging.info('RMF: {0}'.format(rmf))
        edisp = EnergyDispersion.read(rmf)
    else:
        edisp = None

    if extra:
        logging.info('Extra file: {0}'.format(extra))
        extraf = fits.open(extra)
        logging.info('Using effective area with 80% containment from extra file')
        ea = extraf['EA80'].data.field('VAL') / .8  # 100% effective area
        ea_erange = 10. ** np.hstack([extraf['EA80'].data.field('BIN_LO'), extraf['EA80'].data.field('BIN_HI')[-1]])
    else:
        logging.info('Assuming energy independent 80% cut efficiency for ARF file.')
        ea /= .80

    #---------------------------------------------------------------------------
    # Signal

    # log_e_cen = (irf_data[:,0] + irf_data[:,1]) / 2.
    e_cen = 10. ** ((np.log10(ea_erange[1:] * ea_erange[:-1])) / 2.)

    ea_loge_step_mean = np.log10(ea_erange[1:] / ea_erange[:-1]).mean().round(4)
    logging.debug('ea_loge_step_mean = {0}'.format(ea_loge_step_mean))

    # Resample effective area to increase precision
    if ea_loge_step_mean > .1:
        elog10step = .05
        logging.info('Resampling effective area in log10(EA) vs log10(E) (elog10step = {0})'.format(elog10step))
        ea_spl = UnivariateSpline(e_cen, np.log10(ea), s=0, k=1)
        e_cen = 10. ** np.arange(np.log10(e_cen[0]), np.log10(e_cen[-1]), step=elog10step)
        ea = 10. ** ea_spl(e_cen)

    # DEBUG plot
    # plt.loglog(e_cen_s, ea_s, )
    # plt.loglog(e_cen, ea, '+')
    # plt.show()

    func_pl = lambda x, p: p[0] * x ** (-p[1])
    flux_f = lambda x : func_pl(x, (3.45E-11 * flux, 2.63))

    f_test = UnivariateSpline(e_cen, ea * flux_f(e_cen) * 1E4, s=0, k=1)  # m^2 > cm^2

    if rmf:
        log_e_steps = np.log10(edisp.energy_bounds('true'))
    else:
        log_e_steps = np.log10(ea_erange)

    # Calculate event numbers for the RMF bins
    def get_int_rate(emin, emax):
        if emin < e_cen[0] or emax > e_cen[-1]:
            return 0.
        else:
            return f_test.integral(emin, emax)
    int_rate = np.array([get_int_rate(10. ** el, 10. ** eh) for (el, eh) in zip(log_e_steps[:-1], log_e_steps[1:])])
    # Sanity
    int_rate[int_rate < 0.] = 0.

    # DEBUG
    # int_rate_s = int_rate

    if rmf:
        logging.debug('Photon rate before RM = {0}'.format(np.sum(int_rate)))
        # Apply energy distribution matrix
        int_rate = edisp.apply(int_rate)
        logging.debug('Photon rate after RM = {0}'.format(np.sum(int_rate)))

    # DEBUG plots
    # plt.figure(1)
    # plt.semilogy(log_e_steps[:-1], int_rate_s, 'o', label='PRE RMF')
    # #plt.plot(log_e_steps[:-1], int_rate_s, 'o', label='PRE RMF')
    # if rmf :
    #    plt.semilogy(np.log10(rm_ebounds[:-1]), int_rate, '+', label='POST RMF')
    # plt.ylim(1E-6,1.)
    # plt.legend()
    # plt.show()
    # #sys.exit(0)

    # Calculate cumulative event numbers
    int_all = np.sum(int_rate)
    int_rate = np.cumsum(int_rate)

    if rmf:
        # log_e_steps = (np.log10(rm_ebounds[1:]) + np.log10(rm_ebounds[:-1])) / 2.
        log_e_steps = np.log10(edisp.energy_bounds('true'))

    # Filter out low and high values to avoid spline problems at the edges
    istart = np.sum(int_rate == 0.) - 1
    if istart < 0:
        istart = 0
    istop = np.sum(int_rate / int_all > 1. - 1e-4)  # This value dictates the dynamic range at the high energy end

    logging.debug('istart = {0}, istop = {1}'.format(istart, istop))

    # DEBUG plots
    # plt.plot(int_rate[istart:-istop] / int_all, log_e_steps[istart + 1:-istop], '+')
    # plt.show()

    # DEBUG plots
    # plt.hist(int_rate[istart:-istop] / int_all)
    # plt.show()

    ev_gen_f = UnivariateSpline(int_rate[istart:-istop] / int_all,
                                log_e_steps[istart + 1:-istop],
                                s=0, k=1)

    # # DEBUG plot
    # plt.plot(np.linspace(0.,1.,100), ev_gen_f(np.linspace(0.,1.,100)), 'o')
    # plt.show()

    # Test event generator function
    n_a_t = 100.
    a_t = ev_gen_f(np.linspace(0., 1., n_a_t))
    logging.debug('Test ev_gen_f, (v = 0 / #v) = {0}, (v = NaN / #v) = {1}'
                  ''.format(np.sum(a_t == 0.) / n_a_t, np.sum(np.isnan(a_t)) / n_a_t))

    if (np.sum(a_t == 0.) / n_a_t > 0.05) or (np.sum(np.isnan(a_t)) / n_a_t > .05):
        raise Exception('Could not generate event generator function for photons. '
                        'Try to decrease the upper cut-off value in the code.')

    # Calculate total number of photon events
    n_events = int_all * obstime

    logging.debug('Number of photons : {0}'.format(n_events))

    # Generate energy event list
    evlist_e = ev_gen_f(np.random.rand(n_events))

    # Sanity
    logging.debug('Number of photons with E = NaN : {0}'.format(np.sum(np.isnan(evlist_e))))
    evlist_e[np.isnan(evlist_e)] = 0.

    # # DEBUG plot
    # plt.figure(1)
    # plt.hist(evlist_e, range=[-2.,2.], bins=20)
    # #plt.show()
    # sys.exit(0)

    #------------------------------------------------------
    # Apply PSF

    # Broken power law fit function, normalized at break energy
    bpl = lambda p, x : np.where(x < p[0], p[1] * (x / p[0]) ** -p[2], p[1] * (x / p[0]) ** -p[3])
    evlist_psf = None
    if extra:
        d = extraf['ANGRES68'].data
        g = UnivariateSpline((d.field('BIN_LO') + d.field('BIN_HI')) / 2., d.field('VAL'), s=0, k=1)
        evlist_psf = g(evlist_e)
    else:
        psf_p1 = [1.1, 5.5E-2, .42, .19]  # Fit from SubarrayE_IFAE_50hours_20101102
        evlist_psf = bpl(psf_p1, 10. ** evlist_e)
        logging.warning('Using dummy PSF extracted from SubarrayE_IFAE_50hours_20101102')

    evlist_dec = obj_dec + np.random.randn(n_events) * evlist_psf
    evlist_ra = obj_ra + np.random.randn(n_events) * evlist_psf / objcosdec

    evlist_t = t_min + obstime * np.random.rand(n_events) / 86400.

    #---------------------------------------------------------------------------
    # Background

    # plt.figure(1)

    p_rate_area, log_e_cen = None, None
    if extra:
        d = extraf['BGRATED'].data
        p_rate_area = d.field('VAL')
        log_e_cen = (d.field('BIN_LO') + d.field('BIN_HI')) / 2
        # g = scipy.interpolate.UnivariateSpline((d.field('BIN_LO') + d.field('BIN_HI')) / 2., d.field('VAL'), s=0, k=1)
    else:
        logging.warning('Using dummy background rate extracted from SubarrayE_IFAE_50hours_20101102')
        bgrate_p1 = [9., 5.E-4, 1.44, .49]  # Fit from SubarrayE_IFAE_50hours_20101102
        log_e_cen = np.linspace(-1.5, 2., 35.)
        p_rate_area = bpl(bgrate_p1, 10. ** log_e_cen)
        p_rate_area[log_e_cen < -1.] = .4

    # DEBUG plot
    plt.semilogy(log_e_cen, p_rate_area)
    plt.show()

    p_rate_total = np.sum(p_rate_area)

    ev_gen_f = UnivariateSpline(np.cumsum(p_rate_area) / np.sum(p_rate_area),
                                log_e_cen, s=0, k=1)

    cam_acc = lambda p, x: p[0] * x ** 0. * (1. + (x / p[1]) ** p[2]) ** ((0. + p[3]) / p[2])
    cam_acc_par = (1., 1.7, 6., -5.5)

    r_steps = np.linspace(0.001, 4., 150)
    int_cam_acc = np.zeros(150)
    for i, r in enumerate(r_steps):
        int_cam_acc[i] = quad(lambda x: cam_acc(cam_acc_par, x) * x * 2. * np.pi, 0., r)[0]

    n_events_bg = int(p_rate_total * obstime * int_cam_acc[-1])

    logging.debug('Number of protons : {0}'.format(n_events_bg))

    tplt_multi = 5
    evlist_bg_e = ev_gen_f(np.random.rand(n_events_bg * (tplt_multi + 1)))

    temp42 = quad(lambda x: cam_acc(cam_acc_par, x) * 2. * x * np.pi, 0., 4.)[0]
    ev_gen_f2 = UnivariateSpline(int_cam_acc / temp42,
                                 r_steps, s=0, k=1)

    evlist_bg_r = ev_gen_f2(np.random.rand(n_events_bg * (tplt_multi + 1)))

    r_max = 4.
    # evlist_bg_r = np.sqrt(np.random.rand(n_events_bg * (tplt_multi + 1))) * r_max
    rnd = np.random.rand(n_events_bg * (1 + tplt_multi))
    evlist_bg_rx = np.sqrt(rnd) * evlist_bg_r * np.where(np.random.randint(2, size=(n_events_bg * (tplt_multi + 1))) == 0, -1., 1.)
    evlist_bg_ry = np.sqrt(1. - rnd) * evlist_bg_r * np.where(np.random.randint(2, size=(n_events_bg * (tplt_multi + 1))) == 0, -1., 1.)

    # evlist_bg_sky_r = np.sqrt(np.random.rand(n_events_bg * (tplt_multi + 1))) * r_max
    # evlist_bg_sky_r = ev_gen_f2(np.random.rand(n_events_bg * (tplt_multi + 1)))
    rnd = np.random.rand(n_events_bg * (tplt_multi + 1))
    evlist_bg_ra = np.sin(2. * np.pi * rnd) * evlist_bg_r / objcosdec
    evlist_bg_dec = np.cos(2. * np.pi * rnd) * evlist_bg_r

    # plt.hist(evlist_bg_rx ** 2. + evlist_bg_ry**2., bins=50)

    # print float(n_events_bg * (tplt_multi + 1)) / np.sum(p_rate_area) / 86400.
    evlist_bg_t = t_min + obstime * np.random.rand(n_events_bg * (tplt_multi + 1)) / 86400.

    #---------------------------------------------------------------------------
    # Plots & debug

    plt.figure(3)

    objra, objdec = 0., 0.
    H, xedges, yedges = np.histogram2d(
        np.append(evlist_bg_dec, evlist_dec),
        np.append(evlist_bg_ra, evlist_ra),
        bins=[100, 100],
        range=[[objra - 3., objra + 3.], [objdec - 3., objdec + 3.]]
        )
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    plt.imshow(H, extent=extent, interpolation='nearest')
    cb = plt.colorbar()
    cb.set_label('Number of events')

    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')

    test_r = np.sqrt(evlist_bg_ra ** 2. + evlist_bg_dec ** 2.)
    logging.debug('Number of BG events in a circle of area 1 deg^2 = {0}'.format(np.sum(test_r[0:n_events_bg] < np.sqrt(1. / np.pi))))
    logging.debug('Expected number of BG event per area 1 deg^2 = {0}'.format(p_rate_total * obstime))

    obj_r = np.sqrt(((obj_ra - evlist_ra) / objcosdec) ** 2. + (obj_dec - evlist_dec) ** 2.)

    thetamax_on, thetamax_off = .1, .22
    non = np.sum(obj_r < thetamax_on) + np.sum(test_r[0:n_events_bg] < thetamax_on)
    noff = np.sum(test_r[0:n_events_bg] < thetamax_off)
    alpha = thetamax_on ** 2. / thetamax_off ** 2.

    logging.info('N_ON = {0}, N_OFF = {1}, ALPHA = {2}, SIGN = {3}'.format(
        non, noff, alpha, significance_on_off(non, noff, alpha)))

    plt.figure(2)
    plt.hist(obj_r ** 2., bins=30)


    dbase = Time('2011-01-01 00:00:00', scale='utc')
    dstart = Time('2011-01-01 00:00:00', scale='utc')
    dstop = dstart + TimeDelta(obstime, format='sec')

    #---------------------------------------------------------------------------
    # Output to file

    if output_filename_base:
        logging.info('Writing eventlist to file {0}.eventlist.fits'.format(output_filename_base))

        newtable = np_to_evt(evlist_t, evlist_bg_t,
                             evlist_ra, evlist_bg_ra,
                             evlist_dec, evlist_bg_dec,
                             evlist_bg_rx, evlist_bg_ry,
                             evlist_e, evlist_bg_e,
                             pnt_ra, pnt_dec,
                             tplt_multi, obstime,
                             dstart=dstart, dstop=dstop, dbase=dbase)
        # Write eventlist to file
        newtable.write('{0}.eventlist.fits'.format(output_filename_base))

        if write_pha:
            logging.info('Writing PHA to file {0}.pha.fits'.format(output_filename_base))
            # Prepare data
            dat, t = np.histogram(10. ** evlist_e, bins=edisp.energy_bounds('true'))
            dat = np.array(dat, dtype=float)
            dat_err = np.sqrt(dat)
            chan = np.arange(len(dat))
            # Data to PHA
            tbhdu = np_to_pha(counts=dat, stat_err=dat_err, channel=chan, exposure=obstime,
                              obj_ra=obj_ra, obj_dec=obj_dec,
                              quality=np.where((dat == 0), 1, 0),
                              dstart=dstart, dstop=dstop, dbase=dbase, creator='pfsim',
                              telescope='CTASIM')
            tbhdu.header.update('ANCRFILE', os.path.basename(arf), 'Ancillary response file (ARF)')
            if rmf:
                tbhdu.header.update('RESPFILE', os.path.basename(rmf), 'Redistribution matrix file (RMF)')

            # Write PHA to file
            tbhdu.writeto('{0}.pha.fits'.format(output_filename_base))

    if do_graphical_output:
        plt.show()


def np_to_evt(evlist_time, evlist_bg_time,
              evlist_ra, evlist_bg_ra,
              evlist_dec, evlist_bg_dec,
              evlist_bg_rx, evlist_bg_ry,
              evlist_energy, evlist_bg_energy,
              pnt_ra, pnt_dec,
              tplt_multi, obstime,
              dstart, dstop, dbase=None,
              stat_err=None, quality=None, syserr=None,
              obj_ra=0.0, obj_dec=0.0,
              obj_name='DUMMY', creator='DUMMY',
              telescope='DUMMY', instrument='DUMMY', filter='NONE'):
    """Create EVT FITS table extension from numpy arrays.

    TODO: document.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    n_events, n_events_bg = len(evlist_time), len(evlist_bg_time)
    table = Table()
    table['TIME'] = Column(np.append(evlist_time, evlist_bg_time), unit='deg'),
    table['RA'] = Column(np.append(evlist_ra, evlist_bg_ra), unit='deg'),
    table['DEC'] = Column(np.append(evlist_dec, evlist_bg_dec), unit='deg'),
    # Position source at position (DETX, DETY) = (0, 0.5)
    table['DETX'] = Column(np.append(np.zeros(n_events), evlist_bg_rx), unit='deg'),
    table['DETY'] = Column(np.append(np.ones(n_events) * .5, evlist_bg_ry), unit='deg'),
    table['ENERGY'] = Column(10. ** np.append(evlist_energy, evlist_bg_energy), unit='tev'),
    hil_data = np.append(np.zeros(n_events + n_events_bg), 5. * np.ones(n_events_bg * tplt_multi))
    table['HIL_MSW'] = Column(hil_data)
    table['HIL_MSL'] = Column(hil_data)


    header = table.meta

    header['RA_OBJ'] = obj_ra, 'Target position RA [deg]'
    header['DEC_OBJ'] = obj_dec, 'Target position dec [deg]'
    header['RA_PNT'] = pnt_ra, 'Observation position RA [deg]'
    header['DEC_PNT'] = pnt_dec, 'Observation position dec [deg]'
    header['EQUINOX'] = 2000.0, 'Equinox of the object'
    header['RADECSYS'] = 'FK5', 'Co-ordinate frame used for equinox'
    header['CREATOR'] = 'gammapy v{0}'.format(version), 'Program'
    header['DATE'] = datetime.datetime.today().strftime('%Y-%m-%d'), 'FITS file creation date (yyyy-mm-dd)'
    header['TELESCOP'] = 'CTASIM', 'Instrument name'
    header['EXTNAME'] = 'EVENTS' , 'HESARC standard'
    header['DATE-OBS'] = dstart.datetime.strftime('%Y-%m-%d'), 'Obs. start date (yy-mm-dd)'
    header['TIME-OBS'] = dstart.datetime.strftime('%H:%M:%S'), 'Obs. start time (hh:mm::ss)'
    header['DATE-END'] = dstop.datetime.strftime('%Y-%m-%d'), 'Obs. stop date (yy-mm-dd)'
    header['TIME-END'] = dstop.datetime.strftime('%H:%M:%S'), 'Obs. stop time (hh:mm::ss)'
    header['TSTART'] = 0., 'Mission time of start of obs [s]'
    header['TSTOP'] = obstime, 'Mission time of end of obs [s]'
    header['MJDREFI'] = int(dstart.mjd), 'Integer part of start MJD [s] '
    header['MJDREFF'] = dstart.mjd - int(dstart.mjd), 'Fractional part of start MJD'
    header['TIMEUNIT'] = 'days' , 'Time unit of MJD'
    header['TIMESYS'] = 'TT', 'Terrestrial Time'
    header['TIMEREF'] = 'local', ''
    header['TELAPSE'] = obstime, 'Diff of start and end times'
    # Note: we are assuming deadtime = 0 here.
    header['ONTIME'] = obstime, 'Tot good time (incl deadtime)'
    header['LIVETIME'] = obstime, 'Deadtime=ONTIME/LIVETIME'
    header['DEADC'] = 1., 'Deadtime fraction'
    header['TIMEDEL'] = 1., 'Time resolution'
    header['EUNIT'] = 'TeV', 'Energy unit'
    header['EVTVER'] = 'v1.0.0', 'Event-list version number'

    return table


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
    if not nbins:
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
    if exreg:
        ex_a = np.zeros(len(r))
        t = np.ones(len(r))
        for reg in exreg:
            ex_a += (circle_circle_intersection_array(bins[1:], t * reg[0], t * reg[1])
                     - circle_circle_intersection_array(bins[:-1], t * reg[0], t * reg[1]))
        ex_a /= r_a
    # Fit the data
    fitter = None
    if fit:
        # fitfunc = lambda p, x: p[0] * x ** p[1] * (1. + (x / p[2]) ** p[3]) ** ((p[1] + p[4]) / p[3])
        if not fitfunc:
            fitfunc = lambda p, x: p[0] * x ** 0. * (1. + (x / p[1]) ** p[2]) ** ((0. + p[3]) / p[2])
            # fitfunc = lambda p, x: p[0] * x ** 0. * (1. + (x / p[1]) ** p[2]) ** ((0. + p[3]) / p[2]) + p[4] / (np.exp(p[5] * (x - p[6])) + 1.)            
        if not p0:
            p0 = [n[0] / r_a[0], 1.5, 3., -5.]  # Initial guess for the parameters
            # p0 = [.5 * n[0] / r_a[0], 1.5, 3., -5., .5 * n[0] / r_a[0], 100., .5] # Initial guess for the parameters            
        fitter = ChisquareFitter(fitfunc)
        m = (n > 0.) * (nerr > 0.) * (r_a != 0.) * ((1. - ex_a) != 0.)
        if np.sum(m) <= len(p0):
            logging.error('Could not fit camera acceptance (dof={0}, bins={1})'.format(len(p0), np.sum(m)))
        else:
            # ok, this _should_ be improved !!!
            x, y, yerr = r[m], n[m] / r_a[m] / (1. - ex_a[m]) , nerr[m] / r_a[m] / (1. - ex_a[m])
            m = np.isfinite(x) * np.isfinite(y) * np.isfinite(yerr) * (yerr != 0.)
            if np.sum(m) <= len(p0):
                logging.error('Could not fit camera acceptance (dof={0}, bins={1})'.format(len(p0), np.sum(m)))
            else:
                fitter.fit_data(p0, x[m], y[m], yerr[m])
    return (n, bins, nerr, r, r_a, ex_a, fitter)


def get_sky_mask_circle(r, bin_size):
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


def get_exclusion_region_map(map, rarange, decrange, exreg):
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
            for reg in exreg:
                if reg.contains(SkyCoord(yval, xval)):
                    sky_mask[x, y] = 0.
    return sky_mask


def oversample_sky_map(sky, mask, exmap=None):
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
    if exmap != None:
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
    from scipy.interpolate import UnivariateSpline
    import matplotlib.pyplot as plt

    #---------------------------------------------------------------------------
    # Loop over the file list, calculate quantities, & fill histograms

    # Skymap definition
    # skymap_size, skymap_bin_size = 6., 0.05
    rexdeg = .3

    # Intialize some variables
    skycenra, skycendec, pntra, pntdec = None, None, None, None
    if skymap_center:
        skycenra, skycendec = eval(skymap_center)
        logging.info('Skymap center: RA {0}, Dec {1}'.format(skycenra, skycendec))

    ring_bg_r_min, ring_bg_r_max = .3, .7
    if ring_bg_radii:
        ring_bg_r_min, ring_bg_r_max = eval(ring_bg_radii)

    if r_overs > ring_bg_r_min:
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
        try:
            f = fits.open(input_file_name)
            f.close()
            file_list = [input_file_name]
        except:
            # We are dealing with a bankfile
            logging.info('Reading files from bankfile {0}'.format(input_file_name))
            file_list = np.loadtxt(input_file_name, dtype='S', usecols=[0])
        return file_list

    file_list = get_filelist(input_file_name)

    tpl_file_list = None

    # Read in template files
    if template_background:
        tpl_file_list = get_filelist(template_background)
        if len(file_list) != len(tpl_file_list):
            logging.warning('Different number of signal and template background files. '
                            'Switching off template background analysis.')
            template_background = None

    # Main loop over input files
    for i, file_name in enumerate(file_list):

        logging.info('Processing file {0}'.format(file_name))

        def get_evl(file_name):
            # Open fits file
            hdulist = fits.open(file_name)
            # Access header of second extension
            header = hdulist['EVENTS'].header
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
            return hdulist, header, newtable.data

        hdulist, ex1hdr, tbdata = get_evl(file_name)

        # Read in eventlist for template background
        tpl_hdulist, tpl_tbdata = None, None
        if template_background:
            tpl_hdulist, tpl_hdr, tpl_tbdata = get_evl(tpl_file_list[i])

        #---------------------------------------------------------------------------
        # Intialize & GTI

        objra, objdec = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ']
        if firstloop :
            # If skymap center is not set, set it to the object position of the first run
            if skycenra == None or skycendec == None:
                skycenra, skycendec = objra, objdec
                logging.debug('Setting skymap center to skycenra, skycendec = {0}, {1}'.format(skycenra, skycendec))
            if 'TELESCOP' in ex1hdr:
                telescope = ex1hdr['TELESCOP']
                logging.debug('Setting TELESCOP to {0}'.format(telescope))
            if 'OBJECT' in ex1hdr:
                object_ = ex1hdr['OBJECT']
                logging.debug('Setting OBJECT to {0}'.format(object_))

        mgit, tpl_mgit = np.ones(len(tbdata), dtype=np.bool), None
        if template_background:
            tpl_mgit = np.ones(len(tpl_tbdata), dtype=np.bool)
        try:
            # Note: according to the eventlist format document v1.0.0 Section 10
            # "The times are expressed in the same units as in the EVENTS
            # table (seconds since mission start in terresterial time)."
            for gti in hdulist['GTI'].data:
                mgit *= (tbdata.field('TIME') >= gti[0]) * (tbdata.field('TIME') <= gti[1])
                if template_background:
                    tpl_mgit *= (tpl_tbdata.field('TIME') >= gti[0]) * (tpl_tbdata.field('TIME') <= gti[1])
        except:
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
        for sc in sky_ex_reg:
            if exmask:
                exmask *= sc.c.dist(SkyCoord(tbdata.field('RA'), tbdata.field('DEC'))) < sc.r
            else:
                exmask = sc.c.dist(SkyCoord(tbdata.field('RA'), tbdata.field('DEC'))) < sc.r
        exmask = np.invert(exmask)

        photbdata = tbdata[exmask * mgit]
        if len(photbdata) < 10 :
            logging.warning('Less then 10 events found in file {0} after exclusion region cuts'.format(file_name))

        hadtbdata = None
        if template_background:
            exmask = None
            for sc in sky_ex_reg:
                if exmask:
                    exmask *= sc.c.dist(SkyCoord(tpl_tbdata.field('RA'), tpl_tbdata.field('DEC'))) < sc.r
                else:
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
        if template_background:
            had_acc = get_cam_acc(
                hadtbdata.field('XCAMDIST'),
                exreg=[(sc.r, sc.c.dist(SkyCoord(pntra, pntdec))) for sc in sky_ex_reg],
                fit=True
                )
            had_n, had_fit = had_acc[0], had_acc[6]
            logging.debug('Camera acceptance hadrons fit probability: {0}'.format(had_fit.prob))

        # !!! All photons including the exclusion regions
        photbdata = tbdata[mgit]
        if template_background:
            hadtbdata = tpl_tbdata[tpl_mgit]
        if len(photbdata) < 10:
            logging.warning('Less then 10 events found in file {0} after GTI cut.'.format(file_name))

        tpl_acc_cor_use_interp = True
        tpl_acc_f, tpl_acc = None, None
        if template_background:
            if tpl_acc_cor_use_interp:
                tpl_acc_f = UnivariateSpline(r, n.astype(float) / had_n.astype(float), s=0, k=1)
            else:
                tpl_acc_f = lambda r: fitter.fitfunc(p1, r) / had_fit.fitfunc(had_fit.results[0], r)
            tpl_acc = tpl_acc_f(hadtbdata.field('XCAMDIST'))
            m = hadtbdata.field('XCAMDIST') > r[-1]
            tpl_acc[m] = tpl_acc_f(r[-1])
            m = hadtbdata.field('XCAMDIST') < r[0]
            tpl_acc[m] = tpl_acc_f(r[0])

        #---------------------------------------------------------------------------
        # Skymap - definitions/calculation

        # Object position in the sky
        if firstloop:
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

        if firstloop:
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
        else:
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
        if not fov_acceptance:
            logging.debug('Do _not_ apply FoV acceptance correction')
            acc = acc * 0. + 1.

        # DEBUG plot
        # plt.imshow(acc[::-1], extent=extent, interpolation='nearest')
        # plt.colorbar()
        # plt.title('acc_bg_overs')
        # plt.show()

        if firstloop:
            acc_hist = acc * exposure_run  # acc[0] before
        else:
            acc_hist += acc * exposure_run  # acc[0] before

        if template_background:
            # Create hadron event like map for template background
            tpl_had = np.histogram2d(x=hadtbdata.field('DEC     '), y=hadtbdata.field('RA      '),
                                     bins=[skymap_nbins, skymap_nbins],
                                     # weights=1./accept,
                                     range=[[sky_dec_min, sky_dec_max], [sky_ra_min, sky_ra_max]])
            if firstloop:
                tpl_had_hist = tpl_had[0]
            else:
                tpl_had_hist += tpl_had[0]

            # Create acceptance map for template background
            tpl_acc = np.histogram2d(x=hadtbdata.field('DEC     '), y=hadtbdata.field('RA      '),
                                     bins=[skymap_nbins, skymap_nbins],
                                     weights=tpl_acc,
                                     range=[[sky_dec_min, sky_dec_max], [sky_ra_min, sky_ra_max]])
            if firstloop:
                tpl_acc_hist = tpl_acc[0]
            else:
                tpl_acc_hist += tpl_acc[0]

        # Close fits file
        hdulist.close()
        if tpl_hdulist:
            tpl_hdulist.close()

        # Clean up memory
        newtable = None
        # TODO: this shouldn't be necessary any more.
        # This was an issue with old versions of pyfits.
        # Remove and test that memory is not leaked.
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

    if template_background:

        logging.info('Calculating oversampled template background map ..')
        tpl_had_overs, tpl_had_overs_alpha = oversample_sky_map(tpl_had_hist, sc)

        logging.info('Calculating oversampled template acceptance map ..')
        tpl_acc_overs, tpl_acc_overs_alpha = oversample_sky_map(tpl_acc_hist, sc)

        tpl_exc_overs = sky_overs - tpl_acc_overs
        tpl_alpha_overs = tpl_acc_overs / tpl_had_overs
        tpl_sig_overs = significance_on_off(sky_overs, tpl_had_overs, tpl_alpha_overs)

    #---------------------------------------------------------------------------
    # Write results to file

    if write_output:

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

        for ext, data in outfile_data.items():
            image_to_primaryhdu(data, rarange, decrange, author='PyFACT pfmap',
                                 object_=object_, telescope=telescope).writeto(outfile_base_name + ext)

        if template_background:
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

            for ext, data in outfile_data.items():
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

    if has_matplotlib and do_graphical_output:

        logging.info('Plotting results (matplotlib v{0})'.format(matplotlib.__version__))

        plot_skymaps(sky_overs, rng_exc_overs, rng_sig_overs, sky_bg_ring, rng_alpha_overs, 'Ring BG overs.',
                     sky_ra_min, sky_ra_max, sky_dec_min, sky_dec_max, objcosdec, r_overs, extent,
                     skycenra, skycendec, ring_bg_r_min, ring_bg_r_max, sign_hist_r_max=2.)

        if template_background:
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

    def set_title_and_axlabel(label):
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

    if ring_bg_r_min and ring_bg_r_max:
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


def unique_base_file_name(name, extension=None):
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
    def filename_exists(name, extension):
        exists = False
        if extension:
            try:
                len(extension)
                for ext in extension:
                    if os.path.exists(name + ext):
                        exists = True
                        break
            except:
                if os.path.exists(name + extension):
                    exists = True
                else:
                    if os.path.exists(name):
                        exists = True
        return exists

    if filename_exists(name, extension):
        name += datetime.datetime.now().strftime('_%Y%m%d-%H%M%S')
        if filename_exists(name, extension):
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
    hdu : astropy.io.fits.PrimaryHDU
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
    hdu : astropy.io.fits.PrimaryHDU
        FITS primary HDU containing the skymap.
    """
    decnbins, ranbins = map.shape

    decstep = (decrange[1] - decrange[0]) / float(decnbins)
    rastep = (rarange[1] - rarange[0]) / float(ranbins)

    hdu = None
    if primary:
        hdu = fits.PrimaryHDU(image)
    else:
        hdu = fits.ImageHDU(image)
    header = hdu.header

    # Image definition
    header['CTYPE1'] = 'RA---CAR'
    header['CTYPE2'] = 'DEC--CAR'
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CRVAL1'] = rarange[0]
    header['CRVAL2'] = 0.  # Must be zero for the lines to be rectilinear according to Calabretta (2002)
    header['CRPIX1'] = .5
    header['CRPIX2'] = -decrange[0] / decstep + .5  # Pixel outside of the image at DEC = 0.
    header['CDELT1'] = rastep
    header['CDELT2'] = decstep
    header['RADESYS'] = 'FK5'
    header['BUNIT'] = 'count'

    # Extra data
    header['TELESCOP'] = telescope
    header['OBJECT'] = object
    header['AUTHOR'] = author

    return hdu


def create_spectrum(input_file_names,
                    analysis_position=None,
                    analysis_radius=.125,
                    match_rmf=None,
                    datadir='',
                    write_output_files=False,
                    do_graphical_output=True,
                    loglevel='INFO'):
    """Creates spectra from VHE event lists in FITS format.

    TODO: describe

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    import matplotlib.pyplot as plt
    #---------------------------------------------------------------------------
    # Loop over the file list, calculate quantities, & fill histograms

    # Exclusion radius [this should be generalized in future versions]
    rexdeg = .3
    logging.warning('pfspec is currently using a single exclusion region for background extraction set on the analysis position (r = {0})'.format(rexdeg))
    logging.warning('This should be improved in future versions (tm).')

    # Intialize some variables
    objra, objdec, pntra, pntdec = None, None, None, None
    if analysis_position:
        objra, objdec = eval(analysis_position)
        logging.info('Analysis position: RA {0}, Dec {1}'.format(objra, objdec))
    else:
        logging.info('No analysis position given => will use object position from first file')

    logging.info('Analysis radius: {0} deg'.format(analysis_radius))

    if write_output_files:
        logging.info('The output files can be found in {0}'.format(os.getcwd()))

    theta2_hist_max, theta2_hist_nbins = .5 ** 2., 50
    theta2_on_hist, theta2_off_hist, theta2_offcor_hist = np.zeros(theta2_hist_nbins), np.zeros(theta2_hist_nbins), np.zeros(theta2_hist_nbins)
    non, noff, noffcor = 0., 0., 0.
    sky_ex_reg = None
    firstloop = True

    spec_nbins, spec_emin, spec_emax = 40, -2., 2.
    telescope, instrument = 'DUMMY', 'DUMMY'

    arf_m, arf_m_erange = None, None

    if match_rmf:
        logging.info('Matching total PHA binning to RMF file: {0}'.format(match_rmf))
        edisp = EnergyDispersion(match_rmf)
        ebounds = edisp.energy_bounds('reco')
        erange = edisp.energy_bounds('true')
        spec_nbins = (len(ebounds) - 1)
        spec_emin = np.log10(ebounds[0])
        spec_emax = np.log10(ebounds[-1])
        arf_m_erange = erange

        edisp_header = fits.getheader(match_rmf, extnr=1)
        if 'INSTRUME' in edisp_header:
            instrument = edisp_header['INSTRUME']
        if 'TELESCOP' in edisp_header:
            telescope = edisp_header['TELESCOP']

    spec_on_hist, spec_off_hist, spec_off_cor_hist = np.zeros(spec_nbins), np.zeros(spec_nbins), np.zeros(spec_nbins)
    spec_hist_ebounds = np.linspace(spec_emin, spec_emax, spec_nbins + 1)

    dstart, dstop = None, None

    exposure = 0.  # [s]

    # Read in input file, can be individual fits or bankfile
    logging.info('Opening input file(s) ..')

    # This list will hold the individual file names as strings
    file_list = None

    # Check if we are dealing with a single file or a bankfile
    # and create/read in the file list accordingly
    try:
        f = fits.open(input_file_names[0])
        f.close()
        file_list = [input_file_names]
    except:
        logging.info('Reading files from batchfile {0}'.format(input_file_names[0]))
        file_list = np.loadtxt(input_file_names[0], dtype='S')
        if len(file_list.shape) == 1:
            file_list = np.array([file_list])

    # Sanity checks on input file(s)
    if len(file_list) < 1:
        raise RuntimeError('No entries in bankfile')
    if len(file_list[0]) != 3:
        raise RuntimeError('Bankfile must have three columns (data/arf/rmf)')

    # Shortcuts for commonly used functions
    cci_f, cci_a = circle_circle_intersection_float, circle_circle_intersection_array

    for files in file_list:
        dataf, arf, rmf = datadir + files[0], datadir + files[1], datadir + files[2]
        logging.info('==== Processing file {0}'.format(dataf))

        # Open fits file
        hdulist = fits.open(dataf)

        # Print file info
        # hdulist.info()

        # Access header of second extension
        ex1hdr = hdulist[1].header

        # Print header of the first extension as ascardlist
        # print ex1hdr.ascardlist()

        # Access data of first extension
        tbdata = hdulist[1].data  # assuming the first extension is a table

        # Print table columns
        # hdulist[1].columns.info()

        #---------------------------------------------------------------------------
        # Calculate some useful quantities and add them to the table

        if firstloop:
            # If skymap center is not set, set it to the target position of the first run
            if objra == None or objdec == None :
                objra, objdec = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ']
                logging.info('Analysis position from header: RA {0}, Dec {1}'.format(objra, objdec))

        pntra, pntdec = ex1hdr['RA_PNT'], ex1hdr['DEC_PNT']
        obj_cam_dist = SkyCoord(objra, objdec).dist(SkyCoord(pntra, pntdec))

        # If no exclusion regions are given, use the object position from the first run
        if sky_ex_reg == None:
            sky_ex_reg = [SkyCircle(SkyCoord(objra, objdec), rexdeg)]

        exposure_run = ex1hdr['LIVETIME']
        exposure += exposure_run

        logging.info('RUN Start date/time : {0} {1}'.format(ex1hdr['DATE_OBS'], ex1hdr['TIME_OBS']))
        logging.info('RUN Stop date/time  : {0} {1}'.format(ex1hdr['DATE_END'], ex1hdr['TIME_END']))
        logging.info('RUN Exposure        : {0:.2f} [s]'.format(exposure_run))
        logging.info('RUN Pointing pos.   : RA {0:.4f} [deg], Dec {1:.4f} [deg]'.format(pntra, pntdec))
        logging.info('RUN Obj. cam. dist. : {0:.4f} [deg]'.format(obj_cam_dist))

        run_dstart = datetime.datetime(*[int(x) for x in (ex1hdr['DATE_OBS'].split('-') + ex1hdr['TIME_OBS'].split(':'))])
        run_dstop = datetime.datetime(*[int(x) for x in (ex1hdr['DATE_END'].split('-') + ex1hdr['TIME_END'].split(':'))])
        if firstloop:
            dstart = run_dstart
        dstop = run_dstop

        # Distance from the camera (FOV) center
        camdist = np.sqrt(tbdata.field('DETX    ') ** 2. + tbdata.field('DETY    ') ** 2.)

        # Distance from analysis position
        thetadist = SkyCoord(objra, objdec).dist(SkyCoord(tbdata.field('RA      '), tbdata.field('DEC    ')))

        # # cos(DEC)
        # cosdec = np.cos(tbdata.field('DEC     ') * np.pi / 180.)
        # cosdec_col = fits.Column(name='XCOSDEC', format='1E', array=cosdec)

        # Add new columns to the table
        coldefs_new = fits.ColDefs(
            [fits.Column(name='XCAMDIST', format='1E', unit='deg', array=camdist),
             fits.Column(name='XTHETA', format='1E', unit='deg', array=thetadist)
             ]
            )
        newtable = fits.new_table(hdulist[1].columns + coldefs_new)

        # Print new table columns
        # newtable.columns.info()

        mgit = np.ones(len(tbdata), dtype=np.bool)
        try:
            # Note: according to the eventlist format document v1.0.0 Section 10
            # "The times are expressed in the same units as in the EVENTS
            # table (seconds since mission start in terresterial time)."
            for gti in hdulist['GTI'].data :
                mgit *= (tbdata.field('TIME') >= gti[0]) * (tbdata.field('TIME') <= gti[1])
        except:
            logging.warning('File does not contain a GTI extension')

        # New table data
        tbdata = newtable.data[mgit]

        #---------------------------------------------------------------------------
        # Select signal and background events

        photbdata = tbdata

        on_run = photbdata[photbdata.field('XTHETA') < analysis_radius]
        off_run = photbdata[((photbdata.field('XCAMDIST') < obj_cam_dist + analysis_radius)
                             * (photbdata.field('XCAMDIST') > obj_cam_dist - analysis_radius)
                             * np.invert(photbdata.field('XTHETA') < rexdeg))]

        spec_on_run_hist = np.histogram(np.log10(on_run.field('ENERGY')), bins=spec_nbins, range=(spec_emin, spec_emax))[0]
        spec_on_hist += spec_on_run_hist

        non_run, noff_run = len(on_run), len(off_run)

        alpha_run = analysis_radius ** 2. / ((obj_cam_dist + analysis_radius) ** 2.
                                           - (obj_cam_dist - analysis_radius) ** 2.
                                           - cci_f(obj_cam_dist + analysis_radius, rexdeg, obj_cam_dist) / np.pi
                                           + cci_f(obj_cam_dist - analysis_radius, rexdeg, obj_cam_dist) / np.pi)

        spec_off_run_hist, ebins = np.histogram(np.log10(off_run.field('ENERGY')), bins=spec_nbins, range=(spec_emin, spec_emax))
        spec_off_hist += spec_off_run_hist
        spec_off_cor_hist += spec_off_run_hist * alpha_run

        # DEBUG plot
        # plt.plot(ebins[:-1], spec_on_hist, label='ON')
        # plt.plot(ebins[:-1], spec_off_cor_hist, label='OFF cor.')
        # plt.legend()
        # plt.show()

        def print_stats(non, noff, alpha, pre='') :
            logging.info(pre + 'N_ON = {0}, N_OFF = {1}, ALPHA = {2:.4f}'.format(non, noff, alpha))
            logging.info(pre + 'EXCESS = {0:.2f}, SIGN = {1:.2f}'.format(non - alpha * noff, significance_on_off(non, noff, alpha)))

        non += non_run
        noff += noff_run
        noffcor += alpha_run * noff_run

        print_stats(non_run, noff_run, alpha_run, 'RUN ')
        print_stats(non, noff, noffcor / noff, 'TOTAL ')

        theta2_on_hist += np.histogram(photbdata.field('XTHETA') ** 2., bins=theta2_hist_nbins, range=(0., theta2_hist_max))[0]

        theta2_off_run_hist, theta2_off_run_hist_edges = np.histogram(np.fabs((photbdata[np.invert(photbdata.field('XTHETA') < rexdeg)].field('XCAMDIST') - obj_cam_dist) ** 2.), bins=theta2_hist_nbins, range=(0., theta2_hist_max))

        theta2_off_hist += theta2_off_run_hist

        h_edges_r = np.sqrt(theta2_off_run_hist_edges)

        a_tmp = (
            cci_a(obj_cam_dist + h_edges_r,
                  np.ones(theta2_hist_nbins + 1) * rexdeg,
                  np.ones(theta2_hist_nbins + 1) * obj_cam_dist) / np.pi
            - cci_a(obj_cam_dist - h_edges_r,
                    np.ones(theta2_hist_nbins + 1) * rexdeg,
                    np.ones(theta2_hist_nbins + 1) * obj_cam_dist) / np.pi
            )

        theta2_off_hist_alpha = (
            (theta2_off_run_hist_edges[1:] - theta2_off_run_hist_edges[:-1])
            / (4. * obj_cam_dist * (h_edges_r[1:] - h_edges_r[:-1])
               - (a_tmp[1:] - a_tmp[:-1])
               )
            )
        # logging.debug('theta2_off_hist_alpha = {0}'.format( theta2_off_hist_alpha))
        theta2_offcor_hist += theta2_off_run_hist * theta2_off_hist_alpha

        # Read run ARF file
        logging.info('RUN Reading ARF from : {0}'.format(arf))
        arf_obj = EffectiveAreaTable.read(arf)
        ea = arf_obj.effective_area.value
        ea_erange = np.hstack(arf_obj.energy_lo.value, arf_obj.energy_hi.value[-1])
        del arf_obj

        # If average ARF is not matched to RMF use first ARF as template
        if firstloop and arf_m_erange is None:
            arf_m_erange = ea_erange

        if (len(ea_erange) is not len(arf_m_erange)) or (np.sum(np.fabs(ea_erange - arf_m_erange)) > 1E-5):
            logging.debug('Average ARF - ARF binning does not match RMF for file: {0}'.format(arf))
            logging.debug('Average ARF - Resampling ARF to match RMF EBOUNDS binning')
            from scipy.interpolate import UnivariateSpline
            ea_spl = UnivariateSpline(np.log10(ea_erange[:-1] * ea_erange[1:]) / 2. , np.log10(ea), s=0, k=1)
            ea = 10. ** ea_spl((np.log10(arf_m_erange[:-1] * arf_m_erange[1:]) / 2.))
        if firstloop:
            arf_m = ea * exposure_run
        else:
            arf_m += ea * exposure_run

        # # DEBUG plot
        # plt.errorbar(spec_hist_ebounds[:-1], dat, yerr=dat_err)
        # plt.title(dataf)
        # plt.show()

        # Write run wise data to PHA
        if write_output_files:

            # Create base file name for run wise output files
            run_out_basename = os.path.basename(dataf[:dataf.find('.fits')])

            # Open run RMF file
            logging.info('RUN Reading RMF from : {0}'.format(rmf))
            edisp = EnergyDispersion.read(rmf)
            ebounds = edisp.energy_bounds('reco')

            # Bin data to match EBOUNDS from RMF
            spec_on_run_hist = np.histogram(on_run.field('ENERGY'), bins=ebounds)[0]
            spec_off_run_hist = np.histogram(off_run.field('ENERGY'), bins=ebounds)[0]

            # Prepare excess data
            dat = spec_on_run_hist - alpha_run * spec_off_run_hist  # ON - alpha x OFF = Excess
            dat_err = np.sqrt(spec_on_run_hist + spec_off_run_hist * alpha_run ** 2.)
            quality = np.where(((spec_on_run_hist == 0) | (spec_off_run_hist == 0)), 2, 0)  # Set quality flags
            chan = np.arange(len(dat))

            # Signal PHA
            hdu = np_to_pha(channel=chan, counts=spec_on_run_hist, quality=quality,
                            exposure=exposure_run, obj_ra=objra, obj_dec=objdec,
                            dstart=run_dstart, dstop=run_dstop, creator='pfspec', version=version,
                            telescope=telescope, instrument=instrument)
            header = hdu.header
            header['ANCRFILE'] = os.path.basename(arf), 'Ancillary response file (ARF)'
            header['RESPFILE'] = os.path.basename(rmf), 'Redistribution matrix file (RMF)'
            header['BACKFILE'] = run_out_basename + '_bg.pha.fits', 'Bkgr FITS file'
            header['BACKSCAL'] = alpha_run, 'Background scale factor'            
            header['HDUCLAS2'] = 'TOTAL', 'Extension contains source + bkgd'
            filename = run_out_basename + '_signal.pha.fits'
            logging.info('RUN Writing signal PHA file to {0}'.format(filename))
            hdu.writeto(filename)

            # Background PHA
            hdu = np_to_pha(channel=chan, counts=spec_off_run_hist,
                            exposure=exposure_run, obj_ra=objra, obj_dec=objdec,
                            dstart=run_dstart, dstop=run_dstop, creator='pfspec', version=version,
                            telescope=telescope, instrument=instrument)
            header['ANCRFILE'] = os.path.basename(arf), 'Ancillary response file (ARF)'
            header['RESPFILE'] = os.path.basename(rmf), 'Redistribution matrix file (RMF)'
            header['HDUCLAS2'] = 'TOTAL', 'Extension contains source + bkgd'
            logging.info('RUN Writing background PHA file to {0}'.format(run_out_basename + '_bg.pha.fits'))
            hdu.writeto(run_out_basename + '_bg.pha.fits')

            # Excess PHA
            hdu = np_to_pha(channel=chan, counts=dat, stat_err=dat_err, exposure=exposure_run, quality=quality,
                            obj_ra=objra, obj_dec=objdec,
                            dstart=run_dstart, dstop=run_dstop, creator='pfspec', version=version,
                            telescope=telescope, instrument=instrument)
            header['ANCRFILE'] = os.path.basename(arf), 'Ancillary response file (ARF)'
            header['RESPFILE'] = os.path.basename(rmf), 'Redistribution matrix file (RMF)'
            logging.info('RUN Writing excess PHA file to {0}'.format(run_out_basename + '_excess.pha.fits'))
            hdu.writeto(run_out_basename + '_excess.pha.fits')

        hdulist.close()

        firstloop = False

    #---------------------------------------------------------------------------
    # Write results to file

    arf_m /= exposure

    if write_output_files:
        # Prepare data
        dat = spec_on_hist - spec_off_cor_hist  # ON - alpha x OFF = Excess
        dat_err = np.sqrt(spec_on_hist + spec_off_hist * (spec_off_cor_hist / spec_off_hist) ** 2.)
        quality = np.where(((spec_on_hist == 0) | (spec_off_hist == 0)), 1, 0)  # Set quality flags
        chan = np.arange(len(dat))

        # # DEBUG plot
        # plt.errorbar(spec_hist_ebounds[:-1], dat, yerr=dat_err)
        # plt.title('Total')
        # plt.show()

        # Data to PHA
        hdu = np_to_pha(channel=chan, counts=dat, stat_err=dat_err, exposure=exposure, quality=quality,
                        obj_ra=objra, obj_dec=objdec,
                        dstart=dstart, dstop=dstop, creator='pfspec', version=version,
                        telescope=telescope, instrument=instrument)
        # Write PHA to file
        hdu.header['ANCRFILE'] = os.path.basename('average.arf.fits'), 'Ancillary response file (ARF)'
        hdu.writeto('average.pha.fits')

        # Write ARF
        energy_lo = arf_m_erange[:-1]
        energy_hi = arf_m_erange[1:]
        arf_obj = EffectiveAreaTable(energy_lo, energy_hi, arf_m)
        hdu_list = arf_obj(header='pyfact', telescope=telescope, instrument=instrument)
        hdu_list.writeto('average.arf.fits')
        del arf_obj
    #---------------------------------------------------------------------------
    # Plot results

    try:
        import matplotlib
        has_matplotlib = True
    except:
        has_matplotlib = False

    if has_matplotlib and do_graphical_output:

        import matplotlib
        logging.info('Plotting results (matplotlib v{0})'.format(matplotlib.__version__))

        def set_title_and_axlabel(label):
            plt.xlabel('RA (deg)')
            plt.ylabel('Dec (deg)')
            plt.title(label, fontsize='medium')

        plt.figure()
        x = np.linspace(0., theta2_hist_max, theta2_hist_nbins + 1)
        x = (x[1:] + x[:-1]) / 2.
        plt.errorbar(x, theta2_on_hist, xerr=(theta2_hist_max / (2. *  theta2_hist_nbins)), yerr=np.sqrt(theta2_on_hist),
                     fmt='o', ms=3.5, label=r'N$_{ON}$', capsize=0.)
        plt.errorbar(x, theta2_offcor_hist, xerr=(theta2_hist_max / (2. *  theta2_hist_nbins)),
                     yerr=np.sqrt(theta2_off_hist) * theta2_offcor_hist / theta2_off_hist,
                     fmt='+', ms=3.5, label=r'N$_{OFF} \times \alpha$', capsize=0.)
        plt.axvline(analysis_radius ** 2., ls='--', label=r'$\theta^2$ cut')
        plt.xlabel(r'$\theta^2$ (deg$^2$)')
        plt.ylabel(r'N')
        plt.legend(numpoints=1)

        plt.figure()
        ax = plt.subplot(111)
        ecen = (spec_hist_ebounds[1:] + spec_hist_ebounds[:-1]) / 2.
        plt.errorbar(ecen, spec_on_hist,
                     xerr=(spec_hist_ebounds[1] - spec_hist_ebounds[0]) / 2.,
                     yerr=np.sqrt(spec_on_hist), fmt='o', label='ON')
        plt.errorbar(ecen, spec_off_cor_hist,
                     xerr=(spec_hist_ebounds[1] - spec_hist_ebounds[0]) / 2.,
                     yerr=np.sqrt(spec_off_hist) * spec_off_cor_hist / spec_off_hist, fmt='+', label='OFF cor.')
        dat = spec_on_hist - spec_off_cor_hist
        dat_err = np.sqrt(spec_on_hist + spec_off_hist * (spec_off_cor_hist / spec_off_hist) ** 2.)
        plt.errorbar(ecen, dat, yerr=dat_err, fmt='s', label='ON - OFF cor.')
        plt.xlabel(r'log(E/1 TeV)')
        plt.ylabel(r'N')
        plt.legend(numpoints=1)
        ax.set_yscale('log')

    plt.show()


def root_axis_to_array(ax):
    a = np.zeros(ax.GetNbins() + 1)
    for i in range(ax.GetNbins()):
        a[i] = ax.GetBinLowEdge(i + 1)
    a[-1] = ax.GetBinUpEdge(ax.GetNbins())
    return a


def root_1dhist_to_array(hist):
    nbins = hist.GetXaxis().GetNbins()
    a, e = np.zeros(nbins), np.zeros(nbins)
    for i in range(nbins):
        a[i] = hist.GetBinContent(i + 1)
        e[i] = hist.GetBinError(i + 1)
    return (a, e)


def root_2dhist_to_array(hist2d):
    nbinsx = hist2d.GetXaxis().GetNbins()
    nbinsy = hist2d.GetYaxis().GetNbins()
    a = np.zeros([nbinsx, nbinsy])
    e = np.zeros([nbinsx, nbinsy])
    for x in range(nbinsx):
        for y in range(nbinsy):
            a[x, y] = hist2d.GetBinContent(x + 1, y + 1)
            e[x, y] = hist2d.GetBinError(x + 1, y + 1)
    return (a, e)


def root_th1_to_fitstable(hist, xunit='', yunit=''):
    d, e = root_1dhist_to_array(hist)
    ax = root_axis_to_array(hist.GetXaxis())
    hdu = fits.new_table(
        [fits.Column(name='BIN_LO',
                       format='1E',
                       array=ax[:-1],
                       unit=xunit),
         fits.Column(name='BIN_HI',
                       format='1E',
                       array=ax[1:],
                       unit=xunit),
         fits.Column(name='VAL',
                       format='1E',
                       array=d,
                       unit=yunit),
         fits.Column(name='ERR',
                       format='1E',
                       array=e,
                       unit=yunit)
         ]
        )
    header = hdu.header
    header['ROOTTI'] = hist.GetTitle(), 'ROOT hist. title'
    header['ROOTXTI'] = hist.GetXaxis().GetTitle(), 'ROOT X-axis title'
    header['ROOTYTI'] = hist.GetYaxis().GetTitle(), 'ROOT Y-axis title'
    header['ROOTUN'] = hist.GetBinContent(0), 'ROOT n underflow'
    header['ROOTOV'] = hist.GetBinContent(hist.GetXaxis().GetNbins() + 1), 'ROOT n overflow'
    return hdu


def plot_th1(hist, logy=False):
    import matplotlib.pyplot as plt
    d, e = root_1dhist_to_array(hist)
    ax = root_axis_to_array(hist.GetXaxis())
    if logy:
        plt.semilogy((ax[:-1] + ax[1:]) / 2., d)
    else:
        plt.plot((ax[:-1] + ax[1:]) / 2., d)
    plt.xlabel(hist.GetXaxis().GetTitle(), fontsize='small')
    plt.ylabel(hist.GetYaxis().GetTitle(), fontsize='small')
    plt.title(hist.GetTitle(), fontsize='small')
    fontsize = 'small'
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    return


def fit_th1(fitter, p0, hist, errscale=None, range_=None, xaxlog=True):
    y, yerr = root_1dhist_to_array(hist)
    if errscale:
        yerr = errscale * y
    ax = root_axis_to_array(hist.GetXaxis())
    x = ((ax[:-1] + ax[1:]) / 2.)
    if xaxlog:
        x = 10 ** x
    if range_ is not None:
        m = (x >= range_[0]) * (x <= range_[1])
        x = x[m]
        y = y[m]
        yerr = yerr[m]
    fitter.fit_data(p0, x, y, yerr)
    return (fitter, x, y, yerr)


def cta_irf_root_to_fits(irf_root_file_name, write_output=False):
    """Convert CTA IRF data from ROOT to FITS format.

    This script converts a CTA response stored in a root file into a set of FITS
    files, namely ARF, RMF, and one auxiliary file, which stores all information
    from the response file in simple fits tables.

    Parameters
    ----------
    irf_root_file_name : str
        TODO
    write_output : bool
        Write output?

    Returns
    -------
    TODO: at the moment returns nothing ... files should be written by caller!
    """
    from ROOT import TFile
    from scipy.interpolate import UnivariateSpline
    from scipy.special import erf
    import matplotlib.pyplot as plt

    #---------------------------------------------------------------------------
    # Open CTA response file in root format

    irf_file_name_base = irf_root_file_name.rsplit('.', 1)[0].rsplit('/', 1)[1]

    logging.info('Reading IRF data from file {0}'.format(irf_root_file_name))
    irf_root_file = TFile(irf_root_file_name)
    logging.info('File content (f.ls()) :')
    irf_root_file.ls()

    #---------------------------------------------------------------------------
    # Write ARF & MRF

    #----------------------------------------------
    # Read RM
    h = irf_root_file.Get('MigMatrix')
    rm_erange_log, rm_ebounds_log = None, None
    if h != None:
        # Transpose and normalize RM
        rm = np.transpose(root_2dhist_to_array(h)[0])
        n = np.transpose(np.sum(rm, axis=1) * np.ones(rm.shape[::-1]))
        rm[rm > 0.] /= n[rm > 0.]
        # Read bin enery ranges
        rm_erange_log = root_axis_to_array(h.GetYaxis())
        rm_ebounds_log = root_axis_to_array(h.GetXaxis())
    else:
        logging.info('ROOT file does not contain MigMatrix.')
        logging.info('Will produce RMF from ERes histogram.')

        # Read energy resolution
        h = irf_root_file.Get('ERes')
        d = root_1dhist_to_array(h)[0]
        ax = root_axis_to_array(h.GetXaxis())

        # Resample to higher resolution in energy
        nbins = int((ax[-1] - ax[0]) * 20)  # 20 bins per decade
        rm_erange_log = np.linspace(ax[0], ax[-1], nbins + 1)
        rm_ebounds_log = rm_erange_log

        sigma = UnivariateSpline((ax[:-1] + ax[1:]) / 2., d, s=0, k=1)

        logerange = rm_erange_log
        logemingrid = logerange[:-1] * np.ones([nbins, nbins])
        logemaxgrid = logerange[1:] * np.ones([nbins, nbins])
        logecentergrid = np.transpose(((logerange[:-1] + logerange[1:]) / 2.) * np.ones([nbins, nbins]))

        gauss_int = lambda p, x_min, x_max: .5 * (erf((x_max - p[1]) / np.sqrt(2. * p[2] ** 2.)) - erf((x_min - p[1]) / np.sqrt(2. * p[2] ** 2.)))

        rm = gauss_int([1., 10. ** logecentergrid, sigma(logecentergrid).reshape(logecentergrid.shape) * 10. ** logecentergrid ], 10. ** logemingrid, 10. ** logemaxgrid)
        # rm = gauss_int([1., 10. ** logecentergrid, .5], 10. ** logemingrid, 10. ** logemaxgrid)

    # Create RM hdulist
    hdulist = np_to_rmf(rm,
                        (10. ** rm_erange_log).round(decimals=6),
                        (10. ** rm_ebounds_log).round(decimals=6),
                        1E-5,
                        telescope='CTASIM')

    # Write RM to file
    if write_output:
        hdulist.writeto(irf_file_name_base + '.rmf.fits')

    #----------------------------------------------
    # Read EA
    h = irf_root_file.Get('EffectiveAreaEtrue')  # ARF should be in true energy
    if h == None:
        logging.info('ROOT file does not contain EffectiveAreaEtrue (EA vs E_true)')
        logging.info('Will use EffectiveArea (EA vs E_reco) for ARF')
        h = irf_root_file.Get('EffectiveArea')
    ea = root_1dhist_to_array(h)[0]
    # Read EA bin energy ranges
    ea_erange_log = root_axis_to_array(h.GetXaxis())
    # Re-sample EA to match RM
    resample_ea_to_mrf = True
    # resample_ea_to_mrf = False
    if resample_ea_to_mrf:
            logging.info('Resampling effective area in log10(EA) vs log10(E) to match RM.')
            logea = np.log10(ea)
            logea[np.isnan(logea) + np.isinf(logea)] = 0.
            ea_spl = UnivariateSpline((ea_erange_log[1:] + ea_erange_log[:-1]) / 2., logea, s=0, k=1)
            e = (rm_erange_log[1:] + rm_erange_log[:-1]) / 2.
            ea = 10. ** ea_spl(e)
            ea[ea < 1.] = 0.
            ea_erange_log = rm_erange_log

    tbhdu = np_to_arf(ea,
                      (10. ** ea_erange_log).round(decimals=6),
                      telescope='CTASIM')
    # Write AR to file
    if write_output:
        tbhdu.writeto(irf_file_name_base + '.arf.fits')

    #----------------------------------------------
    # Fit some distributions

    # Broken power law fit function, normalized at break energy
    bpl = lambda p, x : np.where(x < p[0], p[1] * (x / p[0]) ** -p[2], p[1] * (x / p[0]) ** -p[3])
    fitter = ChisquareFitter(bpl)

    h = irf_root_file.Get('BGRatePerSqDeg')
    fit_th1(fitter, [3.2, 1E-5, 2., 1.], h, errscale=.2, range_=(.1, 100))
    fitter.print_results()
    bgrate_p1 = fitter.results[0]
    fitx = np.linspace(-2., 2., 100.)

    h = irf_root_file.Get('AngRes')
    fit_th1(fitter, [1., .6, .5, .2], h, errscale=.1)
    fitter.print_results()
    angres68_p1 = fitter.results[0]

    #----------------------------------------------
    # Read extra information from response file

    aux_tab = []
    plt.figure(figsize=(10, 8))

    h = irf_root_file.Get('BGRate')
    plt.subplot(331)
    plot_th1(h, logy=1)
    tbhdu = root_th1_to_fitstable(h, yunit='Hz', xunit='log(1/TeV)')
    tbhdu.header.update('EXTNAME ', 'BGRATE', 'Name of this binary table extension')
    aux_tab.append(tbhdu)

    h = irf_root_file.Get('BGRatePerSqDeg')
    plt.subplot(332)
    plot_th1(h, logy=1)
    plt.plot(fitx, bpl(bgrate_p1, 10. ** fitx))
    plt.plot(fitx, bpl([9., 5E-4, 1.44, .49], 10. ** fitx))
    tbhdu = root_th1_to_fitstable(h, yunit='Hz/deg^2', xunit='log(1/TeV)')
    tbhdu.header.update('EXTNAME ', 'BGRATED', 'Name of this binary table extension')
    aux_tab.append(tbhdu)

    h = irf_root_file.Get('EffectiveArea')
    plt.subplot(333)
    plot_th1(h, logy=1)
    tbhdu = root_th1_to_fitstable(h, yunit='m^2', xunit='log(1/TeV)')
    tbhdu.header.update('EXTNAME ', 'EA', 'Name of this binary table extension')
    aux_tab.append(tbhdu)

    h = irf_root_file.Get('EffectiveArea80')
    if h != None:
        plt.subplot(334)
        plot_th1(h, logy=True)
        tbhdu = root_th1_to_fitstable(h, yunit='m^2', xunit='log(1/TeV)')
        tbhdu.header.update('EXTNAME ', 'EA80', 'Name of this binary table extension')
        aux_tab.append(tbhdu)

    h = irf_root_file.Get('EffectiveAreaEtrue')
    if h != None:
        plt.subplot(335)
        plot_th1(h, logy=True)
        tbhdu = root_th1_to_fitstable(h, yunit='m^2', xunit='log(1/TeV)')
        tbhdu.header.update('EXTNAME ', 'EAETRUE', 'Name of this binary table extension')
        aux_tab.append(tbhdu)

    h = irf_root_file.Get('AngRes')
    plt.subplot(336)
    plot_th1(h, logy=True)
    plt.plot(fitx, bpl(angres68_p1, 10. ** fitx))
    plt.plot(fitx, bpl([1.1, 5.5E-2, .42, .19], 10. ** fitx))
    tbhdu = root_th1_to_fitstable(h, yunit='deg', xunit='log(1/TeV)')
    tbhdu.header.update('EXTNAME ', 'ANGRES68', 'Name of this binary table extension')
    aux_tab.append(tbhdu)

    h = irf_root_file.Get('AngRes80')
    plt.subplot(337)
    plot_th1(h, logy=True)
    tbhdu = root_th1_to_fitstable(h, yunit='deg', xunit='log(1/TeV)')
    tbhdu.header.update('EXTNAME ', 'ANGRES80', 'Name of this binary table extension')
    aux_tab.append(tbhdu)

    h = irf_root_file.Get('ERes')
    plt.subplot(339)
    plot_th1(h)
    tbhdu = root_th1_to_fitstable(h, xunit='log(1/TeV)')
    tbhdu.header.update('EXTNAME ', 'ERES', 'Name of this binary table extension')
    aux_tab.append(tbhdu)

    plt.subplot(338)
    # plt.set_cmap(plt.get_cmap('Purples'))
    plt.set_cmap(plt.get_cmap('jet'))
    # plt.imshow(np.log10(rm), origin='lower', extent=(rm_ebounds_log[0], rm_ebounds_log[-1], rm_erange_log[0], rm_erange_log[-1]))
    plt.imshow(rm, origin='lower', extent=(rm_ebounds_log[0], rm_ebounds_log[-1], rm_erange_log[0], rm_erange_log[-1]))
    plt.colorbar()
    # plt.clim(-2., 1.)

    plt.subplots_adjust(left=.08, bottom=.08, right=.97, top=.95, wspace=.3, hspace=.35)

    # Create primary HDU and HDU list to be stored in the output file
    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList([hdu] + aux_tab)

    # Write extra response data to file
    if write_output:
        hdulist.writeto(irf_file_name_base + '.extra.fits')

    # Close CTA IRF root file
    irf_root_file.Close()
