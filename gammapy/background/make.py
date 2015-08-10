# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.table import Table
from ..background import CubeBackgroundModel
from ..obs import DataStore
from ..data import EventListDataset

__all__ = ['make_bg_cube_model',
           'define_cube_binning',
           'fill_events',
           'divide_bin_volume',
           'set_zero_level',
           'smooth'
           ]

# TODO: restructure bg models:
#       - rename CubeBackgroundModel -> CubeModel
#       - create a CubeBackgroundModel class that contains:
#          events_cube, livetime_cube, bg_model_cube
#          methods to produce the bg cube model:
#           'define_cube_binning',
#           'fill_events',
#           'divide_bin_volume',
#           'set_zero_level',
#           'smooth'
# @ C. Deil: does it make sense?


def define_cube_binning(n_obs, DEBUG):
    """Define cube binning (E, Y, X).

    The shape of the cube (number of bins on each axis) depends on the
    number of observations.

    (TODO: and the lower boundary of the cube on the energy threshold??!!!)

    Parameters
    ----------
    n_obs : int
        Number of observations.
    DEBUG : int
        Debug level.

    Returns
    -------
    energy_edges : `~astropy.units.Quantity`
        Energy bin edges.
    dety_edges : `~astropy.coordinates.Angle`
        Detector Y bin edges.
    detx_edges : `~astropy.coordinates.Angle`
        Detector X bin edges.
    """

    # define cube binning shape
    n_ebins = 20
    n_ybins = 60
    n_xbins = 60
    if n_obs < 100:       
        minus_bins = int(n_obs/10) - 10
        n_ebins += minus_bins
        n_ybins += 4*minus_bins
        n_xbins += 4*minus_bins
    bg_cube_shape = (n_ebins, n_ybins, n_xbins)

    # define cube edges
    energy_min = Quantity(0.1, 'TeV') # TODO: should this be overwriten by the energy threshold??!!!!
    # TODO: should E_min (= energy_edges[0]) be equal to E_THRES??!!!
    #energy_min = energy_threshold
    energy_max = Quantity(80, 'TeV')
    dety_min = Angle(-0.07, 'radian').to('degree')
    dety_max = Angle(0.07, 'radian').to('degree')
    detx_min = Angle(-0.07, 'radian').to('degree')
    detx_max = Angle(0.07, 'radian').to('degree')
    # TODO: the bin edges (at least for X and Y) should depend on the
    #       experiment/observatory.

    # energy bins (logarithmic)
    log_delta_energy = (np.log(energy_max.value)
                        - np.log(energy_min.value))/bg_cube_shape[0]
    energy_edges = np.exp(np.arange(bg_cube_shape[0] + 1)*log_delta_energy
                          + np.log(energy_min.value))
    energy_edges = Quantity(energy_edges, energy_min.unit)
    # TODO: this function should be reviewed/re-written, when
    # the following PR is completed:
    # https://github.com/gammapy/gammapy/pull/290

    # spatial bins (linear)
    delta_y = (dety_max - dety_min)/bg_cube_shape[1]
    dety_edges = np.arange(bg_cube_shape[1] + 1)*delta_y + dety_min
    delta_x = (detx_max - detx_min)/bg_cube_shape[2]
    detx_edges = np.arange(bg_cube_shape[2] + 1)*delta_x + detx_min

    if DEBUG > 1:
        energy_edges = Quantity([0.01, 0.1, 1., 10., 100.], 'TeV') # log binning
        dety_edges = Angle(np.arange(-5., 6., 1.), 'degree') # stops at 5
        detx_edges = Angle(np.arange(-5., 6., 1.), 'degree') # stops at 5
    if DEBUG:
        print("energy bin edges", energy_edges)
        print("dety bin edges", dety_edges)
        print("detx bin edges", detx_edges)

    return energy_edges, dety_edges, detx_edges


def fill_events(observation_table, fits_path, events_cube, livetime_cube, DEBUG):
    """Fill events and compute corresponding livetime.

    Get data files corresponding to the observation list, histogram
    the events and the livetime and fill the corresponding cube
    containers.

    Parameters
    ----------
    observation_table : `~gammapy.obs.ObservationTable`
        Observation list to use for the histogramming.
    fits_path : str
        Path to the data files.
    events_cube : `~gammapy.background.CubeBackgroundModel`
        Cube container for the events.
    livetime_cube : `~gammapy.background.CubeBackgroundModel`
        Cube container for the livetime.
    DEBUG : int
        Debug level.

    Returns
    -------
    events_cube : `~gammapy.background.CubeBackgroundModel`
        Cube containing the events.
    livetime_cube : `~gammapy.background.CubeBackgroundModel`
        Cube containing the livetime.
    """
    # stack events
    data_store = DataStore(dir=fits_path)
    event_list_files = data_store.make_table_of_files(observation_table, filetypes=['events'])
    data_set = EventListDataset.vstack_from_files(event_list_files['filename'])
    # TODO: the stacking can be long, if many runs are read: maybe it would be faster to grab the needed columns and stack them manually?!!!
    if DEBUG:
        print(data_set)
        print(data_set.event_list)
    print('Total number of events: {}'.format(len(data_set.event_list)))
    print('Total number of GTIs: {}'.format(len(data_set.good_time_intervals)))

    # loop over event files to get necessary infos from header
    livetime = Quantity(0. , 'second')
    # TODO: this loop is slow: can we accelerate it???!!! (or avoid it?)
    for i_file in event_list_files['filename']:
        if DEBUG > 2:
            print(' infile: {}'.format(i_file))
        ev_list_ds = EventListDataset.read(i_file)
        livetime += Quantity(ev_list_ds.event_list.meta['LIVETIME'],
                             ev_list_ds.event_list.meta['TIMEUNIT'])
        # TODO: the best would be to fill the livetime directly into a cube!!!
        if DEBUG > 2:
            print(' livetime {0} {1}'.format(ev_list_ds.event_list.meta['LIVETIME'],
                                             ev_list_ds.event_list.meta['TIMEUNIT']))
    if DEBUG:
        print('Total livetime = {}'.format(livetime.to('h')))

    # loop over effective area files to get necessary infos from header
    energy_threshold = 999. # TODO: define as quantity!!!
    # TODO: I need Aeff for the E_THRES!!!
    #  mi mayer takes the min E_THRES of all runs in the bin (alt-az, or zen bin, ...)
    #  and the E_THRES of each run (for PA) is stored in the header of the run-specific Aeff fits file: header["LO_THRES"]
    # TODO: I think this definition of E_th is not good:
    #  - some runs might not have events at that energy, but they still contribute to the livetime
    #  - max of all runs would be more conservative but more correct (if using this, redefine initial value for energy_threshold)
    #  - best I think: use the E_th of each run, and correct the livetime accordingly: livetime = f(E): fill, for each run, livetime only for bins above its threshold (and for events, fill only events above the corr. threshold)
    #  - C.Deil says: fill all events of all runs, ignoring the E_th
    aeff_list_files = data_store.make_table_of_files(observation_table, filetypes=['effective area'])
    # TODO: can we avoid the loop???!!! (or combine it with the loop over event files?)!!!
    for i_file in aeff_list_files['filename']:
        if DEBUG > 2:
            print(' infile: {}'.format(i_file))
        aeff_list_ds = EventListDataset.read(i_file)
        energy_threshold = min(energy_threshold,
                               aeff_list_ds.event_list.meta['LO_THRES'])
        if DEBUG > 2:
            print(' energy threshold {}'.format(aeff_list_ds.event_list.meta['LO_THRES']))
    energy_threshold = Quantity(energy_threshold, 'TeV') # TODO: units hard coded!!! units are in the fits file, but are not read into the dataset meta infos!!!
    if DEBUG:
        print('Total energy threshold = {}'.format(energy_threshold))
    if (energy_threshold < Quantity(1.e-6, 'TeV')) or (energy_threshold > Quantity(100., 'TeV')):
        raise ValueError("Enargy threshold sees incorrect: {}".format(energy_threshold))

    # apply mask to filer out events too close to known sources??!!!!
    # correct livetime accordingly) !!!
    # For the moment I will restrict only to runs far away from
    # sources, so no need for this.
    # THIS SHOULD GO LATER: when building the datacubes for events and livetime
    # no: earlier, since I loose the RA/Dec info otherwhise... right?
    # then I might have to define the cubes earlier, and fill run by run, instead of stacking them

    # construct events cube (energy, X, Y)
    # TODO: UNITS ARE MISSING??!!! -> look in the fits tables!!!
    # in header there is EUNIT (TeV)!!!
    # hard coding the units for now !!!
    ev_DETX = Angle(data_set.event_list['DETX'], 'degree')
    ev_DETY = Angle(data_set.event_list['DETY'], 'degree')
    ev_energy = Quantity(data_set.event_list['ENERGY'],
                         data_set.event_list.meta['EUNIT'])
    ev_cube_table = Table([ev_energy, ev_DETY, ev_DETX],
                          names=('ENERGY', 'DETY', 'DETX'))
    if DEBUG:
        print(ev_cube_table)

    # fill events

    # get correct data cube format for histogramdd
    ev_cube_array = np.vstack([ev_cube_table['ENERGY'], ev_cube_table['DETY'], ev_cube_table['DETX']]).T

    # fill data cube into histogramdd
    ev_cube_hist, ev_cube_edges = np.histogramdd(ev_cube_array,
                                                 [events_cube.energy_bins,
                                                  events_cube.dety_bins,
                                                  events_cube.detx_bins])
    ev_cube_hist = Quantity(ev_cube_hist, '') # counts
    ev_cube_edges[0] = Quantity(ev_cube_edges[0], ev_cube_table['ENERGY'].unit)
    ev_cube_edges[1] = Angle(ev_cube_edges[1], ev_cube_table['DETY'].unit)
    ev_cube_edges[2] = Angle(ev_cube_edges[2], ev_cube_table['DETX'].unit)

    # store in container class
    events_cube.background = ev_cube_hist
    livetime_cube.background = livetime

    return events_cube, livetime_cube


def divide_bin_volume(cube):
    """Divide by the bin volume.

    Parameters
    ----------
    cube : `~gammapy.background.CubeBackgroundModel`
        Cube containing the data to process.

    Returns
    -------
    cube : `~gammapy.background.CubeBackgroundModel`
        Cube divided by the bin volume.
    """
    delta_energy = cube.energy_bins[1:] - cube.energy_bins[:-1]
    delta_y = cube.dety_bins[1:] - cube.dety_bins[:-1]
    delta_x = cube.detx_bins[1:] - cube.detx_bins[:-1]
    # define grid of deltas (i.e. bin widths for each 3D bin)
    delta_energy, delta_y, delta_x = np.meshgrid(delta_energy, delta_y, delta_x, indexing='ij')
    bin_volume = delta_energy.to('MeV')*(delta_y*delta_x).to('sr') # TODO: use TeV!!!
    cube.background /= bin_volume

    return cube


def set_zero_level(cube):
    """Setting level 0 to something very small.

    Parameters
    ----------
    cube : `~gammapy.background.CubeBackgroundModel`
        Cube containing the data to process.

    Returns
    -------
    cube : `~gammapy.background.CubeBackgroundModel`
        Cube with 0-level applied.
    """
    zero_level = Quantity(1.e-10, cube.background.unit)
    zero_level_mask = cube.background < zero_level
    cube.background[zero_level_mask] = zero_level

    return cube


def smooth(bg_cube, n_counts):
    """
    Smooth background cube model.

    Smooth method:

    1. slice model in energy bins: 1 image per energy bin
    2. calculate integral of the image
    3. determine times to smooth (N) depending on number of entries in the cube
    4. smooth image N times with root TH2::Smooth
       default smoothing kernel: **k5a**

       .. code:: python

           k5a = [ [ 0, 0, 1, 0, 0 ],
                   [ 0, 2, 2, 2, 0 ],
                   [ 1, 2, 5, 2, 1 ],
                   [ 0, 2, 2, 2, 0 ],
                   [ 0, 0, 1, 0, 0 ] ]

       Reference: https://root.cern.ch/root/html/TH2.html#TH2:Smooth
    5. scale with the cocient of the old integral div by the new integral
    6. fill the values of the image back in the cube

    Parameters
    ----------
    bg_cube : `~gammapy.background.CubeBackgroundModel`
        Cube background model to smooth.
    n_counts : int
        Number of events used to fill the cube background model.

    Returns
    -------
    bg_cube : `~gammapy.background.CubeBackgroundModel`
        Smoothed cube background model.
    """
    from scipy import ndimage

    # smooth images

    # integral of original images
    dummy_delta_energy = np.zeros_like(bg_cube.energy_bins[1:])
    delta_y = bg_cube.dety_bins[1:] - bg_cube.dety_bins[:-1]
    delta_x = bg_cube.detx_bins[1:] - bg_cube.detx_bins[:-1]
    # define grid of deltas (i.e. bin widths for each 3D bin)
    dummy_delta_energy, delta_y, delta_x = np.meshgrid(dummy_delta_energy, delta_y, delta_x, indexing='ij')
    bin_area = (delta_y*delta_x).to('sr')
    integral_image = bg_cube.background*bin_area
    integral_image = integral_image.sum(axis=(1,2))

    # number of times to smooth
    if n_counts >= 1.e6:
        n_smooth = 3
    elif (n_counts < 1.e6) and (n_counts >= 1.e5):
        n_smooth = 4
    else:
        n_smooth = 5

    # smooth images

    # define smoothing kernel as k5a in root:
    # https://root.cern.ch/root/html/TH2.html#TH2:Smooth
    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 2, 2, 2, 0],
                       [1, 2, 5, 2, 1],
                       [0, 2, 2, 2, 0],
                       [0, 0, 1, 0, 0]])

    # loop over energy bins (i.e. images)
    for i_energy in np.arange(len(bg_cube.energy_bins) - 1):
        # loop over number of times to smooth
        for i_smooth in np.arange(n_smooth):
            data = bg_cube.background[i_energy]
            image_smooth = ndimage.convolve(data, kernel)

            # overwrite bg image with smoothed bg image
            bg_cube.background[i_energy] = Quantity(image_smooth, bg_cube.background.unit)

    # integral of smooth images
    integral_image_smooth = bg_cube.background*bin_area
    integral_image_smooth = integral_image_smooth.sum(axis=(1,2))

    # scale images to preserve original integrals

    # loop over energy bins (i.e. images)
    for i_energy in np.arange(len(bg_cube.energy_bins) - 1):
        bg_cube.background[i_energy] *= (integral_image/integral_image_smooth)[i_energy]

    return bg_cube


def make_bg_cube_model(observation_table, fits_path, DEBUG):
    """Create a bg model from an observation table.

    Produce a background cube using the data from an observation list.
    Steps:

    1. define binning
    2. fill events and livetime correction in cubes
    3. fill bg cube
    4. smooth
    5. correct for bin volume
    6. set 0 level

    TODO: review steps!!!

    Parameters
    ----------
    observation_table : `~gammapy.obs.ObservationTable`
        Observation list to use for the histogramming.
    fits_path : str
        Path to the data files.
    DEBUG : int
        Debug level.

    Returns
    -------
    events_cube : `~gammapy.background.CubeBackgroundModel`
        Cube containing the events.
    livetime_cube : `~gammapy.background.CubeBackgroundModel`
        Cube containing the livetime.
    bg_cube : `~gammapy.background.CubeBackgroundModel`
        Cube background model.
    """

    # DEBUG: 0: no output, 1: output, 2: run fast, 3: more verbose
    # TODO: remove the DEBUG variable, when the logger works!!!
    # TODO: I need to pass the logger or at least the log level!!!!
    #       and remove the DEBUG option!!!
    #       Look how the DataStore does it (when importing a file).

    ##################
    # define binning #
    ##################

    energy_edges, dety_edges, detx_edges = define_cube_binning(len(observation_table), DEBUG)


    ####################################################
    # create empty cubes: events, livetime, background #
    ####################################################

    empty_cube_data = np.zeros((len(energy_edges) - 1,
                                len(dety_edges) - 1,
                                len(detx_edges) - 1))

    if DEBUG:
        print("cube shape", empty_cube_data.shape)

    events_cube = CubeBackgroundModel(detx_bins=detx_edges,
                                      dety_bins=dety_edges,
                                      energy_bins=energy_edges,
                                      background=empty_cube_data)

    livetime_cube = CubeBackgroundModel(detx_bins=detx_edges,
                                        dety_bins=dety_edges,
                                        energy_bins=energy_edges,
                                        background=empty_cube_data)

    bg_cube = CubeBackgroundModel(detx_bins=detx_edges,
                                  dety_bins=dety_edges,
                                  energy_bins=energy_edges,
                                  background=empty_cube_data)


    ############################
    # fill events and livetime #
    ############################

    # TODO: filter out possible sources in the data
    #       for now, the observation table should not contain any
    #        run at or near an existing source
    # TODO: move this TODO to its rightful place!!!

    events_cube, livetime_cube = fill_events(observation_table, fits_path,
                                             events_cube, livetime_cube,
                                             DEBUG)


    ################
    # fill bg cube #
    ################

    # Weight the counts with something meaningful:
    # divide by livetime times the cube bin volume.
    # The bin volume division is done after the smoothing.

    bg_cube.background = events_cube.background/livetime_cube.background
    # TODO: rename the datamemeber background to data!!!
    # TODO: test: 1st smooth events, then divide by livetime, bin vol and then set level 0 !!!


    ##########
    # smooth #
    ##########

    bg_cube = smooth(bg_cube, events_cube.background.sum())


    ##########################################
    # correct for bin volume and set 0 level #
    ##########################################

    # divide by the bin volume and setting level 0 AFTER smoothing
    bg_cube = divide_bin_volume(bg_cube)
    bg_cube = set_zero_level(bg_cube)


    ######################
    # return the 3 cubes #
    ######################

    return events_cube, livetime_cube, bg_cube
