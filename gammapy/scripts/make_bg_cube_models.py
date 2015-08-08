# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from astropy.io import fits
from astropy.wcs import WCS
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from ..utils.scripts import get_parser
from ..obs import ObservationTable, DataStore
from .. import datasets
from ..data import EventListDataset, SpectralCube
from ..background import CubeBackgroundModel

__all__ = ['make_bg_cube_models']


# TODO: remove all these global options: if needed, define as arguments to parse!!!
DEBUG = 2 # 0: no output, 1: output, 2: run fast, 3: more verbose
GRAPH_DEBUG = 1 # 0: no plots, 1: make plots, 2: wait between steps (bins), 3: draw 3D scatter plots (not implemented)
SAVE = 1

HESSFITS_MPP = '/home/mapaz/astropy/gammapy_tutorial/HESS_fits_data/pa/Model_Deconvoluted_Prod26/Mpp_Std'
BG_OBS_TABLE_FILE = 'bg_observation_table.fits'

def main(args=None):
    parser = get_parser(make_bg_cube_models)
    parser.add_argument("-l", "--log", dest="loglevel",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR',
                                 'CRITICAL'],
                        help="Set the logging level")
    parser.add_argument('--fitspath', type=str,
                        default=HESSFITS_MPP,
                        help='Dir path to input event list fits files '
                        '(default is {}).'.format(HESSFITS_MPP))
##    parser.add_argument('run_list', type=str,
##                        help='Input run list file name')
##    parser.add_argument('exclusion_list', type=str,
##                        help='Input exclusion list file name')
##    parser.add_argument('reference_file', type=str,
##                        help='Input FITS reference cube file name')
##    parser.add_argument('out_file', type=str,
##                        help='Output FITS counts cube file name')
##    parser.add_argument('--overwrite', action='store_true',
##                        help='Overwrite existing output file?')
    args = parser.parse_args(args)
    make_bg_cube_models(**vars(args))


##def make_bg_cube_models(run_list,
##                        exclusion_list,
##                        reference_file,
##                        out_file,
##                        overwrite):
def make_bg_cube_models(loglevel,
                        fitspath):
    """Create background cube models from off runs.

    Starting with only gamma-ray event lists, make background templates.
    (I actually need also Aeff for E_THRES!!!)

    This is an example showcasing some of the Gammapy features and what
    needs to be implemented. In this case, bg cube templates (X, Y, ENERGY)
    are created.

    With just ~ 100 lines of high-level code we can do this:

    - Make a global event list from a datastore
    - Filter the runs keeping only the ones far from known sources
    - Group the runs according to similar observation conditions (i.e. alt, az)
    - Bin/histogram events into a histogram.
    - Store the bg models histograms into CubeBackgroundModel objects and save them.
    - Plot the models if requested.

    TODO: SLOW IF DOING ALL MODELS!!! (maybe because of the plots???!!!)

    You can use this script to run certain steps by commenting in or out the functions in main().

    TODO: revise doc!!!

    Here the steps communicate via FITS files.

    Parameters
    ----------
    fits_path : str
        path to dir containing event list fits files and a list of them
    """
    if (loglevel):
        logging.basicConfig(level=getattr(logging, loglevel), format='%(levelname)s - %(message)s')

    create_bg_observation_list(fitspath)
    group_observations()
    stack_observations(fitspath)


# TODO: can I read/write fits.gz files? (I would save some disk space...)!!! (try/test it)!!!
# if it works, try to compress current result files

# define observation binning
# define a binning in altitude angle
# TODO: ObservationGroups
# https://github.com/mapazarr/gammapy/blob/bg-api/dev/background-api.py#L55
altitude_edges = Angle([0, 20, 23, 27, 30, 33, 37, 40, 44, 49, 53, 58, 64, 72, 90], 'degree')
if DEBUG > 1:
    altitude_edges = Angle([0, 45, 90], 'degree')

# define a binning in azimuth angle
# TODO: ObservationGroups
# https://github.com/mapazarr/gammapy/blob/bg-api/dev/background-api.py#L55
azimuth_edges = Angle([-90, 90, 270], 'degree')
if DEBUG > 1:
    azimuth_edges = Angle([90, 270], 'degree')


#define cube binning shape
def get_cube_shape(nobs):
    """Define shape of bg cube (E, Y, X)."""
    nebins = 20
    nybins = 60
    nxbins = 60
    if nobs < 100:       
        minusbins = int(nobs/10) - 10
        nebins += minusbins
        nybins += 4*minusbins
        nxbins += 4*minusbins
    return (nebins, nybins, nxbins)


def create_bg_observation_list(fits_path):
    """Make total observation list and filter the observations.

    In a first version, all obs taken within 3 deg of a known source will be rejected. If a source is extended, twice the extension is added to the corresponding exclusion region radius of 3 deg.

    TODO: on a second version, one could only filter out the runs
          too close to the galacic plane, and afterwards use masks
          to cut out sources, for runs taken on extragalactic objects
          (correcting the livetime accordingly).

    Parameters
    ----------
    fits_path : str
        path to dir containing list of input fits event files
    """
    if DEBUG:
        print()
        print("#######################################")
        print("# Starting create_bg_observation_list #")
        print("#######################################")

    # get full list of H.E.S.S. observations
    # TODO: shouldn't observatory='HESS' be a parameter to specify which experiment we are using??!!!
    data_store = DataStore(dir=fits_path)
    observation_table = data_store.make_observation_table()

    # For testing, only process a small subset of observations
    if DEBUG > 1:
        observation_table = observation_table.select_linspace_subset(num=100)
    if DEBUG:
        print()
        print("full observation table")
        print(observation_table)

    # TODO: the column format is not the accepted format in Gammapy!!! -> write converter? or adapt data_store?
    # https://gammapy.readthedocs.org/en/latest/dataformats/observation_lists.html
    # TODO: GLON GLAT are missing the units!!!

    # filter observations: load catalog and reject obs too close to sources

    # load catalog(s): HESS/TeVCAT (what about unpublished sources?)
    # there is no HESS catalog function? (only hess_galactic?)
    catalog = datasets.load_catalog_tevcat()

    # For testing, only process a small subset of sources
    if DEBUG > 1:
        catalog = catalog[:10]
    if DEBUG:
        print()
        print("TeVCAT catalogue")
        print(catalog)
        print("colnames: {}".format(catalog.colnames))

    # sources coordinates
    sources_coord = SkyCoord(catalog['coord_ra'], catalog['coord_dec'])

    # sources sizes (x, y): radius
    sources_size = Angle([catalog['size_x'], catalog['size_y']])
    sources_size = sources_size.reshape(len(catalog), 2)
    # substitute nan with 0
    sources_size[np.isnan(sources_size)] = 0
    # sources max size
    sources_max_size = np.amax(sources_size, axis=1)

    # sources exclusion radius = 2x max size + 3 deg (fov + 0.5 deg?)
    sources_excl_radius = 2*sources_max_size + Angle(3., 'degree')

    # mask all obs taken within the excl radius of any of the sources
    # loop over sources
    obs_coords = SkyCoord(observation_table['RA'], observation_table['DEC'])
    for i_source in range(len(catalog)):
        # TODO: call filer observations!!!
        # calculate separations (ang distances) of obs to source
        obs_sep_source = obs_coords.separation(sources_coord[i_source])
        source_mask = obs_sep_source <= sources_excl_radius[i_source]
        source_mask = np.invert(source_mask)
        observation_table = observation_table[source_mask]
        obs_coords = obs_coords[source_mask]

    # TODO: is there a way to quickly filter out sources in a region of the sky, where H.E.S.S. can't observe????!!!! -> don't loose too much time on this (detail)

    # save the bg observation list to a fits file
    if SAVE:
        outfile = BG_OBS_TABLE_FILE
        if DEBUG:
            print("outfile", outfile)
        observation_table.write(outfile, overwrite=True)


def group_observations():
    """Group list of runs according to observation properties into observation groups (bins).

    Parameters
    ----------
    fits_path : str
        path to dir containing event list fits files
    """
    if DEBUG:
        print()
        print("###############################")
        print("# Starting group_observations #")
        print("###############################")

    # read bg observation table from file
    # TODO: clean header from unnecessary info!!! (I only need obs
    #       table specific stuff, no FITS header stuff!!!)
    #       i.e. MJDREFI MJDREFF (observatory missing!!!)
    observation_table = ObservationTable.read(BG_OBS_TABLE_FILE)

    # split observation table according to binning
    # TODO: could be done by FindObservations (i.e. findruns)
    # https://github.com/mapazarr/gammapy/blob/bg-api/dev/background-api.py#L30
    # or by observation_selection (like in hgps example)
    # https://github.com/mapazarr/hess-host-analyses/blob/master/hgps_survey_map/hgps_survey_map.py#L62

    # define a binning in altitude angle
    # TODO: ObservationGroups
    # https://github.com/mapazarr/gammapy/blob/bg-api/dev/background-api.py#L55
    if DEBUG:
        print()
        print("altitude bin boundaries")
        print(repr(altitude_edges))

    # define a binning in azimuth angle
    # TODO: ObservationGroups
    # https://github.com/mapazarr/gammapy/blob/bg-api/dev/background-api.py#L55
    if DEBUG:
        print()
        print("azimuth bin boundaries")
        print(repr(azimuth_edges))

    # get observation altitude and azimuth angles
    altitude = Angle(observation_table['ALT_PNT'])
    azimuth = Angle(observation_table['AZ_PNT'])
    # wrap azimuth angles to (-90, 270) deg
    # TODO: needs re-thinking if azimuth angle definitions change!!!
    #       or if user-defined azimuth angle bins are allowed!!!
    azimuth = azimuth.wrap_at(Angle(270., 'degree'))

    if DEBUG:
        print()
        print("full list of observation altitude angles")
        print(repr(altitude))
        print("full list of observation azimuth angles")
        print(repr(azimuth))

    # loop over altitude and azimuth angle bins: remember 1 bin less than bin boundaries
    for i_alt in range(len(altitude_edges) - 1):
        if DEBUG:
            print()
            print("bin alt", i_alt)
        for i_az in range(len(azimuth_edges) - 1):
            if DEBUG:
                print()
                print("bin az", i_az)

            # filter observation table
            # TODO: this could be an extra selection criterion for FindObservations
            observation_table_filtered = observation_table
            altitude_filtered = altitude
            azimuth_filtered = azimuth
            altitude_mask = (altitude_edges[i_alt] <= altitude_filtered) & (altitude_filtered < altitude_edges[i_alt + 1])
            observation_table_filtered = observation_table_filtered[altitude_mask]
            altitude_filtered = altitude_filtered[altitude_mask]
            azimuth_filtered = azimuth_filtered[altitude_mask]
            azimuth_mask = (azimuth_edges[i_az] <= azimuth_filtered) & (azimuth_filtered < azimuth_edges[i_az + 1])
            observation_table_filtered = observation_table_filtered[azimuth_mask]
            altitude_filtered = altitude_filtered[azimuth_mask]
            azimuth_filtered = azimuth_filtered[azimuth_mask]
            if DEBUG:
                print(observation_table_filtered)

            # save the observation list to a fits file
            if SAVE:
                outfile = 'bg_observation_table_alt{0}_az{1}.fits'.format(i_alt, i_az)
                if DEBUG:
                    print("outfile", outfile)
                observation_table_filtered.write(outfile, overwrite=True)


def stack_observations(fits_path):
    """Stack events for each observation group (bin).
    """
    if DEBUG:
        print()
        print("###############################")
        print("# Starting stack_observations #")
        print("###############################")

    # loop over altitude and azimuth angle bins: remember 1 bin less than bin boundaries
    for i_alt in range(len(altitude_edges) - 1):
        if DEBUG:
            print()
            print("bin alt", i_alt)
        for i_az in range(len(azimuth_edges) - 1):
            if DEBUG:
                print()
                print("bin az", i_az)

            filename = 'bg_observation_table_alt{0}_az{1}.fits'.format(i_alt, i_az)

            # read group observation table from file
            # TODO: clean header from unnecessary info!!! (I only need obs
            #       table specific stuff, no FITS header stuff!!!)
            #       i.e. MJDREFI MJDREFF (observatory missing!!!)
            observation_table = ObservationTable.read(filename)

            if DEBUG:
                print(observation_table)

            # skip bins with no obs
            if len(observation_table) == 0:
                continue # skip the rest

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
                    print(' filename: {}'.format(i_file))
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
                    print(' filename: {}'.format(i_file))
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

            # bin events

            # define cube binning
            bg_cube_shape = get_cube_shape(len(observation_table))
            #energy_min = Quantity(0.1, 'TeV') # TODO: this should be overwriten by the energy threshold??!!!!
            # TODO: should E_min (= energy_edges[0]) be equal to E_THRES??!!!
            energy_min = energy_threshold
            energy_max = Quantity(80, 'TeV')
            dety_min = Angle(-0.07, 'radian').to('degree')
            dety_max = Angle(0.07, 'radian').to('degree')
            detx_min = Angle(-0.07, 'radian').to('degree')
            detx_max = Angle(0.07, 'radian').to('degree')

            # TODO: flag for make_test_bg_cube_model, in order to create an empty bg cube (only the binning, but no content)!!!

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
            ev_cube = np.zeros((len(energy_edges), len(detx_edges), len(dety_edges)))
            print("ev_cube shape", ev_cube.shape)

            # fill events

            # get correct data cube format for histogramdd
            ev_cube_array = np.vstack([ev_cube_table['ENERGY'], ev_cube_table['DETY'], ev_cube_table['DETX']]).T

            # fill data cube into histogramdd
            ev_cube_hist, ev_cube_edges = np.histogramdd(ev_cube_array, [energy_edges, dety_edges, detx_edges])
            n_counts = ev_cube_hist.sum()
            ev_cube_hist = Quantity(ev_cube_hist, '') # counts
            ev_cube_edges[0] = Quantity(ev_cube_edges[0], ev_cube_table['ENERGY'].unit)
            ev_cube_edges[1] = Angle(ev_cube_edges[1], ev_cube_table['DETY'].unit)
            ev_cube_edges[2] = Angle(ev_cube_edges[2], ev_cube_table['DETX'].unit)
            # Weight the counts with something meaningful:
            # divide by livetime times the cube bin volume
            #  mi mayer uses: histo.Scale(1.0/duration) (i.e. livetime)
            #  what about units of energy and solid angle?!!! -> he does this AFTER THE SMOOTHING!!! (but why?)
            ev_cube_hist /= livetime
            delta_energy = ev_cube_edges[0][1:] - ev_cube_edges[0][:-1]
            delta_y = ev_cube_edges[1][1:] - ev_cube_edges[1][:-1]
            delta_x = ev_cube_edges[2][1:] - ev_cube_edges[2][:-1]
            # define grid of deltas (i.e. bin widths for each 3D bin)
            delta_energy, delta_y, delta_x = np.meshgrid(delta_energy, delta_y, delta_x, indexing='ij')
            bin_volume = delta_energy.to('MeV')*(delta_y*delta_x).to('sr') # TODO: use TeV!!!

            # store in container class
            bg_cube_model = CubeBackgroundModel(detx_bins=detx_edges,
                                                dety_bins=dety_edges,
                                                energy_bins=energy_edges,
                                                background=ev_cube_hist)

            # smooth method
            # 1: slice model in energy bins -> 1 image per energy bin
            # 2: calculate integral of the image
            # 3: determine times to smooth (N) depending on number of entries in the cube
            # 4: smooth image N times with root TH2::Smooth function
            #    https://root.cern.ch/root/html/TH2.html#TH2:Smooth
            #    default smoothing kernel: k5a
            #    Double_t k5a[5][5] =  { { 0, 0, 1, 0, 0 },
            #                            { 0, 2, 2, 2, 0 },
            #                            { 1, 2, 5, 2, 1 },
            #                            { 0, 2, 2, 2, 0 },
            #                            { 0, 0, 1, 0, 0 } };
            # 5: scale with the cocient of the old integral div by the new integral
            # 6: fill the values of the image back in the cube
            # AFTER smoothing (when exporting to fits) he does:
            # 7: divide by the bin volumes
            # 8: set the 0 level to smthg very small

            # smooth images

            # integral of original images
            bin_area = (delta_y*delta_x).to('sr')
            integral_image = bg_cube_model.background*bin_area
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
            for i_energy in np.arange(len(bg_cube_model.energy_bins) - 1):
                # loop over number of times to smooth
                for i_smooth in np.arange(n_smooth):
                    data = bg_cube_model.background[i_energy]
                    image_smooth = ndimage.convolve(data, kernel)

                    # overwrite bg image with smoothed bg image
                    bg_cube_model.background[i_energy] = Quantity(image_smooth, bg_cube_model.background.unit)

            # integral of smooth images
            integral_image_smooth = bg_cube_model.background*bin_area
            integral_image_smooth = integral_image_smooth.sum(axis=(1,2))

            # scale images to preserve original integrals

            # loop over energy bins (i.e. images)
            for i_energy in np.arange(len(bg_cube_model.energy_bins) - 1):
                bg_cube_model.background[i_energy] *= (integral_image/integral_image_smooth)[i_energy]

            # correct by the bin volume and setting level 0 AFTER smoothing
            bg_cube_model.background /= bin_volume
            zero_level_mask = bg_cube_model.background < Quantity(1.e-10, '1 / (s sr MeV)')
            bg_cube_model.background[zero_level_mask] = Quantity(1.e-10, '1 / (s sr MeV)')

            # TODO: interpolate method!!!
            #       (not needed here, but useful for applying the models)

            # save model to file
            if SAVE:
                oufile = 'bg_cube_model_alt{0}_az{1}'.format(i_alt, i_az)
                if DEBUG:
                    print("outfile", '{}_table.fits'.format(oufile))
                    print("outfile", '{}_image.fits'.format(oufile))
                bg_cube_model.write('{}_table.fits'.format(oufile), format='table', clobber=True)
                bg_cube_model.write('{}_image.fits'.format(oufile), format='image', clobber=True)

            # check model: do some plots
            # TODO: move to an extra function?!!!
            if GRAPH_DEBUG:
                fig, axes = plt.subplots(nrows=1, ncols=3)
                fig.set_size_inches(30., 8., forward=True)
                plt.suptitle('altitude bin {0} azimuth bin {1}'.format(i_alt, i_az))

                # plot images
                #bg_cube_model.plot_image(energy=Quantity(0.5, 'TeV'), ax=axes[0])
                bg_cube_model.plot_image(energy=Quantity(5., 'TeV'), ax=axes[0])
                bg_cube_model.plot_image(energy=Quantity(50., 'TeV'), ax=axes[1])

                # plot spectra
                bg_cube_model.plot_spectrum(det=Angle([0., 0.], 'degree'), ax=axes[2],
                                            style_kwargs=dict(color='blue',
                                                              label='(0, 0) deg'))
                bg_cube_model.plot_spectrum(det=Angle([2., 2.], 'degree'), ax=axes[2],
                                            style_kwargs=dict(color='red',
                                                              label='(2, 2) deg'))
                axes[2].set_title('')
                axes[2].legend()

                #plt.tight_layout()
                plt.draw()

                if GRAPH_DEBUG > 1:
                    plt.show() # wait until image is closed

    # TODO: use random data (write a random data generator (see bg API))
    #       then write a similar script inside gammapy as example
    #       what about IRFs (i.e. Aeff for the E_THRESH?)?

    if GRAPH_DEBUG:
        plt.show() # don't leave at the end
