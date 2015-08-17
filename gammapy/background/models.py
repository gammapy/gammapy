# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Background models.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.modeling.models import Gaussian1D
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from ..utils.wcs import (linear_wcs_to_arrays,
                         linear_arrays_to_wcs)
from ..utils.fits import table_to_fits_table
from ..background import Cube
from ..obs import DataStore
from ..data import EventListDataset

__all__ = ['GaussianBand2D',
           'CubeBackgroundModel',
           ]

DEFAULT_SPLINE_KWARGS = dict(k=1, s=0)


class GaussianBand2D(object):

    """Gaussian band model.

    This 2-dimensional model is Gaussian in ``y`` for a given ``x``,
    and the Gaussian parameters can vary in ``x``.

    One application of this model is the diffuse emission along the
    Galactic plane, i.e. ``x = GLON`` and ``y = GLAT``.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table of Gaussian parameters.
        ``x``, ``amplitude``, ``mean``, ``stddev``.
    spline_kwargs : dict
        Keyword arguments passed to `~scipy.interpolate.UnivariateSpline`
    """

    def __init__(self, table, spline_kwargs=DEFAULT_SPLINE_KWARGS):
        self.table = table
        self.parnames = ['amplitude', 'mean', 'stddev']

        from scipy.interpolate import UnivariateSpline
        s = dict()
        for parname in self.parnames:
            x = self.table['x']
            y = self.table[parname]
            s[parname] = UnivariateSpline(x, y, **spline_kwargs)
        self._par_model = s

    def _evaluate_y(self, y, pars):
        """Evaluate Gaussian model at a given ``y`` position.
        """
        return Gaussian1D.evaluate(y, **pars)

    def parvals(self, x):
        """Interpolated parameter values at a given ``x``.
        """
        x = np.asanyarray(x, dtype=float)
        parvals = dict()
        for parname in self.parnames:
            par_model = self._par_model[parname]
            shape = x.shape
            parvals[parname] = par_model(x.flat).reshape(shape)

        return parvals

    def y_model(self, x):
        """Create model at a given ``x`` position.
        """
        x = np.asanyarray(x, dtype=float)
        parvals = self.parvals(x)
        return Gaussian1D(**parvals)

    def evaluate(self, x, y):
        """Evaluate model at a given position ``(x, y)`` position.
        """
        x = np.asanyarray(x, dtype=float)
        y = np.asanyarray(y, dtype=float)
        parvals = self.parvals(x)
        return self._evaluate_y(y, parvals)


class CubeBackgroundModel(object):

    """Cube background model.

    Container class for cube background model *(X, Y, energy)*.
    *(X, Y)* are detector coordinates (a.k.a. nominal system coordinates).
    This class defines 3 cubes of type `~gammapy.background.Cube`:

    - **events_cube**: to store the counts that participate in the
      model creation.

    - **livetime_cube**: to store the livetime correction.

    - **background_cube**: to store the background model.

    The class defines methods to define the binning, fill and smooth
    the cubes.

    - TODO: review this doc!!!
    - TODO: review this class!!!
    - TODO: review high-level doc!!!

    Parameters
    ----------
    detx_edges : `~astropy.coordinates.Angle`
        Spatial bin edges vector (low and high) for the cubes.
        X coordinate.
    dety_edges : `~astropy.coordinates.Angle`
        Spatial bin edges vector (low and high) for the cubes.
        Y coordinate.
    energy_edges : `~astropy.units.Quantity`
        Energy bin edges vector (low and high) for the cubes.

    Examples
    --------
    Access cube bg model data:

    .. code:: python

        energy_bin = bg_cube_model.find_energy_bin(energy=Quantity(2., 'TeV'))
        det_bin = bg_cube_model.find_det_bin(det=Angle([0., 0.], 'degree'))
        bg_cube_model.background[energy_bin, det_bin[1], det_bin[0]]
    """

    events_cube = Cube()
    livetime_cube = Cube()
    background_cube = Cube()

    def __init__(self, detx_edges, dety_edges, energy_edges):
        # define cube binning

        empty_cube_data = np.zeros((len(energy_edges) - 1,
                                    len(dety_edges) - 1,
                                    len(detx_edges) - 1))

        self.events_cube.coordx_edges = detx_edges
        self.events_cube.coordy_edges = dety_edges
        self.events_cube.energy_edges = energy_edges
        self.events_cube.data = Quantity(empty_cube_data, '') # counts
        self.events_cube.scheme = 'bg_counts_cube'

        self.livetime_cube.coordx_edges = detx_edges
        self.livetime_cube.coordy_edges = dety_edges
        self.livetime_cube.energy_edges = energy_edges
        self.livetime_cube.data = Quantity(empty_cube_data, 'second')
        self.livetime_cube.scheme = 'bg_livetime_cube'

        self.background_cube.coordx_edges = detx_edges
        self.background_cube.coordy_edges = dety_edges
        self.background_cube.energy_edges = energy_edges
        self.background_cube.data = Quantity(empty_cube_data, '1 / (s TeV sr)')
        self.background_cube.scheme = 'bg_cube'

    @classmethod
    def define_cube_binning(cls, n_obs, DEBUG):
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
        bg_cube_model : `~gammapy.background.CubeBackgroundModel`
            Cube background model object.
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
        #       or at least they should be read as parameters

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

    #    if DEBUG > 1:
    #        energy_edges = Quantity([0.01, 0.1, 1., 10., 100.], 'TeV') # log binning
    #        dety_edges = Angle(np.arange(-5., 6., 1.), 'degree') # stops at 5
    #        detx_edges = Angle(np.arange(-5., 6., 1.), 'degree') # stops at 5
        if DEBUG:
            print("energy bin edges", energy_edges)
            print("dety bin edges", dety_edges)
            print("detx bin edges", detx_edges)

        return cls(detx_edges, dety_edges, energy_edges)

    def fill_events(self, observation_table, fits_path, DEBUG):
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
        DEBUG : int
            Debug level.
        """
        # stack events
        data_store = DataStore(dir=fits_path)
        event_list_files = data_store.make_table_of_files(observation_table,
                                                 	  	  filetypes=['events'])
        aeff_list_files = data_store.make_table_of_files(observation_table,
                                                         filetypes=['effective area'])

        # loop over observations
        for i_ev_file, i_aeff_file in zip(event_list_files['filename'],
                                          aeff_list_files['filename']):
            if DEBUG > 2:
                print(' ev infile: {}'.format(i_ev_file))
                print(' aeff infile: {}'.format(i_aeff_file))
            ev_list_ds = EventListDataset.read(i_ev_file)
            livetime = ev_list_ds.event_list.observation_live_time_duration
            aeff_hdu = fits.open(i_aeff_file)['EFFECTIVE AREA']
            # TODO: Gammapy needs a class that interprets IRF files!!!
            if aeff_hdu.header.comments['LO_THRES'] == '[TeV]':
                energy_threshold_unit = 'TeV'
            energy_threshold = Quantity(aeff_hdu.header['LO_THRES'],
                                        energy_threshold_unit)
            if DEBUG > 2:
                print(' livetime {}'.format(livetime))
                print(' energy threshold {}'.format(energy_threshold))

            # fill events above energy threshold, correct livetime accordingly
            data_set = ev_list_ds.event_list
            data_set = data_set.select_energy((energy_threshold, energy_threshold*1.e6))

            # construct events cube (energy, X, Y)
            # TODO: UNITS ARE MISSING??!!! -> also in the fits tables!!!
            #       in header there is EUNIT (TeV)!!!
            #       hard coding the units for now !!!
            # TODO: try to cast units, if it doesn't work, use hard coded ones!!!
            ev_DETX = Angle(data_set['DETX'], 'degree')
            ev_DETY = Angle(data_set['DETY'], 'degree')
            ev_energy = Quantity(data_set['ENERGY'],
                                 data_set.meta['EUNIT'])
            ev_cube_table = Table([ev_energy, ev_DETY, ev_DETX],
                                  names=('ENERGY', 'DETY', 'DETX'))
            if DEBUG > 2:
                print(ev_cube_table)

            # TODO: filter out possible sources in the data
            #       for now, the observation table should not contain any
            #       run at or near an existing source

            # fill events

            # get correct data cube format for histogramdd
            ev_cube_array = np.vstack([ev_cube_table['ENERGY'],
                                       ev_cube_table['DETY'],
                                       ev_cube_table['DETX']]).T

            # fill data cube into histogramdd
            ev_cube_hist, ev_cube_edges = np.histogramdd(ev_cube_array,
                                                         [self.events_cube.energy_edges,
                                                          self.events_cube.coordy_edges,
                                                          self.events_cube.coordx_edges])
            ev_cube_hist = Quantity(ev_cube_hist, '') # counts

            # fill cube
            self.events_cube.data += ev_cube_hist

            # fill livetime for bins where E_max > E_thres
            energy_max = self.events_cube.energy_edges[1:]
            dummy_dety_max = np.zeros_like(self.events_cube.coordy_edges[1:])
            dummy_detx_max = np.zeros_like(self.events_cube.coordx_edges[1:])
            # define grid of max values (i.e. bin max values for each 3D bin)
            energy_max, dummy_dety_max, dummy_detx_max = np.meshgrid(energy_max,
                                                                     dummy_dety_max,
                                                                     dummy_detx_max,
                                                                     indexing='ij')
            mask = energy_max > energy_threshold

            # fill cube
            self.livetime_cube.data += livetime*mask

    def smooth(self):
        """
        Smooth background cube model.

        Smooth method:

        1. slice model in energy bins: 1 image per energy bin
        2. calculate integral of the image
        3. determine times to smooth (N) depending on number of entries (events) in the cube
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
        """
        from scipy import ndimage

        # smooth images

        # integral of original images
        dummy_delta_energy = np.zeros_like(self.background_cube.energy_edges[:-1])
        delta_y = self.background_cube.coordy_edges[1:] - self.background_cube.coordy_edges[:-1]
        delta_x = self.background_cube.coordx_edges[1:] - self.background_cube.coordx_edges[:-1]
        # define grid of deltas (i.e. bin widths for each 3D bin)
        dummy_delta_energy, delta_y, delta_x = np.meshgrid(dummy_delta_energy, delta_y,
                                                           delta_x, indexing='ij')
        bin_area = (delta_y*delta_x).to('sr')
        integral_image = self.background_cube.data*bin_area
        integral_image = integral_image.sum(axis=(1, 2))

        # number of times to smooth
        n_counts = self.events_cube.data.sum()
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
        for i_energy in np.arange(len(self.background_cube.energy_edges) - 1):
            # loop over number of times to smooth
            for i_smooth in np.arange(n_smooth):
                data = self.background_cube.data[i_energy]
                image_smooth = ndimage.convolve(data, kernel)

                # overwrite bg image with smoothed bg image
                self.background_cube.data[i_energy] = Quantity(image_smooth, self.background_cube.data.unit)

        # integral of smooth images
        integral_image_smooth = self.background_cube.data*bin_area
        integral_image_smooth = integral_image_smooth.sum(axis=(1, 2))

        # scale images to preserve original integrals

        # loop over energy bins (i.e. images)
        for i_energy in np.arange(len(self.background_cube.energy_edges) - 1):
            self.background_cube.data[i_energy] *= (integral_image/integral_image_smooth)[i_energy]
