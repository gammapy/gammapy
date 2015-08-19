# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Background models.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.modeling.models import Gaussian1D
import astropy.units as u
from astropy.units import Quantity, UnitsError
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
    - TODO: review high-level doc!!!

    Parameters
    ----------
    events_cube : `~gammapy.background.Cube`, optional
        Cube to store counts.
    livetime_cube : `~gammapy.background.Cube`, optional
        Cube to store livetime correction.
    background_cube : `~gammapy.background.Cube`, optional
        Cube to store background model.
    """

    def __init__(self, events_cube=None, livetime_cube=None, background_cube=None):
        self.events_cube = events_cube
        self.livetime_cube = livetime_cube
        self.background_cube = background_cube

    @classmethod
    def read(cls, filename, format='table'):
        """Read cube background model from fits file.

        Several input formats are accepted, depending on the value
        of the **format** parameter:

        * table (default and preferred format): all 3 cubes as
          `~astropy.io.fits.HDUList` of `~astropy.io.fits.BinTableHDU`
        * image (alternative format): bg cube saved as
          `~astropy.io.fits.PrimaryHDU`, with the energy binning
          stored as `~astropy.io.fits.BinTableHDU`

        The events and livetime cubes are optional.

        This method calls `~gammapy.background.Cube.read`,
        forwarding all arguments.

        Parameters
        ----------
        filename : str
            Name of file with the cube.
        format : str, optional
            Format of the cube to read.

        Returns
        -------
        bg_cube_model : `~gammapy.background.CubeBackgroundModel`
            Cube background model object.
        """
        hdu = fits.open(filename)
        events_scheme_dict = Cube.define_scheme('bg_counts_cube')
        livetime_scheme_dict = Cube.define_scheme('bg_livetime_cube')
        background_scheme_dict = Cube.define_scheme('bg_cube')

        try:
            events_cube = Cube.read(filename, format, scheme='bg_counts_cube')
            livetime_cube = Cube.read(filename, format, scheme='bg_livetime_cube')
        except:
            # no events/livetime cube found: read only bg cube
            events_cube = Cube()
            livetime_cube = Cube()

        background_cube = Cube.read(filename, format, scheme='bg_cube')

        return cls(events_cube=events_cube,
                   livetime_cube=livetime_cube,
                   background_cube=background_cube)

    def write(self, outfile, format='table', **kwargs):
        """Write cube to fits file.

        Several output formats are accepted, depending on the value
        of the **format** parameter:

        * table (default and preferred format): all 3 cubes as
          `~astropy.io.fits.HDUList` of `~astropy.io.fits.BinTableHDU`
        * image (alternative format): bg cube saved as
          `~astropy.io.fits.PrimaryHDU`, with the energy binning
          stored as `~astropy.io.fits.BinTableHDU`

        The events and livetime cubes are optional.

        This method calls `~astropy.io.fits.HDUList.writeto`,
        forwarding the **kwargs** arguments.

        Parameters
        ----------
        outfile : str
            Name of file to write.
        format : str, optional
            Format of the cube to write.
        kwargs
            Extra arguments for the corresponding `io.fits` `writeto` method.
        """
        if ((self.events_cube.data.sum() == 0) or
            (self.livetime_cube.data.sum() == 0)):
            # empty envets/livetime cube: save only bg cube
            self.background_cube.write(outfile, format, **kwargs)
        else:
            if format == 'table':
                hdu_list = fits.HDUList([fits.PrimaryHDU(), # empty primary HDU
                                         self.events_cube.to_fits_table(),
                                         self.livetime_cube.to_fits_table(),
                                         self.background_cube.to_fits_table()])
                hdu_list.writeto(outfile, **kwargs)
            elif format == 'image':
                # save only bg cube: DS9 understands only one (primary) HDU
                self.background_cube.write(outfile, format, **kwargs)
            else:
                raise ValueError("Invalid format {}.".format(format))

    @classmethod
    def set_cube_binning(cls, detx_edges, dety_edges, energy_edges, do_not_fill=False):
        """
        Set cube binning from function parameters.

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
        do_not_fill : bool, optional
            Flag to avoid filling empty data (zeros) in the cubes.

        Returns
        -------
        bg_cube_model : `~gammapy.background.CubeBackgroundModel`
            Cube background model object.
        """
        empty_cube_data = np.zeros((len(energy_edges) - 1,
                                    len(dety_edges) - 1,
                                    len(detx_edges) - 1))

        events_cube = Cube(coordx_edges = detx_edges,
                           coordy_edges = dety_edges,
                           energy_edges = energy_edges,
                           data = Quantity(empty_cube_data, ''), # counts
                           scheme = 'bg_counts_cube')

        livetime_cube = Cube(coordx_edges = detx_edges,
                             coordy_edges = dety_edges,
                             energy_edges = energy_edges,
                             data = Quantity(empty_cube_data, 'second'),
                             scheme = 'bg_livetime_cube')

        background_cube = Cube(coordx_edges = detx_edges,
                               coordy_edges = dety_edges,
                               energy_edges = energy_edges,
                               data = Quantity(empty_cube_data, '1 / (s TeV sr)'),
                               scheme = 'bg_cube')

        return cls(events_cube=events_cube,
                   livetime_cube=livetime_cube,
                   background_cube=background_cube)

    @classmethod
    def define_cube_binning(cls, n_obs, do_not_fill=False):
        """Define cube binning (E, Y, X).

        The shape of the cube (number of bins on each axis) depends on the
        number of observations.

        (TODO: and the lower boundary of the cube on the energy threshold??!!!)

        Parameters
        ----------
        n_obs : int
            Number of observations.
        do_not_fill : bool, optional
            Flag to avoid filling empty data (zeros) in the cubes.

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

        return cls.set_cube_binning(detx_edges, dety_edges, energy_edges, do_not_fill)

    def fill_events(self, observation_table, fits_path):
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
        """
        # stack events
        observatory_name = observation_table.meta['OBSERVATORY_NAME']
        if observatory_name == 'HESS':
            scheme = 'hess'
        else:
            s_error = "Warning! Storage scheme for {}".format(observatory_name)
            s_error += "not implemented. Only H.E.S.S. scheme is available."
            raise RuntimeError(s_error)

        data_store = DataStore(dir=fits_path, scheme=scheme)
        event_list_files = data_store.make_table_of_files(observation_table,
                                                 	  	  filetypes=['events'])
        aeff_table_files = data_store.make_table_of_files(observation_table,
                                                          filetypes=['effective area'])

        # loop over observations
        for i_ev_file, i_aeff_file in zip(event_list_files['filename'],
                                          aeff_table_files['filename']):
            ev_list_ds = EventListDataset.read(i_ev_file)
            livetime = ev_list_ds.event_list.observation_live_time_duration
            aeff_hdu = fits.open(i_aeff_file)['EFFECTIVE AREA']
            # TODO: Gammapy needs a class that interprets IRF files!!!
            if aeff_hdu.header.comments['LO_THRES'] == '[TeV]':
                energy_threshold_unit = 'TeV'
            energy_threshold = Quantity(aeff_hdu.header['LO_THRES'],
                                        energy_threshold_unit)
            # TODO: please avoid storing important info (like units) in comments!!!

            # fill events above energy threshold, correct livetime accordingly
            data_set = ev_list_ds.event_list
            data_set = data_set.select_energy((energy_threshold, energy_threshold*1.e6))

            # construct events cube (energy, X, Y)
            # TODO: units are missing in the H.E.S.S. fits event
            #       lists; this should be solved in the next (prod03)
            #       H.E.S.S. fits production
            # workaround: try to cast units, if it doesn't work, use hard coded ones
            try:
                ev_DETX = Angle(data_set['DETX'])
                ev_DETY = Angle(data_set['DETY'])
                ev_energy = Quantity(data_set['ENERGY'])
            except UnitsError:
                ev_DETX = Angle(data_set['DETX'], 'degree') # hard-coded!!!
                ev_DETY = Angle(data_set['DETY'], 'degree') # hard-coded!!!
                ev_energy = Quantity(data_set['ENERGY'],
                                     data_set.meta['EUNIT']) # half hard-coded!!!
            ev_cube_table = Table([ev_energy, ev_DETY, ev_DETX],
                                  names=('ENERGY', 'DETY', 'DETX'))

            # TODO: filter out possible sources in the data;
            #       for now, the observation table should not contain any
            #       observation at or near an existing source

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
