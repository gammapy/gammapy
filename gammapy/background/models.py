# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Background models."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.modeling.models import Gaussian1D
from astropy.table import Table
from astropy.units import Quantity
from ..data import EventList
from ..utils.energy import EnergyBounds
from .energy_offset_array import EnergyOffsetArray
from .fov_cube import _make_bin_edges_array, FOVCube

__all__ = [
    'GaussianBand2D',
    'FOVCubeBackgroundModel',
    'EnergyOffsetBackgroundModel',
]

DEFAULT_SPLINE_KWARGS = dict(k=1, s=0)


def _add_column_and_sort_table(sources, pointing_position):
    """Sort the table and add the column separation (offset from the source) and phi (position angle from the source).

    Parameters
    ----------
    sources : `~astropy.table.Table`
            Table of excluded sources.
    pointing_position : `~astropy.coordinates.SkyCoord`
            Coordinates of the pointing position

    Returns
    -------
    sources : `~astropy.table.Table`
        given sources table sorted with extra column "separation" and "phi"
    """
    sources = sources.copy()
    source_pos = SkyCoord(sources["RA"], sources["DEC"], unit="deg")
    sources["separation"] = pointing_position.separation(source_pos)
    sources["phi"] = pointing_position.position_angle(source_pos)
    sources.sort("separation")
    return sources


def _compute_pie_fraction(sources, pointing_position, fov_radius):
    """Compute the fraction of the pie over a circle.

    Parameters
    ----------
    sources : `~astropy.table.Table`
        Table of excluded sources.
        Required columns: RA, DEC, Radius
    pointing_position : `~astropy.coordinates.SkyCoord`
        Coordinates of the pointing position
    fov_radius : `~astropy.coordinates.Angle`
        Field of view radius

    Returns
    -------
    pie fraction : float
        If 0: nothing is excluded
    """
    sources = _add_column_and_sort_table(sources, pointing_position)
    radius = Angle(sources["Radius"])[0]
    separation = Angle(sources["separation"])[0]
    if separation > fov_radius:
        return 0
    else:
        return (2 * np.arctan(radius / separation) / (2 * np.pi)).value


def _select_events_outside_pie(sources, events, pointing_position, fov_radius):
    """Table row indices of events outside the pie.

    Parameters
    ----------
    sources : `~astropy.table.Table`
        Table of excluded sources.
        Required columns: RA, DEC, Radius
    events : `gammapy.data.EventList`
        List of events for one observation
    pointing_position : `~astropy.coordinates.SkyCoord`
        Coordinates of the pointing position
    fov_radius : `~astropy.coordinates.Angle`
        Field of view radius

    Returns
    -------
    idx : `~numpy.ndarray`
        Table row indices of the events that are outside the pie
    """
    sources = _add_column_and_sort_table(sources, pointing_position)
    radius = Angle(sources["Radius"])[0]
    phi = Angle(sources["phi"])[0]
    separation = Angle(sources["separation"])[0]
    if separation > fov_radius:
        return np.arange(len(events.table))
    else:
        phi_min = phi - np.arctan(radius / separation)
        phi_max = phi + np.arctan(radius / separation)
        phi_events = pointing_position.position_angle(events.radec)
        if phi_max.degree > 360:
            phi_max = phi_max - Angle(360, "deg")
            idx = np.where((phi_events > phi_max) & (phi_events < phi_min))
        else:
            idx = np.where((phi_events > phi_max) | (phi_events < phi_min))

        return idx[0]


def _poisson_gauss_smooth(counts, bkg):
    """Adaptive Poisson method to compute the smoothing kernel width from the available counts.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Count histogram 1D in offset
    bkg : `~numpy.ndarray`
        Count histogram 1D in offset

    Returns
    -------
    bkg_smooth : `~numpy.ndarray`
        Count histogram 1D in offset
    """
    from scipy.ndimage import convolve
    Nev = np.sum(counts)
    Np = len(counts)

    # Number of pixels per sigma of the kernel gaussian to have more than 150 events/sigma
    Npix_sigma = (150 / Nev) * Np

    # For high statistic, we impose a minimum of 4pixel/sigma
    Npix_sigma = np.maximum(Npix_sigma, 4)

    # For very low statistic, we impose a maximum lenght of the kernel equal of the number of bin
    # in the counts histogram
    Npix_sigma = np.minimum(Npix_sigma, Np / 6)

    # kernel gaussian define between -3 and 3 sigma
    x = np.linspace(-3, 3, 6 * Npix_sigma)
    kernel = np.exp(-0.5 * x ** 2)
    bkg_smooth = convolve(bkg, kernel / np.sum(kernel), mode="reflect")
    return bkg_smooth


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
        from scipy.interpolate import UnivariateSpline

        self.table = table
        glon = Angle(self.table['GLON']).wrap_at('180d')

        splines, units = {}, {}

        for column in table.colnames:
            y = self.table[column].quantity
            spline = UnivariateSpline(glon.degree, y.value, **spline_kwargs)
            splines[column] = spline
            units[column] = y.unit

        self._splines = splines
        self._units = units

    def _interpolate_parameter(self, parname, glon):
        glon = glon.wrap_at('180d')
        y = self._splines[parname](glon.degree)
        return y * self._units[parname]

    def peak_brightness(self, glon):
        """Peak brightness at a given longitude.

        Parameters
        ----------
        glon : `~astropy.coordinates.Longitude`
            Galactic Longitude.
        """
        return self._interpolate_parameter('Surface_Brightness', glon)

    def peak_brightness_error(self, glon):
        """Peak brightness error at a given longitude.

        Parameters
        ----------
        glon : `~astropy.coordinates.Longitude` or `~astropy.coordinates.Angle`
            Galactic Longitude.
        """
        return self._interpolate_parameter('Surface_Brightness_Err', glon)

    def width(self, glon):
        """Width at a given longitude.

        Parameters
        ----------
        glon : `~astropy.coordinates.Longitude` or `~astropy.coordinates.Angle`
            Galactic Longitude.
        """
        return self._interpolate_parameter('Width', glon)

    def width_error(self, glon):
        """Width error at a given longitude.

        Parameters
        ----------
        glon : `~astropy.coordinates.Longitude` or `~astropy.coordinates.Angle`
            Galactic Longitude.
        """
        return self._interpolate_parameter('Width_Err', glon)

    def peak_latitude(self, glon):
        """Peak position at a given longitude.

        Parameters
        ----------
        glon : `~astropy.coordinates.Longitude` or `~astropy.coordinates.Angle`
            Galactic Longitude.
        """
        return self._interpolate_parameter('GLAT', glon)

    def peak_latitude_error(self, glon):
        """Peak position error at a given longitude.

        Parameters
        ----------
        glon : `~astropy.coordinates.Longitude` or `~astropy.coordinates.Angle`
            Galactic Longitude.
        """
        return self._interpolate_parameter('GLAT_Err', glon)

    def evaluate(self, position):
        """Evaluate model at a given position.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position on the sky.
        """
        glon, glat = position.galactic.l, position.galactic.b
        width = self.width(glon)
        amplitude = self.peak_brightness(glon)
        mean = self.peak_latitude(glon)
        return Gaussian1D.evaluate(glat, amplitude=amplitude, mean=mean, stddev=width)


class FOVCubeBackgroundModel(object):
    """Field of view (FOV) cube background model.

    Container class for cube background model *(X, Y, energy)*.
    *(X, Y)* are detector coordinates (a.k.a. nominal system coordinates).
    This class defines 3 cubes of type `~gammapy.background.FOVCube`:

    - **counts_cube**: to store the counts (a.k.a. events) that
      participate in the model creation.

    - **livetime_cube**: to store the livetime correction.

    - **background_cube**: to store the background model.

    The class defines methods to define the binning, fill and smooth
    of the background cube models.

    Parameters
    ----------
    counts_cube : `~gammapy.background.FOVCube`, optional
        FOVCube to store counts.
    livetime_cube : `~gammapy.background.FOVCube`, optional
        FOVCube to store livetime correction.
    background_cube : `~gammapy.background.FOVCube`, optional
        FOVCube to store background model.
    """

    def __init__(self, counts_cube=None, livetime_cube=None, background_cube=None):
        self.counts_cube = counts_cube
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

        The counts and livetime cubes are optional.

        This method calls `~gammapy.background.FOVCube.read`,
        forwarding all arguments.

        Parameters
        ----------
        filename : str
            Name of file with the cube.
        format : str, optional
            Format of the cube to read.

        Returns
        -------
        bg_cube_model : `~gammapy.background.FOVCubeBackgroundModel`
            FOVCube background model object.
        """
        try:
            counts_cube = FOVCube.read(filename, format, scheme='bg_counts_cube')
            livetime_cube = FOVCube.read(filename, format, scheme='bg_livetime_cube')
        except:
            # no counts/livetime cube found: read only bg cube
            counts_cube = FOVCube()
            livetime_cube = FOVCube()

        background_cube = FOVCube.read(filename, format, scheme='bg_cube')

        return cls(counts_cube=counts_cube,
                   livetime_cube=livetime_cube,
                   background_cube=background_cube)

    def write(self, outfile, format='table', **kwargs):
        """Write cube to FITS file.

        Several output formats are accepted, depending on the value
        of the **format** parameter:

        * table (default and preferred format): all 3 cubes as
          `~astropy.io.fits.HDUList` of `~astropy.io.fits.BinTableHDU`
        * image (alternative format): bg cube saved as
          `~astropy.io.fits.PrimaryHDU`, with the energy binning
          stored as `~astropy.io.fits.BinTableHDU`

        The counts and livetime cubes are optional.

        This method calls `~astropy.io.fits.HDUList.writeto`,
        forwarding the **kwargs** arguments.

        Parameters
        ----------
        outfile : str
            Name of file to write.
        format : str, optional
            Format of the cube to write.
        kwargs
            Extra arguments for the corresponding `astropy.io.fits` ``writeto`` method.
        """
        if ((self.counts_cube.data.sum() == 0) or
                (self.livetime_cube.data.sum() == 0)):
            # empty envets/livetime cube: save only bg cube
            self.background_cube.write(outfile, format, **kwargs)
        else:
            if format == 'table':
                hdu_list = fits.HDUList([fits.PrimaryHDU(),  # empty primary HDU
                                         self.counts_cube.to_fits_table(),
                                         self.livetime_cube.to_fits_table(),
                                         self.background_cube.to_fits_table()])
                hdu_list.writeto(outfile, **kwargs)
            elif format == 'image':
                # save only bg cube: DS9 understands only one (primary) HDU
                self.background_cube.write(outfile, format, **kwargs)
            else:
                raise ValueError("Invalid format {}.".format(format))

    @classmethod
    def set_cube_binning(cls, detx_edges, dety_edges, energy_edges):
        """Set cube binning from function parameters.

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

        Returns
        -------
        bg_cube_model : `~gammapy.background.FOVCubeBackgroundModel`
            FOVCube background model object.
        """
        empty_cube_data = np.zeros((len(energy_edges) - 1,
                                    len(dety_edges) - 1,
                                    len(detx_edges) - 1))

        counts_cube = FOVCube(coordx_edges=detx_edges,
                              coordy_edges=dety_edges,
                              energy_edges=energy_edges,
                              data=Quantity(empty_cube_data, ''),  # counts
                              scheme='bg_counts_cube')

        livetime_cube = FOVCube(coordx_edges=detx_edges,
                                coordy_edges=dety_edges,
                                energy_edges=energy_edges,
                                data=Quantity(empty_cube_data, 'second'),
                                scheme='bg_livetime_cube')

        background_cube = FOVCube(coordx_edges=detx_edges,
                                  coordy_edges=dety_edges,
                                  energy_edges=energy_edges,
                                  data=Quantity(empty_cube_data, '1 / (s TeV sr)'),
                                  scheme='bg_cube')

        return cls(counts_cube=counts_cube,
                   livetime_cube=livetime_cube,
                   background_cube=background_cube)

    @classmethod
    def define_cube_binning(cls, observation_table, method='default'):
        """Define cube binning (E, Y, X).

        The shape of the cube (number of bins on each axis) depends on the
        number of observations.
        The binning is slightly altered in case a different method
        as the *default* one is used. In the *michi* method:

        * Minimum energy (i.e. lower boundary of cube energy
          binning) is equal to minimum energy threshold of all
          observations in the group.

        Parameters
        ----------
        observation_table : `~gammapy.data.ObservationTable`
            Observation list to use for the *michi* binning.
        data_dir : str
            Data directory
        method : {'default', 'michi'}, optional
            Bg cube model calculation method to apply.

        Returns
        -------
        bg_cube_model : `~gammapy.background.FOVCubeBackgroundModel`
            FOVCube background model object.
        """
        # define cube binning shape
        n_ebins = 20
        n_ybins = 60
        n_xbins = 60
        n_obs = len(observation_table)
        if n_obs < 100:
            minus_bins = int(n_obs / 10) - 10
            n_ebins += minus_bins
            n_ybins += 4 * minus_bins
            n_xbins += 4 * minus_bins
        bg_cube_shape = (n_ebins, n_ybins, n_xbins)

        # define cube edges
        energy_min = Quantity(0.1, 'TeV')
        energy_max = Quantity(100, 'TeV')
        dety_min = Angle(-0.07, 'radian').to('deg')
        dety_max = Angle(0.07, 'radian').to('deg')
        detx_min = Angle(-0.07, 'radian').to('deg')
        detx_max = Angle(0.07, 'radian').to('deg')
        # TODO: the bin min/max edges should depend on
        #       the experiment/observatory.
        #       or at least they should be read as parameters
        #       The values here are good for H.E.S.S.

        # energy bins (logarithmic)
        log_delta_energy = (np.log(energy_max.value) - np.log(energy_min.value)) / bg_cube_shape[0]
        energy_edges = np.exp(np.arange(bg_cube_shape[0] + 1) * log_delta_energy + np.log(energy_min.value))
        energy_edges = Quantity(energy_edges, energy_min.unit)
        # TODO: this function should be reviewed/re-written, when
        # the following PR is completed:
        # https://github.com/gammapy/gammapy/pull/290

        # spatial bins (linear)
        delta_y = (dety_max - dety_min) / bg_cube_shape[1]
        dety_edges = np.arange(bg_cube_shape[1] + 1) * delta_y + dety_min
        delta_x = (detx_max - detx_min) / bg_cube_shape[2]
        detx_edges = np.arange(bg_cube_shape[2] + 1) * delta_x + detx_min

        return cls.set_cube_binning(detx_edges, dety_edges, energy_edges)

    def fill_obs(self, observation_table, data_store):
        """Fill events and compute corresponding livetime.

        Get data files corresponding to the observation list, histogram
        the counts and the livetime and fill the corresponding cube
        containers.

        Parameters
        ----------
        observation_table : `~gammapy.data.ObservationTable`
            Observation list to use for the histogramming.
        data_store : `~gammapy.data.DataStore`
            Data store
        """
        for obs in observation_table:
            events = data_store.obs(obs_id=obs['OBS_ID']).events

            # TODO: filter out (mask) possible sources in the data
            #       for now, the observation table should not contain any
            #       run at or near an existing source

            self.counts_cube.fill_events([events])
            self.livetime_cube.data += events.observation_live_time_duration

    def smooth(self):
        """Smooth background cube model.

        Smooth method:

        1. slice model in energy bins: 1 image per energy bin
        2. calculate integral of the image
        3. determine times to smooth (N) depending on number of
           entries (counts) used to fill the cube
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

        # integral of original images
        integral_images = self.background_cube.integral_images

        # number of times to smooth
        n_counts = self.counts_cube.data.sum()
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
                self.background_cube.data[i_energy] = Quantity(image_smooth,
                                                               self.background_cube.data.unit)

        # integral of smooth images
        integral_images_smooth = self.background_cube.integral_images

        # scale images to preserve original integrals

        # loop over energy bins (i.e. images)
        for i_energy in np.arange(len(self.background_cube.energy_edges) - 1):
            self.background_cube.data[i_energy] *= (integral_images / integral_images_smooth)[i_energy]

    def compute_rate(self):
        """Compute background_cube from count_cube and livetime_cube."""
        bg_rate = self.counts_cube.data / self.livetime_cube.data

        bg_rate /= self.counts_cube.bin_volume
        # bg_rate.set_zero_level()

        # import IPython; IPython.embed()
        bg_rate = bg_rate.to('1 / (MeV sr s)')

        self.background_cube.data = bg_rate


class EnergyOffsetBackgroundModel(object):
    """EnergyOffsetArray background model.

    Container class for `EnergyOffsetArray` background model *(energy, offset)*.
    This class defines 3 `EnergyOffsetArray` of type `~gammapy.background.EnergyOffsetArray`

    Parameters
    ----------
    energy : `~gammapy.utils.energy.EnergyBounds`
         energy bin vector
    offset : `~astropy.coordinates.Angle`
        offset bin vector
    counts : `~numpy.ndarray`, optional
        data array (2D): store counts.
    livetime : `~numpy.ndarray`, optional
        data array (2D): store livetime correction
    bg_rate : `~numpy.ndarray`, optional
        data array (2D): store background model
    counts_err : `~numpy.ndarray`, optional
        data array (2D): store errors on the counts.
    bg_rate_err : `~numpy.ndarray`, optional
        data array (2D): store errors on the background model
    """

    def __init__(self, energy, offset, counts=None, livetime=None, bg_rate=None, counts_err=None, bg_rate_err=None):
        self.counts = EnergyOffsetArray(energy, offset, counts, data_err=counts_err)
        self.livetime = EnergyOffsetArray(energy, offset, livetime, "s")
        self.bg_rate = EnergyOffsetArray(energy, offset, bg_rate, "MeV-1 sr-1 s-1", data_err=bg_rate_err)

    def write(self, filename, **kwargs):
        """Write to FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        self.to_table().write(filename, format='fits', **kwargs)

    def to_table(self):
        """Convert to `~astropy.table.Table`.

        Returns
        -------
        table : `~astropy.table.Table`
            Table containing the `EnergyOffsetBackgroundModel`: counts, livetime and bg_rate
        """
        table = Table()
        table['THETA_LO'] = Quantity([self.counts.offset[:-1]], unit=self.counts.offset.unit)
        table['THETA_HI'] = Quantity([self.counts.offset[1:]], unit=self.counts.offset.unit)
        table['ENERG_LO'] = Quantity([self.counts.energy[:-1]], unit=self.counts.energy.unit)
        table['ENERG_HI'] = Quantity([self.counts.energy[1:]], unit=self.counts.energy.unit)
        table['counts'] = self.counts.to_table()['data']
        if self.counts.data_err is not None:
            table['counts_err'] = self.counts.to_table()['data_err']
        table['livetime'] = self.livetime.to_table()['data']
        table['bkg'] = self.bg_rate.to_table()['data']
        if self.bg_rate.data_err is not None:
            table['bkg_err'] = self.bg_rate.to_table()['data_err']
        table.meta['HDUNAME'] = "bkg_2d"
        return table

    @classmethod
    def read(cls, filename):
        """Create from FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        table = Table.read(filename)
        return cls.from_table(table)

    @classmethod
    def from_table(cls, table):
        """Create from `~astropy.table.Table`."""
        offset_edges = _make_bin_edges_array(table['THETA_LO'].squeeze(), table['THETA_HI'].squeeze())
        offset_edges = Angle(offset_edges, table['THETA_LO'].unit)
        energy_edges = _make_bin_edges_array(table['ENERG_LO'].squeeze(), table['ENERG_HI'].squeeze())
        energy_edges = EnergyBounds(energy_edges, table['ENERG_LO'].unit)
        counts = Quantity(table['counts'].squeeze(), table['counts'].unit)
        if "counts_err" in table.colnames:
            counts_err = Quantity(table['counts_err'].squeeze(), table['counts_err'].unit)
        else:
            counts_err = None
        livetime = Quantity(table['livetime'].squeeze(), table['livetime'].unit)
        bg_rate = Quantity(table['bkg'].squeeze(), table['bkg'].unit)
        if "bkg_err" in table.colnames:
            bg_rate_err = Quantity(table['bkg_err'].squeeze(), table['bkg_err'].unit)
        else:
            bg_rate_err = None
        return cls(energy_edges, offset_edges, counts, livetime, bg_rate, counts_err=counts_err,
                   bg_rate_err=bg_rate_err)

    def fill_obs(self, obs_ids, data_store, excluded_sources=None, fov_radius=Angle(2.5, "deg")):
        """Fill events and compute corresponding livetime.

        Get data files corresponding to the observation list, histogram
        the counts and the livetime and fill the corresponding cube
        containers.

        Parameters
        ----------
        obs_ids : list
            List of observation IDs
        data_store : `~gammapy.data.DataStore`
            Data store
        excluded_sources : `~astropy.table.Table`
            Table of excluded sources.
            Required columns: RA, DEC, Radius
        fov_radius : `~astropy.coordinates.Angle`
            Field of view radius
        """
        for obs_id in obs_ids:
            obs = data_store.obs(obs_id=obs_id)
            events = obs.events

            if excluded_sources:
                pie_fraction = _compute_pie_fraction(excluded_sources, events.pointing_radec, fov_radius)
                idx = _select_events_outside_pie(excluded_sources, events, events.pointing_radec, fov_radius)
                events = EventList(events.table[idx])
            else:
                pie_fraction = 0
            self.counts.fill_events([events])
            self.livetime.data += obs.observation_live_time_duration * (1 - pie_fraction)

    def compute_rate(self):
        """Compute background rate cube from count_cube and livetime_cube."""
        bg_rate = self.counts.data / self.livetime.data

        bg_rate /= self.counts.bin_volume

        bg_rate = bg_rate.to('MeV-1 sr-1 s-1')

        self.bg_rate.data = bg_rate
        self.bg_rate.data_err = (np.sqrt(self.counts.data) / (self.counts.bin_volume * self.livetime.data)).to(
            'MeV-1 sr-1 s-1')

    def smooth(self):
        """Smooth the bkg rate with a gaussian 1D kernel.

        Calling this method modifies the ``bg_rate`` data member, replacing it with a smoothed version.

        This method uses an adaptive Poisson method to compute the smoothing Kernel width
        from the available counts (see code and inline comments for details).
        """
        for idx_energy in range(len(self.counts.energy) - 1):
            counts = self.counts.data[idx_energy, :]
            bkg = self.bg_rate.data[idx_energy, :]
            n_events = np.sum(counts).value
            # For zero counts, the background rate is zero and smoothing would not change it.
            # For speed we're skipping the smoothing in that case
            if n_events > 0:
                acceptance_convolve = _poisson_gauss_smooth(counts, bkg)
                self.bg_rate.data[idx_energy, :] = Quantity(acceptance_convolve, self.bg_rate.data.unit)
