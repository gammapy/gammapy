# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.table import Table
from astropy import units as u

from .utils import get_random_state


class InverseCDFSampler:
    """Inverse CDF sampler.
       
   It determines a set of random numbers and calculate the cumulative 
   distribution function.
   
   Parameters
   ----------
   pdf : `~gammapy.maps.Map`
        Map of the predicted source counts.
   axis : int
        Axis along which sampling the indexes.
   random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.   
    """

    def __init__(self, pdf, axis=None, random_state=0):
        self.random_state = get_random_state(random_state)
        self.axis = axis

        if axis is not None:
            self.cdf = np.cumsum(pdf, axis=self.axis)
            self.cdf /= self.cdf[:, [-1]]
        else:
            self.pdf_shape = pdf.shape

            pdf = pdf.ravel() / pdf.sum()
            self.sortindex = np.argsort(pdf, axis=None)

            self.pdf = pdf[self.sortindex]
            self.cdf = np.cumsum(self.pdf)

    def sample_axis(self):
        """Sample along a given axis.

        Returns
        -------
        index : tuple of `~numpy.ndarray`
            Coordinates of the drawn sample.

        """
        choice = self.random_state.uniform(high=1, size=len(self.cdf))

        # find the indices corresponding to this point on the CDF
        index = np.argmin(np.abs(choice.reshape(-1, 1) - self.cdf), axis=self.axis)

        return index + self.random_state.uniform(low=-0.5, high=0.5, size=len(self.cdf))

    def sample(self, size):
        """Draw sample from the given PDF.

        Parameters
        ----------
        size : int
            Number of samples to draw.

        Returns
        -------
        index : tuple of `~numpy.ndarray`
            Coordinates of the drawn sample.
        """
        # pick numbers which are uniformly random over the cumulative distribution function
        choice = self.random_state.uniform(high=1, size=size)

        # find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        index = self.sortindex[index]

        # map back to multi-dimensional indexing
        index = np.unravel_index(index, self.pdf_shape)
        index = np.vstack(index)

        index = index + self.random_state.uniform(low=-0.5, high=0.5, size=index.shape)
        return index



class MapEventSampler:
    """Map event sampler.

    Parameters
    ----------

    npred_map : `~gammapy.maps.Map`
            Predicted number of counts map.
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.
    lc : `~gammapy.time.models.LightCurveTableModel`
            Input light-curve model of the source, given with columns labelled
            as "time" (in seconds) and "normalization" (arbitrary units): the bin time
            HAS to be costant.
    phase_lc : `~gammapy.time.models.PhaseCurveTableModel`
            Input phase-curve model of the source, given with columns labelled
            as "Phase" and "normalization" (arbitrary units): the bin time
            HAS to be costant.
    tmin : float
            Start time of the sampling, given in seconds.
    tmax : float
            Stop time of the sampling, given in seconds.
    """

    def __init__(self, npred_map, random_state=0,
                  lc = None, phase_lc=None, tmin=0, tmax=3600):
        self.random_state = get_random_state(random_state)
        self.npred_map = npred_map
        self.lc = lc
        self.phase_lc = phase_lc
        self.tmin=tmin
        self.tmax=tmax
    
    def npred_total(self):
        """ Calculate the total number of the sampled predicted events.
            
        Returns
        -------
        random_state.poisson() : int
                        Number of predicted events.
        """

        return self.random_state.poisson(np.sum(self.npred_map.data))

    def sample_npred(self):
        """ Calculate energy and Galactic coordinates of the sampled source events.
            
        Returns
        -------
        coords : `~numpy.ndarray`
                Array with coordinates and energies of the sampled events.
        """

        self.n_events = self.npred_total()
        
        cdf_sampler = InverseCDFSampler(self.npred_map.data, random_state=self.random_state)
        
        pix_coords = cdf_sampler.sample(self.n_events)
        coords = self.npred_map.geom.pix_to_coord(pix_coords[::-1])

        return coords
        
    def sample_timepred(self):
        """ Calculate the times of arrival of the sampled source events.

        Returns
        -------
        ToA : `~numpy.array`
            array with times of the sampled events.
        """
        
        n_events = self.n_events
        if (self.lc is not None) and (self.tmax>0):
            n_tbin = self.tmax - self.tmin
            t = np.linspace(self.tmin,self.tmax,n_tbin)
            normalization = self.lc._interpolator(t,ext=3)
            time_sampler = InverseCDFSampler(normalization, random_state=self.random_state)
            ToA = time_sampler.sample(n_events)[0]

        if (self.phase_lc is not None) and (self.tmax>0):
            n_tbin = self.tmax - self.tmin
            t = np.linspace(self.tmin,self.tmax,n_tbin)
            normalization = self.phase_lc.evaluate_norm_at_time(t)
            time_sampler = InverseCDFSampler(normalization, random_state=self.random_state)
            ToA = time_sampler.sample(n_events)[0]

        else:
            ToA = self.tmin + self.random_state.uniform(high=(self.tmax-self.tmin), size=n_events)

        return ToA

    def sample_events(self):
        """It converts the given sampled event list into an astropy table.
            
        Returns
        -------
        events : `~astropy.table`
            Sampled event list in an astropy table format.
        """

        coords = self.sample_npred()

        events = Table()
        events['lon_true'] = coords[0] * u.deg
        events['lat_true'] = coords[1] * u.deg
        events['e_true'] = coords[2] * u.TeV
        if ((self.lc is not None) and (self.tmax>0)) or (self.tmax>0):
            events['TIME'] = self.sample_timepred() * u.s
        
        return events
