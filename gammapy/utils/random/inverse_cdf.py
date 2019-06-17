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
   axis : integer
        Axis along which sampling the indexes.
   random_state : integer
        Take a `numpy.random.RandomState` instance.
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
    random_state : integer
            Take a `numpy.random.RandomState` instance.
    lc : `~gammapy.time.models.LightCurveTableModel`
            Input light-curve of the source, given with columns labelled
            as "time" and "normalization" (arbitrary units): the bin time
            HAS to be costant.
    tmin : start time of the sampling, in seconds.
    tmax : stop time of the sampling, in seconds.
    """

    def __init__(self, npred_map, random_state=0,
                  lc = None, tmin=0, tmax=3600):
        self.random_state = get_random_state(random_state)
        self.npred_map = npred_map
        self.lc = lc
        self.tmin=tmin
        self.tmax=tmax
    
    def npred_total(self):
        """ Calculate the total number of the sampled predicted events.
            
        Returns
        -------
        random_state.poisson() : Number of predicted events.
        """

        return self.random_state.poisson(np.sum(self.npred_map.data))

    def sample_npred(self):
        """ Calculate energy and coordinates of the sampled source events.
            
        Returns
        -------
        coords : array with coordinates and energies of the sampled events.
        """

        self.n_events = self.npred_total()
        
        cdf_sampler = InverseCDFSampler(self.npred_map.data, random_state=self.random_state)
        
        pix_coords = cdf_sampler.sample(self.n_events)
        self.coords = self.npred_map.geom.pix_to_coord(pix_coords[::-1])

        return self.coords
        
    def sample_timepred(self):
        """ Calculate the times of arrival of the sampled source events.

        Returns
        -------
        ToA : array with times of the sampled events.
        """
        
        n_events = self.n_events
        if self.lc is not None:
            start_lc = self.lc.table['TIME'].data[0]
            stop_lc = self.lc.table['TIME'].data[-1]
            dt = self.lc.table['TIME'].data[1] - self.lc.table['TIME'].data[0]
            
            if (self.tmax < start_lc) or (self.tmin > stop_lc):
                # if the lc does not fit into the range [tmin,tmax]
                self.ToA = self.tmin + self.random_state.uniform(high=(self.tmax-self.tmin), size=n_events)

            else:
                #define a time array in the range [t_min,t_max]
                times = np.arange(self.tmin,self.tmax,dt)

                #select the lc times in the time range [t_min,t_max]
                common_idx = np.where((times>=start_lc) & (times<=stop_lc))
                common_time = times[common_idx]
#                uncommon_idx = np.where((times<start_lc) | (times>stop_lc))
#                uncommon_time = times[uncommon_idx]

                #calculate the normalization in the common times and
                #the mean lc normalization in the rest
                norm = self.lc.evaluate_norm_at_time(common_time)
                mean_norm = self.lc.mean_norm_in_time_interval(start_lc,stop_lc)

                #define a new normalization array defined in the range [tmin,tmax]
                normalization = np.full_like(times, mean_norm)
                normalization[common_idx] = norm
                
                # sample the times
                time_sampler = InverseCDFSampler(normalization,random_state=0)
                self.ToA = time_sampler.sample(n_events)[0]

            return self.ToA

    def sample_events(self):
        """It converts the given sampled event list into an astropy table.
            
        Parameters
        ----------
        coords : output of sample_npred()
        ToA : output of sample_timepred()

        Returns
        -------
        events : event list in an astropy table format.
        """

        events = Table()
        events['lon_true'] = self.coords[0] * u.deg
        events['lat_true'] = self.coords[1] * u.deg
        events['e_true'] = self.coords[2] * u.TeV
        try:
           events['time'] = self.ToA * u.s
        except Exception:
            print("<<Warning: no event times have been sampled. \n>>")
        return events

