# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tools to create profiles (i.e. 1D "slices" from 2D images)"""
import numpy as np

__all__ = ['compute_binning', 'FluxProfile']

def compute_binning(data, n_bins, method='equal width', eps=1e-10):
    """Computes 1D array of bin edges.

    Note that bin_edges has n_bins bins, i.e. length n_bins + 1.

    Parameters
    ----------
    data : array_like
        Data to be binned (any dimension)
    n_bins : int
        Number of bins
    method : str
        One of: 'equal width', 'equal entries'
    eps : float
        added to range so that the max data point is inside the
        last bin. If eps=0 it falls on the right edge of the last
        data point and thus would be not cointained.

    Returns
    -------
    bin_edges : 1D ndarray
        Array of bin edges.
    """
    if method == 'equal width':
        bin_edges = np.linspace(np.nanmin(data), np.nanmax(data) + eps, n_bins + 1)
    elif method == 'equal entries':
        raise NotImplementedError
        # sort and sub-sample; test many cases
    else:
        raise ValueError('Invalid option: method = {0}'.format(method))
    return bin_edges


class FluxProfile(object):
    """Compute flux profiles"""
    
    def __init__(self, x_edges, x, counts, background, exposure, mask=None):
        """
        x : Defines which pixel goes in which bin in combination with x_edges
        x_edges : Defines binning in x (could be GLON, GLAT, DIST, ...)
        
        mask : possibility to mask pixels (i.e. ignore in computations)

        Note: over- and underflow is ignored and not stored in the profile
        
        Note: this is implemented by creating bin labels and storing the
        input 2D data in 1D pandas.DataFrame tables.
        The 1D profile is also stored as a pandas.DataFrame and computed
        using the fast and flexible pandas groupby and apply functions.

        TODO: take mask into account everywhere
        TODO: decide on under- and overflow handling in x range
        TODO: separate FluxProfile.profile into a separate ProfileStack or HistogramStack class?
        """
        import pandas as pd
        # Make sure input is numpy arrays
        x_edges = np.asanyarray(x_edges)
        x = np.asanyarray(x)
        counts = np.asanyarray(counts)
        background = np.asanyarray(background)
        exposure = np.asanyarray(exposure)
        mask = np.asanyarray(mask)
        
        # Remember the shape of the 2D input arrays
        self.shape = x.shape

        # Store all input data as 1D vectors in a pandas.DataFrame
        d = pd.DataFrame(index=np.arange(x.size))
        d['x'] = x.flat
        # By default np.digitize uses 0 as the underflow bin.
        # Here we ignore under- and overflow, thus the -1
        d['label'] = np.digitize(x.flat, x_edges) - 1
        d['counts'] = counts.flat
        d['background'] = background.flat
        d['exposure'] = exposure.flat
        if mask:
            d['mask'] = mask.flat
        else:
            d['mask'] = np.ones_like(d['x'])
        self.data = d

        self.bins = np.arange(len(x_edges) + 1)
        
        # Store all per-profile bin info in a pandas.DataFrame
        p = pd.DataFrame(index=np.arange(x_edges.size - 1))
        p['x_lo'] = x_edges[:-1]
        p['x_hi'] = x_edges[1:]
        p['x_center'] = 0.5 * (p['x_hi'] + p['x_lo'])
        p['x_width'] = p['x_hi'] - p['x_lo']
        self.profile = p

        # The x_edges array is longer by one than the profile arrays,
        # so we store it separately
        self.x_edges = x_edges
        
    def compute(self, method='sum_first'):
        """
        Compute the flux profile.
        
        method : 'sum_first' or 'divide_first'
        
        Note: the current implementation is very inefficienct in speed and memory.
        There are various fast implementations, but none is flexible enough to
        allow combining many input quantities (counts, background, exposure) in a
        flexlible way:
        - numpy.histogram
        - scipy.ndimage.labeled_comprehension and special cases

        pandas DataFrame groupby followed by apply is flexible enough, I think:
        http://pandas.pydata.org/pandas-docs/dev/groupby.html
        """
        # Shortcuts to access class info needed in this method
        d = self.data
        # Here the pandas magic happens: we group pixels by label
        g = d.groupby('label')
        p = self.profile

        # Compute number of entries in each profile bin
        p['n_entries'] = g['x'].aggregate(len)
        p['counts_sum'] = g['counts'].sum()
        #p['counts_sum'] = np.histogram(d['label'], weights=d['counts'], bins=p.index)[0]
        p['counts_mean'] = p['counts_sum'] / p['n_entries']
        p['flux'] = 42

    def plot(self, which='n_entries', xlabel='Distance (deg)', ylabel=None):
        import matplotlib.pyplot as plt
        if ylabel == None:
            ylabel = which
        p = self.profile
        x = p['x_center']
        xerr = 0.5 * p['x_width']
        y = p[which]
        plt.errorbar(x, y, xerr=xerr, fmt='o');
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        #plt.ylim(-10, 20)
        
    def save(self, filename):
        """Save all profiles to a FITS file"""
        pass
        
