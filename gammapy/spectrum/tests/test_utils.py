# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from tempfile import NamedTemporaryFile
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.time import Time
from astropy.table import Table
from ..utils import np_to_pha

class TestPHA(object):

    def setup_class(self):
        """Create test PHA file."""
    
        counts = np.array([0., 0., 5., 10., 20., 15., 10., 5., 2.5, 1., 1., 0.])
        stat_err = np.sqrt(counts)
        channel = (np.arange(len(counts)))
        
        exposure = 3600.
        
        dstart = Time('2011-01-01 00:00:00', scale='utc')
        dstop = Time('2011-01-31 00:00:00', scale='utc')
        dbase = Time('2011-01-01 00:00:00', scale='utc')
    
        pha = np_to_pha(channel=channel, counts=counts, exposure=exposure,
                        dstart=dstart, dstop=dstop, dbase=dbase,
                        stat_err=stat_err)

        #self.PHA_FILENAME = tmpdir.join('dummy.pha').strpath
        self.PHA_FILENAME = NamedTemporaryFile(suffix='.pha', delete=False).name        
        self.PHA_SUM = np.sum(counts)
    
        pha.writeto(self.PHA_FILENAME)    
    
        
    def test_pha(self):
        pha = Table.read(self.PHA_FILENAME)
        assert_allclose(pha['COUNTS'].sum(), 69.5)
