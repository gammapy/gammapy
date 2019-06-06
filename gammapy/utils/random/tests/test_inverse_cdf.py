# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.table import Table
from ..inverse_cdf import InverseCDFSampler
from gammapy.time.models import LightCurveTableModel
import numpy as np


def ct_rate(x):
    return np.exp(-x/20)

def random_light_curve():
    x = np.linspace(0,100,100)
    table = Table()
    table['time'] = x
    table['rate'] = ct_rate(x)
    lc = LightCurveTableModel(table)

    t_data = lc.table['time'].data
    count_rate = lc.table['rate'].data
    lc = np.vstack([times,ctss])

    plt.plot(t_data,count_rate)

    sampled_times = InverseCDFSampler(pdf=count_rate,random_state=0)
    times=sampled_times.sample(10000)
