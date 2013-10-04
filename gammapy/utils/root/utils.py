# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""ROOT utility functions"""
from __future__ import print_function, division
import numpy as np


def fill_acceptance(offset_image, acceptance_curve, offset_max=2):
    """Fill acceptance image.

    Parameters
    ----------
    offset_image : 2D array
        Image of offset wrt. pointing position as value
    acceptance_curve : ROOT.TH1F
        Acceptance curve lookup histogram
    
    Returns
    -------
    acceptance_image : 2D array
        Acceptance image filled according to the offset_image and acceptance_curve
    """
    shape = offset_image.shape
    psi2 = (offset_image ** 2).flatten()
    acceptance_image = np.empty_like(psi2, dtype='float32')
    for ii in range(len(psi2)):
        jj = acceptance_curve.FindBin(psi2[ii])
        acceptance_image[ii] = acceptance_curve.GetBinContent(jj)

    acceptance_image = np.where(psi2 > offset_max ** 2,
                                0, acceptance_image)

    return acceptance_image.reshape(shape)


def measure_hist(hist, position='max'):
    """Measure a special position and value in a ROOT histogram.
    
    Parameters
    ----------
    hist : ROOT histogram
        Histogram
    position : one of 'max', 'min', 'center'
        Position to measure
    
    Returns
    -------
    dict with keys 'binx', 'biny', 'x', 'y', 'value'
    """
    import ROOT
    # Find binx, biny
    if position == 'max':
        binglobal = hist.GetMaximumBin()
        binx, biny, binz = ROOT.Long(), ROOT.Long(), ROOT.Long()
        hist.GetBinXYZ(binglobal, binx, biny, binz)
    elif position == 'min':
        binglobal = hist.GetMinimumBin()
        binx, biny, binz = ROOT.Long(), ROOT.Long(), ROOT.Long()
        hist.GetBinXYZ(binglobal, binx, biny, binz)
    elif position == 'center':
        binx = hist.GetXaxis().GetNbins() / 2
        biny = hist.GetYaxis().GetNbins() / 2
    else:
        raise RuntimeError('Unknown position = %s' % position)

    x = hist.GetXaxis().GetBinCenter(binx)
    y = hist.GetYaxis().GetBinCenter(biny)
    value = hist.GetBinContent(binx, biny)

    return dict(binx=binx, biny=biny, x=x, y=y, value=value)
