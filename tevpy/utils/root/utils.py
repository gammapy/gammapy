"""ROOT utility functions"""
import numpy as np


def fill_acceptance(psi, acceptance, psi_max=2):
    """Fill acceptance image.
    psi = np.array of offset values
    acceptance = ROOT.TH1F lookup histogram"""
    shape = psi.shape
    psi2 = (psi * psi).flatten()
    result = np.empty_like(psi2, dtype='float32')
    for ii in range(len(psi2)):
        jj = acceptance.FindBin(psi2[ii])
        result[ii] = acceptance.GetBinContent(jj)

    result = np.where(psi2 > psi_max ** 2, 0, result)

    return result.reshape(shape)
