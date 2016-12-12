"""Compute flux points

This is an example script that show how to compute flux points in Gammapy.
TODO: Refactor and add to FluxPointsComputation class or so
"""

from gammapy.spectrum import (
    SpectrumObservation,
    SpectrumFit,
    FluxPoints,
    SpectrumResult
)
import astropy.units as u
from astropy.table import Table
import numpy as np
import copy
import matplotlib.pyplot as plt

plt.style.use('ggplot')
obs = SpectrumObservation.read('$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits')

fit = SpectrumFit(obs)
fit.run()
best_fit = copy.deepcopy(fit.result[0].fit)

# Define Flux points binning
emin = np.log10(obs.lo_threshold.to('TeV').value)
emax = np.log10(40)
binning = np.logspace(emin, emax, 8) * u.TeV

# Fix index
fit.model.gamma.freeze()

# Fit norm in bands
diff_flux = list()
diff_flux_err = list()
e_err_hi = list()
e_err_lo = list()
energy = list()
for ii in range(len(binning) - 1):
    energ = np.sqrt(binning[ii] * binning[ii + 1])
    energy.append(energ)
    e_err_hi.append(binning[ii + 1])
    e_err_lo.append(binning[ii])
    fit.fit_range = binning[[ii, ii + 1]]
    fit.run()
    res = fit.result[0].fit
    diff_flux.append(res.model(energ).to('cm-2 s-1 TeV-1'))
    err = res.model_with_uncertainties(energ.to('keV').value)
    diff_flux_err.append(err.s * u.Unit('cm-2 s-1 keV-1'))

table = Table()
table['e_ref'] = energy
table['e_min'] = e_err_lo
table['e_max'] = e_err_hi
table['dnde'] = diff_flux
table['dnde_err'] = diff_flux_err

points = FluxPoints(table)
result = SpectrumResult(fit=best_fit, points=points)
result.plot_spectrum()
plt.savefig('fluxpoints.png')
