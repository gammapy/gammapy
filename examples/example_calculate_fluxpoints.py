"""Compute flux points

This is an example script that show how to compute flux points in Gammapy. 
TODO: Refactor and add to FluxPointsComputation class or so
"""

from gammapy.spectrum import (
    SpectrumObservation,
    SpectrumFit,
    DifferentialFluxPoints,
    SpectrumResult
)
import astropy.units as u
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
    e_err_hi.append(binning[ii + 1] - energ)
    e_err_lo.append(energ - binning[ii])
    fit.fit_range = binning[[ii, ii + 1]]
    fit.run()
    res = fit.result[0].fit
    diff_flux.append(res.model(energ).to('cm-2 s-1 TeV-1'))
    err = res.model_with_uncertainties(energ.to('keV').value)
    diff_flux_err.append(err.s * u.Unit('cm-2 s-1 keV-1'))

points = DifferentialFluxPoints.from_arrays(energy=energy, diff_flux=diff_flux,
                                            diff_flux_err_hi=diff_flux_err,
                                            diff_flux_err_lo=diff_flux_err,
                                            energy_err_hi=e_err_hi,
                                            energy_err_lo=e_err_lo)
result = SpectrumResult(fit=best_fit, points=points)
result.plot_spectrum()
plt.savefig('fluxpoints.png')
