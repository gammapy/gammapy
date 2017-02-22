"""
Example script showing how to simulate expected counts
in the CTA energy range
"""
from gammapy.spectrum.models import LogParabola
from gammapy.scripts import CTAPerf
from gammapy.scripts.cta_utils import CTASpectrumObservation, Target, ObservationParameters

import astropy.units as u
import time

# Observation parameters
alpha = 0.2 * u.Unit('')
livetime = 5. * u.h
emin = 0.03 * u.TeV
emax = 5 * u.TeV
obs_param = ObservationParameters(alpha=alpha, livetime=livetime,
                                  emin=emin, emax=emax)

# Target, PKS 2155-304 from 3FHL
name = "2155"
# model parameters
alpha = 1.88 * u.Unit('')
beta = 0.15 * u.Unit('')
reference = 18.3 * u.GeV
amplitude = 7.7e-11 * u.Unit('cm-2 s-1 GeV-1')
model = LogParabola(alpha=alpha, beta=beta, reference=reference, amplitude=amplitude)
# redshift
redshift = 0.116
# EBL model
ebl_model_name = 'dominguez'
target = Target(name=name, model=model,
                redshift=redshift,
                ebl_model_name=ebl_model_name)

# Performance
filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz'
cta_perf = CTAPerf.read(filename)

# Simulation
t_start = time.clock()
simu = CTASpectrumObservation.simulate_obs(perf=cta_perf,
                                           target=target,
                                           obs_param=obs_param)
t_end = time.clock()
print(simu)
print('\nsimu done in {} s'.format(t_end-t_start))
CTASpectrumObservation.plot_simu(simu, target)
