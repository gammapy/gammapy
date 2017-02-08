"""
Example script showing how to simulate expected counts
in the CTA energy range

TODO: should use `~gammapy.spectrum.SpectrumSimulation`

"""
from gammapy.spectrum.models import PowerLaw
from gammapy.scripts import CTAPerf
from gammapy.scripts.cta_utils import CTASpectrumObservation, Target, ObservationParameters

import astropy.units as u

# Observation parameters
alpha = 0.2 * u.Unit('')
livetime = 5. * u.h
offset = 0.5 * u.degree
emin = 0.05 * u.TeV
emax = 5 * u.TeV
obs_params = ObservationParameters(alpha=alpha, livetime=livetime,
                                   offset=offset,
                                   emin=emin, emax=emax)

# Target
name = "golden_src"
# model
index = 3.5 * u.Unit('')
amplitude = 6. * 1e-12 * u.Unit('cm-2 s-1 TeV-1')
reference = 1 * u.TeV
model = PowerLaw(index=index, amplitude=amplitude, reference=reference)
# redshift
redshift = 0.4
# EBL model
ebl_model_name = 'dominguez'
target = Target(name=name, model=model,
                redshift=redshift,
                ebl_model_name=ebl_model_name)

# Performance
filename = '$GAMMAPY_EXTRA/datasets/cta/irf/prod2/Prod2_Mars_IRFs/South_5h.fits'
cta_perf = CTAPerf.read(filename)

# Simulation
obs = CTASpectrumObservation(perf=cta_perf, target=target)
obs.simulate_obs(obs_params)
print(obs.simu)
obs.peek()
simu = obs.simu

