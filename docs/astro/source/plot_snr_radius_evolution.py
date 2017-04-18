"""Plot SNR radius evolution versus time."""
import numpy as np
import matplotlib.pyplot as plt
from astropy.units import Quantity
from gammapy.astro.source import SNR, SNRTrueloveMcKee

snr_models = [SNR, SNRTrueloveMcKee]
densities = Quantity([1, 0.1], 'cm^-3')
linestyles = ['-', '--']
t = Quantity(np.logspace(0, 5, 100), 'yr')

for density in densities:
    for linestyle, snr_model in zip(linestyles, snr_models):
        snr = snr_model(n_ISM=density)
        label = snr.__class__.__name__ + ' (n_ISM = {0})'.format(density.value)
        x = t.value
        y = snr.radius(t).to('pc').value
        plt.plot(x, y, label=label, linestyle=linestyle)

plt.xlabel('time [years]')
plt.ylabel('radius [pc]')
plt.legend(loc=4)
plt.loglog()
plt.show()
