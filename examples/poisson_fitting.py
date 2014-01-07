# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Perform a binned Poisson maximum likelihood fit.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Gaussian1D
from gammapy.stats import PoissonLikelihoodFitter

model = Gaussian1D(amplitude=1000, mean=2, stddev=3)
print('True parameters: ', model.parameters)
x = np.arange(-10, 20, 0.1)
dx = 0.1 * np.ones_like(x)
np.random.seed(0)
y = np.random.poisson(dx * model(x))

fitter = PoissonLikelihoodFitter()
model = fitter(model, x, y, dx, fit_statistic='cstat')
print('Parameters after fit: ', model.parameters)
y_model = dx * model(x)

plt.plot(x, y, 'o')
plt.plot(x, y_model, 'r-')
plt.savefig('poisson_fitting.png')
