"""Runs commands to produce convolved predicted counts map in current directory.
"""
import numpy as np
import matplotlib.pyplot as plt
from gammapy.stats import significance
from gammapy.image.utils import disk_correlate
from npred_general import prepare_images

model, gtmodel, ratio, counts = prepare_images()

# Top hat correlation
correlation_radius = 3

correlated_gtmodel = disk_correlate(gtmodel, correlation_radius)
correlated_counts = disk_correlate(counts, correlation_radius)
correlated_model = disk_correlate(model, correlation_radius)

# Fermi significance
fermi_significance = np.nan_to_num(significance(correlated_counts, gtmodel,
                                                method='lima'))
# Gammapy significance
significance = np.nan_to_num(significance(correlated_counts, correlated_model,
                                          method='lima'))
# Plotting

vmin, vmax = 0, 10

fig, axes = plt.subplots(nrows=1, ncols=2)

results = [significance, fermi_significance]
titles = ['Gammapy Significance', 'Fermi Tools Significance']

for i in np.arange(2):
    im = axes.flat[i].imshow(results[i],
                         interpolation='nearest',
                         origin="lower", vmin=vmin, vmax=vmax,
                         cmap=plt.get_cmap())

    axes.flat[i].set_title(titles[i], fontsize=12)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.3, 0.025, 0.4])
fig.colorbar(im, cax=cbar_ax)
a = fig.get_axes()[0]
b = fig.get_axes()[1]
a.set_axis_off()
b.set_axis_off()
