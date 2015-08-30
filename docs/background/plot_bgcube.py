import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.units import Quantity
from astropy.coordinates import Angle
from gammapy.background import CubeBackgroundModel
from gammapy import datasets

filename = datasets.get_path('../test_datasets/background/bg_cube_model_test.fits',
                             location='remote')
bg_model = CubeBackgroundModel.read(filename, format='table')

fig, axes = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches(30., 8., forward=True)

# plot images
bg_model.plot_image(energy=Quantity(0.5, 'TeV'), ax=axes[0], style_kwargs=dict(norm=LogNorm()))
bg_model.plot_image(energy=Quantity(50., 'TeV'), ax=axes[1], style_kwargs=dict(norm=LogNorm()))

# plot spectra
bg_model.plot_spectrum(det=Angle([0., 0.], 'degree'), ax=axes[2],
                       style_kwargs=dict(color='blue', label='(0, 0) deg'))
bg_model.plot_spectrum(det=Angle([2., 2.], 'degree'), ax=axes[2],
                       style_kwargs=dict(color='red', label='(2, 2) deg'))
axes[2].set_title('')
axes[2].legend()

plt.show()
