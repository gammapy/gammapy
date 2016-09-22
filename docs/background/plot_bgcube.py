import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.units import Quantity
from astropy.coordinates import Angle
from gammapy.background import FOVCube
from gammapy.datasets import gammapy_extra

filename = gammapy_extra.filename('test_datasets/background/bg_cube_model_test1.fits')
bg_cube_model = FOVCube.read(filename, format='table', scheme='bg_cube', hdu='BACKGROUND')

fig, axes = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches(16, 5., forward=True)

# plot images
bg_cube_model.plot_image(energy=Quantity(0.5, 'TeV'), ax=axes[0],
                         style_kwargs=dict(norm=LogNorm()))
bg_cube_model.plot_image(energy=Quantity(50., 'TeV'), ax=axes[1],
                         style_kwargs=dict(norm=LogNorm()))

# plot spectra
bg_cube_model.plot_spectrum(coord=Angle([0., 0.], 'degree'), ax=axes[2],
                            style_kwargs=dict(label='(0, 0) deg'))
bg_cube_model.plot_spectrum(coord=Angle([2., 2.], 'degree'), ax=axes[2],
                            style_kwargs=dict(label='(2, 2) deg'))
axes[2].set_title('')
axes[2].legend()

plt.tight_layout()
plt.show()
