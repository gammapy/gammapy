import matplotlib.pyplot as plt
from astropy.units import Quantity
from gammapy.background import CubeBackgroundModel
from gammapy import datasets

filename = '../test_datasets/background/bg_cube_model_test.fits'
filename = datasets.get_path(filename, location='remote')
bg_model = CubeBackgroundModel.read(filename, format='table')

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(16., 8., forward=True)

axes[0] = bg_model.plot_image(energy=Quantity(2., 'TeV'), ax=axes[0])
axes[1] = bg_model.plot_image(energy=Quantity(20., 'TeV'), ax=axes[1])

plt.draw()
