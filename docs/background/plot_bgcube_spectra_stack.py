import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from gammapy.background import CubeBackgroundModel
from gammapy import datasets

filename = '../test_datasets/background/bg_cube_model_test.fits'
filename = datasets.get_path(filename, location='remote')
bg_model = CubeBackgroundModel.read(filename, format='table')

fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_size_inches(8., 8., forward=True)

ax = bg_model.plot_spectrum(det=Angle([0., 0.], 'degree'),
                            ax=ax,
                            style_kwargs=dict(color='blue',
                                              label='(0, 0) deg'))
ax = bg_model.plot_spectrum(det=Angle([2., 2.], 'degree'),
                            ax=ax,
                            style_kwargs=dict(color='red',
                                              label='(2, 2) deg'))
ax.set_title('')
ax.legend()

plt.draw()
