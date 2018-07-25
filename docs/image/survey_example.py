"""Plot a Galactic plane survey image in two panels."""
from astropy.coordinates import Angle
import matplotlib.pyplot as plt
from gammapy.image import MapPanelPlotter
from gammapy.maps import Map


filename = '$GAMMAPY_EXTRA/datasets/fermi_survey/all.fits.gz'
survey_map = Map.read(filename, hdu='counts')
survey_map.data = survey_map.data.astype('float')
smoothed_map = survey_map.smooth(radius=Angle(0.2, unit='deg'))


fig = plt.figure(figsize=(15, 8))
xlim = Angle([70, 262], unit='deg')
ylim = Angle([-4, 4], unit='deg')
plotter = MapPanelPlotter(figure=fig, xlim=xlim, ylim=ylim, npanels=4, top=0.98,
                          bottom=0.07, right=0.98, left=0.05, hspace=0.15)
axes = plotter.plot(smoothed_map, cmap='inferno', stretch='log', vmax=50)
plt.savefig('survey_example.png')
