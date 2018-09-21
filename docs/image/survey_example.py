"""Plot a Galactic plane survey image in two panels."""
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from gammapy.maps import Map
from gammapy.image import MapPanelPlotter

filename = "$GAMMAPY_DATA/fermi_survey/all.fits.gz"
survey_map = Map.read(filename, hdu="counts")
survey_map.data = survey_map.data.astype("float")
smoothed_map = survey_map.smooth(width=Angle(0.1, unit="deg"))

fig = plt.figure(figsize=(15, 8))
xlim = Angle([70, 262], unit="deg")
ylim = Angle([-4, 4], unit="deg")
plotter = MapPanelPlotter(
    figure=fig,
    xlim=xlim,
    ylim=ylim,
    npanels=4,
    top=0.98,
    bottom=0.07,
    right=0.98,
    left=0.05,
    hspace=0.15,
)
axes = plotter.plot(smoothed_map, cmap="inferno", stretch="log", vmax=50)
plt.savefig("survey_example.png")
