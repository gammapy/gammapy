"""Plot a Galactic plane survey image in two panels."""
from aplpy import FITSFigure
from gammapy.datasets import FermiGalacticCenter
from gammapy.image import GalacticPlaneSurveyPanelPlot


class GPSFermiPlot(GalacticPlaneSurveyPanelPlot):    

    def main(self, figure, box):
        filename = FermiGalacticCenter.filenames()['counts']
        self.fits_figure = FITSFigure(filename, figure=figure, subplot=box)
        self.fits_figure.show_colorscale(vmin=1, vmax=10)
        self.fits_figure.ticks.set_xspacing(2)

plot = GPSFermiPlot(npanels=3, center=(0, 0), fov=(30, 3))
plot.draw_panels('all')
plot.figure.savefig('survey_example.png')
