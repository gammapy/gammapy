"""Fit gamma-ray images with Sherpa.
"""
import sherpa.astro.ui as ui
from kapteyn import wcs, positions
try:
    from astropy.io import fits
except:
    import pyfits as fits

filename = 'skymap_ex.fits'
nomposstr = '05h34m31.94s 22d00m52.2s'
header = fits.getheader(filename)
proj = wcs.Projection(header)
xc, yc = float(header['NAXIS1']) / 2., float(header['NAXIS2']) / 2.
ui.load_image(filename)
ui.notice2d('circle({0}, {1}, {2})'.format(xc, yc, float(header['NAXIS2']) / 4.))
ui.set_source(ui.gauss2d.g1 + ui.gauss2d.g2)
g1.xpos = xc
g1.ypos = yc
g2.fwhm = g1.fwhm = 3.
ui.link(g2.xpos, g1.xpos)
ui.link(g2.ypos, g1.ypos)
g2.ampl = 50.
g1.ampl = 50.
ui.guess()
ui.fit()
ui.image_fit()
ui.covar()
conf = ui.get_covar_results()
conf_dict = dict([(n,(v, l, h)) for n,v,l,h in
                   zip(conf.parnames, conf.parvals, conf.parmins, conf.parmaxes)])
x, y = proj.toworld((conf_dict['g1.xpos'][0], conf_dict['g1.ypos'][0]))
xmin, ymin = proj.toworld((conf_dict['g1.xpos'][0] + conf_dict['g1.xpos'][1],
                           conf_dict['g1.ypos'][0] + conf_dict['g1.ypos'][1]))
xmax, ymax = proj.toworld((conf_dict['g1.xpos'][0] + conf_dict['g1.xpos'][2],
                           conf_dict['g1.ypos'][0] + conf_dict['g1.ypos'][2]))
nompos = positions.str2pos(nomposstr, proj)    
print('{0} ({1}-{2}) vs {3}'.format(x, xmin, xmax, nompos[0][0][0]))
print('{0} ({1}-{2}) vs {3}'.format(y, ymin, ymax, nompos[0][0][1]))
