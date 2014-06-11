# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Make plots illustrating exposure in the FOV.

"""


def plot_exposure_image(filename):
    """Plot FOV image of exposure for one given energy slice"""
    from astropy.io import fits
    from aplpy import FITSFigure
    fig = FITSFigure(filename, dimensions=(0, 1), slices=[10], figsize=(5, 5))
    header = fits.getheader(filename)
    fig.show_grayscale()
    fig.add_colorbar()
    ra, dec = header['CRVAL1'], header['CRVAL2']

    # Bug: Marker doesn't show up at the center of the run
    # Bug: aplpy show_circles doesn't show a circle in degress.
    fig.show_markers(ra, dec)
    fig.show_circles(ra, dec, 1.)

    fig.tick_labels.set_xformat('dd')
    fig.tick_labels.set_yformat('dd')
    fig.ticks.set_xspacing(1)
    fig.ticks.set_yspacing(1)
    fig.colorbar.set_axis_label_text('Effective Area (cm^2)')

    fig.save('exposure_image.png')


def plot_exposure_curve(filename):
    from astropy.io import fits
    import matplotlib.pyplot as plt
    energy = fits.getdata(filename, 'ENERGIES')['Energy']
    cube = fits.getdata(filename, 0)
    exposure_1 = cube[:, 25, 25]
    exposure_2 = cube[:, 10, 25]
    plt.figure(figsize=(5, 4))
    plt.plot(energy, exposure_1, lw=2)
    plt.plot(energy, exposure_2, lw=2)
    plt.xlabel('Energy (TeV)')
    plt.ylabel('Effective Area (cm^2)')
    plt.xlim(0.1, 100)
    plt.ylim(1e6, 1e10)
    plt.loglog()
    plt.tight_layout()
    plt.savefig('exposure_curve.png')


def plot_psf_image(filename):
    from astropy.io import fits
    from aplpy import FITSFigure
    pass


def plot_psf_curve(filename):
    from astropy.io import fits
    import matplotlib.pyplot as plt
    pass


if __name__ == '__main__':
    DIR = '/Users/deil/work/host/data_formats/'
    exposure_filename = DIR + 'Aeff_23544.fits'
    psf_filename = DIR + 'psf_23544.fits'

    plot_exposure_image(exposure_filename)
    # plot_exposure_curve(exposure_filename)
    #plot_psf_image(psf_filename)
    #plt_psf_curve(psf_filename)
