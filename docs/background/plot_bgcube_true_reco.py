"""
Script to produce plots comparing 2 background cube models.

Details in stringdoc of the plot_bg_cube_model_comparison function.

Inspired on the Gammapy examples/plot_bg_cube_model_comparison.py script.

The resulting image should be similar to:
https://github.com/gammapy/gammapy-extra/blob/master/figures/bg_cube_model_true_reco.png
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.units import Quantity
from astropy.coordinates import Angle
from gammapy.background import FOVCubeBackgroundModel
from gammapy.spectrum import power_law_evaluate, power_law_integral_flux
from gammapy.datasets import gammapy_extra

E_REF = Quantity(1., 'TeV')
NORM = 1
INDEX = 1.5


def get_normed_pl(plot_data, E_0, norm, index):
    """Define a power-law (PL) function normalized to the specified data.

    Parameters
    ----------
    plot_data : `~matplotlib.lines.Line2D`
        plot data (X, Y) points to emulate
    E_0 : `~astropy.units.Quantity`
        PL reference energy.
    norm : `~astropy.units.Quantity` or float
        PL normalization factor.
    index : float
        PL spectral index.
    """
    plot_data_x = plot_data.get_xydata()[:, 0]
    plot_data_y = plot_data.get_xydata()[:, 1]
    plot_data_int = np.trapz(y=plot_data_y, x=plot_data_x)
    energy_band = np.array([plot_data_x[0], plot_data_x[-1]])
    model_int = power_law_integral_flux(f=norm, g=index, e=E_0,
                                        e1=energy_band[0], e2=energy_band[1])
    normed_pl = plot_data_int / model_int * power_law_evaluate(energy=plot_data_x,
                                                               norm=norm, gamma=index,
                                                               energy_ref=E_0)

    return plot_data_x, normed_pl


def plot_bg_cube_model_comparison(input_file1, name1,
                                  input_file2, name2):
    """
    Plot background cube model comparison.

    Produce a figure for comparing 2 bg cube models (1 and 2).

    Plot strategy in each figure:

    * Images:
        * rows: similar energy bin
        * cols: same bg cube model set
    * Spectra:
        * rows: similar det bin
        * cols: compare both bg cube model sets

    Parameters
    ----------
    input_file1, input_file2 : str
        File where the corresponding bg cube model is stored.
    name1, name2 : str
        Name to use for plot labels/legends.
    """
    # get cubes
    filename1 = input_file1
    filename2 = input_file2
    bg_cube_model1 = FOVCubeBackgroundModel.read(filename1,
                                                 format='table').background_cube
    bg_cube_model2 = FOVCubeBackgroundModel.read(filename2,
                                                 format='table').background_cube

    # normalize 1 w.r.t. 2 (i.e. true w.r.t. reco)
    # normalize w.r.t. cube integral
    integral1 = bg_cube_model1.integral
    integral2 = bg_cube_model2.integral
    bg_cube_model1.data *= integral2 / integral1

    # make sure that both cubes use the same units for the plots
    bg_cube_model2.data = bg_cube_model2.data.to(bg_cube_model1.data.unit)

    # plot
    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(30., 15., forward=True)
    group_info = 'group 27: ALT = [72.0, 90.0) deg, AZ = [90.0, 270.0) deg'
    plt.suptitle(group_info)

    # plot images
    #  rows: similar energy bin
    #  cols: same file
    bg_cube_model1.plot_image(energy=Quantity(1., 'TeV'), ax=axes[0, 0])
    axes[0, 0].set_title("{0}: {1}".format(name1, axes[0, 0].get_title()))
    bg_cube_model1.plot_image(energy=Quantity(10., 'TeV'), ax=axes[1, 0])
    axes[1, 0].set_title("{0}: {1}".format(name1, axes[1, 0].get_title()))
    bg_cube_model2.plot_image(energy=Quantity(1., 'TeV'), ax=axes[0, 1])
    axes[0, 1].set_title("{0}: {1}".format(name2, axes[0, 1].get_title()))
    bg_cube_model2.plot_image(energy=Quantity(10., 'TeV'), ax=axes[1, 1])
    axes[1, 1].set_title("{0}: {1}".format(name2, axes[1, 1].get_title()))

    # plot spectra
    #  rows: similar det bin
    #  cols: compare both files
    bg_cube_model1.plot_spectrum(coord=Angle([0., 0.], 'degree'),
                                 ax=axes[0, 2],
                                 style_kwargs=dict(color='blue',
                                                   label=name1))
    spec_title1 = axes[0, 2].get_title()
    bg_cube_model2.plot_spectrum(coord=Angle([0., 0.], 'degree'),
                                 ax=axes[0, 2],
                                 style_kwargs=dict(color='red',
                                                   label=name2))
    spec_title2 = axes[0, 2].get_title()
    if spec_title1 != spec_title2:
        s_error = "Expected same det binning, but got "
        s_error += "\"{0}\" and \"{1}\"".format(spec_title1, spec_title2)
        raise ValueError(s_error)
    else:
        axes[0, 2].set_title(spec_title1)

    # plot normalized models on top
    plot_data_x, normed_pl = get_normed_pl(plot_data=axes[0, 2].get_lines()[0],
                                           E_0=E_REF, norm=NORM, index=INDEX)
    axes[0, 2].plot(plot_data_x, normed_pl, color='blue',
                    linestyle='dotted', linewidth=2,
                    label='model index = {}'.format(INDEX))
    plot_data_x, normed_pl = get_normed_pl(plot_data=axes[0, 2].get_lines()[0],
                                           E_0=E_REF, norm=NORM, index=INDEX + 1)
    axes[0, 2].plot(plot_data_x, normed_pl, color='blue',
                    linestyle='dashed', linewidth=2,
                    label='model index = {}'.format(INDEX + 1))

    axes[0, 2].legend()

    bg_cube_model1.plot_spectrum(coord=Angle([2., 2.], 'degree'),
                                 ax=axes[1, 2],
                                 style_kwargs=dict(color='blue',
                                                   label=name1))
    spec_title1 = axes[1, 2].get_title()
    bg_cube_model2.plot_spectrum(coord=Angle([2., 2.], 'degree'),
                                 ax=axes[1, 2],
                                 style_kwargs=dict(color='red',
                                                   label=name2))
    spec_title2 = axes[1, 2].get_title()
    if spec_title1 != spec_title2:
        s_error = "Expected same det binning, but got "
        s_error += "\"{0}\" and \"{1}\"".format(spec_title1, spec_title2)
        raise ValueError(s_error)
    else:
        axes[1, 2].set_title(spec_title1)

    # plot normalized models on top
    plot_data_x, normed_pl = get_normed_pl(plot_data=axes[1, 2].get_lines()[0],
                                           E_0=E_REF, norm=NORM, index=INDEX)
    axes[1, 2].plot(plot_data_x, normed_pl, color='blue',
                    linestyle='dotted', linewidth=2,
                    label='model index = {}'.format(INDEX))
    plot_data_x, normed_pl = get_normed_pl(plot_data=axes[1, 2].get_lines()[0],
                                           E_0=E_REF, norm=NORM, index=INDEX + 1)
    axes[1, 2].plot(plot_data_x, normed_pl, color='blue',
                    linestyle='dashed', linewidth=2,
                    label='model index = {}'.format(INDEX + 1))

    axes[1, 2].legend()

    plt.show()


if __name__ == '__main__':
    """Main function: define arguments and launch the whole analysis chain.
    """
    input_file1 = gammapy_extra.filename('test_datasets/background/bg_cube_model_true.fits.gz')
    name1 = 'true'

    input_file2 = gammapy_extra.filename('test_datasets/background/bg_cube_model_reco.fits.gz')
    name2 = 'reco'

    plot_bg_cube_model_comparison(input_file1, name1,
                                  input_file2, name2)
