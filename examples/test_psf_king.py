import numpy as np
from astropy.coordinates import Angle
from gammapy.irf import PSFKing
from gammapy.background import EnergyOffsetArray


def load_psf(chain='hd', tool='gammapy'):
    """Load a test PSF."""
    if chain == 'hd':
        filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2-input/run023400-023599/run023523/hess_psf_king_023523.fits.gz'
        if tool == 'gammalib':
            filename += '[PSF_2D_KING]'
    elif chain == 'pa':
        filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-pa/run23400-23599/run23523/psf_king_23523.fits.gz'
    else:
        raise ValueError('Invalid chain: {}'.format(chain))

    print('Reading {}'.format(filename))

    if tool == 'gammapy':
        return PSFKing.read(filename)
    elif tool == 'gammalib':
        from gammalib import GCTAPsfKing
        return GCTAPsfKing(filename)
    else:
        raise ValueError('Invalid tool: {}'.format(tool))


def test_psf_king():
    psf = load_psf(chain='hd')
    print('HD PSF')
    print(psf.info())

    psf = load_psf(chain='pa')
    print('PA PSF')
    print(psf.info())


def containment_radius(psf, fraction, energy, offset):
    """Compute containment radius with Gammalib.

    http://cta.irap.omp.eu/gammalib/doxygen/classGCTAPsfKing.html#ffa8ff988290ccc0927ea3e73ca639e2
    """
    radius = np.empty((len(energy), len(offset)))
    for ii in range(len(energy)):
        for jj in range(len(offset)):
            logE = float(np.log10(energy.value[ii]))
            theta = float(offset.radian[jj])
            val = psf.containment_radius(float(fraction), logE, theta)
            # print(ii, jj, fraction, logE, theta, val)
            radius[ii, jj] = val

    return Angle(radius.T, 'radian').to('deg')


def plot_psf_king(chain='hd'):
    psf = load_psf(chain=chain)
    gamma = EnergyOffsetArray(energy=psf.energy, offset=psf.offset,
                              data=psf.gamma, data_units=psf.gamma.unit)
    sigma = EnergyOffsetArray(energy=psf.energy, offset=psf.offset,
                              data=psf.sigma, data_units=psf.sigma.unit)

    # For now, let's use Gammalib to compute containment radii
    gpsf = load_psf(chain=chain, tool='gammalib')
    r68 = containment_radius(gpsf, 0.68, psf.energy, psf.offset)
    r68 = EnergyOffsetArray(energy=psf.energy, offset=psf.offset,
                            data=r68, data_units='deg')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    gamma.plot_image(ax=ax[0])
    sigma.plot_image(ax=ax[1])
    r68.plot_image(ax=ax[2], vmin=0, vmax=0.3)
    # import IPython; IPython.embed()
    # ax[2].colorbar()
    fig.tight_layout()
    # fig.show()
    filename = 'test_psf_king_{}.png'.format(chain)
    print('Writing {}'.format(filename))
    fig.savefig(filename)
    # input()


if __name__ == '__main__':
    # test_psf_king()
    plot_psf_king(chain='hd')
    plot_psf_king(chain='pa')
