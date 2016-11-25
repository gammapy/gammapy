"""This script calculated WStat using different implemented methods.

It's purpose is to aid the decision which is the 'correct' WStat to be used.
The sherpa values are furthermore used as reference values for
gammapy/stats/test/test_fit_statistics"""
import numpy as np
import sys
from gammapy.stats.tests.test_fit_statistics import test_data


def calc_wstat_sherpa(mu_sig, n_on, n_off, alpha):
    import sherpa.stats as ss
    from sherpa.astro.data import DataPHA
    from sherpa.models import Const1D
    wstat = ss.WStat()

    model = Const1D()
    model.c0 = mu_sig
    data = DataPHA(counts=np.atleast_1d(n_on),
                   name='dummy',
                   channel=np.atleast_1d(1),
                   backscal=1,
                   exposure=1)
    background = DataPHA(counts=np.atleast_1d(n_off),
                         name='dummy background',
                         channel=np.atleast_1d(1),
                         backscal=np.atleast_1d(1. / alpha),
                         exposure=1)

    data.set_background(background, 1)

    # Docstring for ``calc_stat``
    # https://github.com/sherpa/sherpa/blob/fe8508818662346cb6d9050ba676e23318e747dd/sherpa/stats/__init__.py#L219

    stat = wstat.calc_stat(model=model, data=data)
    print("Sherpa stat: {}".format(stat[0]))
    print("Sherpa fvec: {}".format(stat[1]))



def calc_wstat_gammapy(mu_sig, n_on, n_off, alpha):
    from gammapy.stats import wstat, get_wstat_mu_bkg, get_wstat_gof_terms

    # background estimate
    bkg = get_wstat_mu_bkg(mu_sig=mu_sig, n_on=n_on, n_off=n_off, alpha=alpha)
    print("Gammapy mu_bkg: {}".format(bkg))

#    # without extra terms
#    statsvec = wstat(extra_terms=False, mu_sig=mu_sig, n_on=n_on, n_off=n_off,
#                     alpha = alpha)
#
#    print("Gammapy stat: {}".format(np.sum(statsvec)))
#    print("Gammapy statsvec: {}".format(statsvec))
#
#    print("---> with extra terms")
#    extra_terms = _get_wstat_extra_terms(n_on = n_on,
#                                         n_off = n_off)
#    print("Gammapy extra terms: {}".format(extra_terms))

    statsvec = wstat(extra_terms=True, mu_sig=mu_sig, n_on=n_on, n_off=n_off,
                     alpha=alpha)

    print("Gammapy stat: {}".format(np.sum(statsvec)))
    print("Gammapy statsvec: {}".format(statsvec))


def calc_wstat_xspec(mu_sig, n_on, n_off, alpha):
    from xspec_stats import xspec_wstat as wstat
    from xspec_stats import xspec_wstat_f, xspec_wstat_d

    td = test_data()

    # alpha = t_s / t_b
    t_b = 1. / alpha
    t_s = 1

    d = xspec_wstat_d(t_s, t_b, mu_sig, n_on, n_off)
    f = xspec_wstat_f(n_on, n_off, t_s, t_b, mu_sig, d)
    bkg = f * t_b
    stat = wstat(t_s, t_b, mu_sig, n_on, n_off)

    print("XSPEC mu_bkg (f * t_b): {}".format(bkg))
    print("XSPEC stat: {}".format(stat))


def test_case(args):
    mu_sig = float(args[1])
    n_on = float(args[2])
    n_off = float(args[3])
    alpha = float(args[4])

    print("mu_sig: {}".format(mu_sig))
    print("n_on: {}".format(n_on))
    print("n_off: {}".format(n_off))
    print("alpha: {}".format(alpha))
    print("\n")

    calc_wstat_sherpa(mu_sig, n_on, n_off, alpha)
    print("\n")
    calc_wstat_gammapy(mu_sig, n_on, n_off, alpha)
    print("\n")
    calc_wstat_xspec(mu_sig, n_on, n_off, alpha)

if __name__ == "__main__":

    if (len(sys.argv) != 5) & (len(sys.argv) != 2):
        print('Usage: {} <mu_sig> <n_on> <n_off> <alpha>'.format(sys.argv[0]))
        print('Usage: {} ref-vals-for-test'.format(sys.argv[0]))
        sys.exit()

    if len(sys.argv) == 5:
        test_case(sys.argv)
    else:
        td = test_data()
        for mu_sig, n_on, n_off, alpha in zip(
                td['mu_sig'], td['n_on'], td['n_off'], td['alpha']):
            test_case([None, mu_sig, n_on, n_off, alpha])
            print('\n-------------------------\n')
