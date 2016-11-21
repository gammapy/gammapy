"""This script calculated WStat using different implemented methods.

It's purpose is to aid the decision which is the 'correct' WStat to be used.
The sherpa values are furthermore used as reference values for
gammapy/stats/test/test_fit_statistics"""
import numpy as np
from gammapy.stats.tests.test_fit_statistics import test_data


def test_data_wstat():
    test_data_wstat = test_data()
    test_data_wstat.pop('staterror')
    return test_data_wstat


def calc_wstat_sherpa():
    import sherpa.stats as ss
    wstat = ss.WStat()

    # This is how sherpa want the background
    # We assume equal exposure
    bkg = dict(bkg=test_data_wstat()['n_off'],
               exposure_time=[1, 1],
               backscale_ratio=1. / test_data_wstat()['alpha'],
               data_size=len(test_data_wstat()['n_on'])
               )

    stat = wstat.calc_stat(test_data_wstat()['n_on'],
                           test_data_wstat()['mu_sig'],
                           staterror=test_data()['staterror'],
                           extra_args=bkg)
    print("Sherpa stat: {}".format(stat[0]))
    print("Sherpa fvec: {}".format(stat[1]))


def calc_wstat_gammapy():
    from gammapy.stats import wstat
    from gammapy.stats.fit_statistics import (
        _get_wstat_background,
        _get_wstat_extra_terms,
    )

    # background estimate
    bkg = _get_wstat_background(**test_data_wstat())
    print("Gammapy mu_bkg: {}".format(bkg))

    statsvec = wstat(**test_data_wstat())

    print("Gammapy stat: {}".format(np.sum(statsvec)))
    print("Gammapy statsvec: {}".format(statsvec))

    print("---> with extra terms")
    extra_terms = _get_wstat_extra_terms(test_data_wstat()['n_on'],
                                         test_data_wstat()['n_off'])
    print("Gammapy extra terms: {}".format(extra_terms))

    statsvec = wstat(extra_terms=True,
                     **test_data_wstat())

    print("Gammapy stat: {}".format(np.sum(statsvec)))
    print("Gammapy statsvec: {}".format(statsvec))


def calc_wstat_xspec():
    data, model, staterr, off_vec, alpha = get_test_data()
    from xspec_stats import xspec_wstat as wstat
    from xspec_stats import xspec_wstat_f, xspec_wstat_d

    # alpha = t_s / t_b
    t_b = 1. / alpha
    t_s = 1

    d = xspec_wstat_d(t_s, t_b, model, data, off_vec)
    f = xspec_wstat_f(data, off_vec, t_s, t_b, model, d)
    bkg = f * t_b
    stat = wstat(t_s, t_b, model, data, off_vec)

    print("XSPEC mu_bkg (f * t_b): {}".format(bkg))
    print("XSPEC stat: {}".format(stat))

if __name__ == "__main__":
    td = test_data()
    print("Test data")
    print("n_on: {}".format(td['n_on']))
    print("n_off: {}".format(td['n_off']))
    print("alpha: {}".format(td['alpha']))
    print("n_pred: {}".format(td['mu_sig']))
    print("\n")
    calc_wstat_sherpa()
    print("\n")
    calc_wstat_gammapy()
    print("\n")
    # calc_wstat_xspec()
