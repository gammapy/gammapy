"""This script calculated WStat using different implemented methods. It's
purpose is to aid the decision which is the 'correct' WStat to be used"""
import numpy as np
from gammapy.utils.random import get_random_state

def get_test_data():
    n_bins = 1
    random_state = get_random_state(3)
    model = random_state.rand(n_bins) * 10
    data = random_state.poisson(model)
    staterror = np.sqrt(data) 
    off_vec = random_state.poisson(0.7 * model)
    alpha = np.array([0.2] * len(model))
    return data, model, staterror, off_vec, alpha

def calc_wstat_sherpa():
    data, model, staterr, off_vec, alpha = get_test_data()
    import sherpa.stats as ss
    wstat = ss.WStat()
    
    # We assume equal exposure
    bkg = dict(bkg = off_vec,
               exposure_time=[1, 1],
               backscale_ratio=1./alpha,
               data_size=len(model) 
              )

    stat = wstat.calc_stat(data, model, staterror=staterr, bkg=bkg)
    print("Sherpa stat: {}".format(stat[0]))
    print("Sherpa fvec: {}".format(stat[1]))

def calc_wstat_gammapy():
    data, model, staterr, off_vec, alpha = get_test_data()
    from gammapy.stats import wstat
    statsvec = wstat(n_on=data,
                     mu_signal=model,
                     n_off=off_vec,
                     alpha=alpha)

    print("Gammapy stat: {}".format(np.sum(statsvec)))
    print("Gammapy statsvec: {}".format(statsvec))

def calc_wstat_xspec():
    data, model, staterr, off_vec, alpha = get_test_data()
    from xspec_stats import xspec_wstat as wstat
    
    # Assume t_b = alpha * t_s
    # I think this is what Sherpa does
    t_b = 1. / alpha
    t_s = 1
    stat = wstat(t_s, t_b, model, data, off_vec)

    print("XSPEC stat: {}".format(stat))


if __name__ == "__main__":
    data, model, staterr, off_vec, alpha = get_test_data()
    print("Test data")
    print("n_on: {}".format(data))
    print("n_off: {}".format(off_vec))
    print("alpha: {}".format(alpha))
    print("n_pred: {}".format(model))
    print("\n")
    calc_wstat_sherpa()
    print("\n")
    calc_wstat_gammapy()
    print("\n")
    calc_wstat_xspec()
