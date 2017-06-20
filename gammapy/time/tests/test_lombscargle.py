import numpy as np
from lombscargle_gammapy import lomb_scargle, plotting

def test_lombscargle():
    rand = np.random.RandomState(42)
    t = np.linspace(0, 100, 1000)
    t_obs = np.sort(rand.choice(t, 500, replace=False))
    n_outliers = 50
    dmag = np.random.normal(0, 1, 1000) * -1**(rand.randint(2, size=1000))
    dmag_obs = dmag[np.searchsorted(t, t_obs)]
    outliers = rand.randint(0, t.size, n_outliers)
    amplitude = 2
    mag = amplitude * np.sin(2 * np.pi * t / 5) + dmag
    for n in range(n_outliers):
        mask = (t >= outliers[n])
        mag[mask] = mag[mask] + 10 * amplitude * np.exp(-1 * (t[mask] - outliers[n]))
    mag_obs = mag[np.searchsorted(t, t_obs)]
    K = 4 # oversampling factor
    FAP = 0.05 # false alarm probability
    N_bootstraps = 100 # number of bootstrap resamplings
    freq, PLS, best_period, quant_pre, quant_cvm, quant_nll, quant_boot, PLS_win = lomb_scargle(t_obs, mag_obs, dmag_obs, K, N_bootstraps, FAP)
    plotting(t, mag, dmag, freq, PLS, best_period, quant_pre, quant_cvm, quant_nll, quant_boot, N_bootstraps, PLS_win)
    print('Best period: ' + str(best_period))
    assert np.array([freq, PLS, best_period, quant_pre, quant_cvm, quant_nll, quant_boot, PLS_win]) == True
