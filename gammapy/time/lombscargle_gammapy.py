# from lombscargle
from astropy.stats import LombScargle
# from PyAstronomy.pyTiming.pyPeriod import Gls
# from PyAstronomy.pyTiming import pyPeriod
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
from matplotlib import rcParams
plt.style.use('ggplot')
# import pylab
from scipy.stats import beta
from scipy import optimize

__all__ = [
    'lombscargle',
    'plotting',
]

def _nll(param,*args):
    """
    Negative log likelihood function for beta
    <param>: list for parameters to be fitted.
    <args>: 1-element array containing the sample data.    
    Return <nll>: negative log-likelihood to be minimized.
    """
    a, b = param
    data = args[0]
    pdf = beta.pdf(data, a, b, loc=0, scale=1)
    lg = np.log(pdf)
    mask = np.isfinite(lg)
    nll = -lg[mask].sum()
    return nll

def _cvm(param, *args):
    """Cramer-von-Mises distance minimization
    """
    a, b = param
    data = args[0]
    ordered_data = np.sort(data, axis=None)
    for n in range(len(data)):
        sumbeta =+ (beta.cdf(ordered_data[n], a, b, loc=0, scale=1) - (n - 0.5) / len(data))**2.0
    cvm_dist = (1. / len(data)) * sumbeta + 1. / (12 * (len(data)**2.))
    mask = np.isfinite(cvm_dist)
    cvm = cvm_dist[mask]
    return cvm

def _window_function(t):
    least_exponent = str(t[0])[::-1].find('.')
    dt = 10**-least_exponent
    t_length = (np.max(t) - np.min(t)) / dt
    t_win = np.linspace(dt, np.around(np.max(t), decimals=abs(least_exponent+1)), t_length, endpoint=True)
    window = np.zeros(len(t_win))
    window[np.searchsorted(t_win, np.around(t, decimals=abs(least_exponent+1)))] = 1

    return t_win, window

def _bootstrap(t, mag, dmag, freq, N_bootstraps):
    rand = np.random.RandomState(42)
    FAP = 0.05
    N_bootstraps = 100
    max_periods = np.empty(N_bootstraps)
    for m in range(N_bootstraps):
        ind = rand.randint(0, len(mag), len(mag))
        PLS_boot = LombScargle(t, mag[ind], dmag[ind]).power(freq)
        max_periods[m] = np.max(PLS_boot)

    return max_periods

def _grid(t, K):
    max_period = (np.max(t) - np.min(t)) / 10 # see Halpern
    min_period = 2 * np.median(t[1:-1] - t[0:-2])    
    d_p = min_period / K    
    n_periods = max_period / d_p
    periods = np.linspace(min_period, max_period, n_periods)
    grid = 1. / periods

    return grid

def _significance(t, mag, dmag, freq, PLS, FAP, N_bootstraps):
    percentile = 1 - FAP
    # compute pre-defined Beta-Distribution
    a = (3-1)/2
    b = (len(t)-3)/2
    quant_pre = beta.ppf(percentile**(1./len(freq)), a, b)
    
    # compute CvM-minimization-fitted Beta-Distribution
    # initial guess for CvM-minimization
    theta_1 = -1. * (np.mean(PLS) * (-np.mean(PLS) + np.mean(PLS)**2 + np.var(PLS))) / (np.var(PLS))
    if theta_1 < 0:
        theta_1 = 0.00001
    theta_2 = (theta_1 - theta_1 * np.mean(PLS)) / np.mean(PLS)
    if theta_2 < 0:
        theta_2 = 0.00001
    x0 = [theta_1, theta_2]
    cvm_minimize = optimize.fmin(_cvm, [theta_1, theta_2], args=(PLS,))
    quant_cvm = beta.ppf(percentile**(1./len(freq)), cvm_minimize[0], cvm_minimize[1])
    
    # compute NLL-minimization-fitted Beta-Distribution
    nll_minimize = optimize.fmin(_nll, [a, b], args=(PLS,))
    quant_nll = beta.ppf(percentile**(1./len(freq)), nll_minimize[0], nll_minimize[1])

    # bootstrap
    max_periods = _bootstrap(t, mag, dmag, freq, N_bootstraps)
    sig = np.percentile(max_periods, 100*percentile, axis=0)

    return quant_pre, quant_cvm, quant_nll, sig

def lomb_scargle(t, mag, dmag, K, N_bootstraps, FAP):
    """This function processes the lomb-scargle-algorithmus in autopower mode
    """
    ### set up lomb-scargle-algorithm    
    freq = _grid(t, K)
    PLS = LombScargle(t, mag, dmag).power(freq)
    # spectral window function
    t_win, window = _window_function(t)
    PLS_win = LombScargle(t_win, window, 1).power(freq)
    
    ### significance criteria
    quant_pre, quant_cvm, quant_nll, quant_boot = _significance(t, mag, dmag, freq, PLS, FAP, N_bootstraps)    
    # find significant periods
    best_period_bar = 0
    best_period = 0
    for f in range(len(freq)):
        if PLS[f] > quant_boot:#     here a significance criteria has to be chosen
            if PLS[f] > best_period_bar:
                best_period_bar = PLS[f]
                best_period = 1. / freq[f]
    if best_period == 0:
        best_period = np.nan
    
    return freq, PLS, best_period, quant_pre, quant_cvm, quant_nll, quant_boot, PLS_win

def plotting(t, mag, dmag, freq, PLS, best_period, quant_pre, quant_cvm, quant_nll, quant_boot, N_bootstraps, PLS_win):
    ### plotting
    # set up the figure & axes for plotting
    fig = plt.figure(figsize=(16, 9))
    # rcParams['axes.labelsize'] = 16
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif':['Computer Modern']})
    gs = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    # plot the raw data
    ax1.errorbar(t, mag, dmag, fmt='ok', elinewidth=1.5, capsize=0)
    ax1.set(xlabel=r'\textbf{time} (d)',
            ylabel=r'\textbf{magnitude} (a.u.)')
    # plot the periodogram
    ax2.plot(1. / freq, PLS)
    ax2.plot((1. / freq[0], 1. / freq[-1]), [quant_pre, quant_pre], label='pre-defined beta-distribution')
    ax2.plot((1. / freq[0], 1. / freq[-1]), [quant_cvm, quant_cvm], label='cvm-fitted beta-distribution')
    ax2.plot((1. / freq[0], 1. / freq[-1]), [quant_nll, quant_nll], label='nll-fitted beta-distribution')
    ax2.plot((1. / freq[0], 1. / freq[-1]), [quant_boot, quant_boot], ':', label='{} bootstraps'.format(N_bootstraps))
    ax2.set(#xlabel=r'\textbf{period} (d)'
            ylabel=r'\textbf{power}',
            xlim=(0, np.max(1. / freq)),
            ylim=(0, 1)
            );
    ax2.legend(loc='upper right')
    # plot the window function
    ax3.plot(1. / freq, PLS_win)
    ax3.set(xlabel=r'\textbf{period} (d)'
              , ylabel=r'\textbf{power}'
              , xlim=(0, np.max(1. / freq))
              # , ylim=(0, 1)
              );    
    # save plot
    plt.tight_layout()
    plt.savefig('Lomb-Scargle periodogram', bbox_inches='tight')
