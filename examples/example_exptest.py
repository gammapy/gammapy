#  An example how to compute exptest for multiple runs.
from gammapy.time import exptest
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import norm
import numpy as np





def exptest_multi(table):
    """
    Parameter
    ----------
     table : astropy.table as the input

     plot : This function returns a histogram for the distribution of the Mr value of each run.
            A normal distribution is expected for non-flaring sources.
    """
    m_value = []
    size = len(table['runnum'])
    individual_run = []
    num_of_events = []
    time_delta_all = []

    for i in range(0, size - 1):
        if table['runnum'][i] != table['runnum'][i + 1]:
            individual_run.append(table['runnum'][i])
    individual_run.append(table['runnum'][size - 1])
    num_of_runs = len(individual_run)

    for i in range(0, len(individual_run)):
        num_of_events_temp = 0
        for j in range(0, size):
            if individual_run[i] == table['runnum'][j]:
                num_of_events_temp += 1
        num_of_events.append(num_of_events_temp)

    for i in range(0, num_of_runs):
        time_delta_each_run = []
        for j in range(0, size - 1):
            if num_of_events[i] > 20 and individual_run[i] == table[
                    'runnum'][j] and table['runnum'][j] == table['runnum'][j + 1]:
                time_delta_each_run.append(
                    (table['mjd'][
                        j + 1] - table['mjd'][j]) * 0.5 * (
                        table['expCount'][
                            j + 1] + table['expCount'][j]))
                time_delta_all.append(
                    (table['mjd'][
                        j + 1] - table['mjd'][j]) * 0.5 * (
                        table['expCount'][
                            j + 1] + table['expCount'][j]))
        if len(time_delta_each_run) == 0:
            continue
        m = exptest(time_delta_each_run)
        m_value.append(m)
        print("m value: ", m, "   ", individual_run[i])
        del time_delta_each_run[:]
    overallm = exptest(time_delta_all)
    print("Mr for the whole dataset: ", overallm)
    return m_value


def plot(m_value):
    (mu, sigma) = norm.fit(m_value)
    n, bins, patches = plt.hist(
        m_value, bins=30, normed=1, facecolor='green', alpha=0.75)
    plt.mlab.normpdf(bins, mu, sigma)
    print("mu:{:10.3f}".format(mu), " sigma:{:10.4f}".format(sigma))
    plt.xlabel('Mr value')
    plt.ylabel('counts')
    plt.title(
        r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' %
        (mu, sigma))
    plt.grid(True)
    plt.show()


def simulation(number_of_runs):
    """

    """
    table = Table()
    num_of_events_per_run = np.random.random_integers(20, 30, number_of_runs)
    size_cumsum = np.cumsum(num_of_events_per_run)
    size = size_cumsum[number_of_runs - 1]
    runnum = []
    for i in range(0, number_of_runs):
        for j in range(0, num_of_events_per_run[i]):
            runnum.append(i + 1)
    table['runnum'] = runnum
    mjd_random = np.random.uniform(0, size, size)
    mjd = np.sort(mjd_random)
    table['mjd'] = mjd
    for i in range(0, size):
        expcount = 1
    table['expCount'] = expcount

    return table


if __name__ == '__main__':
    table = simulation(100)
    m_value = exptest_multi(table)
    plot(m_value)
