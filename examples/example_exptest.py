"""
An example for the exptest variability test.

- Simulate constant rate events for several observations.
- Check that the ``mr`` distribution is a standard normal, as expected.
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from astropy.table import Table
from gammapy.time import exptest


def simulation(n_obs):
    """Simulate time series data.

    Produce table with one row per event.
    - obs_id -- Observation ID
    - mjd -- event time
      - mjd is filled randomly between 0 and 1
    - expCount -- expected counts, aka acceptance
      - expcount is filled with 1
    """
    table = Table()

    # For every observation, decide somewhat randomly how many events to simulate
    n_event_per_obs = np.random.random_integers(20, 30, n_obs)
    n_events = n_event_per_obs.sum()

    table["obs_id"] = np.repeat(np.arange(len(n_event_per_obs)), n_event_per_obs)

    mjd_random = np.random.uniform(0, 1, n_events)
    mjd = np.sort(mjd_random)
    table["mjd"] = mjd

    table["expCount"] = 1

    return table


def exptest_multi(table_events):
    """Compute mr value for each run and whole dataset.

    Parameter
    ---------
    table : `astropy.table.Table`
        Input data

    Returns
    -------
    mr_list : list
        List of `mr` values.
    """
    # Make table with one row per observation
    table_obs = Table()
    table_obs["obs_id"] = np.unique(table_events["obs_id"])
    res = table_events.group_by("obs_id")
    n_events = np.array(res.groups.aggregate(sum)["expCount"])
    table_obs["n_events"] = n_events

    mr_list = []

    time_delta_all = []

    for i in range(0, len(table_obs)):
        time_delta_each_run = []

        for j in range(0, len(table_events) - 1):
            if (
                table_obs["n_events"][i] > 20
                and table_obs["obs_id"][i] == table_events["obs_id"][j]
                and table_events["obs_id"][j] == table_events["obs_id"][j + 1]
            ):
                time_delta = (table_events["mjd"][j + 1] - table_events["mjd"][j]) / 2
                time_delta *= (
                    table_events["expCount"][j + 1] + table_events["expCount"][j]
                )
                time_delta_each_run.append(time_delta)
                time_delta_all.append(time_delta)

        if len(time_delta_each_run) == 0:
            continue
        mr = exptest(time_delta_each_run)
        mr_list.append(mr)
        print("mr value: ", mr, "   ", table_obs["obs_id"][i])
        del time_delta_each_run[:]

    overallm = exptest(time_delta_all)
    print("Mr for the whole dataset: ", overallm)

    return mr_list


def plot(m_value):
    """Plot histogram of mr value for each run.

    A normal distribution is expected for non-flaring sources.
    """
    (mu, sigma) = norm.fit(m_value)
    n, bins, patches = plt.hist(
        m_value, bins=30, normed=1, facecolor="green", alpha=0.75
    )
    plt.mlab.normpdf(bins, mu, sigma)
    print("mu:{:10.3f}".format(mu), " sigma:{:10.4f}".format(sigma))
    plt.xlim(-3, 3)
    plt.xlabel("Mr value")
    plt.ylabel("counts")
    title = "Histogram of IQ: mu={:.3f}, sigma={:.3f}".format(mu, sigma)
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    table = simulation(100)
    m_value = exptest_multi(table)
    plot(m_value)
