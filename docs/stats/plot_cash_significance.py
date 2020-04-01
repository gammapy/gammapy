"""Example plot showing the profile of the Cash statistic and its connection to significance."""
import numpy as np
import matplotlib.pyplot as plt
from gammapy.stats import CashCountsStatistic

count_statistic = CashCountsStatistic(n_on=13, mu_bkg=5.5)
excess = count_statistic.excess

# We compute the Cash statistic profile
mu_signal = np.linspace(-1.5, 25, 100)
stat_values = count_statistic._stat_fcn(mu_signal)

xmin, xmax = -1.5, 25
ymin, ymax = -42, -28.0
plt.figure(figsize=(5, 5))
plt.plot(mu_signal, stat_values, color="k")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.xlabel(r"Number of expected signal event, $\mu_{sig}$")
plt.ylabel(r"Cash statistic value, TS ")
plt.vlines(
    excess,
    ymin=ymin,
    ymax=count_statistic.TS_max,
    linestyle="dashed",
    color="k",
    label="Best fit",
)
plt.hlines(
    count_statistic.TS_max, xmin=xmin, xmax=excess, linestyle="dashed", color="k"
)
plt.hlines(
    count_statistic.TS_null,
    xmin=xmin,
    xmax=0,
    linestyle="dotted",
    color="k",
    label="Null hypothesis",
)
plt.vlines(0, ymin=ymin, ymax=count_statistic.TS_null, linestyle="dotted", color="k")

plt.vlines(excess, ymin=count_statistic.TS_max, ymax=count_statistic.TS_null, color="r")
plt.hlines(count_statistic.TS_null, xmin=0, xmax=excess, linestyle="dotted", color="r")
plt.legend()
