"""Example plot showing the profile of the WStat statistic and its connection to excess errors."""
import numpy as np
import matplotlib.pyplot as plt
from gammapy.stats import WStatCountsStatistic

count_statistic = WStatCountsStatistic(n_on=13, n_off=11, alpha=0.5)
excess = count_statistic.n_sig

errn = count_statistic.compute_errn(1.0)
errp = count_statistic.compute_errp(1.0)

errn_2sigma = count_statistic.compute_errn(2.0)
errp_2sigma = count_statistic.compute_errp(2.0)

# We compute the WStat statistic profile
mu_signal = np.linspace(-2.5, 26, 100)
stat_values = count_statistic._stat_fcn(mu_signal)

xmin, xmax = -2.5, 26
ymin, ymax = 0, 15
plt.figure(figsize=(5, 5))
plt.plot(mu_signal, stat_values, color="k")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.xlabel(r"Number of expected signal event, $\mu_{sig}$")
plt.ylabel(r"WStat value, TS ")

plt.hlines(
    count_statistic.stat_max + 1,
    xmin=excess - errn,
    xmax=excess + errp,
    linestyle="dotted",
    color="r",
    label="1 sigma (68% C.L.)",
)
plt.vlines(
    excess - errn,
    ymin=ymin,
    ymax=count_statistic.stat_max + 1,
    linestyle="dotted",
    color="r",
)
plt.vlines(
    excess + errp,
    ymin=ymin,
    ymax=count_statistic.stat_max + 1,
    linestyle="dotted",
    color="r",
)

plt.hlines(
    count_statistic.stat_max + 4,
    xmin=excess - errn_2sigma,
    xmax=excess + errp_2sigma,
    linestyle="dashed",
    color="b",
    label="2 sigma (95% C.L.)",
)
plt.vlines(
    excess - errn_2sigma,
    ymin=ymin,
    ymax=count_statistic.stat_max + 4,
    linestyle="dashed",
    color="b",
)
plt.vlines(
    excess + errp_2sigma,
    ymin=ymin,
    ymax=count_statistic.stat_max + 4,
    linestyle="dashed",
    color="b",
)

plt.legend()
