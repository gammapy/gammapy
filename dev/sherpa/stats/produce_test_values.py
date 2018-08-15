"""This script calculates test statistic values using sherpa

These values are used as reference values for
gammapy/stats/test/test_fit_statistics"""

import numpy as np
import sys
from gammapy.stats.tests.test_fit_statistics import test_data


def get_data(mu_sig, n_on, n_off, alpha):
    from sherpa.astro.data import DataPHA
    from sherpa.models import Const1D

    model = Const1D()
    model.c0 = mu_sig
    data = DataPHA(
        counts=np.atleast_1d(n_on),
        name="dummy",
        channel=np.atleast_1d(1),
        backscal=1,
        exposure=1,
    )
    background = DataPHA(
        counts=np.atleast_1d(n_off),
        name="dummy background",
        channel=np.atleast_1d(1),
        backscal=np.atleast_1d(1. / alpha),
        exposure=1,
    )

    data.set_background(background, 1)

    return model, data


def test_case(args):
    import sherpa.stats as ss

    mu_sig = float(args[1])
    n_on = float(args[2])
    n_off = float(args[3])
    alpha = float(args[4])

    model, data = get_data(mu_sig, n_on, n_off, alpha)
    wstat, cash, cstat = ss.WStat(), ss.Cash(), ss.CStat()

    print('wstat: {}'.format(wstat.calc_stat(model=model, data=data)[0]))
    print('cash: {}'.format(cash.calc_stat(model=model, data=data)[0]))
    print('cstat: {}'.format(cstat.calc_stat(model=model, data=data)[0]))


if __name__ == "__main__":

    td = test_data()
    for mu_sig, n_on, n_off, alpha in zip(
        td["mu_sig"], td["n_on"], td["n_off"], td["alpha"]
    ):
        test_case([None, mu_sig, n_on, n_off, alpha])
        print("\n-------------------------\n")
