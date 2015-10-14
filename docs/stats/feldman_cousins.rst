.. _feldman_cousins:

Feldman and Cousins Confidence Intervals
========================================
Feldman and Cousins solved the problem on how to choose confidence intervals
in a unified way (that is without basing the choice on the data)
[Feldman1998]_. The functions ``gammapy.stats.fc_*`` give you access to a
numerical solution to the Feldman Cousins algorithm. It can be used for any type
of statistical distribution and is not limited to a Poisson process with
background or a Gaussian with a boundary at the origin.

The basic ingredient to `~gammapy.stats.fc_construct_acceptance_intervals_pdfs` is a matrix of
P(X|mu) (see e.g. equation (3.1) and (3.2) in [Feldman1998]_). Every row is a
probability density function (PDF) of x and the columns are built up by varying
the signal strength mu. The other parameter is the desired confidence level
(C.L.). The function will return another matrix of acceptance intervals where 1
means the point is part of the acceptance interval and 0 means it is outside.
This can be easily converted to upper and lower limits (`~gammapy.stats.fc_get_upper_and_lower_limit`),
which simply connect the outside 1s for different mus. An upper or lower limit
is obtained by drawing a vertical line at the measured value and calculating the
intersection (`~gammapy.stats.fc_find_limit`).

Examples
--------
The first plot reproduces Fig. 7 from [Feldman1998]_ (Poisson process with
background 3.0). It should be noted that the plot in the paper is inconsistent
with Table IV from the same paper: the lower limit is off by one bin to the
left.

.. plot:: stats/plot_fc_poisson.py

This plot reproduces Fig. 10 from [Feldman1998]_ (Gaussian with a boundary at the
origin).

.. plot:: stats/plot_fc_gauss.py

Acceptance Interval Fixing
--------------------------
Feldman and Cousins point out that sometimes the set of intersected horizontal
lines is not simply connected. These artefacts are corrected with
`~gammapy.stats.fc_fix_upper_and_lower_limit`.

The following script in the ``examples`` directory demonstrates the problem:
:download:`fc_demonstrate_artefact.py <../../examples/fc_demonstrate_artefact.py>`

For mu = 0.745 the 90% acceptance interval is [0,8] and for mu = 0.750 it is
[1,8]. A lot of the fast algorithms that do not compute the full confidence belt
will come to the conclusion that the 90% confidence interval is [0, 0.745] and
thus the upper limit when zero is measured should be 0.745 (one example is
TFeldmanCousins that comes with `ROOT`, but is has the additional bug of making
the confidence interval one mu bin to big, thus reporting 0.75 as upper limit).

For mu = 1.035 the 90% acceptance interval is [0,8] again and only starting
mu = 1.060 will 0 no longer be in the 90% acceptance interval. Thus the correct
upper limit according to the procedure described in [Feldman1998]_ should be
1.055, which is also the value given in the paper (rounded to 1.06).

General Case
------------
In the more general case, one may not know the underlying PDF of P(X|mu). One
way would be to generate P(X|mu) from Monte Carlo simulation. With a dictionary
of mu values and lists of X values from Monte Carlo one can use `~gammapy.stats.fc_construct_acceptance_intervals`
to construct the confidence belts.

Sensitivity
-----------
[Feldman1998]_ also defines experimental sensitivity as the average upper limit
that would be obtained by an ensemble of experiments with the expected
background and no true signal. It can be calculated using `~gammapy.stats.fc_find_average_upper_limit`.

Verification
------------
To verify that the numerical solution is working, the example plots can also be
produced using the analytical solution. They look consistent. The scripts for
the analytical solution are given in the ``examples`` directory:
:download:`fc_poisson_analytical.py <../../examples/fc_poisson_analytical.py>`
:download:`fc_gauss_analytical.py <../../examples/fc_gauss_analytical.py>`
