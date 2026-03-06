.. include:: ../../references.txt

.. _lstat_derivation:

Derivation of the LStat formula
--------------------------------

The LStat statistic uses a Bayesian approach to marginalize over the background
parameter, which is different from the profile likelihood method in WStat.

Bayesian Background Marginalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Following [Loredo1992]_, instead of using profile likelihood to estimate
:math:`\mu_{\mathrm{bkg}}`, we integrate it out using Bayesian inference.

We start from the same likelihood as WStat:

.. math::
    L (n_{\mathrm{on}}, n_{\mathrm{off}}, \alpha; \mu_{\mathrm{sig}},
    \mu_{\mathrm{bkg}}) = \frac{(\mu_{\mathrm{sig}}+
    \mu_{\mathrm{bkg}})^{n_{\mathrm{on}}}}{n_{\mathrm{on}} !}
    \exp{(-(\mu_{\mathrm{sig}}+ \mu_{\mathrm{bkg}}))}\times
    \frac{(\mu_{\mathrm{bkg}}/\alpha)^{n_{\mathrm{off}}}}{n_{\mathrm{off}}
    !}\exp{(-\mu_{\mathrm{bkg}}/\alpha)},

Using a uniform (flat) prior on :math:`\mu_{\mathrm{bkg}}`, the marginal
likelihood is computed as:

.. math::
    L_{\mathrm{marg}}(n_{\mathrm{on}}, n_{\mathrm{off}}, \alpha; \mu_{\mathrm{sig}}) =
    \int_0^\infty L(n_{\mathrm{on}}, n_{\mathrm{off}}, \alpha; \mu_{\mathrm{sig}},
    \mu_{\mathrm{bkg}}) d\mu_{\mathrm{bkg}}

This integral has an analytical solution:

.. math::
    L_{\mathrm{marg}} = \frac{\Gamma(n_{\mathrm{on}} + n_{\mathrm{off}} + 1)}{\Gamma(n_{\mathrm{on}} + 1) \Gamma(n_{\mathrm{off}} + 1)} \times
    \frac{\alpha^{n_{\mathrm{on}}}}{(1 + \alpha)^{n_{\mathrm{on}} + n_{\mathrm{off}} + 1}} \times
    \frac{1}{[\mu_{\mathrm{sig}} + (n_{\mathrm{on}} + n_{\mathrm{off}})/(1 + \alpha)]^{n_{\mathrm{on}} + n_{\mathrm{off}} + 1}}

Final LStat Formula
^^^^^^^^^^^^^^^^^^^

Taking -2 times the log of the marginal likelihood and simplifying:

.. math::
    L = 2 \big[-\mu_{\mathrm{sig}} + (n_{\mathrm{on}} + n_{\mathrm{off}} + 1) \log(1 + \frac{\alpha \mu_{\mathrm{sig}}}{n_{\mathrm{on}} + n_{\mathrm{off}}}) - n_{\mathrm{on}} \log(\alpha)\big]

This is what's implemented in XSpec and in `~gammapy.stats.lstat`.

Comparing with WStat
^^^^^^^^^^^^^^^^^^^^

The key difference between LStat and WStat comes down to how they treat the background:

- **WStat** uses profile likelihood: it finds the value of :math:`\mu_{\mathrm{bkg}}` that maximizes the likelihood
- **LStat** uses Bayesian marginalization: it integrates over all possible values of :math:`\mu_{\mathrm{bkg}}` with a flat prior

This leads to some practical differences:

- LStat generally gives larger (more conservative) uncertainties
- LStat doesn't require finding the ML estimate of the background
- Both converge to similar results when you have good statistics
- The difference can be significant in low-count regimes

References
^^^^^^^^^^

* `Loredo (1992) <https://ui.adsabs.harvard.edu/abs/1992scma.conf..275L/abstract>`_
* `Knoetig (2014) <https://iopscience.iop.org/article/10.1088/0004-637X/790/2/106#apj497435s5>`_
* `D'Amico et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022Univ....8...90D/abstract>`_
* `XSpec manual <https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSappendixStatistics.html>`_
