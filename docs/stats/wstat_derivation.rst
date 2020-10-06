.. include:: ../references.txt

.. _wstat_derivation:

Derivation of the WStat formula
-------------------------------

you can write down the likelihood formula as

.. math::
    L (n_{\mathrm{on}}, n_{\mathrm{off}}, \alpha; \mu_{\mathrm{sig}},
    \mu_{\mathrm{bkg}}) = \frac{(\mu_{\mathrm{sig}}+
    \mu_{\mathrm{bkg}})^{n_{\mathrm{on}}}}{n_{\mathrm{on}} !}
    \exp{(-(\mu_{\mathrm{sig}}+ \mu_{\mathrm{bkg}}))}\times
    \frac{(\mu_{\mathrm{bkg}}/\alpha)^{n_{\mathrm{off}}}}{n_{\mathrm{off}}
    !}\exp{(-\mu_{\mathrm{bkg}}/\alpha)},

where :math:`\mu_{\mathrm{sig}}` and :math:`\mu_{\mathrm{bkg}}` are respectively
the number of expected signal and background counts in the ON region,
as defined in the :ref:`stats-introduction`. By taking two
time the negative log likelihood and neglecting model independent and thus
constant terms, we define the **WStat**.

.. math::
    W = 2 \big(\mu_{\mathrm{sig}} + (1 + 1/\alpha)\mu_{\mathrm{bkg}}
    - n_{\mathrm{on}} \log{(\mu_{\mathrm{sig}} + \mu_{\mathrm{bkg}})}
    - n_{\mathrm{off}} \log{(\mu_{\mathrm{bkg}}/\alpha)}\big)

In the most general case, where :math:`\mu_{\mathrm{src}}` and
:math:`\mu_{\mathrm{bkg}}` are free the minimum of :math:`W` is at

.. math::
    \mu_{\mathrm{sig}} = n_{\mathrm{on}} - \alpha\,n_{\mathrm{off}} \\
    \mu_{\mathrm{bkg}} = \alpha\,n_{\mathrm{off}}


Profile Likelihood
^^^^^^^^^^^^^^^^^^

Most of the times you probably won't have a model in order to get
:math:`\mu_{\mathrm{bkg}}`. The strategy in this case is to treat
:math:`\mu_{\mathrm{bkg}}` as so-called nuisance parameter, i.e. a free
parameter that is of no physical interest.  Of course you don't want an
additional free parameter for each bin during a fit. Therefore one calculates an
estimator for :math:`\mu_{\mathrm{bkg}}` by analytically minimizing the
likelihood function. This is called 'profile likelihood'.

.. math::
    \frac{\mathrm d \log L}{\mathrm d \mu_{\mathrm{bkg}}} = 0

This yields a quadratic equation for :math:`\mu_{\mathrm{bkg}}`

.. math::
    \frac{\alpha\,n_{\mathrm{on}}}{\mu_{\mathrm{sig}}+\alpha
    \mu_{\mathrm{bkg}}} + \frac{n_{\mathrm{off}}}{\mu_{\mathrm{bkg}}} - (\alpha
    + 1) = 0

with the solution

.. math::
    \mu_{\mathrm{bkg}} = \frac{C + D}{2\alpha(\alpha + 1)}

where

.. math::
    C = \alpha(n_{\mathrm{on}} + n_{\mathrm{off}}) - (\alpha+1)\mu_{\mathrm{sig}} \\
    D^2 = C^2 + 4 (\alpha+1)\alpha n_{\mathrm{off}} \mu_{\mathrm{sig}}

Goodness of fit
^^^^^^^^^^^^^^^

The best-fit value of the WStat as defined now contains no information about the
goodness of the fit. We consider the likelihood of the data
:math:`n_{\mathrm{on}}` and :math:`n_{\mathrm{off}}` under the expectation of
:math:`n_{\mathrm{on}}` and :math:`n_{\mathrm{off}}`,

.. math::
    L (n_{\mathrm{on}}, n_{\mathrm{off}}; n_{\mathrm{on}}, n_{\mathrm{off}}) =
    \frac{n_{\mathrm{on}}^{n_{\mathrm{on}}}}{n_{\mathrm{on}} !}
    \exp{(-n_{\mathrm{on}})}\times
    \frac{n_{\mathrm{off}}^{n_{\mathrm{off}}}}{n_{\mathrm{off}} !}
    \exp{(-n_{\mathrm{off}})}

and add twice the log likelihood

.. math::
     2 \log L (n_{\mathrm{on}}, n_{\mathrm{off}}; n_{\mathrm{on}},
     n_{\mathrm{off}}) = 2 (n_{\mathrm{on}} ( \log{(n_{\mathrm{on}})} - 1 ) +
     n_{\mathrm{off}} ( \log{(n_{\mathrm{off}})} - 1))

to WStat. In doing so, we are computing the likelihood ratio:

.. math::
    -2 \log \frac{L(n_{\mathrm{on}},n_{\mathrm{off}},\alpha;
    \mu_{\mathrm{sig}},\mu_{\mathrm{bkg}})}
    {L(n_{\mathrm{on}},n_{\mathrm{off}};n_{\mathrm{on}},n_{\mathrm{off}})}

Intuitively, this log-likelihood ratio should asymptotically behave like a
chi-square with ``m-n`` degrees of freedom, where ``m`` is the number of
measurements and ``n`` the number of model parameters.

Final result
^^^^^^^^^^^^

.. math::
    W = 2 \big(\mu_{\mathrm{sig}} + (1 + \alpha)\mu_{\mathrm{bkg}} -
    n_{\mathrm{on}} - n_{\mathrm{off}} - n_{\mathrm{on}}
    (\log{(\mu_{\mathrm{sig}} + \alpha \mu_{\mathrm{bkg}}) -
    \log{(n_{\mathrm{on}})}}) - n_{\mathrm{off}} (\log{(\mu_{\mathrm{bkg}})} -
    \log{(n_{\mathrm{off}})})\big)

Special cases
^^^^^^^^^^^^^

The above formula is undefined if :math:`n_{\mathrm{on}}` or
:math:`n_{\mathrm{off}}` are equal to zero, because of the :math:`n\log{{n}}`
terms, that were introduced by adding the goodness of fit terms. These cases are
treated as follows.

If :math:`n_{\mathrm{on}} = 0` the likelihood formulae read

.. math::
    L (0, n_{\mathrm{off}}, \alpha; \mu_{\mathrm{sig}}, \mu_{\mathrm{bkg}}) =
    \exp{(-(\mu_{\mathrm{sig}}+\alpha \mu_{\mathrm{bkg}}))}\times
    \frac{(\mu_{\mathrm{bkg}})^{n_{\mathrm{off}}}}{n_{\mathrm{off}}
    !}\exp{(-\mu_{\mathrm{bkg}})},

and

.. math::
    L (0, n_{\mathrm{off}}; 0, n_{\mathrm{off}}) =
    \frac{n_{\mathrm{off}}^{n_{\mathrm{off}}}}{n_{\mathrm{off}} !}
    \exp{(-n_{\mathrm{off}})}

WStat is derived by taking 2 times the negative log likelihood and adding the
goodness of fit term as ever

.. math::
    W = 2 \big(\mu_{\mathrm{sig}} + (1 + \alpha)\mu_{\mathrm{bkg}} -
    n_{\mathrm{off}} - n_{\mathrm{off}} (\log{(\mu_{\mathrm{bkg}})} -
    \log{(n_{\mathrm{off}})})\big)

Note that this is the limit of the original Wstat formula for
:math:`n_{\mathrm{on}} \rightarrow 0`.

The analytical result for
:math:`\mu_{\mathrm{bkg}}` in this case reads:

.. math::
    \mu_{\mathrm{bkg}} = \frac{n_{\mathrm{off}}}{\alpha + 1}

When inserting this into the WStat we find the simplified expression.

.. math::
    W = 2\big(\mu_{\mathrm{sig}} + n_{\mathrm{off}} \log{(1 + \alpha)}\big)

If :math:`n_{\mathrm{off}} = 0` Wstat becomes

.. math::
    W = 2 \big(\mu_{\mathrm{sig}} + (1 + \alpha)\mu_{\mathrm{bkg}} -
    n_{\mathrm{on}} - n_{\mathrm{on}} (\log{(\mu_{\mathrm{sig}} + \alpha
    \mu_{\mathrm{bkg}}) - \log{(n_{\mathrm{on}})}})

and

.. math::
    \mu_{\mathrm{bkg}} = \frac{n_{\mathrm{on}}}{1+\alpha} -
    \frac{\mu_{\mathrm{sig}}}{\alpha}

For :math:`\mu_{\mathrm{sig}} > n_{\mathrm{on}} (\frac{\alpha}{1 + \alpha})`,
:math:`\mu_{\mathrm{bkg}}` becomes negative which is unphysical.

Therefore we distinct two cases. The physical one where

:math:`\mu_{\mathrm{sig}} < n_{\mathrm{on}} (\frac{\alpha}{1 + \alpha})`.

is straightforward and gives

.. math::
    W = -2\big(\mu_{\mathrm{sig}} \left(\frac{1}{\alpha}\right) +
    n_{\mathrm{on}} \log{\left(\frac{\alpha}{1 + \alpha}\right)\big)}

For the unphysical case, we set :math:`\mu_{\mathrm{bkg}}=0` and arrive at

.. math::
    W = 2\big(\mu_{\mathrm{sig}} + n_{\mathrm{on}}(\log{(n_{\mathrm{on}})} -
    \log{(\mu_{\mathrm{sig}})} - 1)\big)

