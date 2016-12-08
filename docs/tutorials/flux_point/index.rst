.. _tutorials-flux_point:

Where to stick your Spectral Points?
====================================

The `gammapy.spectrum` module offers a number of options for positioning data points within  an energy band. This example offers a comparison between
the log center and Lafferty & Wyatt (described in [Lafferty1994]_) methods. See `gammapy.spectrum.compute_differential_flux_points` for documentation on usage.

Lafferty & Wyatt vs. Log Center
-------------------------------

In measurements of spectra which do not vary rapidly, it is usually sufficient to represent the data within a bin using the log bin center.
However, it is sometimes the case where the region of interest lies within a low frequency regime such that
large bin widths must be chosen to ensure statistical errors remain small. In cases where the spectrum varies rapidly (for
instance, exponentially) over wide bins, simply choosing the log bin center does not offer a good representation of the true underlying spectrum.

Instead, Lafferty & Wyatt [Lafferty1994]_ offer an alternative approach for such situations. In a bin of width :math:`\Delta E` between bounds
:math:`E_1` and :math:`E_2` for energy :math:`E`, the expectation :math:`<g_{meas}>` of the true underlying spectrum, :math:`g(E)` is defined as

.. math::
    <g_{meas}> = \frac{1}{\Delta E}\int_{E_1}^{E_2}{g(E) dE}

As the bin size tends to zero, the expectation of the spectrum tends to it's true value. The value of :math:`E` within a bin for
which the expectation should be regarded as a measurement of the true spectrum is determined by Lafferty & Wyatt as the energy at
which the expectation value is equal to the mean value of the underlying true spectrum within that bin, noted as :math:`E_{lw}`. Thus knowledge of the true spectrum
:math:`g(E)` or an estimate for this (determined by fitting) is required.

So it follows that, in setting expectation equal to :math:`g(E)` at :math:`E_{lw}`, the position of :math:`E_{lw}` is given by the following equation:

.. math::
    E_{lw} = g^{-1}\left(\frac{1}{\Delta E}\int_{E_1}^{E_2}{g(E) dE}\right)

For instances where a power law of spectral index 2 is taken, it can be analytically shown that the Lafferty & Wyatt method and log center method are
coincident. In the case of steeper power laws (e.g. spectral index 3), the Lafferty & Wyatt method
returns a lower coordinate on the energy axis than the log bin center, and the reverse effect is seen for less steep power laws (e.g. spectral index 1).

Power Law Assumption
--------------------

In many "real world" examples, the nature of the true underlying spectrum is unknown and can only be estimated, either by using a
fitting algorithm with data points or by assuming a certain spectral form. In this example, the true spectrum (being a piecewise power law
function of spectral indices 1 through to 5 for bins of increasing energy) is shown in blue for illustration. This would be
unknown to the user who, in this example, assumes the spectrum follows a simple power law with spectral index 4 across all energies.

The plot demonstrates that in these cases where the true spectrum is not known, the Lafferty & Wyatt method of positioning the data
points offers a closer representation than the log center method. Residuals showing the percentage difference of each data point from the true
spectrum are also shown.

.. plot:: tutorials/flux_point/flux_point_demo.py
   :include-source:

Method Evaluation
-----------------

The dependence of the positioning of the x coordinate of the flux point with assumed spectral index is demonstrated below.
As before, the underlying spectrum is a simple power law of index 4. The position is given as the fraction of the size of
the bin in Log Energy, for instance the Log Center method always positions the flux point at the center of the bin for a
Log Energy scale, and so the position is 0.5. The coincidence of the two methods for the power law case of index 2 is evident,
with the Lafferty method positioning the flux point to the right of the Log Center position for power law spectra of index
less than 2, and to the left for index greater than 2.

.. plot:: tutorials/flux_point/x_position_plot.py
   :include-source:

The ability of the two methods to represent an underlying spectrum is demonstrated in the residual images below. Here,
flux points are positioned using the Lafferty method (left image) and Log Center method (central image) for different
'true' and 'assumed' power law spectral indices. Residuals are determined as the difference between the reconstructed flux
compared to the true flux of the underlying spectrum. The closer the residual value is to zero (white), the better the
representation offered.

The residuals log ratio image (right) shows the log ratio of residuals of the Lafferty method to those of the Log Center method,
indicating the regions of assumed/true parameter space for which the Lafferty method offers an improved representation
of the underlying frequency over the Log Center method (blue regions), and where it does not (red regions).

.. plot:: tutorials/flux_point/residuals_images.py
   :include-source:

Caveats
-------

It should be noted that in some cases (for instance, a power law with spectral index lower than 2), the Lafferty & Wyatt approach
does not offer a better representation of the underlying spectrum compared to simply taking the log bin center. Indeed, the
improvements that are seen in the Lafferty & Wyatt approach over using the bin Log Center occurs due to Lafferty & Wyatt
simply positioning the flux point further to the left of the bin. Thus there is reason to consider further methods, namely:

* Positioning the flux point such that the integral flux on either side of it within the energy bin are equal.

* Simply choosing a position towards the left of the bin, for instance at a position 0.2 bin lengths from the left of the bin in log energy.
