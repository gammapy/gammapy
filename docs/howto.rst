.. include:: references.txt

.. _howto:

How To
======

TODO: add short entries explaining how to do something specific in Gammapy.

Probably each HOWTO should be a short section with just 1-2 sentences and links to tutorials and API docs,
or if it should be small mini-tutorials with code snippets, possibly on sub-pages.

See docs PIG: https://github.com/gammapy/gammapy/pull/2463

How to compute the significance of a source?
--------------------------------------------

Estimate the significance of a source, or more generally of an additional model component
(such as e.g. a spectral line on top of a power-law spectrum), is done via a hypothesis test.
You fit two models, with and without the extra source or component, then use the test statistic
values from both fits to compute the significance or p-value.

TODO: update this entry once https://github.com/gammapy/gammapy/issues/2149
and https://github.com/gammapy/gammapy/issues/1540 are resolved, linking to the documentation
developed there.

How to use Gammapy with astrophysical modeling packages?
--------------------------------------------------------

It is possible to combine Gammapy with astrophysical modeling codes, if they provide a Python interface.
Usually this requires some glue code to be written, e.g. `~gammapy.modeling.models.NaimaSpectralModel` is
an example of a Gammapy wrapper class around the Naima spectral model and radiation classes, which then
allows modeling and fitting of Naima models within Gammapy (e.g. using CTA, H.E.S.S. or Fermi-LAT data).

TOOD: give more and better examples.

Other Ideas
-----------

Below some examples what "How to" entries could be, taken
from https://github.com/gammapy/gammapy/pull/2463#issuecomment-544126183

See also https://github.com/gammapy/gammapy/pull/2463#issuecomment-545309352
for links to examples what other projects put as HOWTO or FAQ.

Modeling
++++++++

- How to test the spectral and/or spatial model?
- How to calculate lower/upper limits for a spectral parameter?

SED
+++

- How to calculate integral/differential flux and upper limits?
- How to calculate spectral points and residuals?
- How to plot the SED with errors?

Source Detection
++++++++++++++++

- How to build and display the on region?
- How to get the significance?
- How to get excess and its error?
- How to get background counts?

2D Morphology
+++++++++++++

- How to define/get position and spatial dimensions at different energy thresholds?
- How to calculate surface brightness or radial profile in within a specific mask/region?
- How to calculate a spectrum within a specific mask/region?
- How to overlay significance and excess on maps?

Light Curves
++++++++++++

- How to do analysis of light curves and upper limits