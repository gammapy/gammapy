.. _tutorials-background:

Background Estimation
=====================

Here we introduce a background estimation method based on significance clipping.

Significance Clipping
---------------------
TODO: Add a link to the proceeding, and summarise here the method & intro from
the proceeding.

The algorithm is demonstrated in the example below, where it is applied to
5 years of Fermi-LAT counts data in the Galactic Plane, in line with the proceeding study.
4 iterations are shown here with parameters selected so as to exaggerate the action of the
algorithm.

.. plot:: tutorials/background/source_diffuse_estimation.py
	:include-source:

* The images on the **left** show the background estimation with each iteration.
* The images on the **right** show the residual significance image with each iteration.
* The **contours** show the exclusion mask applied at each iteration.

The source mask is shown by the contours. This includes the regions
excluded above the 5 sigma significance threshold (determined by the Li & Ma method [LiMa1983]_)
in computing the background estimation images above.
