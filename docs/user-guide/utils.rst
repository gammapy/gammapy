.. include:: ../references.txt

.. _utils:

Utility functions
=================

``gammapy.utils`` is a collection of utility functions that are used in many
places or don't fit in one of the other packages.

Since the various sub-modules of ``gammapy.utils`` are mostly unrelated, they
are not imported into the top-level namespace. Here are some examples of how to
import functionality from the ``gammapy.utils`` sub-modules:

.. testcode::

    from gammapy.utils.random import sample_sphere
    sample_sphere(size=10)

    from gammapy.utils import random
    random.sample_sphere(size=10)

.. _time_handling:

Time handling in Gammapy
------------------------

See `gammapy.utils.time`.

Time format and scale
+++++++++++++++++++++
In Gammapy, `astropy.time.Time` objects are used to represent times:

.. testcode::

    from astropy.time import Time
    Time(['1999-01-01T00:00:00.123456789', '2010-01-01T00:00:00'])

Note that Astropy chose ``format='isot'`` and ``scale='utc'`` as default and in
Gammapy these are also the recommended format and time scale.

.. warning::

   Actually what's written here is not true. In CTA it hasn't been decided if
   times will be in ``utc`` or ``tt`` (terrestrial time) format.

   Here's a reminder that this needs to be settled / updated:
   https://github.com/gammapy/gammapy/issues/284


When other time formats are needed it's easy to convert, see the :ref:`time
format section and table in the Astropy docs <astropy:time-format>`.

E.g. sometimes in astronomy the modified Julian date ``mjd`` is used and for
passing times to matplotlib for plotting the ``plot_date`` format should be
used:

.. testcode::

    from astropy.time import Time
    time = Time(['1999-01-01T00:00:00.123456789', '2010-01-01T00:00:00'])
    print(time.mjd)
    print(time.plot_date)

.. testoutput::

    [51179.00000143 55197.        ]
    [10592.00000143 14610.        ]


Converting to other time scales is also easy, see the :ref:`time scale section,
diagram and table in the Astropy docs <astropy:time-scale>`.

E.g. when converting celestial (RA/DEC) to horizontal (ALT/AZ) coordinates, the
`sidereal time <https://en.wikipedia.org/wiki/Sidereal_time>`__ is needed. This
is done automatically by `astropy.coordinates.AltAz` when the
`astropy.coordinates.AltAz.obstime` is set with a `~astropy.time.Time` object in
any scale, no need for explicit time scale transformations in Gammapy (although
if you do want to explicitly compute it, it's easy, see `here
<https://docs.astropy.org/en/latest/api/astropy.time.Time.html#astropy.time.Time.sidereal_time>`__).

The `Fermi-LAT time systems in a nutshell`_ page gives a good, brief explanation
of the differences between the relevant time scales ``UT1``, ``UTC`` and ``TT``.

.. _MET_definition:

Mission elapsed times (MET)
+++++++++++++++++++++++++++

:term:`MET` time references are times representing UTC seconds after a specific
origin. Each experiment may have a different MET origin that should be included
in the header of the corresponding data files. For more details see `Fermi-LAT
time systems in a nutshell`_.

It's not clear yet how to best implement METs in Gammapy, it's one of the tasks
here: https://github.com/gammapy/gammapy/issues/284

For now, we use the `gammapy.utils.time.time_ref_from_dict`,
`gammapy.utils.time.time_relative_to_ref` and `gammapy.utils.time.absolute_time` functions
to convert MET floats to `~astropy.time.Time` objects via the reference times
stored in FITS headers.

Time differences
++++++++++++++++

TODO: discuss when to use `~astropy.time.TimeDelta` or `~astropy.units.Quantity`
or :term:`MET` floats and where one needs to convert between those and what to watch
out for.
