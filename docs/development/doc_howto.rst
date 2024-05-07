.. include:: ../references.txt

.. _doc_howto:

********************
Documentation How To
********************

Documentation building
----------------------

Generating the HTML docs for Gammapy is straight-forward::

    make docs-sphinx
    make docs-show

Or one can equivalently use tox::

     tox -e build_docs

Generating the PDF docs is more complex.
This should work::

    # build the latex file
    cd docs
    python -m sphinx . _build/latex -b latex -j auto
    # first generation of pdf file
    cd _build/latex
    pdflatex -interaction=nonstopmode gammapy.tex
    # final generation of pdf file
    pdflatex -interaction=nonstopmode gammapy.tex
    # clean the git repo
    git reset --hard
    # open the pdf file
    open gammapy.pdf

You need a bunch or LaTeX stuff, specifically ``texlive-fonts-extra`` is needed.

Check Python code
-----------------

Code in RST files
+++++++++++++++++

Most of the documentation of Gammapy is present in RST files that are converted into HTML pages using
Sphinx during the build documentation process. You may include snippets of Python code in these RST files
within blocks labelled with ``.. code-block:: python`` Sphinx directive. However, this code could not be
tested, and it will not be possible to know if it fails in following versions of Gammapy. That's why we
recommend using the ``.. testcode::`` directive to enclose code that will be tested against the results
present in a block labelled with ``.. testoutput::`` directive. If not ``.. testoutput::`` directive is provided,
only execution tests will be performed.

For example, we could check that the code below does not fail, since it does not provide any output.

.. code-block:: text

    .. testcode::

        from gammapy.astro import source
        from gammapy.astro import population
        from gammapy.astro import darkmatter

On the contrary, we could check the execution of the following code as well as the output values produced.

.. code-block:: text

    .. testcode::

        from astropy.time import Time
        time = Time(['1999-01-01T00:00:00.123456789', '2010-01-01T00:00:00'])
        print(time.mjd)

    .. testoutput::

        [51179.00000143 55197.        ]

In order to perform tests of these snippets of code present in RST files, you may run the following command.

.. code-block:: bash

    pytest --doctest-glob="*.rst" docs/

Code in docstrings in Python files
++++++++++++++++++++++++++++++++++

It is also advisable to add code snippets within the docstrings of the classes and functions present in Python files.
These snippets show how to use the function or class that is documented, and are written in the docstrings using the
following syntax.

.. code-block:: text

        Examples
        --------
        >>> from astropy.units import Quantity
        >>> from gammapy.data import EventList
        >>> event_list = EventList.read('events.fits') # doctest: +SKIP

In the case above, we could check the execution of the first two lines importing the ``Quantity`` and ``EventList``
modules, whilst the third line will be skipped. On the contrary, in the example below we could check the execution of
the code as well as the output value produced.

.. code-block:: text

        Examples
        --------
        >>> from regions import Regions
        >>> regions = Regions.parse("galactic;circle(10,20,3)", format="ds9")
        >>> print(regions[0])
        Region: CircleSkyRegion
        center: <SkyCoord (Galactic): (l, b) in deg
            (10., 20.)>
        radius: 3.0 deg

To allow the code block to be placed correctly over multiple lines utilise the "...":

.. code-block:: text

        Examples
        --------
        >>> from gammapy.maps import WcsGeom, MapAxis
        >>> energy_axis_true = MapAxis.from_energy_bounds(
        ...            0.5, 20, 10, unit="TeV", name="energy_true"
        ...        )


For a larger code block it is also possible to utilise the following syntax.

.. code-block:: text

        Examples
        --------
        .. testcode::

            from gammapy.maps import MapAxis
            from gammapy.irf import EnergyDispersion2D
            filename = '$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz'
            edisp2d = EnergyDispersion2D.read(filename, hdu="EDISP")


In order to perform tests of these snippets of code present in the docstrings of the Python files, you may run the
following command.

.. code-block:: bash

    pytest --doctest-modules --ignore-glob=*/tests gammapy

If you get a zsh error try using putting to ignore block inside quotes 

.. code-block:: bash

    pytest --doctest-modules "--ignore-glob=*/tests" gammapy

Sphinx gallery extension
------------------------

The documentation built-in process uses the `sphinx-gallery <https://sphinx-gallery.github.io/stable/>`__
extension to build galleries of illustrated examples on how to use Gammapy (i.e.
:ref:`model-gallery`). The Python scripts used to produce the model gallery are placed in
``examples/models`` and ``examples/tutorials``. The configuration of the ``sphinx-gallery`` module is done in ``docs/conf.py``.


Choose a thumbnail and tooltip for the tutorial gallery
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

The Gammapy :ref:`tutorials` are Python scripts in the Sphinx Gallery format.
They are displayed as a gallery with picture thumbnails and tooltips. You can
choose the thumbnail for the tutorial by adding a comment before the plot:

.. code-block:: python

    # The next line sets the thumbnail for the second figure in the gallery
    # (plot with negative exponential in orange)
    # sphinx_gallery_thumbnail_number = 2
    plt.figure()
    plt.plot(x, -np.exp(-x), color='orange', linewidth=4)
    plt.xlabel('$x$')
    plt.ylabel(r'$-\exp(-x)$')
    # To avoid matplotlib text output
    plt.show()

The example is taken from the `sphinx-gallery documentation <https://sphinx-gallery.github.io/stable/auto_examples/plot_4_choose_thumbnail.html>`__,
please refer to it for more details. 

The tooltip is the text that appears when you hover over the thumbnail. It is taken from the first line 
of the docstring of the tutorial. You can change it by editing the docstring. See e.g.
`Analysis 1 Tutorial <https://github.com/gammapy/gammapy/blob/main/examples/tutorials/starting/analysis_1.py#L5>`__.


Dealing with links
------------------

Links in tutorials are just handled via normal RST syntax.

Links to other tutorials
++++++++++++++++++++++++

From docstrings and RST documentation files in Gammapy you can link to other tutorials 
and gallery examples by using RST syntax like this:

.. code-block:: rst

    :doc:`/tutorials/starting/analysis_2`

This will link to the tutorial :doc:`/tutorials/starting/analysis_2` from the tutorial base folder. The file
suffix will be automatically inferred by Sphinx.


API Links
+++++++++

Links to Gammapy API are handled via normal Sphinx syntax in comments:

.. code-block:: python

   # Create an `~gammapy.analysis.AnalysisConfig` object and edit it to
   # define the analysis configuration:

   config = AnalysisConfig()

This will resolve to a link to the ``AnalysisConfig`` class in the API documentation.

.. _dev-check_html_links:

Check broken links
++++++++++++++++++

To check for broken external links you can use ``tox``:

.. code-block:: bash

   $ tox -e linkcheck

Include png files as images
----------------------------

In the RST files
++++++++++++++++

Gammapy has a ``gp-image`` directive to include an image from ``$GAMMAPY_DATA/figures/``,
use the ``gp-image`` directive instead of the usual Sphinx ``image`` directive like this:

.. code-block:: rst

    .. gp-image:: detect/fermi_ts_image.png
        :scale: 100%

More info on the `image directive <http://www.sphinx-doc.org/en/stable/rest.html#images>`__.

Documentation guidelines
------------------------

Like almost all Python projects, the Gammapy documentation is written in a format called
`restructured text (RST)`_ and built using `Sphinx`_.
We mostly follow the :ref:`Astropy documentation guidelines <astropy:documentation-guidelines>`,
which are based on the `Numpy docstring standard`_,
which is what most scientific Python packages use.

.. _restructured text (RST): http://sphinx-doc.org/rest.html
.. _Sphinx: http://sphinx-doc.org/
.. _Numpy docstring standard: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

There's a few details that are not easy to figure out by browsing the Numpy or Astropy
documentation guidelines, or that we actually do differently in Gammapy.
These are listed here so that Gammapy developers have a reference.

Usually the quickest way to figure out how something should be done is to browse the Astropy
or Gammapy code a bit (either locally with your editor or online on GitHub or via the HTML docs),
or search the Numpy or Astropy documentation guidelines mentioned above.
If that doesn't quickly turn up something useful, please ask by putting a comment on the issue or
pull request you're working on GitHub, or email the Gammapy mailing list.

Functions or class methods that return a single object
++++++++++++++++++++++++++++++++++++++++++++++++++++++

For functions or class methods that return a single object, following the
Numpy docstring standard and adding a *Returns* section usually means
that you duplicate the one-line description and repeat the function name as
return variable name.
See `~astropy.cosmology.LambdaCDM.w` or `~astropy.time.Time.sidereal_time`
as examples in the Astropy codebase. Here's a simple example:

.. testcode::

    def circle_area(radius):
        """Circle area.

        Parameters
        ----------
        radius : `~astropy.units.Quantity`
            Circle radius

        Returns
        -------
        area : `~astropy.units.Quantity`
            Circle area
        """
        return 3.14 * (radius ** 2)

In these cases, the following shorter format omitting the *Returns* section is recommended:

.. testcode::

    def circle_area(radius):
        """Circle area (`~astropy.units.Quantity`).

        Parameters
        ----------
        radius : `~astropy.units.Quantity`
            Circle radius
        """
        return 3.14 * (radius ** 2)

Usually the parameter description doesn't fit on the one line, so it's
recommended to always keep this in the *Parameters* section.

A common case where the short format is appropriate are class properties,
because they always return a single object.
As an example see `~gammapy.data.EventList.radec`, which is reproduced here:

.. testcode::

    @property
    def radec(self):
        """Event RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`)."""
        lon, lat = self['RA'], self['DEC']
        return SkyCoord(lon, lat, unit='deg', frame='icrs')


Class attributes
++++++++++++++++

Class attributes (data members) and properties are currently a bit of a mess.
Attributes are listed in an *Attributes* section because I've listed them in a class-level
docstring attributes section as recommended
`here <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`__.
Properties are listed in separate *Attributes summary* and *Attributes Documentation*
sections, which is confusing to users ("what's the difference between attributes and properties?").

One solution is to always use properties, but that can get very verbose if we have to write
so many getters and setters. We could start using descriptors.

TODO: make a decision on this and describe the issue / solution here.
