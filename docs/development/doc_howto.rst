.. include:: ../references.txt

.. _doc_howto:

********************
Documentation How To
********************

Documentation building
----------------------

Generating the HTML docs for Gammapy is straight-forward::

    make docs-all
    make docs-show

Generating the PDF docs is more complex.
This should work::

    # build the notebooks
    python -m gammapy.utils.notebooks_process
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

Jupyter notebooks present in ``docs/tutorials`` folder have stripped output cells.
Although notebooks are code clean formatted, tested, and filled during the process of documentation
building, where they are also converted to Sphinx formatted HTML files and ``.py`` scripts, 
**you must always use stripped and clean formatted notebooks in your pull requests**.
See :ref:`common-taks-notebooks` for the commands used for these tasks.

The Sphinx formatted versions of the notebooks provide links to the raw ``.ipynb`` Jupyter
files and ``.py`` script versions stored in ``docs/_static/notebooks`` folder, as well as
a link pointing to its specific Binder space in the
`gammapy-webpage <https://github.com/gammapy/gammapy-webpage>`__ repository (not for
the deve version of the docs). Since notebooks are evolving with Gammapy features and documentation, 
the different versions of the notebooks are linked to versioned Binder environments.

Once the documentation is built you can optimize the speed of eventual re-building,
for example in case you are modifying or adding new text, and you would like to check
these changes are displayed nicely. For that purpose, you may run ``make docs-sphinx`` so
that notebooks are not executed during the docs build.

In the case one single notebook is modified or added to the documentation, you can
execute the build doc process with the ``src`` parameter with value the name of the
considered notebook. i.e. ``make docs-all src=docs/tutorials/my-notebook.ipynb``

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
``examples/models`` and the configuration of the ``sphinx-gallery`` module is done in ``docs/conf.py``.

Working with notebooks
----------------------

.. _common-taks-notebooks:

Common tasks
++++++++++++

    * test with tutorials env: ``gammapy jupyter --src mynotebook.ipynb test --tutor``
    * strip the output cells: ``gammapy jupyter --src mynotebook.ipynb strip``
    * clean format code cells: ``gammapy jupyter --src mynotebook.ipynb  black``
    * diff stripped notebooks: ``git diff mynotbook.pynb``
  

Add a notebook into a folder other than tutorials folder
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Most of the Gammapy notebooks are placed in the ``tutorials`` folder, and are displayed in a
:ref:`tutorials` Gallery. However, we can choose to place a notebook in a different folder of the
documentation folder structure. In this way we can write some parts of the documentation as notebooks
instead of RST files. Once we have placed the notebook in the folder we choose we can link it from the
``index.rst`` file using the name of the notebook filename **without the extension** and the Sphinx
``toctree`` directive as shown below.

.. code-block:: text

    .. toctree::

        mynotebook


.. _skip-nb-execution:

Skip notebooks from being executed
++++++++++++++++++++++++++++++++++

You may choose if a notebook is not executed during the documentation building process, and hence
it will be published without the output cells in its static HTML version. To do this you may add
the following code to the notebook metadata:

.. code-block:: javascript

  "gammapy": {
    "skip_run": true
  }

Choose a thumbnail and tooltip for the tutorial gallery
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

The Gammapy :ref:`tutorials` are Jupyter notebooks that are displayed as a gallery with picture thumbnails and tooltips.
You can choose the thumbnail for the tutorial and add the tooltip editing the metadata of the code cell that produces
the picture that you've chosen. You can open the notebook in a text editor, and edit the internal code there. It may
sound risky, but it is much simpler. Then, find the code cell that produces the figure that you would like for the
gallery, and then replace the ``"metadata": {},`` bit above the code cell with the snippet below:

.. code-block:: javascript

    "metadata": {
     "nbsphinx-thumbnail": {
      "tooltip": "Learn how to do perform a Fit in gammapy."
     }},

Note that you may write whatever you like after "tooltip".

Dealing with links
------------------

All Jupyter notebooks in Gammpay documentation are converted to HTML files using
`nb_sphinx <http://nbsphinx.readthedocs.io/>`__ Sphinx extension which provides a source parser
for ``.ipynb`` files.

Links to notebooks
++++++++++++++++++

From docstrings and RST documentation files in Gammapy you can link to the built fixed-text HTML formatted
versions of the notebooks and subsections providing its filename with the ``.ipynb`` file extension
and the relative path to the folder where they are placed::

    `Maps section in Gammapy overview tutorial <../tutorials/overview.ipynb#Maps>`__

Links within notebooks
++++++++++++++++++++++


From MD cells in notebooks you can link to other notebooks, as well as to RST documentation files,
and subsections using the Markdown syntax to declare links to resources, as shown in the examples below:

.. code-block:: rst

    - [Maps section in Gammapy overview tutorial](overview.ipynb#Maps)
    - [Help!](../getting-started.rst#help)

You can also link to the Gammapy API reference documentation using the same Sphinx syntax that is used
when writing RST files. All links to the API reference classes and methods should start with ``~gammapy.``
and enclosed within quotation marks. This syntax will be translated into relative links to the API in the
HTML formatted versions of the notebooks, and to absolute links pointing to the online Gammapy documentation
in the ``.ipynb`` notebook files available to download. During the documentation building process a warning
will be raised for each detected broken link to the API.

Examples:

- `gammapy.maps`
- `gammapy.maps.Geom`
- `gammapy.maps.Geom.is_image`
- `gammapy.maps.Geom.is_image()`

The example links above could be created within MD cells in notebooks with the syntax below:

.. code-block:: rst

    - `~gammapy.maps`
    - `~gammapy.maps.Geom`
    - `~gammapy.maps.Geom.is_image`
    - `~gammapy.maps.Geom.is_image()`

When building the documentation of a release, the links declared in the MD cells as absolute links pointing
to the ``dev`` version of the online Gammapy documentation will be transformed to relative links in the built
HTML formatted notebooks and to absolute links pointing to that specific released version of the online docs
in the downloadable ``.ipynb`` files.

.. _dev-check_html_links:

Check broken links
++++++++++++++++++

To check for broken external links from the Sphinx documentation:

.. code-block:: bash

   $ cd docs; make linkcheck

You may also use `brök <https://github.com/smallhadroncollider/brok>`__ software, which will also check
the links present in the notebooks files.

.. code-block:: bash

   $ brok docs/tutorials/*.ipynb | grep "Failed|Could"


Include png files as images
----------------------------

In Jupyter notebooks
++++++++++++++++++++

You may include static images in notebooks using the following markdown directive:

.. code-block:: rst

    ![](images/my_static_image.png)

Please note that your images should be placed inside a `images` folder, accessed with that relative
path from your notebook.

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
