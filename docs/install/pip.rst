.. include:: ../references.txt

.. _install-pip-setuppy:

Installation with pip or setup.py
=================================

.. _install-pip:

pip
---

To install the latest Gammapy **stable** version (see `Gammapy page on PyPI`_) using `pip`_:

.. code-block:: bash

   $ python -m pip install gammapy

To install the current Gammapy **development** version using `pip`_:

.. code-block:: bash

   $ python -m pip install git+https://github.com/gammapy/gammapy.git#egg=gammapy

If that doesn't work because the download from PyPI or Github is blocked by your network,
but you have some other means of copying files onto that machine,
you can get the tarball (``.tar.gz`` file) from PyPI or ``.zip`` file from Github, and then
``python -m pip install <filename>``.

.. _install-setuppy:

setup.py
--------

To download the latest development version of Gammapy:

.. code-block:: bash

   $ git clone https://github.com/gammapy/gammapy.git
   $ cd gammapy

Now you install, run tests or build the documentation:

.. code-block:: bash

   $ python setup.py install
   $ python setup.py test
   $ python setup.py build_docs

Also you have easy access to the Python scripts from the tutorials and examples:

.. code-block:: bash

   $ cd docs/tutorials
   $ cd examples

If you want to contribute to Gammapy, but are not familiar with Python or
git or Astropy yet, please have a look at the
`Astropy developer documentation <http://docs.astropy.org/en/latest/#developer-documentation>`__.

