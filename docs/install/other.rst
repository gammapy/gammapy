.. include:: ../references.txt

.. _install-other:

Other package managers
======================

Gammapy has been packaged for some Linux package managers. E.g. on Debian, you
can install Gammapy via:

.. code-block:: bash

    sudo apt-get install python3-gammapy

To get a more fully featured scientific Python environment, you can install
other Python packages using the system package manager (``apt-get`` in this
example), and then on top of this install more Python packages using ``pip``.

Example:

.. code-block:: bash

    sudo apt-get install \
        python3-pip python3-matplotlib \
        ipython3-notebook python3-gammapy

    python3 -m pip install antigravity

Note that also on Linux, the recommended way to install Gammapy is via
``conda``. The ``conda`` install is maintained by the Gammapy team and gives you
usually a very recent version (releases every 2 months), whereas the Linux
package managers typically have release cycles of 6 months, or yearly or longer,
meaning that you'll get an older version of Gammapy. But you can always get a
recent version via pip:

.. code-block:: bash

    sudo apt-get install python3-gammapy
    pip install -U gammapy
