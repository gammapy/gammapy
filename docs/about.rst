.. include:: references.txt

.. _about:

About Gammapy
=============

.. _about-overview:

Overview
--------

Gammapy is a community-developed, open-source Python package for gamma-ray astronomy.

It is an in-development `affiliated package`_ of `Astropy`_ that builds on the core
`scientific Python stack`_ to provide tools to simulate and analyse the gamma-ray sky
for telescopes such as `CTA`_, `H.E.S.S.`_, `VERITAS`_, `MAGIC`_, `HAWC`_ and `Fermi-LAT`_.

Gammapy is a place for Python-coding gamma-ray astronomers to share their code and collaborate.

Likelihood fitting of the morphology and spectrum of gamma-ray sources (using `Sherpa`_),
including multi-mission joint likelihood analysis and physical SED modeling (using `Naima`_)
is one important feature we're working on.
But Gammapy has a broader scope, we currently have code e.g. for data handling, background modeling,
source detection, easy access to commonly used datasets and catalogs, statistical methods,
even simulating Galactic source populations.

Feature requests and contributions welcome!

Gammapy is under very active development
(see the `Gammapy project summary on Open HUB`_ and the `Gammapy contributors page on Github`_).
A 1.0 release and a paper are planned for 2016.

For now, please see the Gammapy
`poster <https://indico.cern.ch/event/344485/session/142/contribution/695/attachments/1136524/1626432/gammapy-icrc2015-poster.pdf>`_
and `proceeding <http://arxiv.org/abs/1509.07408>`__
from `ICRC 2015 <https://indico.cern.ch/event/344485/session/142/contribution/695>`__.

.. _about-support:

Acknowledging or Citing Gammapy
-------------------------------

If you have used Gammapy in your scientific work we would appreciate it if you would acknowledge it.

Thank you, in advance, for your support.

For publications
++++++++++++++++

For a publication, we recommend the following line be added to the conclusion or acknowledgements

    This research has made use of Gammapy, a community-developed, open-source
    Python package for gamma-ray astronomy (citation).

For now, the citation is to the `Gammapy ICRC 2015 <http://adsabs.harvard.edu/abs/2015arXiv150907408D>`__
conference proceeding.

If the journal allows please also include a link to https://github.com/gammapy/gammapy .

For presentations and posters
+++++++++++++++++++++++++++++

If you are making a presentation or poster featuring work/research that makes use of Gammapy,
please include the Gammapy banner:

.. image:: _static/gammapy_banner.png
    :width: 400px

There's also a smaller logo and variants with white text instead of black available
`here <https://github.com/gammapy/gammapy-extra/tree/master/logo>`__)

.. _about-contributors:

Contributors
------------

The following people have contributed to Gammapy (first name alphabetical order):

- Arjun Voruganti (`@vorugantia <https://github.com/vorugantia>`__)
- Arpit Gogia (`@arpitgogia <https://github.com/arpitgogia>`__)
- Axel Donath (`@adonath <https://github.com/adonath>`__)
- Brigitta Sipocz (`@bsipocz <https://github.com/bsipocz>`__)
- Christoph Deil (`@cdeil <https://github.com/cdeil>`__)
- Dirk Lennarz (`@dlennarz <https://github.com/dlennarz>`__)
- Domenico Tiziani (`@dltiziani <https://github.com/dltiziani>`__)
- Ellis Owen (`@ellisowen <https://github.com/ellisowen>`__)
- Fabio Acero (`@facero <https://github.com/facero>`__)
- Helen Poon (`@helen-poon <https://github.com/helen-poon>`__)
- Ignasi Reichardt (`@ignasi-reichardt <https://github.com/ignasi-reichardt>`__)
- Johannes King (`@joleroi <https://github.com/joleroi>`__)
- Jonathan Harris (`@JonathanDHarris <https://github.com/JonathanDHarris>`__)
- Julien Lefaucheur (`@jjlk <https://github.com/jjlk>`__)
- Lars Mohrmann (`@lmohrmann <https://github.com/lmohrmann>`__)
- Léa Jouvin (`@JouvinLea <https://github.com/JouvinLea>`__)
- Luigi Tibaldo (`@tibaldo <https://github.com/tibaldo>`__)
- Manuel Paz Arribas (`@mapazarr <https://github.com/mapazarr>`__)
- Matthew Wood (`@woodmd <https://github.com/woodmd>`__)
- Nachiketa Chakraborty (`@cnachi <https://github.com/cnachi>`__)
- Olga Vorokh (`@OlgaVorokh <https://github.com/OlgaVorokh>`__)
- Régis Terrier (`@registerrier <https://github.com/registerrier>`__)
- Rolf Bühler (`@rbuehler <https://github.com/rbuehler>`__)
- Stefan Klepser (`@klepser <https://github.com/klepser>`__)
- Victor Zabalza (`@zblz <https://github.com/zblz>`__)
- Zé Vinícius (`@mirca <https://github.com/mirca>`__)


A detailed listing of contributions is here: :ref:`changelog`.

.. _about-thanks:

Thanks
------

We would like to say thank you to the people, institutions and collaborations
that have supported Gammapy development!

- `Werner Hofmann`_ and `Jim Hinton`_ (directors at `MPIK Heidelberg`_) for giving
  PhDs and postdocs in the `H.E.S.S.`_ and `CTA`_ group time to work on Gammapy.
- Google for sponsoring `Manuel Paz Arribas <https://github.com/mapazarr>`__
  to work on background modeling in Gammapy for `GSoC 2015`_.
- `H.E.S.S.`_ for providing a wonderful TeV gamma-ray dataset
  to develop the Gammapy code and methods (to collaboration members only).
  And specifically to the HOST ("HESS data analysis with open source tools") task
  group within H.E.S.S. for exporting the data and IRFs to FITS format,
  making it available to Gammapy and other open source tools.
- `Fermi-LAT`_ for making their data and software freely available and providing
  a wonderful GeV gamma-ray dataset, which was used to develop Gammapy.
- `CTA`_ for promoting open source and working on the specification of open data formats,
  which are the basis of Gammapy data analysis and interoperability with other
  open source analysis packages (e.g. Gammalib/ctools or 3ML)
  and between different collaborations (e.g. H.E.S.S., VERITAS, MAGIC).
- The `Astropy`_ project (core package, affiliated package, people) for creating
  a core Python package for astronomy.
  (Astropy is one of the building blocks on which Gammapy is built.)
- The `Sherpa`_ developers and the `Chandra X-ray observatory (CXC)`_ for creating
  and maintaining a wonderful modeling / fitting package, and making Sherpa an
  `open package on Github <https://github.com/sherpa/sherpa>`__ in 2015.
  (Sherpa is one of the building blocks on which Gammapy is built.)
- `Martin Raue`_ for creating `PyFACT`_ and organising the first CTA data challenge
  in 2011. PyFACT (and a few other similar Python packages) can be considered
  precursors to Gammapy.
- Everyone that contributed to Gammapy or used it for their research.

.. _about-users:

Papers using Gammapy
--------------------

Here's a list of papers using Gammapy.

If something is missing, please send an email to the Gammapy mailing list
or to `Christoph Deil`_ if you prefer private communication.


.. [Owen2015] `Owen et al. (2015) <http://adsabs.harvard.edu/abs/2015arXiv150602319O>`_,
   "The gamma-ray Milky Way above 10 GeV: Distinguishing Sources from Diffuse Emission",

.. [Puelhofer2015] `Pühlhofer et al. (2015) <https://indico.cern.ch/event/344485/session/109/contribution/1299>`_,
   "Search for new supernova remnant shells in the Galactic plane with H.E.S.S.",
