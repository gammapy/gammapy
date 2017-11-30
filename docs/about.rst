.. include:: references.txt

.. _about:

About Gammapy
=============

.. _about-overview:

Overview
--------

Gammapy is a community-developed, open-source Python package for gamma-ray astronomy.
It is a prototype for the `CTA`_ science tools.

It is an in-development `affiliated package`_ of `Astropy`_ that builds on the core
`scientific Python stack`_ to provide tools to simulate and analyse the gamma-ray sky
for telescopes such as `CTA`_, `H.E.S.S.`_, `VERITAS`_, `MAGIC`_, `HAWC`_ and `Fermi-LAT`_.

Gammapy is a place for Python-coding gamma-ray astronomers to share their code and collaborate.
Feature requests and contributions welcome!

Likelihood fitting of the morphology and spectrum of gamma-ray sources (using `Sherpa`_),
including multi-mission joint likelihood analysis and physical SED modeling (using `Naima`_)
is one important feature we're working on.
But Gammapy has a broader scope, we currently have code e.g. for data handling, background modeling,
source detection, easy access to commonly used datasets and catalogs, statistical methods,
even simulating Galactic source populations.

Gammapy is under very active development
(see the `Gammapy project summary on Open HUB`_ and the `Gammapy contributors page on Github`_).
We plan to write a paper about it soon (late 2017), and are working towards a 1.0 release in 2018.

To learn more about Gammapy, the Gammapy ICRC proceedings mentioned in the next section are a good place to start.
For a more hands-on introduction, the Gammapy tutorial Jupyter notebooks are the best place to start.

.. _about-support:

Acknowledging or Citing Gammapy
-------------------------------

If you have used Gammapy in your scientific work we would appreciate if you acknowledge or cite Gammapy.

Being able to show that Gammapy is used for scientific studies helps us to justify
spending time on it, or in the future possibly even to apply for funding
specifically to extend Gammapy to enable new or better science.

For publications
++++++++++++++++

For a publication, we recommend the following line be added to the conclusion or acknowledgements

    This research has made use of Gammapy, a community-developed, open-source
    Python package for gamma-ray astronomy (citation).

For now, no publication for Gammapy in a refereed journal exists,
and the citation should be to one or both of the Gammapy proceedings:

- `2015ICRC...34..789D <https://ui.adsabs.harvard.edu/#abs/2015ICRC...34..789D>`__:
  "Gammapy: An open-source Python package for gamma-ray astronomy"

- `2017arXiv170901751D <https://ui.adsabs.harvard.edu/#abs/2017arXiv170901751D>`__:
  "Gammapy - A prototype for the CTA science tools"

The first one is more generally introducing Gammapy, the second one is focused on CTA.

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
- Bruno Khelifi (`@bkhelifi <https://github.com/bkhelifi>`__)
- Christoph Deil (`@cdeil <https://github.com/cdeil>`__)
- Cosimo Nigro (`@cosimoNigro <https://github.com/cosimoNigro>`__)
- Dirk Lennarz (`@dlennarz <https://github.com/dlennarz>`__)
- Domenico Tiziani (`@dltiziani <https://github.com/dltiziani>`__)
- Ellis Owen (`@ellisowen <https://github.com/ellisowen>`__)
- Fabio Acero (`@facero <https://github.com/facero>`__)
- Helen Poon (`@helen-poon <https://github.com/helen-poon>`__)
- Ignasi Reichardt (`@ignasi-reichardt <https://github.com/ignasi-reichardt>`__)
- Johannes King (`@joleroi <https://github.com/joleroi>`__)
- Jonathan Harris (`@JonathanDHarris <https://github.com/JonathanDHarris>`__)
- José Enrique Ruiz (`@Bultako <https://github.com/Bultako>`__)
- Julien Lefaucheur (`@jjlk <https://github.com/jjlk>`__)
- Lars Mohrmann (`@lmohrmann <https://github.com/lmohrmann>`__)
- Léa Jouvin (`@JouvinLea <https://github.com/JouvinLea>`__)
- Luigi Tibaldo (`@tibaldo <https://github.com/tibaldo>`__)
- Manuel Paz Arribas (`@mapazarr <https://github.com/mapazarr>`__)
- Matthew Wood (`@woodmd <https://github.com/woodmd>`__)
- Matthias Wegen (`@wegenmat <https://github.com/wegenmat>`__)
- Nachiketa Chakraborty (`@cnachi <https://github.com/cnachi>`__)
- Olga Vorokh (`@OlgaVorokh <https://github.com/OlgaVorokh>`__)
- Peter Deiml (`@pdeiml <https://github.com/pdeiml>`__)
- Régis Terrier (`@registerrier <https://github.com/registerrier>`__)
- Roberta Zanin (`@robertazanin <https://github.com/robertazanin>`__)
- Rolf Bühler (`@rbuehler <https://github.com/rbuehler>`__)
- Rubén López-Coto (`@rlopezcoto <https://github.com/rlopezcoto>`__)
- Stefan Klepser (`@klepser <https://github.com/klepser>`__)
- Thomas Armstrong (`@thomasarmstrong <https://github.com/thomasarmstrong>`__)
- Thomas Vuillaume (`@vuillaut <https://github.com/vuillaut>`__)
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
- Similarly, Régis Terrier and Bruno Khelifi from `APC Paris`_ as well as
  Catherine Boisson from `Paris Observatory`_ are supporting Gammapy development
  by giving PhDs and postdocs time contribute to Gammapy.
- Google for sponsoring Manuel Paz Arribas to work on background modeling
  as well as Olga Vorokh to work on image analysis and source detection in Gammapy
  as part of `Google Summer of Code`_.
- `H.E.S.S.`_ for providing a wonderful TeV gamma-ray dataset
  to develop the Gammapy code and methods (to collaboration members only).
  And specifically to the HOST ("HESS data analysis with open source tools") task
  group within H.E.S.S. for exporting the data and IRFs to FITS format,
  making it available to Gammapy and other open source tools.
- `CTA`_ for promoting open source and working on the specification of open data formats,
  which are the basis of Gammapy data analysis and interoperability with other
  open source analysis packages (e.g. Gammalib/ctools or 3ML)
  and between different collaborations (e.g. H.E.S.S., VERITAS, MAGIC).
- `Fermi-LAT`_ for making their data and software freely available and providing
  a wonderful GeV gamma-ray dataset, which was used to develop Gammapy.
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

Publications using Gammapy
--------------------------

Here's a list of publications using, describing or related to Gammapy (in reverse chronological order).

If something is missing, please let us know (e.g. drop a line to the Gammapy mailing list).


- Deil et al. (2017),
  "Gammapy - A prototype for the CTA science tools",
  `2017arXiv170901751D <https://ui.adsabs.harvard.edu/#abs/2017arXiv170901751D>`__

- Zanin et al. (2017)
  "Observing the Galactic Plane with the Cherenkov Telescope Array"
  `2017arXiv170904354Z <https://ui.adsabs.harvard.edu/#abs/2017arXiv170904354Z>`__

- Trichard et al. (2017),
  "Searching for PeVatrons in the CTA Galactic Plane Survey"
  `2017arXiv170901311T <https://ui.adsabs.harvard.edu/#abs/2017arXiv170901311T>`__

- Lefaucheur et al. (2017)
  "Gammapy: high level data analysis for extragalactic science cases with the Cherenkov Telescope Array"
  `2017arXiv170910169L <https://ui.adsabs.harvard.edu/#abs/2017arXiv170910169L>`__

- Gaté et al. (2017)
  "Studying cosmological gamma-ray propagation with the Cherenkov Telescope Array"
  `2017arXiv170904185G <https://ui.adsabs.harvard.edu/#abs/2017arXiv170904185G>`__

- Wood et al. (2017),
  "Fermipy: An open-source Python package for analysis of Fermi-LAT Data",
  `2017arXiv170709551W <https://ui.adsabs.harvard.edu/#abs/2017arXiv170709551W>`__

- Fioretti et al. (2017),
  "The Cherenkov Telescope array on-site integral sensitivity: observing the Crab",
  `2016SPIE.9906E..3OF <https://ui.adsabs.harvard.edu/#abs/2016SPIE.9906E..3OF>`__

- Voruganti et al. (2017),
  "Gamma-sky.net: Portal to the gamma-ray sky",
  `2017AIPC.1792g0005V <https://ui.adsabs.harvard.edu/#abs/2017AIPC.1792g0005V>`__

- Deil et al. (2017),
  "Open high-level data formats and software for gamma-ray astronomy",
  `2017AIPC.1792g0006D <https://ui.adsabs.harvard.edu/#abs/2017AIPC.1792g0006D>`__

- Gottschall et al. (2017),
  "Discovery of new TeV supernova remnant shells in the Galactic plane with H.E.S.S.",
  `2017AIPC.1792d0030G <https://ui.adsabs.harvard.edu/#abs/2017AIPC.1792d0030G>`__

- Deil (2016),
  "Python in gamma-ray astronomy",
  `2016pyas.confE...4D <https://ui.adsabs.harvard.edu/#abs/2016pyas.confE...4D>`__

- Puelhofer et al. (2015),
  "Search for new supernova remnant shells in the Galactic plane with H.E.S.S.",
  `2015ICRC...34..886P <https://ui.adsabs.harvard.edu/#abs/2015ICRC...34..886P>`__

- Zabalza (2015),
  "Naima: a Python package for inference of particle distribution properties from nonthermal spectra",
  `2015ICRC...34..922Z <https://ui.adsabs.harvard.edu/#abs/2015ICRC...34..922Z>`__

- Donath et al. (2015),
  "Gammapy: An open-source Python package for gamma-ray astronomy",
  `2015ICRC...34..789D <https://ui.adsabs.harvard.edu/#abs/2015ICRC...34..789D>`__

- Owen et al. (2015),
  "The gamma-ray Milky Way above 10 GeV: Distinguishing Sources from Diffuse Emission",
  `2015arXiv150602319O <https://ui.adsabs.harvard.edu/#abs/2015arXiv150602319O>`__
