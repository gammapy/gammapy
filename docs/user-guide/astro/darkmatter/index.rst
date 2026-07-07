.. _astro-darkmatter:

***********
Dark matter
***********

The `gammapy.astro.darkmatter` module implements the physics of indirect dark
matter searches, built around the paradigm of Weakly Interacting Massive
Particles (WIMPs) as the dark matter candidate. It provides the full
toolchain needed to go from a dark matter halo and a particle physics model
to a complete gamma-ray analysis. The module is flexible along every axis of
the analysis:

- **Scenario:** annihilation or decay
- **Data:** real observations or simulated data
- **Analysis type:** 3D (spatial+spectral) or 1D (spectral-only)
- **Output:** upper limits, exclusion curves, and sensitivity (Brazil) bands

For hands-on, step-by-step examples covering the full
analysis pipeline with real code, see the tutorials linked at the bottom of
this page.

Analysis components
====================

A dark matter analysis in Gammapy combines two ingredients:

- **Spectral distribution** -- the expected differential gamma-ray flux per
  annihilation/decay channel, described by
  `~gammapy.astro.darkmatter.DarkMatterAnnihilationSpectralModel` and
  `~gammapy.astro.darkmatter.DarkMatterDecaySpectralModel`. The underlying
  tabulated spectra, via `~gammapy.astro.darkmatter.PrimaryFlux`, can be
  sourced from PPPC4DMID, from CosmiXs, or from a custom user-provided table,
  covering a wide range of masses and channels.

- **Spatial distribution** -- the dark matter halo density profile, described
  through radially symmetric models (see
  `~gammapy.astro.darkmatter.profiles`): NFW, Einasto, Isothermal, Burkert,
  Moore, and Zhao profiles are currently implemented, covering both cuspy and
  cored halo shapes.

These two ingredients are tied together by the astrophysical **J-factor**
(annihilation) or **D-factor** (decay) -- the line-of-sight integral of the
dark matter density, squared for annihilation -- which sets the overall
normalization of the expected signal. `~gammapy.astro.darkmatter.JFactory`
computes this map internally from a chosen spatial profile and an
observer-to-halo distance, making it equally suited to Galactic Center
analyses and to extragalactic targets such as dwarf spheroidal galaxies.
Alternatively, an externally computed J-factor or D-factor -- e.g. from
`CLUMPY`_ -- can be used directly, decoupling the astrophysical modeling from
the gamma-ray analysis itself.

Together, these building blocks plug directly into the standard Gammapy
modeling and fitting workflow, turning a halo and a particle physics
scenario into a ready-to-fit spectral model.

Using gammapy.astro.darkmatter
================================

.. minigallery:: gammapy.astro.darkmatter

.. _CLUMPY: http://lpsc.in2p3.fr/clumpy/
