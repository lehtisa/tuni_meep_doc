===================
Nonlinear Effects
===================

.. _nonlinear_phenomena:

Introduction
============

MEEP supports nonlinear optical simulations via materials with user defined :math:`\chi^{(2)}` or :math:`\chi^{(3)}` nonlinear susceptibilities. In this section, we will provide example simulations of both second and third order nonlinear effects: second harmonic generation under different phase matching conditions and optical bistability. We hope these examples will be useful for new MEEP users looking to simulate nonlinear effects, as the official MEEP documentation provides only a single `example simulation on nonlinearities <https://meep.readthedocs.io/en/latest/Python_Tutorials/Third_Harmonic_Generation/>`_.

First, we will give a very brief introduction on nonlinear optics. Within the linear regime where the optical electric field :math:`\tilde{E}` is not very large, the material polarization :math:`\tilde{P}` is given by

.. math::

   \tilde{P} = \varepsilon_0 \chi^{(1)} \tilde{E},

where :math:`\varepsilon_0` is the vacuum permittivity and :math:`\chi^{(1)}` is the linear susceptibility. However, if the electric field is sufficiently large (i.e. intensity is high), the linear relationship no longer holds and nonlinear correction terms are required. In this case, the material polarization can be written as 

.. math::

   \tilde{P} = \varepsilon_0 \left[ \chi^{(1)} \tilde{E} + \chi^{(2)} \tilde{E}^2 + \chi^{(3)} \tilde{E}^3 + \ldots \right],

where :math:`\chi^{(2)}` and :math:`\chi^{(3)}` are second and third order nonlinear susceptibilities, respectively. The powers of the electric field have severe consequences, some of which can be deduced with a simple treatment.

If we write the optical electric field as the sum of a complex field and it's complex conjugate :math:`\tilde{E} = E e^{-\mathrm{i} \omega t} + E^* e^{\mathrm{i} \omega t}`, the second order nonlinear polarization becomes

.. math::

   \tilde{P}^{(2)} = \varepsilon_0  \chi^{(2)} \tilde{E}^2 = \varepsilon_0  \chi^{(2)} \left(E^2 e^{-\mathrm{i} 2\omega t} + (E^*)^2 e^{\mathrm{i} 2\omega t} + \lvert E \rvert ^2 \right).

We notice that the first and second term inside the brackets are complex conjugates of each other, so the imaginary part cancles out and we are left with a term that is oscillating at twice the initial frequency. The oscillating polarization acts as a new source of light, and therefore frequency doubled light is generated! This phenomenon is known as second harmonic generation, and we will study it closer in demo 1.

We can expand the third order nonlinear polarization in a similar way. If we do this and examine only the term that is oscillating at the incident frequency :math:`\omega`, we obtain

.. math::

   \tilde{P}^{(3)}(\omega) = 3\varepsilon_0  \chi^{(3)}  \lvert E \rvert ^2 E e^{-\mathrm{i} \omega t}.

Now the total polarization oscillating at :math:`\omega` consists of the regular linear contribution, as well as a nonlinear contribution. We can write the total polarization at frequency :math:`\omega` as 

.. math::

   \tilde{P}^{\mathrm{tot}}(\omega) = \varepsilon_0 \chi^{(1)} E e^{-\mathrm{i} \omega t} + 3\varepsilon_0  \chi^{(3)}  \lvert E \rvert ^2 E e^{-\mathrm{i} \omega t} = \varepsilon_0 \left(  \chi^{(1)}  + 3 \chi^{(3)}  \lvert E \rvert ^2 \right) E e^{-\mathrm{i} \omega t} = \varepsilon_0 \chi_{\mathrm{eff}} E e^{-\mathrm{i} \omega t},

where we have defined an effective susceptibility :math:`\chi_{\mathrm{eff}}=\chi^{(1)}  + 3 \chi^{(3)}  \lvert E \rvert ^2` as the sum of linear and nonlinear contributions. If we recall that in general, the refractive index is given by :math:`n=\sqrt{1+\chi}` and that the intensity of the light is proportional to :math:`\lvert E \rvert ^2`, we can observe that intensity of the light affects the effective refractive index experienced by the light! This effect is known as an intensity-dependent refractive index or self-phase modulation, and we will examine one concequence of it, optical bistability, in demo 2.

.. note::

   The above reasoning is sufficient for getting a sense of the nonlinear effects, but a more rigorous treatment would be required for making actual calculations.


Demo 1: Second Harmonic Generation
==================================

This demo provides an example a simulation of nonlinear processes in MEEP with second harmonic generation (SHG). SHG is a second order nonlinear process, where light with frequency :math:`\omega` is injected to a material with a second order nonlinear susceptibility :math:`\chi^{(2)}`, and new light with frequency $2\omega$ is generated. We have used the `example simulation on third harmonic generation <https://meep.readthedocs.io/en/latest/Python_Tutorials/Third_Harmonic_Generation/>`_ from the official documentation as a starting point for this demo, but we will expand on it significantly by studying the evolution of the second harmonic field during propagation under different phase matching conditions.

This demo will discuss the following practical matters of simulation:
- Materials with :math:`\chi^{(2)}`` nonlinearity
- Units with second order nonlinearities
- Materials with predefined dispersion using `meep.materials library <https://meep.readthedocs.io/en/latest/Materials/>`_
- Resolution convergence analysis


SHG without Dispersion: Perfect Phase Patching
----------------------------------------------

First, we will simulate SHG without the presence of dispersion (same refractrive index for all frequencies). The desired simulation behaviour is presented schematically below TODO. We want to place a pump source with wavelength 1064 nm in a :math:`\chi^{(2)}` material, and then measure the output spectrum after propagtion in a 1D simulation. We will use lithium niobate (LiNbo\:sub:3\) as the nonlinear material, which is a common material in second order nonlinear optics applications.

First, we import the required libraries and define parameters:

.. code-block:: python

   import numpy as np
   from matplotlib import pyplot as plt
   import meep as mp
   from meep.materials import LiNbO3

   c = 2.998e8  # speed of light
   a = 1e-6  # charasteristic length scale

   # Nd:YAG laser wavelength 1064 nm converted to MEEP frequency units
   f_pump = a/1064e-9

   # permittivity at source frequency. The .epsilon() returns the
   # permittivity tensor, so we index an element that is on the diagonal
   eps = LiNbO3.epsilon(f_pump)[0,0]
   n0 = np.sqrt(eps)  # refractive index

Next, we define a simulation function that propagates the input pulse in a :math:`\chi^{(2)}` medium and measures the output spectrum, as described in the above figure. A nonlinear optical simulation can be implemented by simply using a material with nonzero nonlinear susceptibility. We are simulating the case without dispersion, so we are using a constant refractive index that corresponds to the refractive index of LiNbo\:sub:3\ at the pump frequency.

SHG with Dispersion: Phase Patching Problem
-------------------------------------------

lorem ipsum

Quase-Phase matching
--------------------

lorem ipsum

Demo 2: Optical Bistability
===========================

lorem ipsum

Conclusions
===========
