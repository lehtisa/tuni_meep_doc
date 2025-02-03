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

This demo provides an example a simulation of nonlinear processes in MEEP with second harmonic generation (SHG). It is a second order nonlinear process, where light with frequency :math:`\omega` is injected to a material with a second order nonlinear susceptibility :math:`\chi^{(2)}`, and new light with frequency :math:`\omega` is generated. We have used the `example simulation on third harmonic generation <https://meep.readthedocs.io/en/latest/Python_Tutorials/Third_Harmonic_Generation/>`_ from the official documentation as a starting point for this demo, but we will expand on it significantly by studying the evolution of the second harmonic field during propagation under different phase matching conditions.

This demo will discuss the following practical matters of simulation:

- Materials with :math:`\chi^{(2)}` nonlinearity
- Units with second order nonlinearities
- Materials with predefined dispersion using `meep.materials library <https://meep.readthedocs.io/en/latest/Materials/>`_
- Resolution convergence analysis

The code used to produce this demo is available at TODO.

SHG without Dispersion: Perfect Phase Patching
----------------------------------------------

First, we will simulate SHG without the presence of dispersion (same refractrive index for all frequencies). The desired simulation behaviour is presented schematically below TODO. We want to place a pump source with wavelength 1064 nm in a :math:`\chi^{(2)}` material, and then measure the output spectrum after propagtion in a 1D simulation. We will use lithium niobate (LiNbO\ :sub:`3`\ ) as the nonlinear material, which is a common material in second order nonlinear optics applications.

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

Next, we define a simulation function that propagates the input pulse in a :math:`\chi^{(2)}` medium and measures the output spectrum, as described in the above figure. A nonlinear optical simulation can be implemented by simply using a material with nonzero nonlinear susceptibility. We are simulating the case without dispersion, so we are using a constant refractive index that corresponds to the refractive index of LiNbO\ :sub:`2`\  at the pump frequency.

.. code-block:: python

   def chi2_propagation(chi2, f_pump, amplitude, resolution):
      """Propagate pulse in a second order nonlinear material and measure
      output spectrum.

      :param chi2: float, second order nonlinear susceptibility
      :param f_pump: float, pump frequency
      :param amplitude: float, pump current amplitude J
      :param resolution: int, resolution of simulation
      :return: (np.ndarray, np.ndarray), output spectral powers and
      corresponding frequencies
      """

      # perfectly matched layers
      pml_size = 2.0
      pml_layers = [mp.PML(pml_size)]

      # define simulation cell (15 µm propagation distance)
      cell_len = 15 + 2*pml_size
      cell = mp.Vector3(0, 0, cell_len)

      # define pump source
      source_loc = mp.Vector3(0, 0, -0.5*cell_len + pml_size)
      f_width = f_pump/20.0
      sources = [
         mp.Source(
               mp.GaussianSource(f_pump, fwidth=f_width),
               component=mp.Ex,
               center=source_loc,
               amplitude=amplitude,
         )
      ]

      # material of the simulation. Note the constant epsilon and hence
      # constant refractive index (no dispersion) and second order nonlinear
      # susceptibility chi2
      default_material = mp.Medium(epsilon=LiNbO3.epsilon(f_pump)[0,0], chi2=chi2)

      # define simulation object
      sim = mp.Simulation(
         cell_size=cell,
         sources=sources,
         boundary_layers=pml_layers,
         default_material=default_material,
         resolution=resolution,
         dimensions=1,
      )

      # define flux object for measuring the spectrum after propagation
      f_min = f_pump/2
      f_max = f_pump*3.5
      n_freq = 600
      end_loc = mp.Vector3(0, 0, 0.5*cell_len - pml_size)
      trans = sim.add_flux(
         0.5*(f_min + f_max),
         f_max-f_min,
         n_freq,
         mp.FluxRegion(end_loc),
      )
      
      # run for sufficiently long such that the pulse has fully passed
      #  through the end of the material
      sim.run(until=250)

      # retrieve spectral powers and corresponding frequencies
      trans_flux = mp.get_fluxes(trans)
      freqs = mp.get_flux_freqs(trans)

      return np.array(trans_flux), np.array(freqs)

Next, we determine the value of :math:`\chi^{(2)}` we will use for our material. It turns out that if we define :math:`\chi^{(2)}` in terms of the dimensionless product :math:`\chi^{(2)}E_\text{pump}`, where :math:`E_\text{pump}` is the pump amplitude, we will spare ourselves the effort of converting the value of :math:`\chi^{(2)}` from MEEP natural units to SI units when comparing simulation results to theoretical predictions later, since the value of the dimensionless product is the same regardless unit system. We choose :math:`\chi^{(2)}E_\text{pump}=0.003`, and then we have to find the value of :math:`E_\text{pump}` in MEEP units and set :math:`\chi^{(2)}=\frac{0.003}{E_\text{pump}}`.

Note that we are using this procedure to deal with the units, not (only) because we are too lazy to convert from MEEP units to SI units, but mainly because it follows the spirit of the official recommended way of dealing with units in nonlinearities. Units of second order nonlinearities are not explicitly discussed in the official documentation, but our approach aligns closely with the `official recommended procedure for Kerr nonlinearities <https://meep.readthedocs.io/en/latest/Units_and_Nonlinearity/#kerr-nonlinearities>`_. If you ever find yourself in an situation where you need to convert units of electric fields or nonlinear susceptibilities from MEEP units to SI units, it is highly likely that you don't actually need to do so and you should instead look for a way of expressing the quantity as a dimensionless ratio or product.

We will use a Gaussian source with current amplitude :math:`J=1`. Because sources in MEEP are current sources, we have to calculate the resulting electric field amplitude when the current is oscillating at the chosen amplitude. In our 1D simulation, the electric amplitude field is given by :math:`E_\text{pump}=\frac{1}{2}ZJ`, where :math:`Z=\sqrt{\frac{\mu}{\varepsilon}}` is the impedance of the medium (note that :math:`\mu=1` in MEEP units) and the factor :math:`\frac{1}{2}` appears because the electric field is split equally between left and and right propagating parts. Note that `there is no general formula for relating current amplitude and electric field amplitude <https://meep.readthedocs.io/en/latest/FAQ/#how-does-the-current-amplitude-relate-to-the-resulting-field-amplitude>`_ in higher dimension, and we can relate them with the above formula only because we are working with a 1D simulation.

Finally, we are using a Gaussian source, but the theory on SHG we will soon encounter assumes a plane wave source. It turns out the Gaussian peak amplitude can be converted to an effective plane wave amplitude by dividing by :math:`\sqrt{2}`. Using all this information, are now ready to determine the value of :math:`\chi^{(2)}` as follows:

.. code-block:: python

   source_amplitude = 1  # source current amplitude J
   Z = np.sqrt(1/eps)  # impedance of medium
   E = Z*source_amplitude/2  # electric field amplitude
   E /= np.sqrt(2)  # Gaussian source

   chi2_E_prod = 0.003  # dimensionless product
   chi2 = chi2_E_prod / E  # in MEEP units

We can now run the simulation. We will first obtain a reference power spectrum without nonlinearities by setting :math:`\chi^{(2)}=0`, after which the simulation is repeated with the desired :math:`\chi^{(2)}` value. We will use a resolution 256 for now, but we will examine the effect of resolution more closely later.

.. code-block:: python

   res = 256
   reference_trans_flux, freqs = chi2_propagation(chi2=0, f_pump=f_pump, amplitude=source_amplitude, resolution=res)
   trans_flux, freqs = chi2_propagation(chi2=chi2, f_pump=f_pump, amplitude=source_amplitude, resolution=res)

The results of the simulation are then plotted as follows:

.. code-block:: python

   fig, ax = plt.subplots()

   # convert frequencies from MEEP units to SI units
   freqs_SI = freqs * c/a

   norm = np.max(reference_trans_flux)
   ax.semilogy(freqs_SI*1e-12, trans_flux/norm, label=fr"$\chi^{{(2)}}={chi2_E_prod}/E_{{\text{{pump}}}}$")
   ax.semilogy(freqs_SI*1e-12, reference_trans_flux/norm, linestyle="--", label=rf"$\chi^{(2)}$={0}")
   ax.set_xlabel("frequency (THz)")
   ax.set_ylabel("transmitted power (a.u.)")
   ax.set_xlim([freqs_SI[0]*1e-12, freqs_SI[-1]*1e-12])
   ax.set_ylim([1e-6, 2])
   ax.legend()
   ax.grid(True)

.. figure:: nonlinear_phenomena_figures/shg_spectrum.png
   :alt: test text
   :width: 80%
   :align: center

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
