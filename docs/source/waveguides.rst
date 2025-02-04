Waveguides
============

.. _waveguides:

Introduction
============

An optical waveguide is a structure that confines and directs light, ensuring it follows a desired path rather than dispersing, being like *a wire for light*. These waveguides play an important role in various applications, enabling technologies such as optical fiber communications and `photonic integrated circuits <https://www.rp-photonics.com/photonic_integrated_circuits.html>`_. In this section, we will discuss the basics on how MEEP allows us to simulate these structures and provide some example simulations to accompany or to further explain the examples (`1 <https://meep.readthedocs.io/en/master/Python_Tutorials/Basics/#fields-in-a-waveguide>`_, `2 <https://meep.readthedocs.io/en/master/Python_Tutorials/Basics/#transmittance-spectrum-of-a-waveguide-bend>`_, and `3 <https://meep.readthedocs.io/en/master/Python_Tutorials/Resonant_Modes_and_Transmission_in_a_Waveguide_Cavity/>`_) provided by the official MEEP documentation.

Before tackling the examples, we will explain briefly the basics behind waveguides to provide an initial understanding and intuition for the simulations. The confinement of light in a waveguide can happen through various ways, the most common of which is *total internal reflection*. Other means include for example the use of metallic reflectors or photonic crystals. Let's concentrate here on the simplest and most common case since it is enough for our purposes. 

A typical waveguide structure consists of a *core* with a higher refractive index :math:`n_1` and a surrounding *cladding* with a lower refractive index :math:`n_2`. Now, if the light is coming from the core towards the cladding, the light refracts at the interface according to Snell's law

.. math::

   n_1 \sin{\theta_1} = n_2 \sin{\theta_2},

where :math:`\theta_1` is the angle of incidence and :math:`\theta_2` is the angle of refraction. Because the light is traveling from optically denser medium to optically rarer medium, total internal reflection is possible if the angle of incidence is large enough. Using Snell's law, we can solve for the critical angle of incidence :math:`\theta_\text{c}` from which the phenomenon starts to occur. Setting :math:`\theta_2 = 90°`, we get the critical angle 

.. math::

   \theta_\text{c}=\arcsin{\frac{n_2}{n_1}}.

Hence, for larger angles of incidence than the critical angle, the light reflected at the interface back to the core. This is essentially the mechanism how the light can be trapped inside the waveguide. 

However, we must remember that this is just the principle explained using ray optics. For further analysis, we would need for example the help of Fresnel coefficients and a more rigorous mathematical treatment of waves as fields. 

Lastly, we briefly mention one phenomena that we will later observe in our waveguide simulations, *evanascent waves*. If we study the total internal reflection case with Maxwell's equations, we naturally get a solution with an incident wave and a reflected wave. Interestingly, the solution does not comply with Maxwell's equations without an exponentialy decaying field penetrating into the cladding material. This phenomenon enables *evanascent coupling* which is utilized for example in photonic integrated circuits. 

