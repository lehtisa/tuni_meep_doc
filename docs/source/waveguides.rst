============
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

   \theta_\text{c}=\arcsin{\left(\frac{n_2}{n_1}\right)}.

Hence, for larger angles of incidence than the critical angle, the light reflected at the interface back to the core. This is essentially the mechanism how the light can be trapped inside the waveguide. 

However, we must remember that this is just the principle explained using ray optics. For further analysis, we would need for example the help of Fresnel coefficients and a more rigorous mathematical treatment of waves as fields. 

Lastly, we briefly mention one phenomena that we will later observe in our waveguide simulations, *evanascent waves*. If we study the total internal reflection case with Maxwell's equations, we naturally get a solution with an incident wave and a reflected wave. Interestingly, the solution does not comply with Maxwell's equations without an exponentialy decaying field penetrating into the cladding material. This phenomenon enables *evanascent coupling* which is utilized for example in photonic integrated circuits. 

==========
Waveguides
==========

.. _waveguides:

Introduction
============

An optical waveguide is a structure that confines and directs light, ensuring it follows a desired path rather than dispersing. In layman's terms, a waveguide is essentially *a wire for light*. These waveguides play an important role in various applications, enabling technologies such as optical fiber communications and `photonic integrated circuits <https://www.rp-photonics.com/photonic_integrated_circuits.html>`_. In this section, we will discuss the basics on how MEEP allows us to simulate these structures and provide some example simulations to accompany or to further explain the examples (`1 <https://meep.readthedocs.io/en/master/Python_Tutorials/Basics/#fields-in-a-waveguide>`_, `2 <https://meep.readthedocs.io/en/master/Python_Tutorials/Basics/#transmittance-spectrum-of-a-waveguide-bend>`_, and `3 <https://meep.readthedocs.io/en/master/Python_Tutorials/Resonant_Modes_and_Transmission_in_a_Waveguide_Cavity/>`_) provided by the official MEEP documentation.

Before tackling the examples, we will explain briefly the basics behind waveguides to provide an initial understanding and intuition for the simulations. The confinement of light in a waveguide can happen through various ways, the most common of which is *total internal reflection*. Other means include for example the use of metallic reflectors or photonic crystals. Let's concentrate here on the simplest and most common case since it is enough for our purposes. 

A typical waveguide structure consists of a *core* with a higher refractive index :math:`n_1` and a surrounding *cladding* with a lower refractive index :math:`n_2`. Now, if the light is coming from the core towards the cladding, the light refracts at the interface according to Snell's law

.. math::

   n_1 \sin{\theta_1} = n_2 \sin{\theta_2},

where :math:`\theta_1` is the angle of incidence and :math:`\theta_2` is the angle of refraction. Because the light is traveling from optically denser medium to optically rarer medium, total internal reflection is possible if the angle of incidence is large enough. Using Snell's law, we can solve for the critical angle of incidence :math:`\theta_\text{c}` from which the phenomenon starts to occur. Setting :math:`\theta_2 = 90°`, we get the critical angle 

.. math::

   \theta_\text{c}=\arcsin{\left(\frac{n_2}{n_1}\right)}.

Hence, for larger angles of incidence than the critical angle, the light reflected at the interface back to the core. This is essentially the mechanism how the light can be trapped inside the waveguide. 

However, we must remember that this is just the principle explained using ray optics. For further analysis, we would need for example the help of Fresnel coefficients and a more rigorous mathematical treatment of waves as fields. 

Lastly, we briefly mention one phenomena that we will later observe in our waveguide simulations, *evanascent waves*. If we study the total internal reflection case with Maxwell's equations, we naturally get a solution with an incident wave and a reflected wave. Interestingly, the solution does not comply with Maxwell's equations without an exponentialy decaying field penetrating into the cladding material. This phenomenon enables *evanascent coupling* which is utilized for example in photonic integrated circuits. 

..
   TODO: Even though in the real world, waveguides are three-dimensional structures, they can often be simulated in two dimensions with the help of *effective refractive index*.

Demo 1: Straight Dielectric Waveguide
============

In this demo, we will get acquainted with the basics of simulating dielectric waveguides. We will simulate how an electric field generated by a continous wave source propagates in a straight dielectric waveguide. We will also learn a way to extract data from the simulation and to visualize the results by creating some figures and an animation. This demo is very similar to the one provided in MEEP's official documentation but here we try explain the workflow in a little bit different way.

In this demo, we will discuss the following matters of simulation in MEEP:
- Creating a simulation cell with dielectric waveguide regions which are simple in shape. 
- Defining a simple continuous wave source which is coupled to the waveguide. 
- Conducting a simple *convergence study* to determine a suitable resolution for a simulation. 
- Extracting data from the simulation during and after the simulation.
- Visualizing the simulation by creating figures and animating. 

Getting started 
----------------
Naturally, we need to import the libraries we need before any coding. In addition to Meep, we import also Numpy, Pyplot from Matplotlib, and h5py for visualization purposes.

.. code-block:: python

   import meep as mp
   import numpy as np
   import h5py
   import matplotlib.pyplot as plt
   from matplotlib.animation import FuncAnimation
   from matplotlib import cm
   from matplotlib.colors import TwoSlopeNorm

The next step is to define the region where MEEP is calculating the fields, i.e. the simulation *cell*. This time we create a 2D cell of the size 10 µm :math:`\times` 5 µm :math:`\times` 0 µm. The cell is defined in a 3D vector whose elements tell the size of the cell. 

.. code-block:: python

   cell_x = 10
   cell_y = 5  
   cell_z = 0

   cell = mp.Vector3(cell_x, cell_y, cell_z)

Now it is time to add the actual waveguide to the cell. To do that, we need to define its dimensions and material. Here we want to simulate a 0.5 µm wide waveguide which is infinitely long and high. We choose our material to be silicon nitride (Si\ :sub:`3`\N\ :sub:`4`\ ). Its refractive index is around 2.0 at the wavelength of 1.55 µm [1]_. which is our wavelength region of interest in this case. 

The material is specified with a ``Medium`` object. We can define its refractive index with the parameter ``index``, or alternatively we could specify its dielectric constant :math:`\varepsilon` with the parameter ``epsilon``. The waveguide geometry can be defined with a ``Block`` object which needs the size, the center and the material. It is customary to store the created geometries in a list called ``geometry``.

.. code-block:: python

   wg_sx = mp.inf # length
   wg_sy = 0.5    # width
   wg_sz = mp.inf # height

   Si3N4 = mp.Medium(index=2.0)

   geometry = [mp.Block(mp.Vector3(wg_sx, wg_sy, wg_sz), 
                        center=mp.Vector3(0, 0, 0),
                        material=Si3N4)]


Next, we introduce the source of the electric field to our simulation. It is located on the left side of our simulation cell in the waveguide and it is defined as a line across the waveguide. We also specify the free space wavelength (in µm) of the field it generates. Here we also use the ``width`` parameter (in Meep time units) to turn on the source gradually in order to avoid exciting other frequencies because of the discontinuity. Also here it is customary to store the sources in a list called ``sources``. 

.. 
   TODO: Tutki tuo width-parameter-juttu?

.. code-block:: python

   # coordinates of the source
   source_x = -3
   source_y = 0
   source_z = 0

   wavelength = 1.55
   width = 5

   sources = [mp.Source(mp.ContinuousSource(wavelength=wavelength, width=width),
                        component=mp.Ez, 
                        center=mp.Vector3(source_x, source_y, source_z),
                        size=mp.Vector3(0, wg_sy, 0))]


It should also be specified what happens at the edges of the simulation cell. In this case, we want that when the field meets the boundary of the simulation cell, it does not reflect and interfere with the field propagating towards the boundaries. This can be done with the perfectly matched layers (PML) which absorb the incident field. Note that they need to have a finite thickness to avoid numerical errors related to the absorption. 

Here we create a 1-µm perfectly matched layers inside our simulation cell, stored in an object named ``pml_layers``. 

.. code-block:: python

   pml_w = 1.0
   pml_layers = [mp.PML(pml_w)]

We must specify the resolution according to which the space and time are discretized. In MEEP, this is done by a single variable ``resolution`` which defines the number of pixels per a distance unit. We will set it to 20 for now, but we will get back to this once we are discussing the convergence study. 

.. code-block:: python

   resolution = 20
   
Finally, we define the simulation object which contains the different objects we have defined earlier.

.. code-block:: python

   sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

Now, before running the simulation, it is a good idea to check that we have set it up correctly. With a function ``plot2D()``, we can see the waveguide geometry, the source, and the perfectly matched layers. Don't forget to show (or save) the figure!

.. code-block:: python

   sim.plot2D()
   plt.show()

.. figure:: waveguide_figures/1_simulation_setup.pdf
   :alt: Simulation setup
   :width: 90%
   :align: center

You can see the waveguide geometry as the black area, the source as the red line and PMLs at the edges of the cell. Everything looks good! 

Running the simulation and basic visualization
----------------------------------------------

Now we can specify the time until which the simulation runs and run the simulation.

.. code-block:: python

   sim_time = 200
   sim.run(until=sim_time)

Now that the simulation has run, we can use ``plot2D()`` to see what kind of result we have obtained. Here we need to tell which field and which component we want to see. We'd like to see the :math:`z`-component of the electric field.

.. code-block:: python
    
   sim.plot2D(fields=mp.Ez)
   plt.show()

.. figure:: waveguide_figures/2_after_first_run.pdf
   :alt: After first run
   :width: 90%
   :align: center

It is nice figure but if you want more control how to figure looks, you could plot it more manually. Here we show one way of extracting the data from the simulation after it has run and visualizing it using Matplotlib. 

.. code-block:: python

   # dielectric constant in the simulation cell (i.e. the waveguide geometry)
    eps_data = sim.get_array(center=mp.Vector3(0, 0, 0), size=cell, component=mp.Dielectric).T 

   # the z-component of the electic field
   ez_data = sim.get_array(center=mp.Vector3(0, 0, 0), size=cell, component=mp.Ez).T

   fig = plt.figure()
   ax = fig.add_subplot(1, 1, 1)
   ax.imshow(eps_data, extent=[0, cell_x, 0, cell_y], cmap='binary')

   # determine the global minimum and maximum of the field
   data_min = ez_data.min()
   data_max = ez_data.max()
   print(f"min: {data_min}, max: {data_max}")

   # create a custom colormap where zero is white
   cmap = cm.RdBu  # Base colormap
   norm = TwoSlopeNorm(vmin=data_min, vcenter=0, vmax=data_max)

   ez_image = ax.imshow(ez_data, extent=[0, cell_x, 0, cell_y], cmap=cmap,
                        norm=norm, alpha=0.9, interpolation="spline16")
   fig.colorbar(ez_image, label="Electric field", ax=ax, orientation='horizontal')
   ax.set_xlabel(r"$x$ (µm)")
   ax.set_ylabel(r"$y$ (µm)")
   plt.show()


.. figure:: waveguide_figures/3_improved_figure.pdf
   :alt: Improved figure
   :width: 90%
   :align: center


Investigating the convergence
-----------------------------

One important question might arise at some point during our numerical experiments: do we know that the current resolution is sufficient? We can determine the sufficient resolution by doing a convergence study. In practice, this can be done by making the simulation with different resolutions and seeing when the simulation results do not change (significantly), i.e. seeing if the simulation has been converged. 

There are multiple ways of doing this but here we choose to inspect the magnitude of the electric field at two different points in and just outside the waveguide after the simulation is run. By doing this, we can get some sort of an idea about the sufficiency of the resolution. 

.. code-block:: python

   resolutions = np.array([5.0, 10.0, 20.0, 40.0, 80.0])

   wg_point = mp.Vector3(4, 0, 0)
   wg_results = np.zeros_like(resolutions, dtype=object)

   outside_point = mp.Vector3(2, 0.27, 0)
   outside_results = np.zeros_like(resolutions, dtype=object)


   for i, resolution in enumerate(resolutions):
      sim = mp.Simulation(
         cell_size=cell,
         boundary_layers=pml_layers,
         geometry=geometry,
         sources=sources,
         resolution=resolution,
         force_complex_fields=True,
      )
      sim.run(until=sim_time)

      wg_results[i] = np.abs(sim.get_field_point(mp.Ez, wg_point))
      outside_results[i] = np.abs(sim.get_field_point(mp.Ez, outside_point))


   fig = plt.figure()

   ax = fig.add_subplot(1, 1, 1)
   ax.plot(
      resolutions, wg_results, "x-", linewidth=2, label="A point inside the waveguide"
   )
   ax.plot(
      resolutions,
      outside_results,
      "o-.",
      linewidth=2,
      label="A point just outside the waveguide",
   )

   ax.set_xscale("log")
   ax.minorticks_off()
   ax.set_xticks(resolutions)
   ax.set_xticklabels([str(int(r)) for r in resolutions])
   ax.set_xlabel("Resolution (px/µm)")
   ax.set_ylabel(r"$|E_z|$ (arb. unit)")
   ax.legend(loc=0)

   plt.show()

This results in the following figure:

.. figure:: waveguide_figures/4_convergence_study.pdf
   :alt: Convergence study
   :width: 90%
   :align: center

We see that the results are quite well converged when the resolution is 20 which is the resolution we used earlier in the demo by a pure coincidence. Increasing the resolution does not change the obtained result very much, so it is not worth the increase in computation time. In this case, we could even set the resolution to 15 if we wanted to run the simulation somewhat faster without compromising the accuracy too much. Setting the resolution is a tradeoff between computation time and accuracy. 


.. note::
   A good rule of thumb for a good resolution would be for example 10 pixels/wavelength in the region with highest refractive index. In this case, this would give us a resolution of

   .. math::

      \frac{10 \text{px}}{\lambda/n}=2\cdot\frac{10 \text{px}}{\cdot 1.55 \text{µm}} \approx 13 \text{px/µm}

   which is in good agreement with our results. Nevertheless, it is not a bad idea to investigate the sufficient resolution with a convergence experiment. 

Animating the results
---------------------
Even though we are now satisfied with our resolution, we might not be as pleased with only some boring static figures about the end state of our simulation. Let's fix the situation by doing an animation! 

Let's get started by redefining the simulation object with a resolution of 20. 

.. code-block:: python

   resolution = 20
   sim = mp.Simulation(
      cell_size=cell,
      boundary_layers=pml_layers,
      geometry=geometry,
      sources=sources,
      resolution=resolution,
   )


To animate the simulation, we naturally need to gather data also during it. This can be done by introduce some *step functions* in our run command. This time we want to store the field data into an h5 file from which we can read the data later. At the beginning of the run, we store the waveguide geometry to the file and at every 0.1 timesteps, we also store the :math:`z`-component of the electric field. Note that we have also reduced the simulation time to avoid an overly long animation.

.. code-block:: python
   filename = "sim_data"
   sim_time = 50
   save_time_step = 0.1
   sim.run(
         mp.to_appended(filename,
                        mp.at_beginning(mp.output_epsilon),
                        mp.at_every(save_time_step, mp.output_efield_z)),
         until=sim_time)

After running the simulation, we should have ended up with a file the name of which end with ``sim_data.h5``. It contains the data about the geometry in a dataset called ``eps`` and the electric data in a dataset called ``ez``. Next we read the data from the file. 

.. code-block:: python

   with h5py.File(filename + ".h5", "r") as f:
      # Remember to transpose!
      eps_data = f["eps"][:,:,0].T
      ez_data = f["ez"][:,:,:].T

And now we create the animation. 

.. code-block:: python

   # Create a figure and axis for plotting
   fig = plt.figure()
   ax = fig.add_subplot(1, 1, 1)

   # Plot the eps_data as the background (fixed layer)
   background = ax.imshow(
      eps_data,
      extent=[0, cell_x, 0, cell_y],
      interpolation="spline36",
      cmap="binary",
      origin="lower",
      alpha=0.8,  # Adjust transparency for blending
   )

   # Determine the global minimum and maximum of the field
   data_min = ez_data.min()
   data_max = ez_data.max()
   print(f"min: {data_min}, max: {data_max}")

   # Create a custom colormap where zero is white
   cmap = cm.RdBu  # Base colormap
   norm = TwoSlopeNorm(vmin=data_min, vcenter=0, vmax=data_max)

   # Initialize the plot with the first time slice of the animated data
   img = ax.imshow(
      ez_data[0,:,:],
      extent=[0, cell_x, 0, cell_y],
      interpolation="spline16",
      cmap=cmap,
      norm=norm,
      origin="lower",
      alpha=0.7,
   )

   # Set up title, labels, etc.
   ax.set_title(r"$t=$0")
   ax.set_xlabel(r"$x$ (µm)")
   ax.set_ylabel(r"$y$ (µm)")

   # Define the update function for each frame of the animation
   def update(frame):
      # Update only the image data for the current time slice
      img.set_data(ez_data[frame,:,:],)
      # Update the title to show the current time
      ax.set_title(fr"$t=${frame*save_time_step}")

      return (img,)  # Return the updated image object

   # Create the animation
   ani = FuncAnimation(fig, update, frames=range(ez_data.shape[0]), interval=100, 
                     blit=True)
   ani.save("5. animation.gif", writer="imagemagick", fps=30, dpi=100)

Now we have a nice animation! Note that we can observe here how to source turns on gradually, how the PML absorbs the incident field, and also the evanascent fields traveling outside the waveguide! 

.. figure:: waveguide_figures/5_animation.gif
   :alt: Convergence study
   :width: 90%
   :align: center

.. note:: 
   Creating an animation this way is only feasible with small simulations. With large simulations, the size of the h5 file can grow rapidly. Tips for outputting data (and animating) the field propagation are presented `here <https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/#a-90-bend>`_ in the official MEEP documentation. 



.. [1] K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, M. Lipson. Broadband mid-infrared frequency comb generation in a Si3N4 microresonator. Opt. Lett. 40, 4823-4826 (2015)
