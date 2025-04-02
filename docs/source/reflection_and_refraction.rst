==========
Refraction
==========

.. _reflection_and_refraction:

In this chapter we show how to simulate refraction in Meep. There are two demos. In the first, one we observe refraction in lens with spherical surface and simulate the intensity of the electric field to estimate the focal length of the lens. The second simulation is about Luneburg Lens.

Demo 1: Refraction in lens
========================== 

First we import the libraries we need in this simulation.

.. code-block :: python

    import meep as mp
    import numpy as np
    import matplotlib.pyplot as plt

We create simulation area cell and perfectly mached layers around it.

.. code-block :: python

    sx=64
    sy=10
    cell = mp.Vector3(sx,sy,0)
    dpml=0.5
    pml_layers=[mp.PML(dpml)]

The source used in this simulation is a continuous source, which is located at 0.5 distance from PML layers of the left side of the cell and which spans over y-direction of the cell.

.. code-block :: python

    freq = 1
    sources = [
    mp.Source(mp.ContinuousSource(frequency = freq, is_integrated=True),
    center=mp.Vector3(-0.5 * sx + dpml+0.5, 0, 0),
    size=mp.Vector3(0,sy),
    component = mp.Ez)
    ]

We create the lens object by creating a sphere with defined radius and then create a block with refractive index of 1 which cuts off the wanted slice of the sphere at distance of lens width. The offset tells how much the lens is moved from the center of the cell. The radius of the lens can be changed, but then you have to notice how much of the lens fits into the cell in y-direction. By changing the variable sy defined before creating the cell you can make the lens fit better to your cell. Also, if you increase the radius of the lens or reduce the refractive index of the lens, you need to make sure that the cell is long enough for you to be able to see the focal point of the lens. You can do this by changing the variable sx.

.. code-block :: python

    offset = -7
    radius = 8
    lens_index = 1.5
    lens_width = 2
    geometry=[mp.Sphere(center=mp.Vector3(offset+radius,0),
        radius=radius,
        material=mp.Medium(index=lens_index)),
        mp.Block(mp.Vector3(2*radius,mp.inf,mp.inf),
        center=mp.Vector3(offset+lens_width+radius,0),
        material=mp.Medium(index=1.0))
        ]

Then we create the simulation object with all the cell, PML layers, geometry, and source created above. You can change the value of resolution to a bigger value, but the simulations might take longer to run. We force complex field on, because then we can easily measure the intensity of the electric field.

.. code-block :: python

    resolution=30
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        force_complex_fields=True)


Now we run the simulation until 75. This time is long enough because then the wave has reached the right side of the cell and we do not observe change. We can use plot2D to easily see the geometry of the simulation. We can plot the electric field at the end of the simulation by using get_array. Now we have to notice that we are using complex fields so to be able to plot them we take absolute values of the field data. To get the imshow to show the plot not upside down, we define origin="lower". The ticks are defined manually because imshow otherwise shows the number of pixels on the axis.

.. code-block :: python

    sim.run(until=75)

    plt.figure(dpi=150)
    sim.plot2D()

.. figure:: refraction_figures/lens_plot2D.png
   :alt: test text
   :width: 90%
   :align: center

.. code-block :: python

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    plt.figure()
    plt.imshow(abs(eps_data).transpose(), interpolation="spline36", cmap="binary", origin="lower")
    plt.imshow(abs(ez_data).transpose(), interpolation="spline36", cmap="RdBu", alpha=0.9, origin="lower")
    plt.xticks(np.linspace(0,eps_data.shape[0],9),np.linspace(-sx/2,sx/2,9))
    plt.yticks(np.linspace(0,eps_data.shape[1],3),np.linspace(-sy/2,sy/2,3))

.. figure:: refraction_figures/lens_end_field.png
   :alt: test text
   :width: 90%
   :align: center

To estimate the focal length, we examine the electric field at the middle of the cell in y-direction and from after the lens until the right side PML layer in x-direction by using get_array and defining corresponding center and size. The values of the fields are spaced equally, so we can create the distance from the lens using linspace. We take both the z-component of the electric field and the x-component of the Poynting vector to show how they relate to intensity. We also calcute the ratio of square of Ez and Sx. A numeric value of focal length is obtained by taking the index of maximum value of Sx and taking the value of distance corresponding to that index. We also calculate a theoretic value for focal length using lensmaker's equation. Then we plot the values to visualize them and save the figure using savefig. The values for simulated and theoretic focal lengths are shown in the figure by plotting straight lines between two points with the same value (the focal length) as x-coordinates and different values for y-coordinates.

.. code-block :: python

    center_dataEz = sim.get_array(center=mp.Vector3((sx/2+offset)/2-dpml), size=mp.Vector3((sx/2-offset-dpml),0,0), component=mp.Ez)
    center_dataSx = sim.get_array(center=mp.Vector3((sx/2+offset)/2-dpml), size=mp.Vector3((sx/2-offset-dpml),0,0), component=mp.Sx)
    pointsEz = np.linspace(0,sx/2-offset-dpml,len(center_dataEz))
    pointsSx = np.linspace(0,sx/2-offset-dpml,len(center_dataSx))

    Ez2Sx = []
    i = 0
    while i < len(center_dataEz):
        Ez2Sx.append(abs(center_dataEz[i])**2/abs(center_dataSx[i]))
        i = i+1

    focal_length = pointsSx[np.argmax(center_dataSx)]
    focal_length_theoretic = radius/(lens_index-1)

    plt.figure()
    plt.plot(pointsEz, abs(center_dataEz), label='Ez')
    plt.plot(pointsEz, abs(center_dataEz) ** 2, label='Ez**2')
    plt.plot(pointsSx, abs(center_dataSx), label='Sx')
    plt.plot(pointsSx, Ez2Sx, label='Ez**2/Sx')
    plt.plot([focal_length_sim, focal_length_sim], [-0.5,2.5], label="simulated f = {:.2f}".format(focal_length_sim))
    plt.plot([focal_length_theoretic, focal_length_theoretic], [-0.5,2.5], label="theoretic f = {:.2f}".format(focal_length_theoretic))
    plt.ylim(0,2.3)
    plt.legend(loc="upper right")
    plt.xlabel('distance from lens')
    plt.ylabel('a.u.')
    plt.savefig("Intensity_after_lens.png", bbox_inches='tight')
    plt.show()

.. figure:: refraction_figures/Intensity_after_lens.png
   :alt: test text
   :width: 90%
   :align: center

We can create an animation of the simulation using Animate2D object and at_every function.

.. code-block :: python

    sim.reset_meep()

    f = plt.figure(dpi=100)
    Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)
    plt.close()

    sim.run(mp.at_every(1, Animate), until=75)
    plt.close()

    filename = "./Lens_animation.mp4"
    Animate.to_mp4(10, filename)

We notice that the ratio of :math:`\left|E_{z}\right|^{2}` and :math:`S_{x}` is 2 other than in the close vicinity from the lens where there is a lot of error. Intensity is defined as the magnitude of the Poynting vector but also corresponds to the square of the electric field [1]:

.. math::

    I=\frac{1}{2}cn\varepsilon_{0}\left|{E}\right|^{2}

In Meep speed of light and vacuum permittivity are defined as 1. Thus, the square of Ez is twice as large as Sz as it should be.

Focal length depends on refractive index and curvature of the lens according to lensmaker's equation [2]:

.. math::

    \frac{1}{f}=(n-1)\left(\frac{1}{R_{1}}-\frac{1}{R_{2}}+\frac{(n-1)d}{nR_{1}R_{2}}\right)

In this simulation, we have a simple case where only one of the surfaces is spherical, while the other one is flat. Thus :math:`R_{2}=\infty` and the equation simplify to:

.. math::

    f=\frac{R_{1}}{n-1}

The focal length is the length in the plot where the maximum intensity is reached. There is some error compared to Lens-Maker's equation because Lens-Maker's equation is based on geometrical optics, where Meep does wave optics. Meep takes into account effects such as diffraction, interference, near-field effects, and reflection at the lens' surface. In our simulation the size of lens is only around 10 times the wavelength, so the lens we consider is microscopic, which causes these phenomena to have more effect. There exist microlenses that can have a diameter as small as 10 micrometers [3]. Meep could be used to simulate larger lenses, which could be more meaningful, but the simulation times might grow long. In Meep the legths are defined as fractions, so by keeping the geometry the same and increasing frequency (decreasing wavelength) of the source we can simulate larger lens size. The resolution used in the simulation can limit the accuracy. When increasing the frequency, we need to increase the resolution to keep the results reasonable. Lensmaker's equation assumes paraxial approximation. Thus, the there is more error for thicker lens with smaller radius where the angle of incidence is larger.

Demo 2: Luneburg lens
=====================

Luneburg lens is a spherically symmetric gradient-index lens. The refractive index of the lens decreases radially from the center of the lens. Certain index profiles have the property to be able to create a perfect geometric image of any two concentric spheres to each other. The simplest solution out of infinite possible solutions for this kind of lens was proposed by Rudolf Luneburg in 1944. [4]

The libraries are imported and cell and perfectly mached layers created as usual. TwoSlopeNorm is used for normalizing the field.

.. code-block :: python

    import meep as mp
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    sx=32
    sy=20
    cell = mp.Vector3(sx,sy,0)

    dpml=1.0
    pml_layers=[mp.PML(dpml)]

We place a dot source at the radius of the lens (the largest sphere we are going to create).

.. code-block :: python

    full_radius = 8
    sources = [mp.Source(mp.ContinuousSource(frequency=1),
                        component=mp.Ez,
                        center=mp.Vector3(-full_radius,0))]

For Luneburg's solution for an ideal Luneburg lens we have the following equation for the refractive index:

.. math::

    n=\sqrt{2-{\left(\frac{r}{R}\right)}^2}

Where R is the full radius of the lens and r is the radial distance from center [4]. We can approximate the lens by creating a large number of overlapping spheres where the radius gets smaller and the refractive index grows larger as we iterate. This can be done easily in Python by using a while loop.

.. code-block :: python

    geometry=[]
    i=0
    sphere_num = 100
    while i<sphere_num:
        geometry.append(mp.Sphere(center=mp.Vector3(0,0),
                radius=full_radius-i*full_radius/sphere_num,
                material=mp.Medium(index=np.sqrt(2-((sphere_num-i)/sphere_num) ** 2))))
        i = i+1

We define the resolution, create the simulation object, and run the simulation. After that we can plot the field at the end of the simulation and save the figure. We use TwoSlopeNorm to normalize the field by the maximum value of the field.


.. code-block :: python

    resolution=20
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    sim.run(until=70)

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    norm = TwoSlopeNorm(vmin=-ez_data.max(), vcenter=0, vmax=ez_data.max())
    plt.figure()
    plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
    plt.imshow(ez_data.transpose(), interpolation="spline36", cmap="RdBu", norm=norm, alpha=0.6)
    plt.xticks(np.linspace(0,eps_data.shape[0],9),np.linspace(-sx/2,sx/2,9))
    plt.yticks(np.linspace(0,eps_data.shape[1],5),np.linspace(-sy/2,sy/2,5))
    plt.savefig("luneburg_end_field.png", bbox_inches='tight')
    plt.show()

.. figure:: refraction_figures/luneburg_end_field.png
   :alt: test text
   :width: 90%
   :align: center

We can also make an animation of the simulation using Animate2D object and at_every funtion.

.. code-block :: python

    sim.reset_meep()

    f = plt.figure(dpi=100)
    Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)
    plt.close()

    sim.run(mp.at_every(1, Animate), until=70)
    plt.close()

    filename = "./Luneburg_lens_animation.mp4"
    Animate.to_mp4(10, filename)

From the figure and the animation we can qualitatively see that after the luneburg lens the electric field has became collimated wave and the focal point of the lens lies at infinity.

.. [1] Wikipedia Intensity available:https://en.wikipedia.org/wiki/Intensity_(physics) referenced 11.2.2025
.. [2] Wikipedia Focal length available:https://en.wikipedia.org/wiki/Focal_length referenced 11.2.2025
.. [3] Wikipedia Microlens available: https://en.wikipedia.org/wiki/Microlens referenced 3.4.2025
.. [4] Wikipedia Luneburg Lens available:https://en.wikipedia.org/wiki/Luneburg_lens referenced 18.2.2025