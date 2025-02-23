======
Theory
======

.. _theory:

.. note::
    This section is adapted from the `offcial Meep documentation <https://meep.readthedocs.io/en/latest/Introduction/>`_,
    which provides a more thorough look on the theoretical background of Meep.

Maxwell's equations
===================

Meep uses Maxwell's equations for its simulations. They are a set of four partial differential equations that describe how
electric (:math:`\mathbf{E}`) and magnetic (:math:`\mathbf{H}`) fields interact with one another. The field can be presented in a way that
shows their time evolution:

.. math::

    \frac{d\mathbf{B}}{dt} = -\nabla\times\mathbf{E} - \mathbf{J}_B - \sigma_B \mathbf{B}

    \mathbf{B} = \mu \mathbf{H}

    \frac{d\mathbf{D}}{dt} = \nabla\times\mathbf{H} - \mathbf{J} - \sigma_D \mathbf{D}

    \mathbf{D} = \varepsilon \mathbf{E}

where :math:`\mathbf{B}` is the magnetic flux density and :math:`\mu` is the magnetic permeability. :math:`\mathbf{B}` is often also
called the magnetic field, but should not be confused with (:math:`\mathbf{H})`. :math:`\mathbf{D}` is the displacement field,
:math:`\varepsilon` is the dielectric constant, :math:`\mathbf{J}` is the current density of electric charge and :math:`\mathbf{J}_B`
is the magnetic-charge current density. The terms :math:`\sigma_B` and :math:`\sigma_D` refer to the frequency-independent magnetic and
electric conductivities respectively. The divergence equations for the Maxwell's equations can be written implicitly as follows:

.. math::

    \nabla \cdot \mathbf{B} = - \int^t \nabla \cdot (\mathbf{J}_B(t') + \sigma_B \mathbf{B}) dt'

    \nabla \cdot \mathbf{D} = - \int^t \nabla \cdot (\mathbf{J}(t') + \sigma_D \mathbf{D})dt' \equiv \rho

Generally,  depends not only on position but also

Usually, the position, frequency (material dispersion) and the field :math:`\mathbf{E}` itself (nonlinearity) affect :math:`\varepsilon`
and may cause loss or gain of absorption in the material. You can read more about the effects of materials and how to apply them
in the `offcial documentation <https://meep.readthedocs.io/en/latest/Materials/>`_.

Meep also supports cylindrical coordinates for simulating Maxwell's equations. You can read more about how to apply them
in the `official documentation <https://meep.readthedocs.io/en/latest/Python_Tutorials/Cylindrical_Coordinates/>`_ as well.

Units in Meep
-------------

Meep uses dimensionless units in its calculations. Most of the results from computation are expressed as a ratio,
which means that units end up cancelling. Maxwell's equations are scale invariant, which means that choosing scale-invariant units
to use when solving problems with Meep is rather convinient.

Let's choose some lengthscale :math:`a` as the unit of distance for a system. Since Meep defines :math:`c=1`, :math:`a` (or :math:`a/c`)
is also the unit of time for the system. The frequency :math:`f` (which corresponds to a time dependence :math:`e^{-i 2\pi f t}`)
is defined in units of :math:`c/a` (and similarly :math:`\omega` is defined in units of :math:`2πc/a`). This means that :math:`f = 1/T`, meaning that
the inverse of the optical period :math:`T` is defined in units of :math:`a/c` and as result, defining :math:`f = a/\lambda`, where :math:`\lambda` is the vacuum wavelength.


Boundary Conditions and Symmetries
==================================

It's possible to set three different kinds of boundary conditions to terminate a Meep simulation: Bloch-periodic boundaries,
metallic walls, and perfectly matched layers (PML) absorbing layers. Boundary conditions are needed to reduce computational requirements and limit the size
of the simulated region. It's possible to also use symmetries of problems to do so, as well.

**Bloch-periodic boundary** is a generalization of a normal periodic boudary. With cell size :math:`L`, a periodic boudary meets satisfies
:math:`f(x+L) = f(x)`, whereas Bloch-periodic boundary satisfies :math:`f(x+L) = e^{ik_x L} f(x)`, where :math:`\mathbf{k}` is a Bloch wavevector.
**Metallic wall** is a simpler boudary condition, in which the fields are forced to be zero on the boudaries. This perfect metal
is assumed to have zero absorption and zero skin depth in the simulation. **PML** is technically not a boundary condition,
but rather a fictitious absorbing material that has zero reflections at its interface. It is placed adjacent to the boundaries.
In a discretized system PML is imperfect, as it has some small reflections, unlike in a theoretical continuous system. As a result,
PML should be given a finite thickness, around half the wavelength.

Meep can exploit symmetries to reduce the size of a computational cell. Using mirror and rotational symmetries, it is easy to
optimize simulations and only simulate necessary parts of a cell. You can read more about how to exploit different symmetries
in Meep simulations in the `offcial documentation <https://meep.readthedocs.io/en/latest/Exploiting_Symmetry/>`_.


Finite-Difference Time-Domain Methods
=====================================

FDTD methods divide space into a discrete, rectangular grid and evolve fields by discrete time steps.
When the temporal and spatial steps are made smaller and smaller, we essentially get exact results from the simulations,
as we approach continuous equations. When using Meep, there are two main effects of using discrete grids that should be taken into account.

The first main point is about resolution of the simulations. With some spatial resolution :math:`\Delta x`, discrete time-step :math:`\Delta t`
is given by :math:`\Delta t = S \Delta x`, where :math:`S` is the Courant factor. For the method to be stable and not diverge,
the Courant factor must satisfy the condition :math:`S < n_\textrm{min} / \sqrt{\mathrm{\# dimensions}}`,
where :math:`n_\textrm{min}` is the minimum refractive index, usually 1. Meep uses :math:`S=0.5` by default,
which is usually good enough for 1-, 2- or 3-dimensional simulations. Essentially what this means is that
doubling the grid resolution doubles the number of time steps involved. This means that in 3-dimensional simulations
time steps increase by 8 times.

The second main point is about discretization using Yee lattice. FDTD methods need to store different field components at
different grid locations so that Maxwell's equations can be discretized with second-order accuracy. They do this using a Yee lattice,
and as a result, the field components need to be interpolated to some common point so that the accuracy stays
when combining, comparing, or outputing these components. Meep does this often automatically, but usually around dielectric interfaces
electric and displacement fields might be less accurate when interpolated linearly. You can read a more about the specifics of a
Yee lattice in the `official documentation <https://meep.readthedocs.io/en/latest/Yee_Lattice/>`_.

The Illusion of Continuity
--------------------------

Dispite using a discrete system, Meep tries to hide this as much as possible to make it seem like the system is actually continuous.
Meep uses subpixel smoothing, a kind of pervasive interpolation, to make this happen, where making changes in the inputs in the
simulation continuously will have Meep respond continuously as well. Meep will try to keep the convergence of the simulation as
smooth and rapid as possible with increased spatial resolution. You can read more about subpixel smoothing and how it works
in the `official documentation <https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/>`_.
