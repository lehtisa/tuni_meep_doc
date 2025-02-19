============
Introduction
============

.. _introduction:

What is Meep?
=============

MIT Electromagnetic Equation Propagation (Meep) is a free and open-source program that uses
the finite-difference time-domain (FDTD) method to make simulations of electromagnetic phenomenon.
Meep simulations evolve fields by discrete time steps that are described by Maxwell’s equations.
With this method we can essentially get exact results as we make the temporal and spatial steps smaller and smaller.

Meep is available for Unix-like operating systems such as Linux, macOS and FreeBSD.
It can also be used in Windows, but it requires Windows Subsystem for Linux (WSL).
Meep simulations can be programmed using Python, Scheme or C++ as the programming language.
This documentation will use Python as the main interface,
as it is a very familiar programming language for many physicists and easy to learn.
Meep has been in active development since 2006 at Massachusetts Institute of Technology,
and is still being maintained by community effort in `GitHub <https://github.com/NanoComp/meep>`_.

Comparison to other electromagnetics programs
=============================================

The main applications of Meep are photonic and optics simulations made using FDTD.
When compared to the finite element method (FEM), another popular method that is used by many programs,
the FDTD method is much simpler. FEM can solve very complex problems, but requires more complex software as a result.
FDTD method is therefore good enough for more simple problems, but still has some flexibility for more complex problems.

To bring up some disadvantages, Meep cannot be installed natively on Windows,
requiring the user to install Windows Subsytem for Linux to run the program.
Also, unlike most other electromagnetics programs, MEEP doesn't have its own seperate graphical user interface (GUI).

One main advantage of Meep when compared to most other similar programs is the fact that it is open-source,
when most others are commercially licensed. This makes MEEP more readily available to the public.
Another main advantage is that MEEP covers a ton of different applications with the FDTD method.
Meep being both open-source and providing a large variety of applications means that developing
seperate programs for different types of research is not as necessary when it comes to using FDTD.

The goal of this documentation
==============================

Meep already has an official documentation that is much larger and more comprehensive `here <https://meep.readthedocs.io/en/master/#>`_.
However, this documentation is mainly meant to be for people new to using Meep.
The goal of this documentation is to help the user get a quick start on using Meep by providing
more easy to read instructions on how to use the program with Python, with a focus on some specific topics as examples.
As such, the documentation is devided into several different sections. The first two sections provide some basics on Meep:

* **Theory** dives briefly into the theoretical background on Maxwell’s equations and the FDTD method

* **Installation** provides a basic guide on how to install the program to computers that run Windows or MacOS

The following sections focus on the different topics, providing a brief overview,
after which some example demos with their coding are given and guided through.
The first two sections provide a good start on the basics on building a Meep simulation:

* **Waveguides** (A good starting point)

* **Double-slit diffraction** (Another good starting point)

* **Symmetries and dimensions** (2- and 3-dimensional simulations)

* **Refraction** (Refraction in a spherical lens, Luneburg Lens)

* **Nonlinear effects** (Second and third order nonlinear effects)
