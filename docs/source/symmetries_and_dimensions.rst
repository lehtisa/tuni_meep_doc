
.. _symmetries_and_dimensions:

This is symmetries and dimensions.

============
Symmetries and dimensions
============

.. _symmetries_and_dimensions:

Demo 1: 2D-case
========================== 


We will firstly import all necessary libraries. In addition to the
libraries mentioned in previous sections we will also import 'h5py'.
H5py is a python package used for storing data in H5DF-form. In this
demo we will introduce only the very basics of using this package. A
more thorough tutorial can be found on LINK.


.. code-block :: python

   import h5py
   import meep as mp
   import matplotlib.pyplot as plt
   import numpy as np
   import os



Our simulation cell will be a 30 by 20 rectangle. For a more modular
code, we will make variable for the center of our cell. We form the
single slit by adding two blocks to the cell. We make sure the wall
is not penetrated by giving the blocks substantial values for
epsilon. The cell will also be surrounded by the PML layer as in most
cases.



.. code-block :: python

   domain = [0, 30, -10, 10]

   center = mp.Vector3(
         (domain[1] + domain[0]) / 2,
         (domain[3] + domain[2]) / 2,
         )

   cell = mp.Vector3(
         domain[1] - domain[0],
         domain[3] - domain[2],
         )

   wallPos = 10
   wallWidth = 0.5
   apertureWidth = 1.5

   wallMaterial = mp.Medium(epsilon = 1e6)
   wallLength = (cell.y-2*apertureWidth)/2

   geometry = [mp.Block(mp.Vector3(wallWidth, wallLength, mp.inf),
               center = mp.Vector3(wallPos - center.x, domain[3] - wallLength/2),
               material = wallMaterial),
               mp.Block(mp.Vector3(wallWidth, wallLength, mp.inf),
               center = mp.Vector3(wallPos - center.x, domain[2] + wallLength/2),
               material = wallMaterial)]

   pmlThickness = 1
   pmlLayers = [mp.PML(pmlThickness)]



Our light source will produce 500 nm light as a wavefront which spans
the whole y-dircetion of the cell. We use Meeps ContinousSource for
this and we set it just outside the left border PML.



.. code-block :: python

   sourceLambda = 0.5  # in μm
   sourceFrequency = 1 / sourceLambda

   source = mp.Source(
         src=mp.ContinuousSource(
            frequency=sourceFrequency,
            is_integrated=True,
            ),
         component=mp.Ez,
         center= mp.Vector3(1, 0, 0) - center,
         size=mp.Vector3(y=cell.y),
         )



The resolution of our simulation will be proportional to the smallest
length of our simulation. We also set the force_complex_fields to
True. This will automatically double the simulation time but
including the complex phase terms of our field is crucial for
precisely simulating interference.



.. code-block :: python

   smallestLength = min(
         sourceLambda,
         wallWidth,
         apertureWidth
   )

   pixelCount = 10
   resolution = int(np.ceil(pixelCount / smallestLength))

   sim = mp.Simulation(
         cell_size=cell,
         sources=[source],
         boundary_layers=pmlLayers,
         geometry=geometry,
         resolution=resolution,
         force_complex_fields=True
         )



We extract the field and dielectricc data into a h5 file. This is a
bit unnecessary for the 2D-case, but will come in very handy when we
increase dimensions to our simulation. Our h5-file consists of a
dataset for the dielectric data, which stays the same during our
simulation and therefore only needs one set of values, and a dataset
for the field values which has two sets of values; one of the initial
state of the simulation and one of the final state.



.. code-block :: python


   # Convenience method to extract Ez and dielectric data
   def getData(sim, cellSize):
         ezData = sim.get_array(
            center=mp.Vector3(), size=cellSize, component=mp.Ez)
         epsData = sim.get_array(
            center=mp.Vector3(), size=cellSize, component=mp.Dielectric)
         return ezData, epsData

   # Where to save the results
   simPath = 'simulation.h5'

   def simulate(sim, simPath):
         
         # Remove previous sim file, if any
         if os.path.exists(simPath):
            os.remove(simPath)
         
         # Save data to an HDF5 binary file
         with h5py.File(simPath, 'a') as f:
            
            # Save initial state as first frame
            sim.init_sim()
            ezData, epsData = getData(sim, cell)
            f.create_dataset(
               'ezData',
               shape=(2, *ezData.shape),
               dtype=ezData.dtype,
               )
            f.create_dataset(
               'epsData',
               shape=epsData.shape,
               dtype=epsData.dtype,
               )
            f['ezData'][0]  = ezData
            f['epsData'][:] = epsData
         
            # Run until the next frame time
            sim.run(until=cell[0]+10)

            # Capture electral field data    
            ezData, _ = getData(sim, cell)
            f['ezData'][1]  = ezData



After defining the simulation and field extraction scheme, we can run
the simulation. The h5-file uses straightforward NumPy and Python
methaphors wich makes extracting the data back into our code trivial.



.. code-block :: python

   simulate(sim, simPath)

   # Grab final simulation snapshot without time-averaging
   with h5py.File(simPath, 'r') as f:
         finalSnap = f['ezData'][1]
         finalEps = f['epsData'][:]



Finally we plot the data. Each y-directional slice of the simulation
is expressed as a vector of one axis. To help visualize the
diffraction pattern, we use NumPys vstack-command.



.. code-block :: python


   # Compute intensity as square of the complex amplitude
   finalSnap = np.abs(finalSnap)**2
   vmax = np.max(finalSnap[-1])

   # Plot simulation
   plt.figure(1)
   plt.imshow(finalEps.T,
               cmap='binary')
   plt.imshow(finalSnap.T,
               interpolation='spline36',
               cmap='inferno',
               alpha=0.9)
   plt.axis('off')
   plt.show()

   plt.figure(2)
   plt.imshow(
            np.vstack(finalSnap[-1]).T,
            cmap='inferno',
            aspect='auto',
            vmax=vmax,
            )
   plt.axis('off')
   plt.show()
