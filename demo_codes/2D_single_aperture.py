import h5py
import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import os

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

   # Method for extracting Ez and dielectric data
def getData(sim, cellSize):
        ezData = sim.get_array(
        center=mp.Vector3(), size=cellSize, component=mp.Ez)
        epsData = sim.get_array(
        center=mp.Vector3(), size=cellSize, component=mp.Dielectric)
        return ezData, epsData

# Where to save the results
simPath = 'simulation.h5'

def simulate(sim, simPath):
        
        # Remove previous sim file
        if os.path.exists(simPath):
        os.remove(simPath)
        
        # Save data to an HDF5 file
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
            
            # Run until the the desired length
            sim.run(until=cell[0]+10)

            # Capture electral field data    
            ezData, _ = getData(sim, cell)
            f['ezData'][1]  = ezData

simulate(sim, simPath)

# Grab dielectric and Ez data from the file
with h5py.File(simPath, 'r') as f:
        finalSnap = f['ezData'][1]
        finalEps = f['epsData'][:]

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