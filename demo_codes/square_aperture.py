import h5py
import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import os

SOL = 299792458e-9

domain = [0, 30, -10, 10, -10, 10]

center = mp.Vector3(
        (domain[1] + domain[0]) / 2,
        (domain[3] + domain[2]) / 2,
        (domain[5] + domain[4]) / 2
        )

cell = mp.Vector3(
        domain[1] - domain[0],
        domain[3] - domain[2],
        domain[5] - domain[4]
        )

symmetries = [mp.Mirror(mp.Y),
                mp.Mirror(mp.Z, phase=-1)]

wallPos = 10
wallWidth = 0.5
apertureWidth = 1.5
wallMaterial = mp.Medium(epsilon = 1e6)
wallLength = (cell[1]-2*apertureWidth)/2

geometry = [mp.Block(mp.Vector3(wallWidth, wallLength, mp.inf),
            center = mp.Vector3(wallPos - center.x, domain[3] - wallLength/2
                                , 0), material = wallMaterial),
            mp.Block(mp.Vector3(wallWidth, wallLength, mp.inf),
            center = mp.Vector3(wallPos - center.x, domain[2] + wallLength/2
                                , 0), material = wallMaterial),
            mp.Block(mp.Vector3(wallWidth, mp.inf, wallLength),
            center = mp.Vector3(wallPos - center.x, 0
                                , domain[3] - wallLength/2), material = wallMaterial),
            mp.Block(mp.Vector3(wallWidth, mp.inf, wallLength),
            center = mp.Vector3(wallPos - center.x, 0
                                , domain[2] - wallLength/2), material = wallMaterial)]

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
        size=mp.Vector3(y=cell[1],z=cell[2]),
        )

smallestLength = min(
        sourceLambda,
        wallWidth,
        apertureWidth
)

pixelCount = 10
#resolution = int(np.ceil(pixelCount / smallestLength))
resolution = 10

sim = mp.Simulation(
        cell_size=cell,
        sources=[source],
        boundary_layers=pmlLayers,
        geometry=geometry,
        resolution=resolution,
        force_complex_fields=True,
        symmetries = symmetries
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

simulate(sim, simPath)

# Grap the final frame
with h5py.File(simPath, 'r') as f:
        finalSnap = f['ezData'][1]

# Compute intensity as square of the complex amplitude
finalSnap = np.abs(finalSnap)**2
vmax = np.max(finalSnap[-1])

plt.figure(2)
plt.imshow(
        finalSnap[-1].T,
        cmap='inferno',
        aspect='auto',
        vmax=vmax,
        )
plt.axis('off')
plt.show()