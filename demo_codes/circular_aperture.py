import h5py
import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import os

dimensions = mp.CYLINDRICAL
cellRadius = 20.0
cellLength = 40.0

cell_size = mp.Vector3(cellRadius,0,cellLength)

wallPos = 10
wallWidth = 0.5
apertureRadius = 1
wallMaterial = mp.Medium(epsilon = 1e6)

geometry = [mp.Block(mp.Vector3(cellRadius - apertureRadius, 1e20, wallWidth),
                        center = mp.Vector3((cellRadius+apertureRadius)/2, 0, -17),
                        material = wallMaterial)]

pmlThickness = 1.0
pmlLayers = [mp.PML(pmlThickness)]

sourceLambda = 0.5
sourceFrequency = 1 / sourceLambda

sources = [mp.Source(mp.ContinuousSource(sourceFrequency,fwidth=0.2*sourceFrequency,is_integrated=True),
                    component=mp.Er,
                    center=mp.Vector3(0.5*cellRadius,0,-0.5*cellLength+1),
                    size=mp.Vector3(cellRadius))]

resolution = 25

sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pmlLayers,
                    resolution=resolution,
                    geometry=geometry,
                    sources=sources,
                    dimensions=dimensions,
                    force_complex_fields=True,
                    m=-1)

sim.run(until=cellLength+10)

nonpmlVol = mp.Volume(center=mp.Vector3(0.5*cellRadius),
                    size=mp.Vector3(cellRadius,0,cellLength))
erData = sim.get_array(component=mp.Er,vol=nonpmlVol)

r = np.linspace(0,cellRadius,erData.shape[1])
z = np.linspace(-0.5*cellLength,0.5*cellLength,erData.shape[0])

theta = np.linspace(0,2*np.pi, 100)

#making up some data    
theta,r = np.meshgrid(theta,r)
values_2d = np.sin(theta)*np.exp(-r)

plt.subplots(1,1,subplot_kw=dict(projection='polar'))
plt.pcolormesh(theta,r,np.tile(np.real(erData[-10]),
                    (100,1)).T, cmap='inferno', shading='gouraud')
plt.axis('off')
plt.show()