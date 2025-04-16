import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm

width = 40
height = 20
cell = mp.Vector3(width, height, 0)

dpml = 1
pml_layers = [mp.PML(dpml)]

# Defining lengths and positions
aperture = 1 # Size of slits
gap = 4 # Gap between centers of slits  
center_length = gap - aperture # Length of center segment of wall
side_length = (height - center_length - 2*aperture)/2 # Length of side wall segments
material = mp.Medium(epsilon=1e7)
thickness = 0.5
wall_xpos = -width/2 + dpml + 3 # Position of wall on x-axis

# Defining the geometry of the cell
geometry = [
    mp.Block(
        mp.Vector3(thickness, side_length, mp.inf),
        center=mp.Vector3(wall_xpos, height/2-side_length/2, 0),
        material=material,
    ),
    mp.Block(
        mp.Vector3(thickness, side_length, mp.inf),
        center=mp.Vector3(wall_xpos, -height/2+side_length/2, 0),
        material=material,
    ),
    mp.Block(
        mp.Vector3(thickness, center_length, mp.inf),
        center=mp.Vector3(wall_xpos, 0, 0),
        material=material,
    )
]

# Defining plane wave current source
frequency = 2.0
wavelength = 1/frequency
sources = [
    mp.EigenModeSource(
        src=mp.ContinuousSource(frequency,
        is_integrated=True,
        width=5),
        center=mp.Vector3(-width/2+dpml+1,0,0),
        size=mp.Vector3(y=height),
        eig_band=1,
        eig_match_freq=True,
    )
]

# Resolution of the simulation
resolution = 20

# Defining the simulation settings
sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

sim.run(until=40)

plt.figure(dpi=300)
sim.plot2D()
plt.savefig('cell_info.pdf')
plt.savefig('cell_info.png')
plt.show()

eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
plt.figure(dpi = 300)
plt.imshow(eps_data.transpose(), extent = [0, width, 0, height], interpolation="spline36", cmap="binary")
plt.imshow(ez_data.transpose(), extent = [0, width, 0, height], interpolation="spline36", cmap="RdBu", alpha=0.9)
plt.xlabel(r"$x$ (µm)")
plt.ylabel(r"$y$ (µm)")
plt.savefig('field_info.pdf')
plt.savefig('field_info.png')
plt.show()
