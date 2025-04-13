import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sx=32
sy=20
cell = mp.Vector3(sx,sy,0)

dpml=1.0
pml_layers=[mp.PML(dpml)]

full_radius = 8
sources = [mp.Source(mp.ContinuousSource(frequency=1),
                     component=mp.Ez,
                     center=mp.Vector3(-full_radius,0))]

geometry=[]
i=0
sphere_num = 100
while i<sphere_num:
    geometry.append(mp.Sphere(center=mp.Vector3(0,0),
            radius=full_radius-i*full_radius/sphere_num,
            material=mp.Medium(index=np.sqrt(2-((sphere_num-i)/sphere_num) ** 2))))
    i = i+1

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
plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary", extent=[-sx/2,sx/2,-sy/2,sy/2])
plt.imshow(ez_data.transpose(), interpolation="spline36", cmap="RdBu", norm=norm, alpha=0.6, extent=[-sx/2,sx/2,-sy/2,sy/2])
plt.savefig("luneburg_end_field.png", bbox_inches='tight')
plt.show()

sim.reset_meep()

f = plt.figure(dpi=100)
Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)
plt.close()

sim.run(mp.at_every(1, Animate), until=70)
plt.close()

filename = "./Luneburg_lens_animation.mp4"
Animate.to_mp4(10, filename)

