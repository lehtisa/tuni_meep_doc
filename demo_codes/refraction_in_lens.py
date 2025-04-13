import meep as mp
import numpy as np
import matplotlib.pyplot as plt
    
sx=64
sy=10
cell = mp.Vector3(sx,sy,0)
dpml=0.5
pml_layers=[mp.PML(dpml)]
    
freq = 1
sources = [
mp.Source(mp.ContinuousSource(frequency = freq, is_integrated=True),
center=mp.Vector3(-0.5 * sx + dpml+0.5, 0, 0),
size=mp.Vector3(0,sy),
component = mp.Ez)
]


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
    
resolution=30
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution,
                    force_complex_fields=True)
                    
sim.run(until=75)

plt.figure(dpi=150)
sim.plot2D()
plt.savefig("lens_plot2D.png", bbox_inches='tight')

eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
plt.figure()
plt.imshow(np.real(eps_data).transpose(), interpolation="spline36", cmap="binary", origin="lower", extent=[-sx/2,sx/2,-sy/2,sy/2])
plt.imshow(np.real(ez_data).transpose(), interpolation="spline36", cmap="RdBu", alpha=0.9, origin="lower", extent=[-sx/2,sx/2,-sy/2,sy/2])
plt.savefig("lens_end_field.png", bbox_inches='tight')

center_dataEz = sim.get_array(center=mp.Vector3((sx/2+offset)/2-dpml), size=mp.Vector3((sx/2-offset-dpml),0,0), component=mp.Ez)
center_dataSx = sim.get_array(center=mp.Vector3((sx/2+offset)/2-dpml), size=mp.Vector3((sx/2-offset-dpml),0,0), component=mp.Sx)
pointsEz = np.linspace(0,sx/2-offset-dpml,len(center_dataEz))
pointsSx = np.linspace(0,sx/2-offset-dpml,len(center_dataSx))

Ez2Sx = []
i = 0
while i < len(center_dataEz):
    Ez2Sx.append(abs(center_dataEz[i])**2/abs(center_dataSx[i]))
    i = i+1

focal_length_sim = pointsSx[np.argmax(center_dataSx)]
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

sim.reset_meep()

f = plt.figure(dpi=100)
Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)
plt.close()

sim.run(mp.at_every(1, Animate), until=75)
plt.close()

filename = "./Lens_animation.mp4"
Animate.to_mp4(20, filename)
