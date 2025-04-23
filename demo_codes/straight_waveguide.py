import meep as mp
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm


cell_x = 10
cell_y = 5
cell_z = 0

cell = mp.Vector3(cell_x, cell_y, cell_z)

wg_sx = mp.inf  # length
wg_sy = 0.5  # width
wg_sz = mp.inf  # height

Si3N4 = mp.Medium(index=2.0)

geometry = [
    mp.Block(
        mp.Vector3(wg_sx, wg_sy, wg_sz), center=mp.Vector3(0, 0, 0), material=Si3N4
    )
]

# coordinates of the source
source_x = -3
source_y = 0
source_z = 0

wavelength = 1.55
width = 5

sources = [
    mp.Source(
        mp.ContinuousSource(wavelength=wavelength, width=width),
        component=mp.Ez,
        center=mp.Vector3(source_x, source_y, source_z),
        size=mp.Vector3(0, wg_sy, 0),
    )
]

pml_w = 1.0
pml_layers = [mp.PML(pml_w)]

resolution = 20

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

sim.plot2D()
plt.savefig("1_simulation_setup.pdf")
plt.show()

sim_time = 200
sim.run(until=sim_time)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Capture the image object returned by sim.plot2D
sim.plot2D(
    fields=mp.Ez,
    ax=ax,
    field_parameters={"colorbar": True},
    colorbar_parameters={"label": "Electric field"},
)

# Label axes
ax.set_xlabel(r"$x$ (µm)")
ax.set_ylabel(r"$y$ (µm)")
plt.savefig("2_after_first_run.pdf")
plt.show()


# dielectric constant in the simulation cell (i.e. the waveguide geometry)
eps_data = sim.get_array(
    center=mp.Vector3(0, 0, 0), size=cell, component=mp.Dielectric
).T

# the z-component of the electic field
ez_data = sim.get_array(center=mp.Vector3(0, 0, 0), size=cell, component=mp.Ez).T

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(
    eps_data, extent=[-cell_x / 2, cell_x / 2, -cell_y / 2, cell_y / 2], cmap="binary"
)

# determine the global minimum and maximum of the field
data_min = ez_data.min()
data_max = ez_data.max()
print(f"min: {data_min}, max: {data_max}")

# create a custom colormap where zero is white
cmap = cm.RdBu  # Base colormap
norm = TwoSlopeNorm(vmin=data_min, vcenter=0, vmax=data_max)

ez_image = ax.imshow(
    ez_data,
    extent=[-cell_x / 2, cell_x / 2, -cell_y / 2, cell_y / 2],
    cmap=cmap,
    norm=norm,
    alpha=0.9,
    interpolation="spline16",
)

fig.colorbar(ez_image, label="Electric field", ax=ax, orientation="horizontal")
ax.set_xlabel(r"$x$ (µm)")
ax.set_ylabel(r"$y$ (µm)")

plt.savefig("3_improved_figure.pdf")
plt.show()

# Convergence study
resolutions = np.array([5.0, 10.0, 20.0, 40.0, 80.0])

wg_point = mp.Vector3(4, 0, 0)
wg_results = np.zeros_like(resolutions, dtype=object)

outside_point = mp.Vector3(2, 0.27, 0)
outside_results = np.zeros_like(resolutions, dtype=object)


for i, resolution in enumerate(resolutions):
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        force_complex_fields=True,
    )
    sim.run(until=sim_time)

    wg_results[i] = np.abs(sim.get_field_point(mp.Ez, wg_point))
    outside_results[i] = np.abs(sim.get_field_point(mp.Ez, outside_point))


fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.plot(
    resolutions, wg_results, "x-", linewidth=2, label="A point inside the waveguide"
)
ax.plot(
    resolutions,
    outside_results,
    "o-.",
    linewidth=2,
    label="A point just outside the waveguide",
)

ax.set_xscale("log")
ax.minorticks_off()
ax.set_xticks(resolutions)
ax.set_xticklabels([str(int(r)) for r in resolutions])
ax.set_xlabel("Resolution (px/µm)")
ax.set_ylabel(r"$|E_z|$ (arb. unit)")
ax.legend(loc=0)

plt.savefig("4_convergence_study.pdf")
plt.show()

# Create the animation using command line tools
resolution = 20
sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

sim_time = 50
output_dir = "out"
save_time_step = 0.1
sim.use_output_directory(output_dir)
sim.run(mp.at_every(save_time_step, mp.output_png(mp.Ez, "-Zc dkbluered")), until=200)

# use e.g. imagemagick of gifski to create the gif at this point!

# Create the animation using Python
filename = "sim_data"
resolution = 20
sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

with h5py.File(filename + ".h5", "r") as f:
    # Remember to transpose!
    eps_data = f["eps"][:, :, 0].T
    ez_data = f["ez"][:, :, :].T

# Create a figure and axis for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Plot the eps_data as the background (fixed layer)
background = ax.imshow(
    eps_data,
    extent=[0, cell_x, 0, cell_y],
    interpolation="spline36",
    cmap="binary",
    origin="lower",
    alpha=0.8,  # Adjust transparency for blending
)

# Determine the global minimum and maximum of the field
data_min = ez_data.min()
data_max = ez_data.max()
print(f"min: {data_min}, max: {data_max}")

# Create a custom colormap where zero is white
cmap = cm.RdBu  # Base colormap
norm = TwoSlopeNorm(vmin=data_min, vcenter=0, vmax=data_max)

# Initialize the plot with the first time slice of the animated data
img = ax.imshow(
    ez_data[0, :, :],
    extent=[0, cell_x, 0, cell_y],
    interpolation="spline16",
    cmap=cmap,
    norm=norm,
    origin="lower",
    alpha=0.7,
)

# Set up title, labels, etc.
ax.set_title(r"$t=$0")
ax.set_xlabel(r"$x$ (µm)")
ax.set_ylabel(r"$y$ (µm)")


# Define the update function for each frame of the animation
def update(frame):
    # Update only the image data for the current time slice
    img.set_data(
        ez_data[frame, :, :],
    )
    # Update the title to show the current time
    ax.set_title(rf"$t=${frame * save_time_step:.0f}")

    return (img,)  # Return the updated image object


# Create the animation
ani = FuncAnimation(
    fig, update, frames=range(ez_data.shape[0]), interval=100, blit=True
)
ani.save("5_animation.gif", writer="imagemagick", fps=30, dpi=100)
