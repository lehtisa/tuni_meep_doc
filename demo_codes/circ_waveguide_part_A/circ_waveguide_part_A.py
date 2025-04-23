import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import gdspy


GDS_DIR = "./gds_files/"

SIM_CELL_LAYER = 1  # simulation cell
SRC_LAYER = 2  # source of the field
REFL_MON_LAYER = 3  # reflection monitor

WG_LAYER_BENT_WG = 4  # bent waveguide
TRAN_MON_LAYER_BENT_WG = 5  # transmission monitor for bent wg

WG_LAYER_STRAIGHT_WG = 8  # straight waveguide
TRAN_MON_LAYER_STRAIGHT_WG = 9  # transmission monitor for straight wg

# Creating GDSII


def circular_bend(cell, x0, wg_w, br, l1, l2, layer):
    """
    Args:
       cell: gdspy cell
       x0: reference point for the drawing, a tuple of the form (x,y)
       wg_w: width of the waveguide
       br: (maximum) bend radius, float
       l1: the length of the first straight section, float
       l2: the length of the second straight section, float
       layer: layer onto which the structure is saved
    """
    # First straight section
    points = [
        (x0[0], x0[1]),
        (x0[0] + l1, x0[1]),
        (x0[0] + l1, x0[1] + wg_w),
        (x0[0], x0[1] + wg_w),
    ]
    poly = gdspy.Polygon(points, layer=layer)
    cell.add(poly)

    # Circular arc
    arc = gdspy.Round(
        center=(x0[0] + l1, x0[1] - br),
        radius=br + wg_w,
        inner_radius=br,
        initial_angle=0,
        final_angle=np.pi / 2,
        tolerance=0.001,
        layer=layer,
    )
    cell.add(arc)

    # Second straight section
    points = [
        (x0[0] + l1 + br, x0[1] - br),
        (x0[0] + l1 + br + wg_w, x0[1] - br),
        (x0[0] + l1 + br + wg_w, x0[1] - br - l2),
        (x0[0] + l1 + br, x0[1] - br - l2),
    ]
    poly = gdspy.Polygon(points, layer=layer)
    cell.add(poly)


def source_regions(cell, x0, wg_w, l1, layer, offset=4):
    path = gdspy.FlexPath(
        [(x0[0] + offset - wg_w, x0[1] + 1 * wg_w), (x0[0] + offset - wg_w, x0[1])],
        0,
        layer=layer,
    )
    cell.add(path)


def refl_mon(cell, x0, wg_w, l1, layer, offset=4):
    path = gdspy.FlexPath(
        [
            (x0[0] + l1 - offset, x0[1] + 1.5 * wg_w),
            (x0[0] + l1 - offset, x0[1] - 0.5 * wg_w),
        ],
        0,
        layer=layer,
    )
    cell.add(path)


def tran_mon_bent_wg(cell, x0, wg_w, br, l1, l2, layer, offset=2):
    path = gdspy.FlexPath(
        [
            (x0[0] + l1 + br - 0.5 * wg_w, x0[1] - br - l2 + offset + wg_w),
            (x0[0] + l1 + br + 1.5 * wg_w, x0[1] - br - l2 + offset + wg_w),
        ],
        0,
        layer=layer,
    )
    cell.add(path)


def straight_wg(cell, x0, tol, wg_w, br, l1, l2, layer):
    # First straight section
    length = sim_cell(cell, x0, tol, wg_w, br, l1, l2, layer, False)[0]
    points = [
        (x0[0], x0[1]),
        (x0[0] + length, x0[1]),
        (x0[0] + length, x0[1] + wg_w),
        (x0[0], x0[1] + wg_w),
    ]
    poly = gdspy.Polygon(points, layer=layer)
    cell.add(poly)


def tran_mon_straight_wg(cell, x0, wg_w, br, l1, l2, layer, offset=2):
    length = sim_cell(cell, x0, tol, wg_w, br, l1, l2, layer, False)[0]
    path = gdspy.FlexPath(
        [
            (x0[0] + length - offset, x0[1] - 0.5 * wg_w),
            (x0[0] + length - offset, x0[1] + wg_w + 0.5 * wg_w),
        ],
        0,
        layer=layer,
    )
    cell.add(path)


def sim_cell(cell, x0, tol, wg_w, br, l1, l2, layer, add=True):
    points = [
        (x0[0], x0[1] + wg_w + tol),
        (x0[0] + l1 + br + tol, x0[1] + wg_w + tol),
        (x0[0] + l1 + br + tol, x0[1] - br - l2 - wg_w),
        (x0[0], x0[1] - br - l2 - wg_w),
    ]
    poly = gdspy.Polygon(points, layer=layer)
    if add:
        cell.add(poly)
    return (l1 + br + tol, 2 * wg_w + tol + br + l2)  # x, y size


wg_w = 0.5  # waveguide width
br = 0.5  # bend radius
l1 = 20  # length of the first straight section
l2 = 10  # length of the second straight section
tol = 3  # spacing used between the waveguide and simulation cell edge

filename = GDS_DIR + f"{br}.gds"

# The GDSII file is called a library, which contains multiple cells.
lib = gdspy.GdsLibrary()
# Geometry must be placed in cells.
cell = lib.new_cell(f"{br}")

# x0 is such that the structure is centered at the origin
x0 = ((-l1 - br - tol) / 2, (l2 + br - tol) / 2)

# common layers
sim_cell(cell, x0, tol, wg_w, br, l1, l2, SIM_CELL_LAYER)
source_regions(cell, x0, wg_w, l1, SRC_LAYER)
refl_mon(cell, x0, wg_w, l1, REFL_MON_LAYER)

# layers for bent wg
circular_bend(cell, x0, wg_w, br, l1, l2, WG_LAYER_BENT_WG)
tran_mon_bent_wg(cell, x0, wg_w, br, l1, l2, TRAN_MON_LAYER_BENT_WG)

# layers for straight wg
straight_wg(cell, x0, tol, wg_w, br, l1, l2, WG_LAYER_STRAIGHT_WG)
tran_mon_straight_wg(cell, x0, wg_w, br, l1, l2, TRAN_MON_LAYER_STRAIGHT_WG)

# layer to identify the center
test = gdspy.Round((0, 0), 1, 1, 0, 2 * np.pi, 0.001)
cell.add(test)

lib.write_gds(filename)  # Save the library in a file

# Importing to Meep

# Define simulation parameters
cell_zmin = 0  # simulation cell zmin
cell_zmax = 0  # simulation cell zmax

pml_w = 1.0

wg_zmin = -100  # waveguide region zmin
wg_zmax = 100  # waveguide region zmax

wg_material = mp.Medium(index=2)

resolution = 20

# wavelength range
wl_begin = 1
wl_end = 10

fcen = (1 / wl_begin + 1 / wl_end) / 2  # central frequency
df = np.abs(1 / wl_end - 1 / wl_begin)  # width in frequency
nfreq = 1000  # number of different frequencies

# Read volumes for cell, geometry, source region
# and flux monitors from the GDSII file
sim_cell = mp.GDSII_vol(filename, SIM_CELL_LAYER, cell_zmin, cell_zmax)

# the straight waveguide is needed for the normalization run
straight_wg = mp.get_GDSII_prisms(
    wg_material, filename, WG_LAYER_STRAIGHT_WG, wg_zmin, wg_zmax
)

# the bent waveguide geometry is for the actual run
bent_wg = mp.get_GDSII_prisms(wg_material, filename, WG_LAYER_BENT_WG, wg_zmin, wg_zmax)

src_vol = mp.GDSII_vol(filename, SRC_LAYER, wg_zmin, wg_zmax)

straight_out_vol = mp.GDSII_vol(filename, TRAN_MON_LAYER_STRAIGHT_WG, wg_zmin, wg_zmax)

bent_out_vol = mp.GDSII_vol(filename, TRAN_MON_LAYER_BENT_WG, wg_zmin, wg_zmax)

in_vol = mp.GDSII_vol(filename, REFL_MON_LAYER, wg_zmin, wg_zmax)

straight_wg_end_pt = straight_out_vol.center
bent_wg_end_pt = bent_out_vol.center

# Define the objects for the simulation
sources = [
    mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, volume=src_vol)
]
straight_geometry = straight_wg
bent_geometry = bent_wg
pml_layers = [mp.PML(pml_w)]

# Create the simulation objects
normalization_sim = mp.Simulation(
    cell_size=sim_cell.size,
    boundary_layers=pml_layers,
    geometry=straight_geometry,
    sources=sources,
    resolution=resolution,
)

actual_sim = mp.Simulation(
    cell_size=sim_cell.size,
    boundary_layers=pml_layers,
    geometry=bent_geometry,
    sources=sources,
    resolution=resolution,
)

straight_refl = normalization_sim.add_flux(
    fcen, df, nfreq, mp.FluxRegion(volume=in_vol)
)
straight_tran = normalization_sim.add_flux(
    fcen, df, nfreq, mp.FluxRegion(volume=straight_out_vol)
)

bend_refl = actual_sim.add_flux(fcen, df, nfreq, mp.FluxRegion(volume=in_vol))
bend_tran = actual_sim.add_flux(fcen, df, nfreq, mp.FluxRegion(volume=bent_out_vol))

normalization_sim.plot2D()
plt.show()
actual_sim.plot2D()

# Running the simulation
decay_by = 1e-3
t_after_decay = 50

# Normalization run
normalization_sim.run(
    until_after_sources=mp.stop_when_fields_decayed(
        t_after_decay, mp.Ez, straight_wg_end_pt, decay_by
    )
)

# save the field data for calculating the reflection later
straight_refl_data = normalization_sim.get_flux_data(straight_refl)

# incident power
straight_tran_flux = mp.get_fluxes(straight_tran)

# information about the field propagating forwards
actual_sim.load_minus_flux_data(bend_refl, straight_refl_data)

# Actual run
actual_sim.run(
    until_after_sources=mp.stop_when_fields_decayed(
        t_after_decay, mp.Ez, bent_wg_end_pt, decay_by
    )
)

# save the reflected flux
bend_refl_flux = mp.get_fluxes(bend_refl)

# save the transmitted flux
bend_tran_flux = mp.get_fluxes(bend_tran)

# save the frequencies
flux_freqs = mp.get_flux_freqs(bend_tran)

wl = []
Rs = []
Ts = []
for i in range(nfreq):
    wl = np.append(wl, 1 / flux_freqs[i])
    # calculation of reflection and transmission
    Rs = np.append(Rs, -bend_refl_flux[i] / straight_tran_flux[i])
    Ts = np.append(Ts, -bend_tran_flux[i] / straight_tran_flux[i])

Ls = 1 - Rs - Ts  # calculation of loss

fig = plt.figure(figsize=(4, 3.25))
ax = fig.add_subplot(111)
colors = ["#CC6666", "#9999CC", "#66CC99"]

# main plot
ax.plot(wl, 100 * Ts, color=colors[0], label="Transmittance")
ax.plot(wl, 100 * Ls, color=colors[1], label="Loss")
ax.plot(wl, 100 * Rs, color=colors[2], label="Reflectance")

ax.set_ylim(bottom=0, top=100)
ax.set_xlabel("Wavelength (μm)")
ax.set_ylabel("Transmittance, Loss, \nand Reflectance (%)")
ax.set_title("")
ax.legend(loc=0)

plt.savefig("one_wg_TRL.pdf", bbox_inches="tight", pad_inches=0.2)
plt.savefig("one_wg_TRL.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
plt.show()
