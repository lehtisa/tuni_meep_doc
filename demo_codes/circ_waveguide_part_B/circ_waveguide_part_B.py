import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import gdspy
from sim_params import *


def gen_circular_bend(cell, x0, wg_w, br, l1, l2, layer):
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


def gen_source_regions(cell, x0, wg_w, l1, layer, offset=4):
    path = gdspy.FlexPath(
        [(x0[0] + offset - wg_w, x0[1] + 1 * wg_w), (x0[0] + offset - wg_w, x0[1])],
        0,
        layer=layer,
    )
    cell.add(path)


def gen_refl_mon(cell, x0, wg_w, l1, layer, offset=4):
    path = gdspy.FlexPath(
        [
            (x0[0] + l1 - offset, x0[1] + 1.5 * wg_w),
            (x0[0] + l1 - offset, x0[1] - 0.5 * wg_w),
        ],
        0,
        layer=layer,
    )
    cell.add(path)


def gen_tran_mon_bent_wg(cell, x0, wg_w, br, l1, l2, layer, offset=2):
    path = gdspy.FlexPath(
        [
            (x0[0] + l1 + br - 0.5 * wg_w, x0[1] - br - l2 + offset + wg_w),
            (x0[0] + l1 + br + 1.5 * wg_w, x0[1] - br - l2 + offset + wg_w),
        ],
        0,
        layer=layer,
    )
    cell.add(path)


def gen_straight_wg(cell, x0, tol, wg_w, br, l1, l2, layer):
    # First straight section
    length = gen_sim_cell(cell, x0, tol, wg_w, br, l1, l2, layer, False)[0]
    points = [
        (x0[0], x0[1]),
        (x0[0] + length, x0[1]),
        (x0[0] + length, x0[1] + wg_w),
        (x0[0], x0[1] + wg_w),
    ]
    poly = gdspy.Polygon(points, layer=layer)
    cell.add(poly)


def gen_tran_mon_straight_wg(cell, x0, wg_w, br, l1, l2, layer, offset=2):
    length = gen_sim_cell(cell, x0, tol, wg_w, br, l1, l2, layer, False)[0]
    path = gdspy.FlexPath(
        [
            (x0[0] + length - offset, x0[1] - 0.5 * wg_w),
            (x0[0] + length - offset, x0[1] + wg_w + 0.5 * wg_w),
        ],
        0,
        layer=layer,
    )
    cell.add(path)


def gen_sim_cell(cell, x0, tol, wg_w, br, l1, l2, layer, add=True):
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


def simulate_bent_wg(
    br,  # bend radius
    resolution,  # simulation resolution
    wl_a,  # beginning of the wavelength range
    wl_b,  # end of the wavelength range
    nfreq,  # number of frequencies (wavelengths)
    wg_w,  # waveguide width
    l1,  # length of the first straight section
    l2,  # length of the second straight section
    tol,  # spacing used between the waveguide and cell edge
    cell_zmin,  # simulation cell zmin
    cell_zmax,  # simulation cell zmax
    wg_material,  # material of the waveguide
    pml_w,  # width of the perfectly mached layer
    wg_zmin,  # waveguide region zmin
    wg_zmax,  # waveguide region zmax
    decay_by,  # intensity decay to start stop counter
    t_after_decay,  # duration sim. keeps running after decay
    show_confirmation=False,  # show the simulation structure before running
):
    # PART I: Create the GDSII file

    filename = GDS_DIR + f"{br}.gds"

    # The GDSII file is called a library, which contains multiple cells.
    lib = gdspy.GdsLibrary()
    # Geometry must be placed in cells.
    cell = lib.new_cell(f"{br}")

    # x0 is such that the structure is centered at the origin
    x0 = ((-l1 - br - tol) / 2, (l2 + br - tol) / 2)

    # common layers
    gen_sim_cell(cell, x0, tol, wg_w, br, l1, l2, SIM_CELL_LAYER)
    gen_source_regions(cell, x0, wg_w, l1, SRC_LAYER)
    gen_refl_mon(cell, x0, wg_w, l1, REFL_MON_LAYER)

    # layers for bent wg
    gen_circular_bend(cell, x0, wg_w, br, l1, l2, WG_LAYER_BENT_WG)
    gen_tran_mon_bent_wg(cell, x0, wg_w, br, l1, l2, TRAN_MON_LAYER_BENT_WG)

    # layers for straight wg
    gen_straight_wg(cell, x0, tol, wg_w, br, l1, l2, WG_LAYER_STRAIGHT_WG)
    gen_tran_mon_straight_wg(cell, x0, wg_w, br, l1, l2, TRAN_MON_LAYER_STRAIGHT_WG)

    # layer to identify the center
    test = gdspy.Round((0, 0), 1, 1, 0, 2 * np.pi, 0.001)
    cell.add(test)

    lib.write_gds(filename)  # Save the library in a file

    # PART II: Import the structure to Meep

    # Read volumes for cell, geometry, source region
    # and flux monitors from the GDSII file
    sim_cell = mp.GDSII_vol(filename, SIM_CELL_LAYER, cell_zmin, cell_zmax)
    straight_wg = mp.get_GDSII_prisms(
        wg_material, filename, WG_LAYER_STRAIGHT_WG, wg_zmin, wg_zmax
    )  # the straight waveguide is needed for the normalization run
    bent_wg = mp.get_GDSII_prisms(
        wg_material, filename, WG_LAYER_BENT_WG, wg_zmin, wg_zmax
    )  # the bent waveguide geometry is for the actual run
    src_vol = mp.GDSII_vol(filename, SRC_LAYER, wg_zmin, wg_zmax)
    straight_out_vol = mp.GDSII_vol(
        filename, TRAN_MON_LAYER_STRAIGHT_WG, wg_zmin, wg_zmax
    )
    bent_out_vol = mp.GDSII_vol(filename, TRAN_MON_LAYER_BENT_WG, wg_zmin, wg_zmax)
    in_vol = mp.GDSII_vol(filename, REFL_MON_LAYER, wg_zmin, wg_zmax)
    straight_wg_end_pt = straight_out_vol.center
    bent_wg_end_pt = bent_out_vol.center

    # Define the objects for the simulation
    fcen = (1 / wl_a + 1 / wl_b) / 2  # central frequency
    df = np.abs(1 / wl_a - 1 / wl_b)  # width in frequency
    sources = [
        mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, volume=src_vol)
    ]
    straight_geometry = straight_wg
    bent_geometry = bent_wg
    pml_layers = [mp.PML(pml_w)]

    # PART III: Set up the simulation

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

    # plot only if wanted
    if show_confirmation:
        normalization_sim.plot2D()
        plt.show()
        actual_sim.plot2D()

    # PART IV: Running the simulation

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

    # PART V: Calculate and return the results
    wls = []
    Rs = []
    Ts = []
    for i in range(nfreq):
        wls = np.append(wls, 1 / flux_freqs[i])
        Rs = np.append(Rs, -bend_refl_flux[i] / straight_tran_flux[i])
        Ts = np.append(Ts, -bend_tran_flux[i] / straight_tran_flux[i])

    Ls = 1 - Rs - Ts

    # return wavelength, transmittance, loss, and reflectance
    return wls, Ts, Ls, Rs


const_args = (
    resolution,  # simulation resolution
    wl_a,  # beginning of the wavelength range
    wl_b,  # end of the wavelength range
    nfreq,  # number of frequencies (wavelengths)
    wg_w,  # waveguide width
    l1,  # length of the first straight section
    l2,  # length of the second straight section
    tol,  # spacing used between the waveguide and cell edge
    cell_zmin,  # simulation cell zmin
    cell_zmax,  # simulation cell zmax
    wg_material,  # material of the waveguide
    pml_w,  # width of the perfectly mached layer
    wg_zmin,  # waveguide region zmin
    wg_zmax,  # waveguide region zmax
    decay_by,  # intensity decay to start stop counter
    t_after_decay,  # duration sim. keeps running after decay
    show_confirmation,  # show the simulation structure before running
)

# Run the simulation and store the results
for br in brs:
    print(30 * "*")
    print(f"Simulating the structure with a bend radius of {br}")
    print(30 * "*")

    args = (br, *const_args)
    wls, Ts, Ls, Rs = simulate_bent_wg(*args)

    results = np.column_stack((wls, Ts, Ls, Rs))
    # save
    print("SAVING...")
    np.savetxt(RESULT_DIR + f"{br}.txt", results)

    print(30 * "*")
    print(f"Simulated the structure with a bend radius of {br} successfully")
    print(30 * "*")

print("Simulation program successfully run!")
