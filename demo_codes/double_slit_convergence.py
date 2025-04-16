import meep as mp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Adjusting font sizes of figure
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

# Defining simulation cell dimensions
width = 40
height = 20
cell = mp.Vector3(width, height, 0)

# Defining PML layer thickness
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

# Figure and axis containers for a specific subplot layout
fig = plt.figure(figsize=(9, 6), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.3)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# Colormap for plots
cmap = mpl.colormaps['copper']

# Calculating distance from slit to PML-layer and maximum angle range to check
slits_to_dpml = (width/2 - dpml) - wall_xpos
theta_max = np.arctan((height/2 - dpml)/slits_to_dpml)

# Looping over different resolutions for convergence testing
for k in range(0,4):

    # Resolution of the simulation
    resolution = 10*(2**k)

    # Color mapping for resolution convergence plot
    plt_color_res = cmap((3-k)/3)

    # Define simulation
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        force_complex_fields=True,
    )

    # Running simulation, time based on propagation speed of light in the cell
    sim.run(until=width+5)
    
    # Saving field data
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    intensity_field = np.abs(ez_data)**2
    Ls = []

    # Extracting intensity distributions at different distances with max resolution
    if k == 3:

        # Dielectric data for plotting propagated field
        eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
        
        # Creating field plot
        ax1.imshow(eps_data.transpose(), extent = [0, width, 0, height], interpolation="spline36", cmap="binary")
        ax1.imshow(np.real(ez_data).transpose(), extent = [0, width, 0, height], interpolation="spline36", cmap="RdBu", alpha=0.9)
        ax1.set_xlabel(r"$x$ (µm)")
        ax1.set_ylabel(r"$y$ (µm)")
        ax1.set_title("(a)")

        # Looping over 3 different distances from slits
        for i in range(0,3):
            # Color map for distance plot
            plt_color_dist = cmap((2-i)/2)

            # Calculating distance to check
            L = slits_to_dpml - i*10
            Ls.append(L)

            # Extracting 1D slice of intensity values along y-axis from set distance
            intensity_slice = intensity_field[(width - dpml - i*10) * resolution]

            # Cropping data vector to fit maximum angle range and remove PML layers
            if i == 0:
                intensity_slice_cropped = intensity_slice[dpml*resolution-1:-dpml*resolution+1]
                # Calculating y value for each data point for later angle calculation
                y_vector = np.linspace(-height/2 + dpml, height/2 - dpml,len(intensity_slice_cropped))
            else:
                y_max = np.tan(theta_max)*L
                y_dif = int(height/2 - y_max)
                intensity_slice_cropped = intensity_slice[y_dif*resolution-1:-y_dif*resolution+1]
                y_vector = np.linspace(-y_max,y_max,len(intensity_slice_cropped))

            # Normalizing intensity data
            i_max = max(intensity_slice_cropped)
            intensity_norm = [x/i_max for x in intensity_slice_cropped]

            # Calculating theta values
            theta = [np.arctan(y/L) for y in y_vector]

            # Adding data relevant subplots
            if i == 0:
                ax3.plot(np.rad2deg(theta), intensity_norm, color = plt_color_res, linewidth = 1.25, label=f"{resolution}")
            ax2.plot(np.rad2deg(theta), intensity_norm, color = plt_color_dist, linewidth = 1.25, label=f"{Ls[i]} µm")

    # Same data extraction process for lower than max resolutions with max L distance
    else:
        intensity_slice = intensity_field[(width - dpml) * resolution]
        intensity_slice_cropped = intensity_slice[dpml*resolution-1:-dpml*resolution+1]
        y_vector = np.linspace(-height/2 + dpml, height/2 - dpml,len(intensity_slice_cropped))
        i_max = max(intensity_slice_cropped)
        intensity_norm = [x/i_max for x in intensity_slice_cropped]
        theta = [np.arctan(y/slits_to_dpml) for y in y_vector]
        ax3.plot(np.rad2deg(theta), intensity_norm, color = plt_color_res, linewidth = 1.25, label=f"{resolution}")

# Calculating Fraunhofer intensity distribution
theta2 = np.linspace(-theta_max,theta_max,1000)
alpha = np.pi * gap * np.sin(theta2) / wavelength
beta = np.pi * aperture * np.sin(theta2) / wavelength
i_theory = ((np.sin(beta)/beta) ** 2) * (np.cos(alpha) ** 2)

# Adding theoretical distribution and text to plots
axs = [ax2, ax3]
titles = ["(b)", "(c)"]

for i in range(2):
    axs[i].plot(np.rad2deg(theta2), i_theory, color=(0.7, 0.7, 0.7, 0.65),
                linewidth=1.5, ls='--', label="Fraunhofer")
    axs[i].set_xlabel("Angle (Degrees)")
    if i == 0:
        axs[i].set_ylabel("Normalized Intensity")
    axs[i].set_title(titles[i])

# Adjusting legends
ax2.legend(loc='lower left',bbox_to_anchor=(-0.2, 1.0),title="L=")
ax3.legend(loc='lower right',bbox_to_anchor=(1.1, 1.0),title="Resolution:")

# Saving figure
fig.tight_layout()
fig.subplots_adjust(top=0.92)
fig.savefig("full_subplot.png", bbox_inches='tight')
fig.savefig("full_subplot.pdf", bbox_inches='tight')