import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from sim_params import *

# interesting wavelengths
iwls = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])


# A helper function for finding the index of the value closest to the target
def clst_i_rng(arr, target):
    if np.min(arr) <= target <= np.max(arr):  # range check
        return np.abs(arr - target).argmin()
    else:
        return None


# A helper function to read the data from a file
def read_data(filename):
    data = np.loadtxt(filename)
    wls = data[:, 0]
    Ts = data[:, 1]
    Ls = data[:, 2]
    Rs = data[:, 3]

    return wls, Ts, Ls, Rs


# Create empty arrays to store the data for each wavelength
iwl_datas = [np.empty((len(brs), 4)) for _ in iwls]

# Read the data for each bend radius (br)
for i, br in enumerate(brs):
    filename = RESULT_DIR + f"{br}.txt"
    wls, Ts, Ls, Rs = read_data(filename)

    # For each wavelength in iwls, find the closest index and store the corresponding values
    for j, iwl in enumerate(iwls):
        k = clst_i_rng(wls, iwl)
        if k is not None:  # Only assign if k is valid
            iwl_datas[j][i] = (wls[k], Ts[k], Ls[k], Rs[k])

# Prepare the figure for plotting
fig = plt.figure(figsize=(9, 5))

# Colors for the wavelengths
n = len(iwls)
cmap = cm.get_cmap("copper")  # winter #TODO:
colors = [cmap(i / (n)) for i in range(n)]  # [::-1]

# Create subplots for each curve type
ax1 = fig.add_subplot(231)  # Transmittance
ax2 = fig.add_subplot(232)  # Loss
ax3 = fig.add_subplot(233)  # Reflectance
# ax4 = fig.add_subplot(224)  # Optional: If you want to add the 4th subplot

# inset plot for reflectance
# axins = inset_axes(ax3, width="50%", height="60%", loc="upper center", borderpad=1)
# Mark inset
# Connecting lines from left-upper and left-lower corners
# mark_inset(ax3, axins, loc1=3, loc2=4, fc="none", ec="black", lw=0.5)


axes = [ax1, ax2, ax3]  # We'll plot on 3 axes

# Plot the Transmittance (T), Loss (L), and Reflectance (R) for each wavelength
for j, iwl in enumerate(iwls):
    # Extract the data for the current wavelength
    iwl_data = iwl_datas[j]
    T_data = iwl_data[:, 1]
    L_data = iwl_data[:, 2]
    R_data = iwl_data[:, 3]

    # Plot on each subplot
    axes[0].plot(brs, 100 * T_data, ".-", lw=2, color=colors[j], label=f"{iwl} µm")
    axes[1].plot(brs, 100 * L_data, ".-", lw=2, color=colors[j], label=f"{iwl} µm")
    axes[2].plot(brs, 100 * R_data, ".-", lw=2, color=colors[j], label=f"{iwl} µm")

    # axins.plot(brs, R_data, ".-", color=curr_col, lw=1.5)

    # axins.set_xlim(0.45, 3)  # Ensure x-axis range
    # axins.set_ylim(0, 0.02)  # Ensure y-axis is 0-1
    # axins.set_xticks([1, 1.25, 1.5, 1.75, 2])
    # axins.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])


# Add titles and labels
ax1.set_ylabel("Transmittance (%)")
ax1.set_title("a)")
ax1.set_ylim(0, 100)  # Ensure y-axis is 0-100
ax2.set_ylabel("Loss (%)")
ax2.set_title("b)")
ax2.set_ylim(0, 100)  # Ensure y-axis is 0-100
ax3.set_ylabel("Reflectance (%)")
ax3.set_title("c)")
ax3.set_ylim(0, 3)

ax3.axhline(y=3, color="gray", linestyle="dashed", linewidth=1)
ax3.text(2.5, 2.7, "Different y-scale", fontsize=10, color="gray", ha="right")

for ax in axes:
    ax.set_xlabel("Bend radius (µm)")

# ax3.legend(loc=0) TODO:
# ax3.text(0.5, 2.7, "note the y-axis")

# Create an inset axis for the **horizontal** colorbar
cbar_ax = ax3.inset_axes([1.1, 0.0, 0.05, 1])  # (x, y, width, height)

# Create a ScalarMappable for colorbar
bounds = np.linspace(iwls[0], iwls[-1], len(iwls) + 1)  # Color boundaries
norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Use boundaries for discrete colormap
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy data for colorbar

cbar = fig.colorbar(sm, cax=cbar_ax, ticks=iwls, orientation="vertical")
cbar.ax.yaxis.set_label_position("right")
cbar.ax.set_ylabel("Wavelength (µm)", fontsize=10)
cbar.ax.tick_params(labelsize=8)


# ✅ Set ticks at midpoints to center them
cbar.ax.yaxis.set_ticks_position("none")
midpoints = (bounds[:-1] + bounds[1:]) / 2  # Middle points between boundaries
cbar.set_ticks(midpoints)
cbar.set_ticklabels([f"{w:.1f}" for w in iwls])  # Ensure labels are correctly mapped
cbar.ax.tick_params(labelsize=8, size=0)


# results with respect to wavelength
# Assuming 'brs' and 'RESULT_DIR' are already defined

# get how many wavelengths
wls, _, _, _ = read_data(RESULT_DIR + f"{br}.txt")
nfreq = len(wls)

# Create empty arrays to store the data for each wavelength
br_datas = [np.empty((nfreq, 4)) for _ in brs]

# Read the data for each bend radius (br)
for i, br in enumerate(brs):
    filename = RESULT_DIR + f"{br}.txt"
    wls, Ts, Ls, Rs = read_data(filename)
    br_datas[i] = (wls, Ts, Ls, Rs)

# Colors for the wavelengths
n = len(brs)
cmap = cm.get_cmap("jet")  # winter #TODO:
colors = [cmap(i / (n)) for i in range(n)]  # [::-1]

# Create subplots for each curve type
ax4 = fig.add_subplot(234)  # Transmittance
ax5 = fig.add_subplot(235)  # Loss
ax6 = fig.add_subplot(236)  # Reflectance

axes = [ax4, ax5, ax6]  # We'll plot on 3 axes

# Plot the Transmittance (T), Loss (L), and Reflectance (R) for each wavelength
for j, br in enumerate(brs):
    # Extract the data for the current wavelength
    br_data = br_datas[j]
    wls = br_data[0]
    T_data = br_data[1]
    L_data = br_data[2]
    R_data = br_data[3]

    # Plot on each subplot
    axes[0].plot(wls, 100 * T_data, "-", lw=2, color=colors[j], label=f"{iwl} µm")
    axes[1].plot(wls, 100 * L_data, "-", lw=2, color=colors[j], label=f"{iwl} µm")
    axes[2].plot(wls, 100 * R_data, "-", lw=2, color=colors[j], label=f"{iwl} µm")

    # axins.plot(brs, R_data, ".-", color=curr_col, lw=1.5)

    # axins.set_xlim(0.45, 3)  # Ensure x-axis range
    # axins.set_ylim(0, 0.02)  # Ensure y-axis is 0-1
    # axins.set_xticks([1, 1.25, 1.5, 1.75, 2])
    # axins.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])


# Add titles and labels
ax4.set_ylabel("Transmittance (%)")
ax4.set_title("d)")
ax4.set_ylim(0, 100)  # Ensure y-axis is 0-100
ax5.set_ylabel("Loss (%)")
ax5.set_title("e)")
ax5.set_ylim(0, 100)  # Ensure y-axis is 0-100
ax6.set_ylabel("Reflectance (%)")
ax6.set_title("f)")
ax6.set_ylim(0, 2.5)

ax6.text(3.5, 2.2, "Different y-scale", fontsize=10, color="gray", ha="right")

for ax in axes:
    ax.set_xlabel("Wavelength (µm)")

# Create an inset axis for the colorbar
cbar_ax = ax6.inset_axes([1.1, 0.0, 0.05, 1])  # (x, y, width, height)

# Create a ScalarMappable for colorbar
bounds = np.linspace(brs[0], brs[-1], len(brs) + 1)  # Color boundaries
norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Use boundaries for discrete colormap
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy data for colorbar

cbar = fig.colorbar(sm, cax=cbar_ax, ticks=brs, orientation="vertical")
cbar.ax.yaxis.set_label_position("right")
cbar.ax.set_ylabel("Bend radius (µm)", fontsize=10, labelpad=-5)
cbar.ax.tick_params(labelsize=8)


# Set ticks at midpoints to center them
cbar.ax.yaxis.set_ticks_position("none")
midpoints = (bounds[:-1] + bounds[1:]) / 2  # Middle points between boundaries
cbar.set_ticks((bounds[0], bounds[-1]))
cbar.set_ticklabels(
    f"{r:.1f}" for r in [bounds[0], bounds[-1]]
)  # Ensure labels are correctly mapped
cbar.ax.tick_params(labelsize=8, size=0)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig("results_radii_wl.pdf")
plt.savefig("results_radii_wl", dpi=300)

iwls, _, _, _ = read_data(RESULT_DIR + f"{brs[0]}.txt")
iwls = iwls[::-1]

# Read the data for each bend radius (br)
iwl_datas = [np.empty((len(brs), 4)) for _ in iwls]
for i, br in enumerate(brs):
    filename = RESULT_DIR + f"{br}.txt"
    wls, Ts, Ls, Rs = read_data(filename)

    # For each wavelength in iwls, find the closest index and store the corresponding values
    for j, iwl in enumerate(iwls):
        k = clst_i_rng(wls, iwl)
        if k is not None:  # Only assign if k is valid
            iwl_datas[j][i] = (wls[k], Ts[k], Ls[k], Rs[k])

# Extract the transmission data (T) for each wavelength and bend radius
T_values = 100 * np.array(
    [[iwl_datas[j][i, 1] for i in range(len(brs))] for j in range(len(iwls))]
)

# Create the plot
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)

# Use pcolormesh for heatmap with normalization
cmap = cm.get_cmap("GnBu")  # Choose a colormap
norm = mcolors.Normalize(vmin=0, vmax=100)  # Normalize from 0% to 100%

c = ax.pcolormesh(brs, iwls, T_values, cmap=cmap, shading="auto", norm=norm)

# Add color bar with normalization
cbar = fig.colorbar(c, ax=ax)
cbar.set_ticks(np.arange(0, 101, 20))  # Tick marks at 0%, 20%, 40%, etc.
cbar.set_label("Transmittance (%)", fontsize=10)

# Add labels and title
ax.set_xlabel("Bend Radius (µm)", fontsize=10)
ax.set_ylabel("Wavelength (µm)", fontsize=10)
ax.set_title("", fontsize=12)

# Set the limits to match the range of data
ax.set_ylim(iwls[0], iwls[-1])
ax.set_xlim(np.min(brs), np.max(brs))

plt.tight_layout()

# Save the heatmap
plt.savefig("heatmap_transmittance.pdf")
plt.savefig("heatmap_transmittance.png", dpi=300)
