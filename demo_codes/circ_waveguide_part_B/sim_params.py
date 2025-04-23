import numpy as np
import meep as mp

GDS_DIR = "./gds_files/"
RESULT_DIR = "./results/"

SIM_CELL_LAYER = 1  # simulation cell
SRC_LAYER = 2  # source of the field
REFL_MON_LAYER = 3  # reflection monitor

WG_LAYER_BENT_WG = 4  # bent waveguide
TRAN_MON_LAYER_BENT_WG = 5  # transmission monitor for bent wg

WG_LAYER_STRAIGHT_WG = 8  # straight waveguide
TRAN_MON_LAYER_STRAIGHT_WG = 9  # transmission monitor for straight wg

# Simulation parameters
brs = np.linspace(0.5, 3, 20)
l1 = 20
l2 = 10

resolution = 20
wl_a = 1
wl_b = 5
nfreq = 200
show_confirmation = False

cell_zmin = 0
cell_zmax = 0
wg_w = 0.5
wg_zmin = -100
wg_zmax = 100
wg_material = mp.Medium(index=2)
pml_w = 1.0
decay_by = 1e-3
t_after_decay = 100
tol = 3
