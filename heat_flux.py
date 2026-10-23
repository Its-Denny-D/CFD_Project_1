"""wall heat flux from the steady fields (part 4)"""
import numpy as np 
import matplotlib.pyplot as plt 

from plot import load_field, field_path, ANALYTICAL, savefig, fig_name, label, COL_WIDTH

def wall_flux(T, meta): 
    "Heat flow into the bar (W/m^2) along the bottom top and left walls."
    dx = meta["L"] / (meta["IMAX"] - 1)
    dy = meta["W"] / (meta["JMAX"] - 1)
    k = meta["k"]
     