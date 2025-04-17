#%% 
from calculate_geomag import *
import numpy as np
# %%
xspace = np.arange(-180, 181, 0.5)
yspace = np.arange(-90, 91, 0.5)
year = 2024
height = 100
# %%
# inclination, equator, magnt, mag_poles, geomag_poles = calculateMag_adaptive(
#         xspace, yspace, year, height, initial_step=4)
# %%
fig = plot_geomagnetic_map(xspace, yspace, year, 
                           height, initial_step=4, 
                           figsize=(12, 8), fontsize=16)
    # plt.show()
# %%
