#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import pyIGRF
import multiprocessing
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm
from idl_colorbars import *


# Method 1: Chunking with joblib for parallel processing
def calculate_chunk(chunk_data, year, height):
    """Process a chunk of lat/lon points in parallel"""
    chunk_y_indices, chunk_x_indices = chunk_data
    results_inc = np.zeros((len(chunk_y_indices), len(chunk_x_indices)))
    results_mag = np.zeros((len(chunk_y_indices), len(chunk_x_indices)))
    
    for i, y_idx in enumerate(chunk_y_indices):
        for j, x_idx in enumerate(chunk_x_indices):
            y = yspace[y_idx]
            x = xspace[x_idx]
            decl, inc, hMag, xMag, yMag, zMag, fMAg = pyIGRF.igrf_value(y, x, height, year)
            results_inc[i, j] = inc
            results_mag[i, j] = fMAg
            
    return chunk_y_indices, chunk_x_indices, results_inc, results_mag

def calculateMag_parallel(xspace, yspace, year, height, n_jobs=-1, chunk_size=20):
    """Parallel version of calculateMag using joblib"""
    # Create the full output arrays
    inclination = np.zeros((len(yspace), len(xspace)))
    magnt = np.zeros((len(yspace), len(xspace)))
    
    # Create chunks of indices
    y_chunks = [list(range(i, min(i+chunk_size, len(yspace)))) 
                for i in range(0, len(yspace), chunk_size)]
    x_chunks = [list(range(i, min(i+chunk_size, len(xspace)))) 
                for i in range(0, len(xspace), chunk_size)]
    
    # Create all combinations of chunks
    chunk_combinations = [(y_chunk, x_chunk) for y_chunk in y_chunks for x_chunk in x_chunks]
    
    # Process chunks in parallel with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_chunk)(chunk, year, height) 
        for chunk in tqdm(chunk_combinations, desc="Processing grid chunks")
    )
    
    # Reconstruct the output arrays
    for y_indices, x_indices, inc_chunk, mag_chunk in results:
        for i, y_idx in enumerate(y_indices):
            for j, x_idx in enumerate(x_indices):
                inclination[y_idx, x_idx] = inc_chunk[i, j]
                magnt[y_idx, x_idx] = mag_chunk[i, j]
    
    # Calculate magnetic equator (where inclination ≈ 0)
    equator = []
    for ii in range(inclination.shape[1]):
        temp = inclination[:, ii]
        sts = np.where((temp > -1) & (temp < 1))
        if len(sts[0]) > 0:  # Check if any points found
            idx = sts[0][np.argmin(abs(temp[sts]))]
            equator.append(yspace[idx])
        else:
            # No points with inclination near zero in this column
            equator.append(np.nan)
    
    return inclination, equator, magnt


# Method 2: Using numba to JIT compile the inner loop
try:
    from numba import jit
    
    @jit(nopython=True)
    def _calculate_grid_numba(yspace, xspace, height, year, inclination, magnt):
        """JIT-compiled inner loop for IGRF calculations"""
        for y_idx in range(len(yspace)):
            for x_idx in range(len(xspace)):
                # Unfortunately, we can't JIT compile pyIGRF.igrf_value directly,
                # but this prepares the structure for a custom IGRF implementation
                decl, inc, hMag, xMag, yMag, zMag, fMAg = pyIGRF.igrf_value(
                    yspace[y_idx], xspace[x_idx], height, year)
                inclination[y_idx, x_idx] = inc
                magnt[y_idx, x_idx] = fMAg
    
    def calculateMag_numba(xspace, yspace, year, height):
        """Version using Numba JIT"""
        inclination = np.zeros((len(yspace), len(xspace)))
        magnt = np.zeros((len(yspace), len(xspace)))
        
        # Call the JIT-compiled function
        _calculate_grid_numba(yspace, xspace, height, year, inclination, magnt)
        
        # Calculate magnetic equator
        equator = []
        for ii in range(inclination.shape[1]):
            temp = inclination[:, ii]
            sts = np.where((temp > -1) & (temp < 1))
            if len(sts[0]) > 0:
                idx = sts[0][np.argmin(abs(temp[sts]))]
                equator.append(yspace[idx])
            else:
                equator.append(np.nan)
        
        return inclination, equator, magnt
except ImportError:
    print("Numba not available - skipping JIT compilation method")


# Method 3: Reduce resolution for initial calculation
import numpy as np
import pyIGRF
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

def calculateMag_adaptive(xspace, yspace, year, height, initial_step=4):
    """
    Adaptive resolution calculation of geomagnetic field
    
    Parameters:
    -----------
    xspace : array_like
        Longitude values from -180 to 180
    yspace : array_like
        Latitude values from -90 to 90
    year : float
        Year for IGRF model
    height : float
        Height above sea level in km
    initial_step : int
        Step size for initial low-resolution calculation
        
    Returns:
    --------
    inclination : ndarray
        Inclination values for each grid point
    equator : list
        Magnetic equator coordinates
    magnt : ndarray
        Total field intensity values for each grid point
    mag_poles : dict
        Dictionary containing locations of magnetic poles
    geomag_poles : dict
        Dictionary containing locations of geomagnetic poles
    """
    # First calculate at lower resolution
    x_low_res = xspace[::initial_step]
    y_low_res = yspace[::initial_step]
    
    inclination_low = np.zeros((len(y_low_res), len(x_low_res)))
    magnt_low = np.zeros((len(y_low_res), len(x_low_res)))
    
    print(f"Initial low-res calculation: {len(y_low_res)}x{len(x_low_res)} points")
    
    # Calculate at low resolution
    for y_idx, y in enumerate(tqdm(y_low_res, desc="Calculating magnetic field")):
        for x_idx, x in enumerate(x_low_res):
            decl, inc, hMag, xMag, yMag, zMag, fMAg = pyIGRF.igrf_value(y, x, height, year)
            inclination_low[y_idx, x_idx] = inc
            magnt_low[y_idx, x_idx] = fMAg
    
    # Now interpolate to full resolution
    interp_inc = RectBivariateSpline(y_low_res, x_low_res, inclination_low)
    interp_mag = RectBivariateSpline(y_low_res, x_low_res, magnt_low)
    
    # Interpolate to full grid
    inclination = interp_inc(yspace, xspace)
    magnt = interp_mag(yspace, xspace)
    
    # Calculate magnetic equator from interpolated data
    equator = []
    for ii in range(inclination.shape[1]):
        temp = inclination[:, ii]
        sts = np.where((temp > -1) & (temp < 1))
        if len(sts[0]) > 0:
            idx = sts[0][np.argmin(abs(temp[sts]))]
            equator.append(yspace[idx])
        else:
            equator.append(np.nan)
    
    # Calculate magnetic poles (where inclination is +/- 90 degrees)
    north_pole_data = find_magnetic_pole(inclination, xspace, yspace, 'north')
    south_pole_data = find_magnetic_pole(inclination, xspace, yspace, 'south')
    
    # Get geomagnetic poles using dipole approximation
    # These are fixed at specific locations for a given year based on the first three Gauss coefficients
    geomag_poles = calculate_geomagnetic_poles(year)
    
    mag_poles = {
        'north': north_pole_data,
        'south': south_pole_data
    }
    
    return inclination, equator, magnt, mag_poles, geomag_poles

def find_magnetic_pole(inclination, xspace, yspace, pole_type='north'):
    """
    Find magnetic pole location by finding extreme inclination values.
    Returns coordinates with proper N/S and E/W formatting.
    
    Parameters:
    -----------
    inclination : ndarray
        2D array of inclination values
    xspace : array_like
        Longitude values from -180 to 180
    yspace : array_like
        Latitude values from -90 to 90
    pole_type : str
        'north' or 'south' to specify which pole to find
        
    Returns:
    --------
    dict
        Dictionary containing the location and inclination of the pole
        with proper direction indicators
    """
    if pole_type == 'north':
        # North magnetic pole: inclination = +90°
        # Find the point with inclination closest to 90 degrees
        idx = np.argmax(inclination)
    else:
        # South magnetic pole: inclination = -90°
        # Find the point with inclination closest to -90 degrees
        idx = np.argmin(inclination)
    
    # Convert flat index to 2D coordinates
    y_idx, x_idx = np.unravel_index(idx, inclination.shape)
    
    # Get the latitude and longitude
    lat = yspace[y_idx]
    lon = xspace[x_idx]
    inc = inclination[y_idx, x_idx]
    
    # Format with proper N/S and E/W indicators
    lat_abs = abs(lat)
    lat_dir = 'N' if lat >= 0 else 'S'
    
    lon_abs = abs(lon)
    lon_dir = 'E' if lon >= 0 else 'W'
    
    return {
        'lat': lat_abs,
        'lat_dir': lat_dir,
        'lon': lon_abs,
        'lon_dir': lon_dir,
        'raw_lat': lat,  # Keep raw values for calculations
        'raw_lon': lon,
        'inclination': inc
    }

def calculate_geomagnetic_poles(year):
    """
    Calculate geomagnetic poles based on the centered dipole approximation
    using the first three Gauss coefficients of the IGRF model
    """
    try:
        # IGRF-13 main field coefficients and secular variations for different epochs
        # Values from the official IGRF-13 model
        
        # Check which epoch range the year falls into
        if 2020 <= year <= 2025:
            # IGRF-13 coefficients for epoch 2020.0
            g10_base = -29404.5  # nT
            g11_base = -1450.7   # nT
            h11_base = 4652.9    # nT
            
            # IGRF-13 secular variation coefficients for 2020-2025
            g10_sv = 5.7  # nT/year
            g11_sv = 7.4  # nT/year
            h11_sv = -25.9  # nT/year
            
            # Reference epoch
            base_epoch = 2020.0
            
        elif 2015 <= year < 2020:
            # DGRF-2015 coefficients (definitive)
            g10_base = -29438.5  # nT
            g11_base = -1501.1   # nT
            h11_base = 4795.8    # nT
            
            # IGRF-13 secular variation coefficients for 2015-2020
            g10_sv = 7.0  # nT/year
            g11_sv = 9.4  # nT/year
            h11_sv = -30.2  # nT/year
            
            # Reference epoch
            base_epoch = 2015.0
            
        elif 2010 <= year < 2015:
            # DGRF-2010 coefficients
            g10_base = -29496.6  # nT
            g11_base = -1586.3   # nT
            h11_base = 4944.3    # nT
            
            # IGRF-12 secular variation coefficients for 2010-2015
            g10_sv = 11.6  # nT/year
            g11_sv = 16.5  # nT/year
            h11_sv = -25.9  # nT/year
            
            # Reference epoch
            base_epoch = 2010.0
            
        elif 2005 <= year < 2010:
            # DGRF-2005 coefficients
            g10_base = -29554.6  # nT
            g11_base = -1669.0   # nT
            h11_base = 5077.9    # nT
            
            # IGRF-11 secular variation coefficients for 2005-2010
            g10_sv = 8.8   # nT/year
            g11_sv = 10.8  # nT/year
            h11_sv = -21.3  # nT/year
            
            # Reference epoch
            base_epoch = 2005.0
            
        elif 2000 <= year < 2005:
            # DGRF-2000 coefficients
            g10_base = -29619.4  # nT
            g11_base = -1728.2   # nT
            h11_base = 5186.1    # nT
            
            # IGRF-10 secular variation coefficients for 2000-2005
            g10_sv = 14.6  # nT/year
            g11_sv = 10.7  # nT/year
            h11_sv = -22.5  # nT/year
            
            # Reference epoch
            base_epoch = 2000.0
            
        elif 1995 <= year < 2000:
            # DGRF-1995 coefficients
            g10_base = -29682.0  # nT
            g11_base = -1789.0   # nT
            h11_base = 5318.0    # nT
            
            # IGRF-9 secular variation coefficients for 1995-2000
            g10_sv = 17.4  # nT/year
            g11_sv = 11.3  # nT/year
            h11_sv = -26.0  # nT/year
            
            # Reference epoch
            base_epoch = 1995.0
            
        else:
            # For years outside our defined ranges, use approximate values
            # This is a simplified approximation 
            # For years before 1995, use different approach
            if year < 1995:
                # Very approximate model for earlier years
                # Historical trend approximation
                g10_base = -30000.0 + 5.0 * (year - 1900)  # nT
                g11_base = -2000.0 + 3.0 * (year - 1900)   # nT
                h11_base = 5700.0 - 4.0 * (year - 1900)    # nT
                
                # Very approximate secular variation
                g10_sv = 5.0   # nT/year
                g11_sv = 3.0   # nT/year
                h11_sv = -4.0  # nT/year
                
                # Reference epoch (nearest decade)
                base_epoch = 10 * (year // 10)
                
            else:  # For years beyond 2025
                # Extrapolate using the latest secular variation
                g10_base = -29404.5  # 2020 value
                g11_base = -1450.7   # 2020 value
                h11_base = 4652.9    # 2020 value
                
                g10_sv = 5.7   # nT/year
                g11_sv = 7.4   # nT/year
                h11_sv = -25.9  # nT/year
                
                base_epoch = 2020.0
        
        # Linear interpolation to the requested year
        time_diff = year - base_epoch
        g10 = g10_base + g10_sv * time_diff
        g11 = g11_base + g11_sv * time_diff
        h11 = h11_base + h11_sv * time_diff
        
        # For the geomagnetic dipole, we need the negative of the coefficients
        # because the dipole points from south to north, but the field
        # direction convention is opposite
        dipole_g10 = -g10
        dipole_g11 = -g11
        dipole_h11 = -h11
        
        # Calculate the location where the dipole axis intersects Earth's surface
        # This is the NORTH geomagnetic pole
        # First calculate the Cartesian components of the dipole moment vector
        mx = dipole_g11  # Points along x-axis (prime meridian at equator)
        my = dipole_h11  # Points along y-axis (90°E at equator)
        mz = dipole_g10  # Points along z-axis (north pole)
        
        # Normalize the vector
        m_mag = np.sqrt(mx**2 + my**2 + mz**2)
        mx_norm = mx / m_mag
        my_norm = my / m_mag
        mz_norm = mz / m_mag
        
        # Convert the Cartesian vector to spherical coordinates
        # North geomagnetic pole (where dipole axis exits in northern hemisphere)
        north_lat = np.degrees(np.arcsin(mz_norm))
        north_lon = np.degrees(np.arctan2(my_norm, mx_norm))
        
        # Ensure longitude is in range -180 to 180
        if north_lon > 180:
            north_lon -= 360
            
        # The south geomagnetic pole is the antipode of the north pole
        south_lat = -north_lat
        south_lon = north_lon + 180
        if south_lon > 180:
            south_lon -= 360
        
        # Format with proper N/S and E/W indicators
        north_lat_abs = abs(north_lat)
        north_lat_dir = 'N' if north_lat >= 0 else 'S'
        
        north_lon_abs = abs(north_lon)
        north_lon_dir = 'E' if north_lon >= 0 else 'W'
        
        south_lat_abs = abs(south_lat)
        south_lat_dir = 'N' if south_lat >= 0 else 'S'
        
        south_lon_abs = abs(south_lon)
        south_lon_dir = 'E' if south_lon >= 0 else 'W'
        
        return {
            'north': {
                'lat': north_lat_abs, 
                'lat_dir': north_lat_dir,
                'lon': north_lon_abs,
                'lon_dir': north_lon_dir,
                'raw_lat': north_lat,  # Keep raw values for calculations
                'raw_lon': north_lon
            },
            'south': {
                'lat': south_lat_abs,
                'lat_dir': south_lat_dir,
                'lon': south_lon_abs,
                'lon_dir': south_lon_dir,
                'raw_lat': south_lat,  # Keep raw values for calculations
                'raw_lon': south_lon
            },
            'coefficients': {
                'g10': g10,
                'g11': g11,
                'h11': h11,
                'epoch': base_epoch,
                'sv_g10': g10_sv,
                'sv_g11': g11_sv,
                'sv_h11': h11_sv
            }
        }
    except Exception as e:
        print(f"Error calculating geomagnetic poles: {e}")
        # Return approximate poles as fallback
        return {
            'north': {
                'lat': 80.65, 
                'lat_dir': 'N',
                'lon': 72.68,
                'lon_dir': 'W',
                'raw_lat': 80.65,  
                'raw_lon': -72.68
            },
            'south': {
                'lat': 80.65,
                'lat_dir': 'S',
                'lon': 107.32,
                'lon_dir': 'E',
                'raw_lat': -80.65,  
                'raw_lon': 107.32
            },
            'coefficients': None
        }


def plot_geomagnetic_map(xspace, yspace, year, height, initial_step=4, figsize=(14, 10), fontsize=12):
    """
    Plot the geomagnetic field intensity over a world map using Cartopy
    
    Parameters:
    -----------
    xspace : array_like
        Longitude values from -180 to 180
    yspace : array_like
        Latitude values from -90 to 90
    year : float
        Year for IGRF model
    height : float
        Height above sea level in km
    initial_step : int
        Step size for initial calculation
    figsize : tuple
        Figure size (width, height) in inches
    fontsize : int
        Base font size for labels and text (default: 12)
    """
    # Calculate the geomagnetic field
    print(f"Calculating geomagnetic field for year {year}...")
    incl, equator, magnt, mag_poles, geomag_poles = calculateMag_adaptive(
        xspace, yspace, year, height, initial_step=initial_step)
    
    # Create figure with a Robinson projection
    plt.figure(figsize=figsize, dpi=300, tight_layout=True)
    ax = plt.axes(projection=ccrs.Mollweide())
    
    # Set map extent to cover the whole globe
    ax.set_global()
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(xspace, yspace)
    
    # Create a custom colormap for the magnetic field intensity
    # cmap = plt.cm.viridis
    # use idl colorbars
    mycmap = getcmap(34)
    
    # Calculate the min/max values for better color scaling
    vmin = np.percentile(magnt, 2)
    vmax = np.percentile(magnt, 98)
    
    # Plot the magnetic field intensity
    cs = ax.contourf(X, Y, magnt, 100, transform=ccrs.PlateCarree(), 
                    cmap=mycmap, 
                    extend='both') #, norm=colors.Normalize(vmin=vmin, vmax=vmax)
                    
    # Add contour lines
    contour_levels = np.linspace(vmin, vmax, 10)  # 10 contour levels between min and max
    contour = ax.contour(X, Y, magnt, levels=contour_levels, transform=ccrs.PlateCarree(),
                       colors='white', linewidths=1, alpha=1.0)
    
    # Add contour labels with increased font size
    plt.clabel(contour, contour.levels[::2], inline=True, fmt='%.0f nT')
    
    # Add coastlines, countries, and other features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='0.5')
    
    # Add gridlines with larger font size for labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    # Turn off longitude labels but keep latitude labels
    gl.top_labels = False
    gl.bottom_labels = False  # Turn off longitude labels at the bottom
    gl.right_labels = False
    gl.left_labels = True     # Keep latitude labels on the left
    # gl.xlabel_style = {'size': fontsize}
    gl.ylabel_style = {'size': fontsize}
    
    # Plot the magnetic equator
    eq_x = xspace
    eq_y = np.array(equator)
    valid_idx = ~np.isnan(eq_y)
    eq_x = eq_x[valid_idx]
    eq_y = eq_y[valid_idx]
    ax.plot(eq_x, eq_y, 'm-', linewidth=2.5, label='Magnetic Equator', transform=ccrs.PlateCarree())
    
    # Plot magnetic poles
    ax.plot(mag_poles['north']['raw_lon'], mag_poles['north']['raw_lat'], 'ro', markersize=8, 
           label='North Magnetic Pole', transform=ccrs.PlateCarree())
    ax.plot(mag_poles['south']['raw_lon'], mag_poles['south']['raw_lat'], 'bo', markersize=8,
           label='South Magnetic Pole', transform=ccrs.PlateCarree())
    
    # Plot geomagnetic poles
    ax.plot(geomag_poles['north']['raw_lon'], geomag_poles['north']['raw_lat'], 'r^', markersize=8, 
           label='North Geomagnetic Pole', transform=ccrs.PlateCarree())
    ax.plot(geomag_poles['south']['raw_lon'], geomag_poles['south']['raw_lat'], 'b^', markersize=8,
           label='South Geomagnetic Pole', transform=ccrs.PlateCarree())
    
    # Add a colorbar with larger font size
    cbar = plt.colorbar(cs, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Magnetic Field Intensity (nT)', fontsize=fontsize-2)
    cbar.ax.tick_params(labelsize=fontsize)
    
    # Add title with larger font
    plt.title(f'Geomagnetic Field Intensity - Year {year}', fontsize=fontsize+2)
    
    # Add legend with larger font
    plt.legend(loc=(0.65, -0.48), framealpha=1, fontsize=fontsize-4)
    
    # Add pole information as text with larger font
    info_text = (
        f"Magnetic Poles:\n"
        f"  North: ({mag_poles['north']['lat']:.2f}°{mag_poles['north']['lat_dir']}, {mag_poles['north']['lon']:.2f}°{mag_poles['north']['lon_dir']})\n"
        f"  South: ({mag_poles['south']['lat']:.2f}°{mag_poles['south']['lat_dir']}, {mag_poles['south']['lon']:.2f}°{mag_poles['south']['lon_dir']})\n\n"
        f"Geomagnetic Poles:\n"
        f"  North: ({geomag_poles['north']['lat']:.2f}°{geomag_poles['north']['lat_dir']}, {geomag_poles['north']['lon']:.2f}°{geomag_poles['north']['lon_dir']})\n"
        f"  South: ({geomag_poles['south']['lat']:.2f}°{geomag_poles['south']['lat_dir']}, {geomag_poles['south']['lon']:.2f}°{geomag_poles['south']['lon_dir']})"
    )
    
    plt.figtext(0.16, -0.01, info_text, fontsize=fontsize-4, bbox=dict(facecolor='white', alpha=0.7))
    
    # Set tick parameters for larger font size
    ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'geomagnetic_field_{int(year)}.png', dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as 'geomagnetic_field_{int(year)}.png'")
    
    return plt.gcf()


#%%

# Example usage
if __name__ == "__main__":
    import time
    
    xspace = np.arange(-180, 181, 0.5)
    yspace = np.arange(-90, 91, 0.5)
    year = 2022.0
    height = 100
    
    # Test parallel version
    print("\nTesting parallel function...")
    start_time = time.time()
    incl_parallel, equator_parallel, magnt_parallel = calculateMag_parallel(
        xspace, yspace, year, height, n_jobs=4, chunk_size=20)
    print(f"Parallel method time: {time.time() - start_time:.2f} seconds")
    
    # Test adaptive resolution
    print("\nTesting adaptive resolution function...")
    start_time = time.time()
    incl_adaptive, equator_adaptive, magnt_adaptive = calculateMag_adaptive(
        xspace, yspace, year, height, initial_step=4)
    print(f"Adaptive method time: {time.time() - start_time:.2f} seconds")
    
    # Compare results visually
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.contourf(xspace, yspace, incl_parallel, 50, cmap='RdBu_r')
    plt.colorbar(label='Inclination (degrees)')
    plt.plot(xspace, equator_parallel, 'k-', linewidth=2)
    plt.title('Inclination from Parallel Method')
    plt.ylabel('Latitude')
    
    plt.subplot(2, 1, 2)
    plt.contourf(xspace, yspace, incl_adaptive, 50, cmap='RdBu_r')
    plt.colorbar(label='Inclination (degrees)')
    plt.plot(xspace, equator_adaptive, 'k-', linewidth=2)
    plt.title('Inclination from Adaptive Method')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig('igrf_comparison.png', dpi=300)
    plt.show()
# %%
