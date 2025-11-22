__authors__ = "Zachary Klouchnikov and Hannah Semple"

# This code file provides answers for Q1 of the PHY407 Lab10, where we graph the Earth as land and water, and calculate
# the land fraction. This is done by selecting a number of random locations and evaluating whether they are land or water. 

"""
IMPORTS
"""
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from math import acos
from scipy.integrate import simpson as simp


"""
FUNCTIONS
"""
def p1(num):
    """
    Returns an angle theta in radians from the p1 probability distribution
    """
    theta = acos(1-2*num)
    return theta

def p2(num):
    """
    Returns an angle phi in radians from the p2 probability distribution
    """
    phi = 2*pi*num
    return phi

def phi_to_lon(phi):
    """
    Returns the longitude coordinate converted from angle phi in radians
    """
    long = phi*180/pi - 180
    return long

def theta_to_lat(theta):
    """
    Returns the latitude coordinate converted from angle theta in radians
    """
    lat = theta*180/pi - 90
    return lat


"""
PART B
"""
N=5000
thetas = []
phis = []

for n in range(N):
    thetas.append(p1(rng.random()))
    phis.append(p2(rng.random()))

thetas, phis = np.array(thetas), np.array(phis)
    
X = phi_to_lon(phis)
Y = theta_to_lat(thetas)

plt.scatter(X, Y, s = 5, color='teal')
plt.xlabel('Longitude [º]')
plt.ylabel('Latitude [º]')
plt.title('5000 Random Lat-Lon Points')
# plt.savefig('random.pdf', bbox_inches='tight')
plt.show()


"""
PART C
"""
flat_earth = data.flatten()  #convert data to a 1D array
A = 4*pi  #surface area of a sphere w/ r=1

lands = flat_earth*A/len(flat_earth)  #converts data such that each value represents amount of land at that coordinate

integral = simp(lands)  #integrate using simpsons method
print('Numerically integrated land fraction is:',integral/A)


"""
PART D
"""
Ns = [50, 500, 5000, 50000]

for N in Ns:
    thetas = []
    phis = []
    num_land = 0

    for n in range(N):  #get N many random locations
        thetas.append(p1(rng.random()))
        phis.append(p2(rng.random()))

    thetas, phis = np.array(thetas), np.array(phis)
    
    #convert to latitude and longitude
    lons = phi_to_lon(phis)
    lats = theta_to_lat(thetas)
    
    #make sure nothing is outside of the boundaries
    lons = np.clip(lons, min(lon_data), max(lon_data))
    lats = np.clip(lats, min(lat_data), max(lat_data))
    
    #nearest interpolator
    interp = RegularGridInterpolator((lon_data, lat_data), data, method='nearest')
    
    coords = np.column_stack((lons, lats))
    nearest_values = interp(coords)
    
    is_land = nearest_values == 1
    is_water = nearest_values == 0
    
    num_land = np.sum(is_land)
    
    print('The land fraction for N={} is:'.format(N), num_land/N)
    
    # Create plot
    plt.figure()
    if np.any(is_water):
        plt.scatter(lons[is_water], lats[is_water], 
                   color='dodgerblue', s=5, label='Water')
    
    if np.any(is_land):
        plt.scatter(lons[is_land], lats[is_land], 
                   color='green', s=5, label='Land')
    
    plt.xlabel('Longitude [º]')
    plt.ylabel('Latitude [º]')
    plt.title('Earth Map (N={} points)'.format(N))
    plt.legend()
#     plt.savefig('map{}.pdf'.format(N), bbox_inches='tight')
    plt.show()

