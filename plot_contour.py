import numpy as np 
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import make_interp_spline
from numpy.fft import fft,ifft,ifftshift,fftfreq
import matplotlib.colors as mcolors

xb_factor = 9.0
n_factor = 4
w_factor = 1800
w = w_factor
x_start = xb_factor
n       = n_factor
n_on    =   2
n_p     =   6
n_off   =   2
n_cycle =   n_on + n_p + n_off      #Total number of cycles within the envelope
I_factor = 2
w       =   45.2/w_factor# Angular frequency if the laser of wavelength 1200 nm
w_ev    =   w*27.2114
T_p     =   2.*np.pi*n_cycle/w
I0      =   I_factor*1.0e13
F0      =   np.sqrt(I0/3.5095e16)
q       =   F0/w**2
eta     =   0.0005
m       =   3
tstart  =   0.0
tend    =   1.01*T_p
dt      =   0.100
Nt      = int((tend-tstart)/dt)
N       = Nt

sigma_factor = 0.90
sigma_norm_factor = 3.0
sigma   =   np.sqrt(sigma_factor)
Norm =  sigma_norm_factor

dens0_file  = np.genfromtxt('data/normalized/hhg_spectra_windowed_normalized_d_w_x_start_9.200_au_N_3.0_sigma_0.90_0.4qBC_w_1800_I_2.dat',dtype=np.float64)

Nh          = 45
N           = len(dens0_file[:])
rho_d       = np.zeros((N,Nh), dtype = np.float64)
z           = np.zeros((Nh), dtype = np.float64)
t           = np.linspace(0, Nh, Nh)*0.25*0.02418884254 #time in fs
for ii in range(1, Nh):
    xb_factor = 7.0 + ii*0.1
    dens_file = np.genfromtxt('data/normalized/hhg_spectra_windowed_normalized_d_w_x_start_%.3f_au_N_%.1f_sigma_%.2f_0.4qBC_w_%04d_I_%01d.dat'%(xb_factor,sigma_norm_factor,sigma_factor, w_factor,I_factor), dtype=np.float64)
    rho_d[:,ii] = np.log10(dens_file[:])

np.savetxt('data/2D_windowed_normalized_N_%.1f_sigma_%.2f_0.4qBC_w_%04d_I_%01d.dat'%(sigma_norm_factor,sigma_factor, w_factor,I_factor),rho_d)
y_axis = np.arange(0, 400, 50)
y_original = np.arange(0, 40, 5)
x_axis = np.arange(0, 45, 10)
x_original = np.arange(7, 11.5, 1)*0.529177
bounds = [ -5.6, -4.8, -4.0, -3.2, -2.4, -1.6, -0.8, 0, 0.8]
colors = ["navy",'navy', 'blue', 'royalblue','white', 'red','red', 'maroon']
cmap = mcolors.ListedColormap(colors)
cmap.set_under('navy')
cmap.set_over('red')
norm = mcolors.BoundaryNorm(bounds, cmap.N)
fig = plt.figure()
ax = fig.add_subplot( )
cs = plt.contourf( rho_d, levels=bounds, norm=norm, cmap = cmap)
cbar = fig.colorbar(cs, ax=ax, boundaries=bounds,ticks=bounds)
plt.xticks(x_axis, np.round(x_original, 2))
plt.yticks(y_axis, y_original)
y_axis_factor = 10/w_ev
plt.xticks(x_axis, np.round(x_original, 2), fontsize = 14)
y_ticks = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])*y_axis_factor  # Define specific y-tick locations
y_tick_labels = ["9", "10",  "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]  # Custom labels
plt.yticks(y_ticks, y_tick_labels, fontsize = 14)  # Apply manual y-ticks and labels
plt.tick_params(axis='both', direction='in')
plt.ylim(8.8*10/w_ev,20.5*10/w_ev)
plt.show()
