import numpy as np 
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from numpy.fft import fft,ifft,ifftshift,fftfreq
xb_factor = 11.5
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
w       =   45.2/w_factor     # Angular frequency if the laser of wavelength 1200 nm
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




J       = np.zeros((Nt,n), dtype = np.complex128)
Jref    = np.zeros((Nt,n), dtype = np.complex128)
d_co    = np.zeros((Nt), dtype = np.complex128)
d_in    = np.zeros(Nt)
print (x_start, sigma_norm_factor, sigma_factor)

for ii in range(n):
    step      = x_start + ii*0.10
    J[:,ii]   = np.genfromtxt('data/dipole/with_barrier_windowed_d_w_x_abs_x_start_%.3f_N_%.1f_sigma_%.2f_.4qBC_wavel_%04d.txt'%(step, sigma_norm_factor, sigma_factor,w_factor), dtype=np.complex128)
    Jref[:,ii]   = np.genfromtxt('data/dipole/without_barrier_windowed_d_w_x_abs_.4qBC_wavel_1800.txt', dtype=np.complex128)
    #d_co   += J[:,ii]/n
    d_co   += J[:,ii]/(n*Jref[:,ii])

HHG_co = abs(d_co**2)
warray = 2.*np.pi*fftfreq(Nt,dt)/w
HHGlog = np.log10(HHG_co)
array = []
for ii in range(Nt):
    if warray[ii]>0 and warray[ii] <=40 :
        array.append(HHG_co[ii])

np.savetxt('data/normalized/hhg_spectra_windowed_normalized_d_w_x_start_%.3f_au_N_%.1f_sigma_%.2f_0.4qBC_w_%04d_I_%01d.dat'%(xb_factor,sigma_norm_factor,sigma_factor, w_factor,I_factor),array)





############ Plot ##############
################################
#plt.plot(warray[0:int(Nt/2)], HHGlog[0:int(Nt/2)])
#plt.grid(which = 'major', linewidth = 1.0)
#plt.grid(which = 'minor', linewidth = 0.2)
#plt.xlim(0,40)
#plt.ylim(-15, 2)
#plt.title('coherent x_start = %.3f, Barrier %.1f, %.2f'%(x_start, sigma_norm_factor, sigma_factor))
##plt.savefig('plots/sum/coherent_sum_4_hhg_x_start_%.3f_N_%.1f_sigma_%.2f_wavel_%04d_I_%02d_0.4qBC.png'%(x_start, sigma_norm_factor, sigma_factor, w_factor, I_factor), dpi = 200)
#plt.show()
