import numpy as np
from numpy import arange,zeros,savetxt
from  scipy.linalg import eig_banded,norm
from numpy.linalg import norm
from matplotlib import pyplot as plt
from numpy.fft import fft,ifft,ifftshift,fftfreq


#The Attractive Potential Function for ionization PE of around 10 eV
def A_potential(x):
    return np.float64(-1.0/np.sqrt(x**2+4.105))

#The Boundary Potential Function
def B_potential(x, mu, sigma, Norm): # Function for Sin2 envelope
        return  (+1/(Norm*np.sqrt(2.0*np.pi)))*np.exp(-((x - mu)**2.0/(2.0*sigma**2.0)))

print(A_potential(0)*27.2)
print(B_potential(10.4, 10.4, 0.9, 3)*27.2)
n_on    =   2
n_p     =   6
n_off   =   2
n_cycle =   n_on + n_p + n_off      #Total number of cycles within the envelope
I_factor = 2
#n_cycle =   12         #Total number of cycles within the envelope
w_factor = 1800
w       =   45.2/w_factor     # Angular frequency if the laser of wavelength 1200 nm
T_p     =   2.*np.pi*n_cycle/w
T_p2    =   2.*np.pi*0.5/w
I0      =   I_factor*1.0e13
F0      =   np.sqrt(I0/3.5095e16)
q       =   F0/w**2
tstart  =   0.0
tend    =   1.01*T_p
dt      =   0.100
Nt      = int((tend-tstart)/dt)

N = 5000
frac = 0.4
xmin    =   -2.0*q
xmax    =   +2.0*q
x0      =   frac*q
dx  = (xmax-xmin)/np.float64(N)
del_x = dx 
xb_factor = 11.5
xb_p     =   xb_factor
xb_m     =   -xb_p
sigma_factor = 0.75
sigma_norm_factor = 3.0
sigma   =   np.sqrt(sigma_factor)
Norm =  sigma_norm_factor
print ("I is ",I_factor)
print ("w is ",w_factor)
print ("f is ",xb_factor)
print ("sigma is ",sigma_factor)
print ("Norm is ",Norm)
print(q)

def F(t):   #Electric field function
    #w       =   w_factor       # Angular frequency if the laser of wavelength 800 nm
    #I0      =   2.0e13
    F0      =   np.sqrt(I0/3.5095e16)
    def TrapezoidalField(t,n_on, n_p, n_off): # Function for Trapezoidal envelope
        t1 = 2*np.pi*n_on/w
        t2 = t1 + 2*np.pi*n_p/w
        t3 = t2 + 2*np.pi*n_off/w
        if t<=t1 and t>0. :
            return t/t1
        elif t<= t2 and t>=t1:
            return 1.
        elif t<= t3 and t>t2:
            return -((t-t3)/(t3-t2))
        else:
            return 0.
    return F0*TrapezoidalField(t,n_on, n_p, n_off)*np.sin(w*t)


######## Half cycle pulse window function ##########
####################################################

def W(t):   #Half cycle window function
        if t<7.5*T_p2:
            return 0.
        elif t>=9.5*T_p2:
            return 0.
        else:
            return F0*np.cos(0.50*np.pi*(t- 8.5*T_p2)/T_p2)**4.


x   =   np.arange(xmin,xmax,dx)
N	=   len(x)

#potential energy function
B1  = np.zeros(N)
B2  = np.zeros(N)
for ii in range(N):
    B1[ii] = B_potential(x[ii], xb_p, sigma_factor, Norm)
    B2[ii] = B_potential(x[ii], xb_m, sigma_factor, Norm)
B = B1 + B2 + A_potential(x)

T = zeros((N,N),dtype='longdouble')                               
V = zeros((N,N),dtype='longdouble')                            
for i in range(N):
    T[i,i] = np.float64(-2./(del_x**2))             
for i in range(N-1):
    T[i,i+1] = np.float64(1./(del_x**2))
    T[i+1,i] = np.float64(1./(del_x**2))
 
T = (-np.float64(0.5))*T                              
#Vdiag = np.float64(A_potential(x))         
Vdiag = np.float64(B)         
for i in range(N):
    V[i,i] = np.float64(Vdiag[i])       

H = T + V                                          

u = 1                                       
a_band = zeros((u+1,N))                     
Vder = np.gradient(Vdiag,dx)






for i in range(N):
    a_band[1,i] = np.float64(H[i,i])   
for i in range(N-1):
    a_band[0,i+1] = np.float64(H[i+1,i])

en,v = eig_banded(a_band)
print(en[0])



x   =   np.linspace(xmin,xmax,N)


## Smooth, boundary-aware mask (no zero tail plateau)
## Between -x0 and +x0 -> 1
## From -x0 to xmin and from +x0 to xmax -> cosine decay to 0 (slope set by distance)
gamma = 1/8.0      # optional softness like your original (set to 0 for pure cosine)
eps = 1e-12        # guard for degenerate widths
eta     =   0.0005
m       =   3

mask = np.zeros(N, dtype=float)

for i in range(N):
    xi = x[i]

    if -x0 <= xi <= x0:
        mval = 1.0

    elif xi < -x0:
        # map [-x0 .. xmin] -> u in [0..1]
        denom = (xmin + x0)  # note: xmin < -x0, so xmin + x0 is negative
        # u should go 0 at -x0, 1 at xmin
        u = (xi + x0) / (denom + eps)  # u in [0, 1]
        u = np.clip(u, 0.0, 1.0)
        mval = 0.5 * (1.0 + np.cos(np.pi * u))

    else:  # xi > +x0
        # map [x0 .. xmax] -> u in [0..1]
        denom = (xmax - x0)
        u = (xi - x0) / (denom + eps)  # u in [0, 1]
        u = np.clip(u, 0.0, 1.0)
        mval = 0.5 * (1.0 + np.cos(np.pi * u))

    # optional softness (keeps your gentle roll-off)
    if gamma != 0:
        mval = mval ** gamma

    mask[i] = mval

Vabs    =   np.zeros(N,dtype='complex64')       # Potential with absorbing boundary conditions

for i in range(N):
    if x[i]>=x0:
        Vabs[i] = Vdiag[i]  - 1j*eta*((x[i]-x0)**m)
        
    elif x[i]<=-x0:
        Vabs[i] = Vdiag[i]  - 1j*eta*(-(x[i]+x0)**m)
    
    else:
        Vabs[i] = Vdiag[i]
##============================================================================================##

u   =   v[:,0] 

k 	    =    2.*np.pi*fftfreq(N,d=dx)
KE_k    =   np.float64(0.5*k**2)                   

d = np.zeros(Nt,dtype = 'clongdouble')


for i in range(Nt):
    t = (i+1)*dt
	
    u_1 =   fft(u)
    ke_x_1 =   np.exp(-1.0j*KE_k*dt/2.0)*u_1   
    u_r =   ifft(ke_x_1)
    ve_x_ur =  np.exp(-1.0j*(Vdiag + x*F(t))*dt)*u_r
    u_2 =  fft(ve_x_ur)
    ke_x_2 = np.exp(-1.0j*KE_k*dt/2.0)*u_2
    u =  ifft(ke_x_2)
    u = u*mask
    dd = np.vdot(u,(-Vder - F(t))*u)
    d[i] = dd
    #d[i] = W(t)*dd

d_w = fft(d)
HHG = abs((d_w)**2)
warray = 2.*np.pi*fftfreq(Nt,dt)/w
HHGlog = np.log10(HHG)


np.savetxt('data/dipole/with_barrier_d_x_abs_x_start_%.3f_N_%.1f_sigma_%.2f_.4qBC_wavel_%04d.txt'%(xb_factor, sigma_norm_factor, sigma_factor,w_factor), d)
np.savetxt('data/dipole/with_barrier_d_w_x_abs_x_start_%.3f_N_%.1f_sigma_%.2f_.4qBC_wavel_%04d.txt'%(xb_factor, sigma_norm_factor, sigma_factor,w_factor), d_w)
#np.savetxt('data/dipole/without_barrier_windowed_d_x_abs_.4qBC_wavel_%04d.txt'%(w_factor), d)
#np.savetxt('data/dipole/without_barrier_windowed_d_w_x_abs_.4qBC_wavel_%04d.txt'%(w_factor), d_w)


#plt.plot(warray[0:int(Nt/2)], HHGlog[0:int(Nt/2)])
#plt.grid(which = 'major', linewidth = 1.0)
#plt.grid(which = 'minor', linewidth = 0.2)
#plt.xlim(0.0,40.0)
##plt.ylim(-14.0,2)
#plt.savefig('plots/HHG_spectra.png', dpi = 512)
#plt.show()
#plt.clf()
####

###Gabor transformation
#warray = 2.*np.pi*fftfreq(Nt,dt)/w
#tpstart     =   0.0
#tpend       =   2000.0
#dtp         =   0.5
#Ntp         =   int((tpend-tpstart)/dtp)
#tp = np.linspace(tpstart, tpstart + (Ntp-1)*dtp, Ntp)  # length Ntp
#t  = np.linspace(0.0, (Nt-1)*dt, Nt)                   # length Nt
#
#d_function  =   np.zeros((Ntp,Nt),dtype='complex64')
#sigma_g   =   1/(3.*w)
#for ii in range(Ntp):
#    tt = ii*dtp
#    d_function[ii,:] =  (1/sigma_g*np.sqrt(2.*np.pi))*np.exp(-((tt - t)**2.)/(2.*sigma_g**2.))*d
#d_gabor = fft(d_function)  #column wise fft of t
#wparray = 2.*np.pi*fftfreq(Ntp,dtp)/w
#
#################Gabor plot###########
######################################
#T_laser = 2.0*np.pi / w          # one optical cycle
#tp_cyc  = tp / T_laser            # time in cycles
#
## use cycles on the x-axis
#Tcyc, Hw = np.meshgrid(tp_cyc, warray)
#Z = np.log(np.transpose(np.abs(d_gabor**2.)))
#
#levels = np.linspace(-60.0, -5.0, 201 )
#cs = plt.contourf(Tcyc, Hw, Z, levels=levels, cmap="jet")
#cbar = plt.colorbar(cs)
#plt.clim(-17.0, -5.0)
#
#plt.xlabel('Time (cycles)')
#plt.ylabel('Harmonic order')
#
## convenient limits/ticks in cycles
#plt.xlim(2.00, 3.25)
##plt.xlim(0.0, tpend / T_laser)
##n_cycles = int(np.ceil(tpend / T_laser))
##plt.xticks(np.arange(0, n_cycles+1, 1))
#plt.ylim(0, 45)
#plt.savefig('plots/Gabor_x_start_%.3f_N_%.1f_sigma_%.2f_entire_pulse_at_0.4q_end_2.0q_1800_2_13.png'%(xb_factor, sigma_norm_factor,sigma_factor), dpi = 512)
#plt.show()
######################################
##cycle     =   2.*np.pi/w
##T, Tp = np.meshgrid(tp, warray)
##levels = np.linspace(-60.0,-5.0,101)
##cs = plt.contourf(T, Tp, np.log(np.transpose(abs(d_gabor**2.))), levels=levels, cmap ="jet")
###s = plt.contourf(T, Tp, np.log(np.transpose(abs(d_gabor**2.))), cmap ="jet")
##cbar = plt.colorbar(cs)
###plt.clim(-30.0, -14.0)
###plt.clim(-20.0, -7.0)
##plt.clim(-17.0, -5.0)
##plt.xlabel('Time')
##plt.ylabel('Harmonic order')
###plt.xlim(0, 1100)
##plt.ylim(0, 60)
###plt.savefig('plots/window/gabor/gabor_without_bump_triple_pulse1_x_abs_0.4q_end_2.0q.png')
###plt.savefig('plots/gabor_with_bump_x_start_%.3f_N_%.1f_sigma_%.2f_entire_pulse_at_0.4q_end_2.0q_1800_2_13.png'%(xb_factor, sigma_norm_factor,sigma_factor), dpi = 200)
##plt.show()
##
