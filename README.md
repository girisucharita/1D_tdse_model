# Code & Data for model 1D TDSE calculation
This repository contains code and data to compute time-dependent dipole moments (in time and frequency domains) 
and to reproduce figures in the manuscript and supplementary materials.
## 1d_tdse.py
Computes time-dependent dipole moments and their Fourier spectra.
Also generates the time–frequency map shown in the Supplementary.
## plot_sum_spectra.py 
Sums spectra over multiple barrier cases to produce the inset of Figure 4.
Can also compute normalized spectra (with respect to the no-barrier case) for Figure 4B.
## plot_contour.py 
Generates contour for Figure. 4B.
## data
Processed data for plotting all figures: data/
Raw data used to generate the figures:
data/dipole/
data/normalized/
Paths in the scripts assume this repository’s directory structure. If you move files, update the paths inside each script.
