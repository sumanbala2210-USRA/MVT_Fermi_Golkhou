from haar_power_mod import haar_power_mod
from numpy import loadtxt

import numpy as np
import warnings

warnings.filterwarnings('ignore', r'divide by zero encountered')
warnings.filterwarnings('ignore', r'invalid value encountered')

min_dt = 1.0e-4

file='grb_lc.txt.gz'
t,rate,drate = loadtxt(file,unpack=True,usecols=(0,2,3))
print("Data loaded successfully.......")

print("Running variability analysis...")
tsnr, tbeta, tmin, dtmin, slope, sigma_tsnr, sigma_tmin = haar_power_mod(
    rate, drate, min_dt=min_dt, max_dt=100., tau_bg_max=0.01, nrepl=2,
    doplot=True, bin_fac=4, zerocheck=False, afactor=-1., snr=3.,
    verbose=True, weight=True, file='test'
)

print("\n--- Analysis Complete: Final Result ---")
