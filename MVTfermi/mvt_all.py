from haar_power_mod import haar_power_mod
from numpy import loadtxt
'''
file='grb_lc.txt.gz'
import numpy as np

# Read the file, skipping the header
data = np.loadtxt('gauss_sigma0.1_bw_.1ms.txt', skiprows=1)

# Split into time and count columns
time = data[:, 0]
count = data[:, 1]

print("Time:", time)
print("Count:", count)
r = count
dr = np.sqrt(r)  # Assuming Poisson noise, error is sqrt of counts
t,r,dr = loadtxt(file,unpack=True,usecols=(0,2,3))



haar_power_mod(r,dr,min_dt=1.e-4,max_dt=100.,tau_bg_max=0.01,nrepl=2,doplot=True,bin_fac=4,zerocheck=False,afactor=1.,snr=3.,verbose=True,weight=True,file='test')
'''
import numpy as np
import warnings
from haar_power_mod import haar_power_mod
# Make sure your other python files (haar_denoise.py, etc.) are also in this directory

# OPTIONAL: This will hide the RuntimeWarning messages for a cleaner output
warnings.filterwarnings('ignore', r'divide by zero encountered')
warnings.filterwarnings('ignore', r'invalid value encountered')

sigma = np.arange(0.1, 1.0, 0.1)  # Example sigma values
for sig in sigma:  # Example bandwidth values
    try:
        # 1. Load the data
        data = np.loadtxt(f'gauss_sigma{round(sig,1)}_bw_.1ms.txt', skiprows=1)
        rate = data[:, 1]
        drate = np.sqrt(rate)
        min_dt = 1.0e-4


        #file='grb_lc.txt.gz'
        #t,rate,drate = loadtxt(file,unpack=True,usecols=(0,2,3))
        #print("Data loaded successfully.")

        # 2. Call the analysis function and CAPTURE THE RETURNED VALUES
        print(f"Running variability analysis sigma={sig}.@@@@@@@@")

        try:
            # Attempt to run the analysis with the provided parameters
            tsnr, tbeta, tmin, dtmin, slope, sigma_tsnr, sigma_tmin = haar_power_mod(
                rate, drate, min_dt=min_dt, max_dt=100., tau_bg_max=0.01, nrepl=2,
                doplot=True, bin_fac=4, zerocheck=False, afactor=-1., snr=3.,
                verbose=True, weight=True, file='test'
            )
                # 3. PRINT THE FINAL RESULTS
            print("\n--- Analysis Complete: Final Result ---")

            if dtmin > 0:
                # This block runs ONLY if a successful measurement was made
                print(f"Variability timescale found (t_min): {tmin:.6f} +/- {dtmin:.6f} s")
                print("A plot file named 'test_haar_mod.png' should have been generated.")
            elif tmin > 0:
                # This block runs if the code returned an upper limit (dtmin will be 0)
                print("No significant variability was found to perform a fit.")
                print(f"An UPPER LIMIT on the timescale has been determined:")
                print(f"Variability timescale (t_min) is less than: {tmin:.6f} s")
                print("(No plot is generated when only an upper limit is found).")
            else:
                # This is a fallback for any other case
                print("Analysis could not determine a measurement or a limit.")
        except Exception as e:
            print("Error!!!")
            #print(f"An error occurred during the analysis: {e}")
            #print("Please check the input data and parameters.")
            #print(f"###############Completed analysis for sigma = {sig:.1f}\n\n")
            #continue
        #print(f"###############Completed analysis for sigma = {sig:.1f}\n\n")
    except FileNotFoundError:
            print("Error: The file 'gauss_sigma0.1_bw_.1ms.txt' was not found.")
            print("Please make sure the data file is in the same directory as the script.")

        

        