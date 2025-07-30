import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV file
det_string = 'best'
df = pd.read_csv(f'Trigger_number_vs_mvt_{det_string}_det_2.csv')

# Filter rows where mvt_ms > 0 and PF64 > 0 and T90 > 0 (to avoid log(0))
df_filtered = df[(df['mvt_ms'] > 0) & (df['PF64'] > 0) & (df['T90'] > 0)].copy()

# Separate rows with upper limits (mvt_error_ms == 0)
upper_limits = df_filtered[df_filtered['mvt_error_ms'] == 0]
normal = df_filtered[df_filtered['mvt_error_ms'] > 0]

# Log-transform the data
normal['log_PF64'] = np.log10(normal['PF64'])
normal['log_T90'] = np.log10(normal['T90'])
normal['log_mvt_ms'] = np.log10(normal['mvt_ms'])

upper_limits['log_PF64'] = np.log10(upper_limits['PF64'])
upper_limits['log_T90'] = np.log10(upper_limits['T90'])
upper_limits['log_mvt_ms'] = np.log10(upper_limits['mvt_ms'])

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot normal data points
ax.scatter(
    normal['log_PF64'], normal['log_T90'], normal['log_mvt_ms'],
    c='blue', label='MVT (normal)', alpha=0.8
)

# Plot upper limit points
'''
ax.scatter(
    upper_limits['log_PF64'], upper_limits['log_T90'], upper_limits['log_mvt_ms'],
    c='red', marker='v', label='Upper limit', alpha=0.6
)
'''
# Axis labels (indicate log)
ax.set_xlabel('log10(PF64)')
ax.set_ylabel('log10(T90 [s])')
ax.set_zlabel('log10(MVT [ms])')
ax.set_title('3D Log-Log-Log Plot: PF64 vs T90 vs MVT')
ax.legend()
plt.tight_layout()

# Save and show
#plt.savefig('3D_log_PF64_T90_MVT_plot.pdf', dpi=300)
plt.show()
