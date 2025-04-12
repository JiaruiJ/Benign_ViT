import matplotlib
from matplotlib import pyplot as plt
import numpy as np

size = 10
matrix = np.loadtxt('matrix2.npy')

plt.imshow(matrix, cmap='viridis', interpolation='nearest', origin='lower')
plt.colorbar()
plt.xlabel('Signal-to-Noise Ratio SNR', size=14)
plt.ylabel('Sample size N', size=14)
x_label = np.array((0.2, 0.6, 1.0, 1.4, 1.8))
y_label = np.array((3000, 4200, 5400, 6600, 7800))
x_ticks = np.array((0, 2, 4, 6, 8))
y_ticks = np.array((0, 2, 4, 6, 8))
plt.xticks(x_ticks, x_label)
plt.yticks(y_ticks, y_label)
plt.savefig('heatmap2.png', dpi=1200, bbox_inches='tight')

matrix[matrix < 0.35] = 0
matrix[matrix >= 0.35] = 1

plt.cla()
plt.clf()

plt.figure(1)
# plt.plot(x, y, color='red', linewidth=5)
plt.imshow(matrix, cmap='viridis', interpolation='nearest', origin='lower')
plt.colorbar()
plt.xlabel('Signal-to-Noise Ratio SNR', size=14)
plt.ylabel('Sample size N', size=14)
plt.xticks(x_ticks, x_label)
plt.yticks(y_ticks, y_label)
# plt.legend(loc='best')
plt.savefig('heatmap_cutoff2.png', dpi=1200,  bbox_inches='tight')
