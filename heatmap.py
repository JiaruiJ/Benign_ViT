import matplotlib
from matplotlib import pyplot as plt
import numpy as np

size = 100
matrix = np.loadtxt('matrix.npy')

x = []
y = []
for i in range(size):
    x.append(i + 1)
    y.append(204800 / (i + 1)**2 - 10)

plt.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
plt.plot(x, y, color='red', linewidth=5, label=r'$N \cdot \mathrm{SNR}^2 = 1000$')
plt.imshow(matrix, cmap='viridis', interpolation='nearest', origin='lower')
plt.colorbar()
plt.xlabel('Signal-to-Noise Ratio SNR', size=14)
plt.ylabel('Sample size N', size=14)
x_label = np.array((0.16, 5, 10, 15))
y_label = np.arange(2, 22, 2)
x_ticks = np.array((0, 31, 63, 95))
y_ticks = np.arange(0, size, 10)
plt.xticks(x_ticks, x_label)
plt.yticks(y_ticks, y_label)
plt.legend(loc='best')
plt.savefig('heatmap.png', dpi=1200, bbox_inches='tight')

matrix[matrix < 0.2] = 0
matrix[matrix >= 0.2] = 1

plt.cla()
plt.clf()

plt.figure(1)
plt.plot(x, y, color='red', linewidth=5, label=r'$N \cdot \mathrm{SNR}^2 = 1000$')
plt.imshow(matrix, cmap='viridis', interpolation='nearest', origin='lower')
plt.colorbar()
plt.xlabel('Signal-to-Noise Ratio SNR', size=14)
plt.ylabel('Sample size N', size=14)
plt.xticks(x_ticks, x_label)
plt.yticks(y_ticks, y_label)
plt.legend(loc='best')
plt.savefig('heatmap_cutoff.png', dpi=1200,  bbox_inches='tight')
