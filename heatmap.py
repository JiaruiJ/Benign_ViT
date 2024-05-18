from matplotlib import pyplot as plt
import numpy as np

size = 100
matrix = np.loadtxt('matrix.npy')

x = []
y = []
for i in range(size):
    x.append(i + 1)
    y.append(204800 / (i + 1)**2 - 10)

plt.plot(x, y, color='red', linewidth=5)
plt.imshow(matrix, cmap='viridis', interpolation='nearest', origin='lower')
plt.colorbar()
plt.xlabel('Signal-to-Noise Ratio SNR', size=14)
plt.ylabel('Sample size N', size=14)
x_label = np.array((1, 20, 40, 60, 80, 100))
y_label = np.arange(2, 22, 2)
x_ticks = np.array((0, 19, 39, 59, 79, 99))
y_ticks = np.arange(0, size, 10)
plt.xticks(x_ticks, x_label)
plt.yticks(y_ticks, y_label)
plt.savefig('heatmap.png', dpi=1200, bbox_inches='tight')

matrix[matrix < 0.2] = 0
matrix[matrix >= 0.2] = 1

plt.cla()
plt.clf()

plt.figure(1)
plt.plot(x, y, color='red', linewidth=5)
plt.imshow(matrix, cmap='viridis', interpolation='nearest', origin='lower')
plt.colorbar()
plt.xlabel('Signal-to-Noise Ratio SNR', size=14)
plt.ylabel('Sample size N', size=14)
plt.xticks(x_ticks, x_label)
plt.yticks(y_ticks, y_label)
plt.savefig('heatmap_cutoff.png', dpi=1200,  bbox_inches='tight')