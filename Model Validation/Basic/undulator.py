from radiation import compute_radiation
import numpy as np
import matplotlib.pyplot as plt

undulator_period = 0.01
n_periods = 2
K = 0.01
gamma = 1000 * 10 / 0.510998
N = 1000

beta = np.sqrt(1 - gamma ** -2)
k = 2 * np.pi / undulator_period
a = undulator_period * K / (2 * np.pi * gamma)
z = np.linspace(0, undulator_period * n_periods, N)
trajectories = np.empty(dtype=np.float64, shape=(1, N, 6))
trajectories[0, :, 0] = a * np.cos(k * z)
trajectories[0, :, 1] = 0
trajectories[0, :, 2] = z
trajectories[0, :, 3] = -(a * k * beta * 299792458) * np.sin(k * z)
trajectories[0, :, 4] = 0
trajectories[0, :, 5] = beta * 299792458
t = z / (beta * 299792458)

thetax = np.linspace(-50e-6, 50e-6, 100)
thetay = np.linspace(-50e-6, 50e-6, 100)
energies = np.array([1e5])

result = compute_radiation(t, trajectories, energies, thetax, thetay)

def plotz(x_min, x_max, x_num, y_min, y_max, y_num, z, **kwargs):
    x, x_step = np.linspace(x_min, x_max, x_num, retstep=True, endpoint=True)
    y, y_step = np.linspace(y_min, y_max, y_num, retstep=True, endpoint=True)
    xmesh, ymesh = np.meshgrid(x, y)
    x_midpoint = np.linspace(x_min - 0.5 * x_step, x_max + 0.5 * x_step, x_num + 1)
    y_midpoint = np.linspace(y_min - 0.5 * y_step, y_max + 0.5 * y_step, y_num + 1)
    plt.pcolormesh(x_midpoint, y_midpoint, z, **kwargs, cmap='inferno')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

plotz(-1000, 1000, 100, -1000, 1000, 100, np.sum(np.abs(result) ** 2, axis=3)[0].T)
plt.colorbar()
plt.savefig('result.png')
