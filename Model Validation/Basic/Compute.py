import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import rad
import radiation
import scipy.interpolate
import os

# trajectory parameters
a0 = 1e-9
step_size = 1e-5 / 299792458
parameters = {
    'x0': 2 * a0,
    'y0': 0,
    'vx0': 0,
    'vy0': 0,
    'gamma_initial': 10 * (1000 / 0.5109989461),
    'ion_atomic_number': 1,
    'plasma_density': 1e24,
    'rho_ion': 0.0,
    'accelerating_field': 0.0,
    'a0': a0,
    'step_size': step_size,
    'steps': int(round(0.1 / (299792458 * step_size)))
}
# compute trajectories
trajectories = rad.comp_traj(**parameters)
t = parameters['step_size'] * np.arange(parameters['steps'] + 1)
z = trajectories[:, 2] + 299792458 * t

# compute radiation parameters
phi_max_urad = 50
phi_points = 200
#energy = np.array([100000])
energy=np.linspace(50000,150000,100)
phis, phi_step = np.linspace(-1e-6 * phi_max_urad, 1e-6 * phi_max_urad, phi_points, endpoint=True, retstep=True)
phis_midpoint = np.linspace(-phi_max_urad - 0.5 * 1e6 * phi_step, phi_max_urad + 0.5 * 1e6 * phi_step, phi_points + 1, endpoint=True)

# compute radiation
new_trajectories = np.empty((1, parameters['steps'] + 1, 6))
new_trajectories[0, :, 0] = trajectories[:, 0]
new_trajectories[0, :, 1] = trajectories[:, 1]
new_trajectories[0, :, 2] = z
new_trajectories[0, :, 3] = trajectories[:, 3] * 299792458
new_trajectories[0, :, 4] = trajectories[:, 4] * 299792458
new_trajectories[0, :, 5] = np.sqrt(1 - trajectories[:, 5] ** -2 - trajectories[:, 3] ** 2 - trajectories[:, 4] ** 2) * 299792458
computed_radiation = radiation.compute_radiation(t, new_trajectories, energy, phis, phis)
result = np.sum(np.abs(computed_radiation) ** 2, axis=3)

energy_spectral_density = np.sum(np.sum(result, axis=1), axis=1) * phi_step * phi_step
plt.plot(energy, energy_spectral_density)
plt.savefig('energy_spectral_density.png')
plt.clf()

# plot radiation


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

range_max = np.max(result)
range_min = np.min(result)

for i in range(result.shape[0]):
    plt.pcolormesh(phis_midpoint, phis_midpoint, result[i].T, cmap='viridis', vmin=range_min, vmax=range_max)
    plt.title(r'$\epsilon = {}$ keV'.format(energy / 1000)) 
    plt.xlim(-phi_max_urad, phi_max_urad)
    plt.ylim(-phi_max_urad, phi_max_urad)
    plt.colorbar(format=ticker.FuncFormatter(fmt)).set_label(r'$\frac{d^2 U}{d\Omega d\epsilon}$')
    plt.xlabel(r'$\phi_x$ ($\mu$rad)')
    plt.ylabel(r'$\phi_y$ ($\mu$rad)')
    plt.savefig(f'frames/picture_{i}.png', dpi=500)
    plt.clf()

os.system('ffmpeg -i \'frames/picture_%d.png\' -vcodec libx264 -vf scale=640:-2,format=yuv420p movie.mp4')







r'''

for i in range(100):
    plot results[i]
    save to results_{i}.png
   
os.system('ffmpeg -i \'results_%d\' -codec ....... movie.mp4')


result = result[0, :, :]

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

plt.pcolormesh(phis_midpoint, phis_midpoint, result, cmap='viridis')
plt.title(r'$\epsilon = {}$ keV'.format(energy / 1000))
plt.xlim(-phi_max_urad, phi_max_urad)
plt.ylim(-phi_max_urad, phi_max_urad)
plt.colorbar(format=ticker.FuncFormatter(fmt)).set_label(r'$\frac{d^2 U}{d\Omega d\epsilon}$')
plt.xlabel(r'$\phi_x$ ($\mu$rad)')
plt.ylabel(r'$\phi_y$ ($\mu$rad)')
plt.savefig('picture.png', dpi=500)
plt.clf()
'''


