#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import rad
import radiation
import scipy.interpolate
import sdf_helper as sh

import os


# Parameters 

for q in [str(n).zfill(4)+'.sdf' for n in np.arange(1, 9, 1)]:
    data = sh.getdata(q)
    s_file=100
    qt=int(q[:4])
    S=qt*s_file
    theta_bin_edges, dtheta = np.linspace(-2e-4, 2e-4, 100, retstep=True)
    n_theta_bins = len(theta_bin_edges) - 1
    px = data.Particles_Px_photons.data
    py = data.Particles_Py_photons.data
    pz = data.Particles_Pz_photons.data
    theta_x = np.arctan2(pz, px)
    theta_y = np.arctan2(py, px)
    energy = np.sqrt(px**2+py**2+pz**2) * 299792458 / 1.60217662e-19
    weights = data.Particles_Weight_photons.data
    energy_times_weights = energy * weights
    dE_dth2 = np.empty((n_theta_bins, n_theta_bins))
    for ix in range(n_theta_bins):
        x_bin_min = theta_bin_edges[ix]
        x_bin_max = theta_bin_edges[ix + 1]
        x_mask = np.logical_and(x_bin_min < theta_x, theta_x < x_bin_max)
        for iy in range(n_theta_bins):
            y_bin_min = theta_bin_edges[iy]
            y_bin_max = theta_bin_edges[iy + 1]
            y_mask = np.logical_and(y_bin_min < theta_y, theta_y < y_bin_max)
            dE_dth2[ix, iy] = ((energy_times_weights)[np.logical_and(x_mask, y_mask)]).sum() / (dtheta ** 2)
    theta_midpoint = np.linspace(theta_bin_edges.min() - 0.0001 * dtheta,
        theta_bin_edges.max() + 0.0001 * dtheta, n_theta_bins)
'''
energies: a 1d numpy array of np.float64 containing values of the photon
            energies in eV. [eV]
phi_xs: a 1d numpy array of np.float64 containing projected angles in x
            [rad]
phi_ys: a 1d numpy array of np.float64 containing projected angles in y
            [rad]
        time_step: step size [s]
'''


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


