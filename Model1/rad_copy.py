import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import scipy.special
import scipy.integrate
import synchrad
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

os.system('mpirun -np 64 ./model1 input')
#os.system('mpirun -np 64 ./model1 input2')

os.system('mkdir -p results_og')

def Sb(x):
    return (9 * np.sqrt(3) / (16 * np.pi)) * (Lambda ** 4) * (x ** 3) * \
        scipy.integrate.quad(lambda v: scipy.integrate.quad(
            lambda u: scipy.special.kv(5/3, u) * np.exp(-Lambda * Lambda * x * x / (2 * v * v)) / (v * v * v),
            v, np.inf)[0], 0, np.inf)[0]
Sb = np.vectorize(Sb)

#.5nC
#4.5um
#3.2e-6
#4e22 m-3
#0.6m

Lambda = 1.7231

output_dict = {}
with open('output', 'r') as f:
    for line in f:
        split = line.split()
        output_dict[split[0]] = (split[2], split[3])

n0 = float(output_dict['plasma_density'][0])
plasma_length = float(output_dict['plasma_length'][0])
gamma = float(output_dict['gamma_initial'][0])
spot_size = float(output_dict['sigma_x'][0])
assert float(output_dict['sigma_x'][0]) == float(output_dict['sigma_y'][0])
beam_charge = 0.5e-9
plasma_angular_wavenumber = np.sqrt(4 * np.pi * 2.817940322719e-15 * n0)
e_crit = (3 /4) * 1.054571817e-34 * 299792458 * Lambda * plasma_angular_wavenumber * plasma_angular_wavenumber * gamma * gamma * spot_size
e_crit_ev = e_crit / 1.602176634e-19
I0b = (1 / (6 * 1.602176634e-19)) * 2.817940322719e-15 * 9.109383701528e-31 * 299792458 * 299792458* (plasma_angular_wavenumber ** 4) * plasma_length * beam_charge * gamma * gamma * spot_size * spot_size
#I0b = (np.sqrt(np.pi) / (3 * 1.602176634e-19)) * 2.817940322719e-15 * 9.109383701528e-31 * 299792458 * 299792458 * (plasma_angular_wavenumber ** 3) * plasma_length * beam_charge * (gamma ** (5/2)) * spot_size
I0b_ev = I0b / 1.602176634e-19
k_beta = plasma_angular_wavenumber / np.sqrt(2 * gamma)

def linspace_midpoint(min, max, num):
    vals, step = np.linspace(min, max, num, endpoint=True, retstep=True)
    vals_midpoint = np.linspace((min - 0.5 * step), (max + 0.5 * step), (num + 1), endpoint=True)
    return (vals, vals_midpoint, step)

def logspace_midpoint(min, max, num):
    vals, step = np.linspace(np.log(min), np.log(max), num, endpoint=True, retstep=True)
    vals_midpoint = np.linspace((np.log(min) - 0.5 * step), (np.log(max) + 0.5 * step), (num + 1), endpoint=True)
    return np.exp(vals), np.exp(vals_midpoint)

assert output_dict['energy_scale'][0] == 'log'

energies, energies_midpoint = logspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), int(output_dict['energy_num'][0]))
energies2, _ = logspace_midpoint(float(output_dict['energy_min'][0]), float(output_dict['energy_max'][0]), 1000)
phi_xs, phi_xs_midpoint, phi_xs_step = linspace_midpoint(float(output_dict['phix_min'][0]), float(output_dict['phix_max'][0]), int(output_dict['phix_num'][0]))
phi_ys, phi_ys_midpoint, phi_ys_step = linspace_midpoint(float(output_dict['phiy_min'][0]), float(output_dict['phiy_max'][0]), int(output_dict['phiy_num'][0]))
particles = int(output_dict['actual_particles'][0])

result = np.fromfile('radiation').reshape((len(energies), len(phi_xs), len(phi_ys), 6))
dd = np.sum(result ** 2, axis=3)
dd *= (beam_charge / (1.602176634e-19 * particles))

'''
result_matched = np.fromfile('radiation2').reshape((len(energies), len(phi_xs), len(phi_ys), 6))
dd_matched = np.sum(result_matched ** 2, axis=3)
dd_matched *= (beam_charge / (1.602176634e-19 * particles))
spectrum_matched = np.sum(dd_matched, axis=(1,2)) * phi_xs_step * phi_ys_step
'''

fig, ax = plt.subplots()
dd2 = np.sum(dd, axis=2) * phi_ys_step
vmin, vmax = dd2.min(), dd2.max()
hm = ax.pcolormesh(energies_midpoint, phi_xs_midpoint * 1e6, dd2.T)
ax.set_xlim(energies.min(), energies.max())
ax.set_ylim(phi_xs.min() * 1e6, phi_xs.max() * 1e6)
ax.set_xscale('log')
ax.set_xlabel('photon energy (eV)')
ax.set_ylabel(f'$\\phi_x$ ($\\mu$rad)')
cbar = fig.colorbar(hm, ax=ax)
cbar.set_label('$\\frac{dI}{d\\phi_x}$ (eV)')
fig.savefig('results_og/double_differential', dpi=300)
plt.close(fig)

'''
fig, ax = plt.subplots()
dd2_matched = np.sum(dd_matched, axis=2) * phi_ys_step
vmin, vmax = dd2_matched.min(), dd2_matched.max()
hm = ax.pcolormesh(energies_midpoint, phi_xs_midpoint * 1e6, dd2_matched.T)
ax.set_xlim(energies.min(), energies.max())
ax.set_ylim(phi_xs.min() * 1e6, phi_xs.max() * 1e6)
ax.set_xscale('log')
ax.set_xlabel('photon energy (eV)')
ax.set_ylabel(f'$\\phi_x$ ($\\mu$rad)')
cbar = fig.colorbar(hm, ax=ax)
cbar.set_label('$\\frac{dI}{d\\phi_x}$ (eV)')
fig.savefig('results_og/double_differential_matched', dpi=300)
plt.close(fig)
'''

dist = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
vmin, vmax = dist.min(), dist.max()
fig, ax = plt.subplots()
ax.pcolormesh(phi_xs_midpoint * 1e6, phi_ys_midpoint * 1e6, dist.T, vmin=vmin, vmax=vmax)
ax.set_xlim(phi_xs.min() * 1e6, phi_xs.max() * 1e6)
ax.set_ylim(phi_ys.min() * 1e6, phi_ys.max() * 1e6)
ax.set_xlabel('$\\phi_x$ ($\\mu$rad)')
ax.set_ylabel('$\\phi_y$ ($\\mu$rad)')
cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax)), ax=ax)
cbar.set_label('$\\frac{dI}{d\\Omega}$ (eV)')
fig.savefig('results_og/dist.png', dpi=300)
plt.close(fig)
'''
dist_matched = np.sum(0.5 * (dd_matched[1:, :, :] + dd_matched[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
vmin, vmax = dist_matched.min(), dist_matched.max()
fig, ax = plt.subplots()
ax.pcolormesh(phi_xs_midpoint * 1e6, phi_ys_midpoint * 1e6, dist_matched.T, vmin=vmin, vmax=vmax)
ax.set_xlim(phi_xs.min() * 1e6, phi_xs.max() * 1e6)
ax.set_ylim(phi_ys.min() * 1e6, phi_ys.max() * 1e6)
ax.set_xlabel('$\\phi_x$ ($\\mu$rad)')
ax.set_ylabel('$\\phi_y$ ($\\mu$rad)')
cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax)), ax=ax)
cbar.set_label('$\\frac{dI}{d\\Omega}$ (eV)')
fig.savefig('results_og/dist_matched.png', dpi=300)
plt.close(fig)
'''

fig, ax = plt.subplots()
custom_lines = []
dist = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
dist_y = np.sum(dist, axis=0) * phi_xs_step
dist_x = np.sum(dist, axis=1) * phi_ys_step
ax.plot(phi_xs * 1e6, dist_x, color='black', linestyle='-', label='$x$')
ax.plot(phi_ys * 1e6, dist_y, color='black', linestyle='--', label='$y$')
ax.legend()
ax.set_xlabel('$\\phi$ ($\\mu$rad)')
ax.set_ylabel('$\\frac{dI}{d\\phi}$')
fig.savefig('results_og/1d_dist.png', dpi=300)
plt.close(fig)

fig, ax = plt.subplots()
energies2 = np.linspace(np.min(energies), np.max(energies), 300)
spectrum = np.sum(dd, axis=(1,2)) * phi_xs_step * phi_ys_step
ax.plot(energies, spectrum, label='numerical', color='C0')
#ax.plot(energies, spectrum_matched, label='numerical', color='C1')
ax.plot(energies2, (I0b_ev / e_crit_ev) * Sb(energies2 / e_crit_ev), label='analytic', color='black', linestyle='dashed')
#ax.legend()
ax.set_xscale('log')
ax.set_xlabel('photon energy (eV)')
ax.set_ylabel('$\\frac{dI}{d\\epsilon}$')
ax.set_yscale('log')
fig.savefig(f'results_og/spectrum.png', dpi=400)
plt.close(fig)

with open('results_og/total.txt', 'w+') as f:
    tot = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis]) * phi_xs_step * phi_ys_step
    f.write(f'{tot:.15e}\n')


r"""


fig, ax = plt.subplots()
vmin, vmax = double_differential.min(), double_differential.max()
ax.pcolormesh(energies_midpoint, thetas_midpoint * 1e3, double_differential[:, :, 0].T / (particles), vmin=vmin / (particles), vmax=vmax / (particles), cmap='inferno')
ax.set_xlim(energies.min(), energies.max())
ax.set_ylim(thetas.min() * 1e3, thetas.max() * 1e3)
ax.set_xscale('log')
ax.set_xlabel('photon energy (eV)')
ax.set_ylabel(r'$\theta$ (mrad)')
ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap='inferno'), ax=ax)
cbar.set_label(r'$\frac{d^2 U}{d\Omega d\epsilon}$ per particle')
cbar.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
fig.savefig('double_differential.png', dpi=300)
plt.close(fig)


electron_rest_energy_mev = 0.510998
c_light = 299792458
classical_electron_radius = 2.8179403227e-15

spectrum = 2 * np.pi * np.sum(double_differential[:, :, 0] * np.sin(thetas[np.newaxis, :]), axis=1) * thetas_step * (0.5e-9 / 1.602176e-19) / 1000

fig, ax = plt.subplots()
ax.plot(energies, spectrum, label='numerical')
ax.plot(energies, spectrum_analytic, label='analytic')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('photon energy (eV)')
ax.set_ylabel(r'$\frac{dU}{d\epsilon}$')
ax.set_ylim(1e8, 1e10)
ax.legend()
fig.savefig('spectrum.png', dpi=300)
plt.close(fig)

r'''
distribution = np.sum(double_differential[:-1, :, 0] * (energies[1:, np.newaxis] - energies[:-1, np.newaxis]), axis=0)

fig, ax = plt.subplots()
ax.plot(thetas * 1e3, distribution / (particles))
ax.set_yscale('log')
ax.set_xlabel(r'$\theta$ (mrad)')
ax.set_ylabel(r'$\frac{dU}{d\Omega}$ per particle (eV)')
fig.savefig('distribution.png', dpi=300)
plt.close(fig)

total_energy = 2 * np.pi * np.sum(distribution * thetas) * thetas_step
with open('result.txt', 'w+') as f:
    f.write('total energy = {:.5e} eV'.format(total_energy))
'''
"""
