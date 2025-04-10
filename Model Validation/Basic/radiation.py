import numpy as np
import scipy.interpolate
import scipy.integrate

c_light = 299792458.
hbar = 6.5821e-16
constant_1 = 0.01359438361 #sqrt(e^2 / (16 pi^3 e0 hbar c))

def compute_radiation(t, trajectories, energies, thetaxs, thetays):
    n_energies = len(energies)
    n_thetaxs = len(thetaxs)
    n_thetays = len(thetays)
    n_particles = trajectories.shape[0]
    n_steps = trajectories.shape[1]
    result = np.zeros(dtype=np.complex128, shape=(n_energies, n_thetaxs, n_thetays, 3))
    r = trajectories[:, :, :3]
    beta = trajectories[:, :, 3:] / c_light # beta: (n_particles, n_steps, 3)
    beta_dot = scipy.interpolate.make_interp_spline(t, beta, k=3, axis=1).derivative(nu=1)(t)
    for i, nx in enumerate(np.sin(thetaxs)):
        for j, ny in enumerate(np.sin(thetays)):
            nz = np.sqrt(1 - nx**2 - ny**2)
            n = np.array((nx, ny, nz)) # n: 3
            kappa2 = (1 - np.dot(beta, n)) ** -2 # kappa: (n_particles, n_steps)
            delta_t = t[np.newaxis, :] - np.dot(r, n)/c_light # particles, steps
            g_t = kappa2[:, :, np.newaxis] * np.cross(n, np.cross(n[np.newaxis, np.newaxis, :] - beta, beta_dot))
            # g_t: (n_particles, n_steps, 3)
            for k, energy in enumerate(energies):
                omega = energy / hbar
                phase = omega*delta_t
                h_t = g_t * np.exp(1j*phase)[:, :, np.newaxis]
                A = scipy.integrate.trapz(h_t, t, axis=1) # (n_particles, 3)
                result[k, i, j, :] = np.sum(A, axis=0)
    return constant_1 * result
