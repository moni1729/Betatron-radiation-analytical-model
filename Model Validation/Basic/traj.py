import numpy as np
import scipy.interpolate



def heatmap_interpolate(x_min, x_max, x_num, y_min, y_max, y_num, z, x_num_interp, y_num_interp, **kwargs):
    x, x_step = np.linspace(x_min, x_max, x_num, retstep=True, endpoint=True)
    y, y_step = np.linspace(y_min, y_max, y_num, retstep=True, endpoint=True)
    x_plot, x_step_plot = np.linspace(x_min, x_max, x_num_interp, retstep=True, endpoint=True)
    y_plot, y_step_plot = np.linspace(y_min, y_max, y_num_interp, retstep=True, endpoint=True)
    z_plot = scipy.interpolate.interp2d(x, y, z, kind='cubic')(x_plot, y_plot)
    x_midpoint_plot = np.linspace(x_min - 0.5 * x_step_plot, x_max + 0.5 * x_step_plot, x_num_interp + 1, endpoint=True)
    y_midpoint_plot = np.linspace(y_min - 0.5 * y_step_plot, y_max + 0.5 * y_step_plot, y_num_interp + 1, endpoint=True)
    plt.pcolormesh(x_midpoint_plot, y_midpoint_plot, z_plot, cmap='inferno', **kwargs)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    return z

z = heatmap(-angle_max_urad, angle_max_urad, plot_points, -angle_max_urad, angle_max_urad, plot_points, compute_vectorized)
plt.colorbar().set_label(r'$\frac{d^2 U}{d\Omega d\epsilon}$')
plt.xlabel(r'$\phi_x$ ($\mu$rad)')
plt.ylabel(r'$\phi_y$ ($\mu$rad)')
plt.savefig('pic.png', dpi=300)
plt.clf()
