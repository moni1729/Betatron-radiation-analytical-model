import numpy as np
import subprocess
import tempfile

parameters_list = [
    'x0',
    'y0',
    'vx0',
    'vy0',
    'gamma_initial',
    'ion_atomic_number',
    'plasma_density',
    'rho_ion',
    'accelerating_field',
    'a0',
    'step_size',
    'steps'
]

def argument_format(argument):
    if isinstance(argument, int):
        return str(argument)
    elif isinstance(argument, float):
        return '{:.9e}'.format(argument)
    else:
        assert False

def comp_traj(**kwargs):
    with tempfile.NamedTemporaryFile() as f_particles:
        args = (
            './gen_traj',
            f_particles.name,
            *[argument_format(kwargs[i]) for i in parameters_list],
        )
        print(' '.join(args))
        subprocess.run(args, check=True)
        return np.fromfile(f_particles).reshape(kwargs['steps'] + 1, 9)

def comp_rad(energies, thetas, phis, trajectories, steps, step_size):
    with tempfile.NamedTemporaryFile() as f_particles:
        with tempfile.NamedTemporaryFile() as f_energies:
            with tempfile.NamedTemporaryFile() as f_thetas:
                with tempfile.NamedTemporaryFile() as f_phis:
                    with tempfile.NamedTemporaryFile() as f_result:
                        trajectories.tofile(f_particles)
                        energies.tofile(f_energies)
                        thetas.tofile(f_thetas)
                        phis.tofile(f_phis)
                        args = (
                            './comp_rad',
                            f_result.name,
                            f_particles.name,
                            f_energies.name,
                            f_thetas.name,
                            f_phis.name,
                            '1',
                            argument_format(steps),
                            argument_format(len(energies)),
                            argument_format(len(thetas)),
                            argument_format(len(phis)),
                            argument_format(step_size)
                        )
                        print(' '.join(args))
                        subprocess.run(args, check=True)
                        result = np.fromfile(f_result.name).reshape((len(energies), len(thetas), len(phis), 6))
                        return result
