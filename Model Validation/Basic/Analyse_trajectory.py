import simulation
import numpy as np
import matplotlib.pyplot as plt
import h5py

#with h5py.File("trajectories.hdf5", "w") as f:

s = simulation.Simulation('output')

trajectories = np.empty(dtype=np.float64, shape=(s['actual_particles'], s['actual_analysis_points'], 6))

for particle in range(s['actual_particles']):

    x = s.phase_space[0, 0, particle] * s['plasma_skin_depth_si']
    y = s.phase_space[0, 2, particle] * s['plasma_skin_depth_si']
    z = s.z_si
    t = s.z_si / 299792458.
    vx = 299792458. * s.phase_space[0, 1, particle] / s.gamma
    vy = 299792458. * s.phase_space[0, 3, particle] / s.gamma
    vz = 299792458. * np.sqrt(1 - np.power(s.gamma, -2) * (1 + s.phase_space[0, 3, particle] ** 2 + s.phase_space[0, 1, particle] ** 2))
    
    trajectories[particle, :, 0] = x
    trajectories[particle, :, 1] = y
    trajectories[particle, :, 2] = z
    trajectories[particle, :, 3] = vx
    trajectories[particle, :, 4] = vy
    trajectories[particle, :, 5] = vz
    

import pickle
with open('results', 'wb+') as f:
    pickle.dump((t, trajectories), f)
        
        

plt.plot(trajectories[:, :, 3], trajectories[:, :, 4])
plt.show()



