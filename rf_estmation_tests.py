import h5py
import numpy as np
from matplotlib import pyplot as plt

from definitions import *


f = h5py.File('../2021-12-14_Pilot/fish1_rec4_neg45um_20211214_laser22_gain_656_mag2_jf7_5dpf_2fps_zeissobj_sphere/fish1_rec4_neg45um_20211214_laser22_gain_656_mag2_jf7_5dpf_2fps_zeissobj_sphere.output.hdf5', 'r')

coords = f['vertex_coords'][:].squeeze().T
# z = f[ZSCORE][117]
z = f[ZSCORE][206]

frames = []
weights = []
for p_id in [2]:

    phase = f[f'phase{p_id}']
    ca_start_frame = phase.attrs['ca_start_frame']
    ca_end_frame = phase.attrs['ca_end_frame']
    states = phase['ddp_vertex_states'][:].astype(np.float64)
    display_times = phase['ddp_time'][:]
    display_times -= display_times[0]

    subz = z[ca_start_frame:ca_end_frame]

    rel_signal_times = np.arange(0, 1, 1 / subz.shape[0]) * (display_times[-1] - display_times[0])

    for i, t in enumerate(rel_signal_times):
        idx = np.argmin(np.abs(t - display_times))
        frames.append(states[idx])
        weights.append(subz[i])

weights = np.array(weights)
frames = np.array(frames)

img = (frames * weights[:,np.newaxis]).sum(axis=0) / weights.sum()

# weight_thresh = 2
# img = (frames * weights[:,np.newaxis] > weight_thresh).sum(axis=0) / (weights > weight_thresh).sum()


ax = plt.subplot(projection='3d')
mappable = ax.scatter(*coords[:,coords[2,:] < 0.5], c=img[coords[2,:] < 0.5], cmap='seismic')
plt.colorbar(mappable=mappable, ax=ax)
plt.show()
