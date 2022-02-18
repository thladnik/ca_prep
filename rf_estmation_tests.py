import os
from typing import Tuple

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
from scipy import signal

from definitions import *
import util


def calculate_rf(phase: h5py.Group, roi_id: int):

    # Fetch data
    z = phase[ZSCORE][roi_id]
    dff = phase[DFF][roi_id]
    coords = np.squeeze(phase['vertex_coords'][:]).T
    frames = phase['ddp_vertex_states'][:]
    if z.shape[0] > frames.shape[0]:
        z = z[:frames.shape[0]]

    # Calculate map
    img = (frames * z[:, np.newaxis]).sum(axis=0) / z.sum()

    # Plot results
    fig = plt.figure(figsize=(20, 10))
    gs = plt.GridSpec(5, 2)

    # Plot signal
    ax_signal = plt.subplot(gs[0, :])
    ax_signal.plot(dff / (dff.max() - dff.min()) - 1, color='black', linewidth=1., label='dF/F')
    ax_signal.plot(z / (z.max() - z.min()), color='red', linewidth=1., label='Z-score')
    ax_signal.legend()
    ax_signal.set_xlabel('Frame')
    ax_signal.set_ylabel('Rel. signal')

    # Plot map
    upper_el = np.pi / 4
    az, el, r = util.cart2sph(*coords)
    subimg = img[el < upper_el]
    subaz = az[el < upper_el]
    subel = el[el < upper_el]

    # Interpolate
    xi = np.linspace(subaz.min(), subaz.max(), 100)
    yi = np.linspace(subel.min(), subel.max(), 100)

    triang = tri.Triangulation(subaz, subel)
    interpolator = tri.LinearTriInterpolator(triang, subimg)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # Plot
    ax_map = plt.subplot(gs[1:, :])
    levels = 20
    ax_map.contour(xi, yi, zi, levels=levels, linewidths=0.5, colors='k')
    cntr1 = ax_map.contourf(xi, yi, zi, levels=levels, cmap="seismic")

    mappable = ax_map.scatter(subaz, subel, c=subimg, s=10. + 10 * np.abs(subel), alpha=1., cmap='seismic')
    plt.colorbar(mappable=cntr1, ax=ax_map)
    ax_map.set_xlabel('Azimuth [rad]')
    ax_map.set_ylabel('Elevation [rad]')

    fig.tight_layout()
    plt.show()


def plot_map_overview(look_at: Tuple[int, int], use_phases=(2, 4)):
    use_phases = (22, )
    # look_at = (3, 410)

    print(look_at)

    base_path = '../20220207_Texture_displacement'
    # base_path = '../2021-12-14_Pilot'

    folders = [fn for fn in os.listdir(base_path) if f'rec{look_at[0]}' in fn]
    if not bool(folders):
        print('Nope')
        quit()
    folder_name = folders[0]
    recording_folder = os.path.join(base_path, folder_name)

    f = h5py.File(f'{recording_folder}/Preprocessed_data.hdf5', 'r')

    z = f[ZSCORE][look_at[1]]
    dff = f[DFF][look_at[1]]

    frames = []
    weights = []
    phase_intervals = []
    for p_id in use_phases:

        phase = f[f'phase{p_id}']
        coords = phase['vertex_coords'][:].squeeze().T
        ca_start_frame = phase.attrs['ca_start_frame']
        ca_end_frame = phase.attrs['ca_end_frame']
        states = phase['ddp_vertex_states'][:].astype(np.float64)
        display_times = phase['ddp_time'][:]
        display_times -= display_times[0]
        phase_intervals.append((ca_start_frame, ca_end_frame))

        subz = z[ca_start_frame:ca_end_frame]

        # use_deconv = False
        # if use_deconv:
        #     dt = np.mean(np.diff(f['frame_time'][:]))
        #     _t = np.arange(10, step=dt)
        #     tau = 1.6
        #     kernel = np.exp(-_t/tau)
        #
        #     subz, _ = signal.deconvolve(subz, kernel)

        rel_signal_times = np.arange(0, 1, 1 / subz.shape[0]) * (display_times[-1] - display_times[0])

        for i, t in enumerate(rel_signal_times):
            idx = np.argmin(np.abs(t - display_times))
            frames.append(states[idx])
            weights.append(subz[i])

    weights = np.array(weights)
    frames = np.array(frames)

    # Do permutations
    do_permutation_test = True
    if do_permutation_test:
        np.random.seed(1)
        weight_above_thresh = weights > 2
        n = 1000
        mean_frames = np.zeros((n, frames.shape[1]))
        for i in range(n):
            random_frames = frames[np.random.randint(frames.shape[0], size=(weight_above_thresh.sum(),))]
            mean_frames[i] = random_frames.mean(axis=0)

        distr_mean = mean_frames.mean(axis=0)
        distr_std = mean_frames.std(axis=0)

        # img = (frames[weight_above_thresh]).mean(axis=0) > (distr_mean + 2 * distr_std)
        img = (frames[weight_above_thresh]).mean(axis=0) > np.percentile(mean_frames, 95, axis=0)
    else:
        img = (frames * weights[:,np.newaxis]).sum(axis=0) / weights.sum()

    # Plot results

    fig_name = f'Recording {look_at[0]} | ROI {look_at[1]}'
    fig = plt.figure(num=fig_name, figsize=(20, 10))

    gs = plt.GridSpec(5, 2)

    # Plot signal
    ax_signal = plt.subplot(gs[0, :])
    ax_signal.plot(dff / (dff.max() - dff.min()) - 1, color='black', linewidth=1., label='dF/F')
    ax_signal.plot(z / (z.max() - z.min()), color='red', linewidth=1., label='Z-score')
    sparsity = ['10%', '20%']
    for i, interval in enumerate(phase_intervals):
        ax_signal.fill_between(interval, -1, 1, alpha=0.5, color='green', label=f'ON phase {sparsity[i]} sparsity')
    ax_signal.legend()
    ax_signal.set_xlabel('Frame')
    ax_signal.set_ylabel('Rel. signal')
    ax_signal.set_title(fig_name)

    # Plot map
    upper_el = np.pi / 4
    az, el, r = util.cart2sph(*coords)
    subimg = img[el < upper_el]
    subaz = az[el < upper_el]
    subel = el[el < upper_el]

    # Interpolate
    xi = np.linspace(subaz.min(), subaz.max(), 100)
    yi = np.linspace(subel.min(), subel.max(), 100)

    triang = tri.Triangulation(subaz, subel)
    interpolator = tri.LinearTriInterpolator(triang, subimg)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # Plot
    ax_map = plt.subplot(gs[1:, :])
    levels = 20
    ax_map.contour(xi, yi, zi, levels=levels, linewidths=0.5, colors='k')
    cntr1 = ax_map.contourf(xi, yi, zi, levels=levels, cmap="seismic")

    mappable = ax_map.scatter(subaz, subel, c=subimg, s=10. + 10 * np.abs(subel), alpha=1., cmap='seismic')
    plt.colorbar(mappable=cntr1, ax=ax_map)
    ax_map.set_xlabel('Azimuth [rad]')
    ax_map.set_ylabel('Elevation [rad]')

    fig.tight_layout()
    save_name = f'roi_map_{look_at[1]}'
    if do_permutation_test:
        save_name += '_permutation_test'
    print(f'Save to {save_name}')
    plt.savefig(os.path.join(recording_folder, f'{save_name}.pdf'), format='pdf')
    # plt.show()


if __name__ == '__main__':

    # (recording, roi)
    # interesting = [(5, 177),
    #                (5, 176),
    #                (5, 116),
    #                (5, 73),
    #                (4, 184),
    #                (4, 206),
    #                (4, 117),
    #                (3, 410),
    #                (3, 376),
    #                (3, 378),
    #                (3, 267),
    #                (2, 396),
    #                (2, 451),
    #                (2, 444),
    #                (2, 403),
    #                (2, 398),
    #                (2, 384),
    #                (2, 284),]
    #
    # for poi in interesting:
    #     plot_map_overview(poi)
    for poi in [(1, i) for i in range(300)]:
        plot_map_overview(poi)
