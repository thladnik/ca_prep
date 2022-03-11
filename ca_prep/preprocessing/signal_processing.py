import os.path

import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.interpolate
import sklearn.mixture

from ca_prep import config, opts, util
from ca_prep.definitions import *

log = logging.getLogger(__name__)


def process_signals(recording_folder: str):

    # Calculate signals from extracted ROI mean fluorescence
    calculate_signals(recording_folder)

    # Add data to individual stimulation phases
    add_phase_data(recording_folder)

    # Plot ROI signals
    display_rois(recording_folder)


def calculate_signals(recording_folder: str):

    log.info(f'Process signals in {recording_folder}')

    with h5py.File(os.path.join(recording_folder, PATH_FN_PREPROCESSED), 'a') as f:

        # The start time is set to the first phase (anything prior to the start time is used as "baseline" signal
        phase1_start_time = f['phase1'].attrs['start_time']

        raw_signals = f[RAW_F][:]
        all_frame_times = f[FRAME_TIME][:]

        # Calculate DFF
        baseline_end_idx = np.argmin(np.abs(all_frame_times-phase1_start_time))
        fun = lambda d: (d - d[:baseline_end_idx].mean()) / d[:baseline_end_idx].mean()
        # dff = np.array([(i - np.mean(i[:baseline_end_idx])) / np.mean(i[:baseline_end_idx]) for i in raw_signals])
        dff = np.array([fun(s) for s in raw_signals])
        util.create_dataset(f, DFF, dff)
        # Calculate z-score
        zscore = scipy.stats.zscore(dff, axis=1)
        util.create_dataset(f, ZSCORE, zscore)


def add_phase_data(recording_folder: str):

    log.info('Attach additional data to stimulation phases and up-sample signals')

    with h5py.File(os.path.join(recording_folder, PATH_FN_PREPROCESSED), 'a') as f:

        all_frame_times = f[FRAME_TIME][:]

        # Go trough all phases and attach info
        for key, phase in f.items():
            # Skip datasets
            if 'phase' not in key:
                continue

            phase_display_times = np.squeeze(phase[f'{config.VAR_DISPLAY_PARAMS_PREFIX}{config.TIME_POSTFIX}'][:])
            # Set phase start index
            frame_start_idx = np.argmin(np.abs(phase_display_times[0] - all_frame_times))
            phase.attrs['ca_start_frame'] = frame_start_idx
            # Set phase end index
            frame_end_idx = np.argmin(np.abs(phase_display_times[-1] - all_frame_times))
            phase.attrs['ca_end_frame'] = frame_end_idx

            # Interpolate DFF
            dff_upsampled = np.array([scipy.interpolate.interp1d(all_frame_times, d, kind='nearest')(phase_display_times) for d in f[DFF]])
            util.create_dataset(phase, DFF, data=dff_upsampled)

            # Interpolate z-score
            zscore_upsampled = np.array([scipy.interpolate.interp1d(all_frame_times, z, kind='nearest')(phase_display_times) for z in f[ZSCORE]])
            util.create_dataset(phase, ZSCORE, data=zscore_upsampled)


def display_rois(recording_folder: str):

    with h5py.File(os.path.join(recording_folder, PATH_FN_PREPROCESSED), 'r') as f:
        dff = f[DFF][:]
        zscore = f[ZSCORE][:]
        times = f[FRAME_TIME][:]

        # Plot filtered/accepted ROIs
        fig_traces, ax_traces = plt.subplots(1, 2, figsize=(20, 12), sharex=True, sharey=True)
        fig_traces.suptitle(recording_folder)
        count = dff.shape[0]
        for i, (d, z) in enumerate(zip(dff, zscore)):
            ax_traces[0].plot(times, i + d, color='black', linewidth=.5)
            ax_traces[1].plot(times, i + z / 4, color='black', linewidth=.5)

        stim_seps = util.get_phase_start_points(f)

        # Plot traces
        ax_traces[0].set_title('dF/F')
        ax_traces[0].set_ylabel('ROI')
        ax_traces[0].set_xlabel('Time [s]')
        ax_traces[0].vlines(times[stim_seps], 0, count, colors='orange', linewidth=.5)
        ax_traces[0].set_yticks(np.arange(count, step=25))

        ax_traces[1].set_title('Z-Score')
        ax_traces[1].set_xlabel('Time [s]')
        ax_traces[1].vlines(times[stim_seps], 0, count, colors='orange', linewidth=.5)
        ax_traces[1].set_yticks(np.arange(count, step=25))
        ax_traces[1].set_ylim(-1, count+2)

        fig_traces.tight_layout()
        plt.savefig(os.path.join(recording_folder, 'ROI_traces.pdf'), format='pdf')

        # Plot map
        fig_map, ax_map = plt.subplots(1, 2, figsize=(20, 12), sharex=True, sharey=True)
        fig_map.suptitle(recording_folder)
        ax_map[0].imshow(dff)
        ax_map[0].set_aspect('auto')
        ax_map[0].set_title('dF/F')
        ax_map[0].set_ylabel('ROI')
        ax_map[0].set_xlabel('Time [s]')
        # ax_map[0].vlines(times[stim_seps], 0, count, colors='orange', linewidth=.5)
        ax_map[0].set_yticks(np.arange(count, step=25))

        ax_map[1].imshow(zscore)
        ax_map[1].set_aspect('auto')
        ax_map[1].set_title('Z-Score')
        ax_map[1].set_xlabel('Time [s]')
        # ax_map[1].vlines(times[stim_seps], 0, count, colors='orange', linewidth=.5)
        ax_map[1].set_yticks(np.arange(count, step=25))
        ax_map[1].set_ylim(-1, count+2)

        fig_map.tight_layout()
        plt.savefig(os.path.join(recording_folder, 'ROI_map.pdf'), format='pdf')

        if opts.PLOT:
            plt.show()


def calculate_zscores(recording_folder: str):

    # Fetch processing (output) file
    processing_filepath = util.get_processing_filepath(recording_folder)

    with h5py.File(processing_filepath, 'a') as f:
        if ZSCORE in f:
            del f[ZSCORE]
        f.create_dataset(ZSCORE, data=scipy.stats.zscore(f[DFF], axis=1))


def test_single(f, trace):
    stim_seps = util.get_phase_start_points(f)

    z = scipy.stats.zscore(trace)

    gh = sklearn.mixture.GaussianMixture(n_components=2, max_iter=300)
    gh.fit(trace[:, np.newaxis])
    s = gh.predict(trace[:, np.newaxis])
    c = np.corrcoef(trace, s)[0, 1]
    if c < 0.:
        c = -c
        s = np.logical_not(s)
    e = np.diff(s) > 0

    fig, ax = plt.subplots(figsize=(24, 2))
    ax.plot(trace, color='black', linewidth=1.)
    plt.plot(s, color='red', alpha=0.5, linewidth=1.)
    ax.vlines(np.where(e)[0], 0., 1., colors='red')
    ax.vlines(stim_seps, 0., 1., colors='orange')
    fig.tight_layout()
    plt.show()
