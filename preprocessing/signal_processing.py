import os.path

import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.interpolate
import sklearn.mixture

from definitions import *
import util

log = logging.getLogger(__name__)


def process_signals(recording_folder: str):

    # Calculate signals from extracted ROI mean fluorescence
    calculate_signals(recording_folder)

    # Add data to individual stimulation phases
    add_phase_data(recording_folder)


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

            phase_display_times = np.squeeze(phase['ddp_time'][:])
            # Set phase start index
            frame_start_idx = np.argmin(np.abs(phase_display_times[0] - all_frame_times))
            phase.attrs['ca_start_frame'] = frame_start_idx
            # Set phase end index
            frame_end_idx = np.argmin(np.abs(phase_display_times[-1] - all_frame_times))
            phase.attrs['ca_end_frame'] = frame_end_idx

            # Interpolate DFF
            dff_upsampled = np.array([scipy.interpolate.interp1d(all_frame_times, d)(phase_display_times) for d in f[DFF]])
            util.create_dataset(phase, DFF, data=dff_upsampled)

            # Interpolate z-score
            zscore_upsampled = np.array([scipy.interpolate.interp1d(all_frame_times, z)(phase_display_times) for z in f[ZSCORE]])
            util.create_dataset(phase, ZSCORE, data=zscore_upsampled)


def filter_rois(recording_folder: str, max_dff_thresh: float, **kwargs):

    # Fetch processing (output) file
    processing_filepath = util.get_processing_filepath(recording_folder)

    calculate_zscores(recording_folder)

    with h5py.File(processing_filepath, 'r') as f:

        times = f[FRAME_TIME][:]
        dff = f[DFF][:]
        zscore = f[ZSCORE][:]

        # fig, ax = plt.subplots(1, 1, num='DFF histogram')
        # counts, bins, _ = ax.hist(dff.flatten(), bins=1000)
        #
        # bin_centers = bins[:-1] + (bins[1] - bins[0]) / 2
        #
        # import scipy.optimize
        # params, cov = scipy.optimize.curve_fit(lambda x, loc, scale, ymax: ymax * scipy.stats.norm.pdf(x, loc=loc, scale=scale), bin_centers, counts)
        # norm = params[2] * scipy.stats.norm.pdf(bin_centers, loc=params[0], scale=params[1])
        # ax.plot(bin_centers, norm / norm.max() * counts.max(), color='red')
        #
        # if kwargs[ARG_PLOT]:
        #     plt.show()

        # Filter by DFF
        max_dff_thresh = dff.mean() + 4 * dff.std()
        selected = dff.max(axis=1) > max_dff_thresh
        selected_dff = dff[selected]
        selected_zscore = zscore[selected]

        # Plot filtered/accepted ROIs
        fig, ax = plt.subplots(1, 2, figsize=(20, 12), sharex=True, sharey=True, num=recording_folder)
        dmult = 2
        zmult = 12
        count = dff.shape[0]
        for i, (s, d, z) in enumerate(zip(selected, dff, zscore)):
            if s:
                ax[0].plot(times, i + d / dmult, color='black', linewidth=.5)
                ax[1].plot(times, i + z / zmult, color='black', linewidth=.5)
            else:
                ax[0].plot(times, i + d / dmult, color='gray', linewidth=.5)
                ax[1].plot(times, i + z / zmult, color='gray', linewidth=.5)

        stim_seps = util.get_phase_start_points(f)

        ax[0].set_title('dF/F')
        ax[0].set_ylabel('ROI')
        ax[0].set_xlabel('Time [s]')
        ax[0].vlines(times[stim_seps], 0, count, colors='orange', linewidth=.5)
        ax[0].set_yticks(np.arange(count, step=25))

        ax[1].set_title('Z-Score')
        ax[1].set_xlabel('Time [s]')
        ax[1].vlines(times[stim_seps], 0, count, colors='orange', linewidth=.5)
        ax[1].set_yticks(np.arange(count, step=25))
        ax[1].set_ylim(-1, count+2)
        fig.tight_layout()
        fig.savefig(os.path.join(recording_folder, 'roi_selection.svg'), format='svg')
        fig.savefig(os.path.join(recording_folder, 'roi_selection.pdf'), format='pdf')

        if kwargs[ARG_PLOT]:
            plt.show()

    # Write to file
    log.info('Write results to file')
    with h5py.File(processing_filepath, 'a') as f:
        if SELECTED_DFF in f:
            del f[SELECTED_DFF]
        f.create_dataset(SELECTED_DFF, data=selected_dff)

        if SELECTED_ZSCORE in f:
            del f[SELECTED_ZSCORE]
        f.create_dataset(SELECTED_ZSCORE, data=selected_zscore)


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


def detect_events(recording_folder):

    # Fetch processing (output) file
    processing_filepath = util.get_processing_filepath(recording_folder)

    with h5py.File(processing_filepath, 'r') as f:

        if 'selected_dff' not in f:
            filter_rois(processing_filepath, 1.2)


        # Fit
        log.info(f'Fit GMMHMM for {processing_filepath}')
        dff = f['selected_dff']
        # Fit to ALL DFFs
        # dffs = np.concatenate([d for d in dff])[:, np.newaxis]
        # lengths = [len(d) for d in dff]
        # gh.fit(dffs, lengths)
        test_single(f, dff[120])

        log.info(f'Predict GMMHMM output for {processing_filepath}')
        states = np.zeros(dff.shape, dtype=bool)
        events = np.zeros((dff.shape[0], dff.shape[1] - 1), dtype=bool)
        corrs = np.zeros(dff.shape[0])
        for i, trace in enumerate(dff):
            log.info(f'DFF {i}')
            gh = GMMHMM(n_components=2, n_mix=2, n_iter=300)
            gh.fit(trace[:, np.newaxis])
            s = gh.predict(trace[:, np.newaxis])
            c = np.corrcoef(trace, s)[0, 1]
            if c < 0.:
                c = -c
                s = np.logical_not(s)
            corrs[i] = c
            states[i] = s
            events[i] = np.diff(s) > 0

        stim_seps = util.get_phase_start_points(f)
        log.info('Plot results')
        fig, ax = plt.subplots(figsize=(6,12))
        lw = 1.
        for i, (trace, s, e) in enumerate(zip(dff, states, events)):
            ax.plot(i + trace, color='black', linewidth=lw)
            # plt.plot(i + s / 2, color='red', alpha=0.5, linewidth=lw)
            ax.vlines(np.where(e)[0], i, i + 0.5, colors='blue')
        ax.vlines(stim_seps, 0, dff.shape[0], colors='orange')
        fig.tight_layout()
        plt.show()

    # Write to file
    log.info('Write results to file')
    with h5py.File(processing_filepath, 'a') as f:
        if 'states' in f:
            del f['states']
        f.create_dataset('states', data=states)

        if 'events' in f:
            del f['events']
        f.create_dataset('events', data=events)