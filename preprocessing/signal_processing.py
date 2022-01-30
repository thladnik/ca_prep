import os.path

import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sklearn.mixture
from hmmlearn.hmm import GMMHMM

from definitions import *
import util

log = logging.getLogger(__name__)


def filter_rois(recording_folder: str, max_dff_thresh: float, **kwargs):

    # Fetch processing (output) file
    processing_filepath = util.get_processing_filepath(recording_folder)

    calculate_zscores(recording_folder)

    with h5py.File(processing_filepath, 'r') as f:

        times = f[FRAME_TIME][:]
        dff = f[DFF][:]
        zscore = f[ZSCORE][:]

        # Filter by DFF
        selected = dff.max(axis=1) > max_dff_thresh
        selected_dff = dff[selected]
        selected_zscore = zscore[selected]

        # Plot filtered/accepted ROIs
        fig, ax = plt.subplots(1, 2, figsize=(20, 12), sharex=True, sharey=True)
        dmult = 2
        zmult = 12
        count = dff.shape[0]
        for i, (s, d, z) in enumerate(zip(selected, dff, zscore)):
            if s:
                ax[0].plot(times, i + d / dmult, color='black', linewidth=.5)
                ax[1].plot(times, i + z / zmult, color='red', linewidth=.5)
            else:
                ax[0].plot(times, i + d / dmult, color='gray', linewidth=.5)
                ax[1].plot(times, i + z / zmult, color='orange', linewidth=.5)

        stim_seps = util.get_phase_start_points(f)

        ax[0].set_title('dF/mF')
        ax[0].set_ylabel('ROI')
        ax[0].set_xlabel('Time [s]')
        ax[0].vlines(times[stim_seps], 0, count, colors='orange')
        ax[0].set_yticks(np.arange(count, step=25))

        ax[1].set_title('Z-Score')
        ax[1].set_xlabel('Time [s]')
        ax[1].vlines(times[stim_seps], 0, count, colors='orange')
        ax[1].set_yticks(np.arange(count, step=25))
        ax[1].set_ylim(-1, count+2)
        fig.tight_layout()
        fig.savefig(os.path.join(recording_folder, 'roi_selection.svg'), format='svg')

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