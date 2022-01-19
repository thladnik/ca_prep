# preproc_file = '..', '2021-12-14_Pilot', 'preprocessed', 'fish1_rec2_neg15um_20211214_laser22_gain_656_mag2_jf7_5dpf_2fps_zeissobj_sphere.hdf5'
# filepath = os.path.join(*preproc_file)
#
# with h5py.File(filepath, 'r') as f:
#     coords = f['vertex_coords'][:].squeeze()
#     az, el, _ = cart2sph(*coords.T)
#     t = f['frame_time'][:]
#     dff = f['selected_dff'][:]
#
#     p2 = f['phase2']
#     p2_t = p2['dff_t'][:]
#     p2_f = p2['dff_matched_display_frames'][:]
#     p4 = f['phase4']
#     p4_t = p4['dff_t'][:]
#     p4_f = p4['dff_matched_display_frames'][:]
#
#     in_p2_bool = np.logical_and(p2_t[0] < t, t < p2_t[-1])
#     in_p4_bool = np.logical_and(p4_t[0] < t, t < p4_t[-1])
#
#     for i, roi in enumerate(dff):
#         gs = plt.GridSpec(10, 10)
#         fig = plt.figure(figsize=(20, 8))
#         ax_roi = fig.add_subplot(gs[:3, :])
#         labels, events = fit_3component_gmm(t, roi, ax=ax_roi)
#
#         p2_events = events[in_p2_bool[1:]]
#         # p2_events = labels[in_p2_bool] == 2
#         p2_subframes = p2_f[:p2_events.shape[0]][p2_events]
#         p4_events = events[in_p4_bool[1:]]
#         # p4_events = labels[in_p4_bool] == 2
#         p4_subframes = p4_f[:p4_events.shape[0]][p4_events]
#         mean_img = np.mean(np.append(p2_subframes, p4_subframes, axis=0), axis=0)
#         # p4_events = events[in_p2_bool]
#
#         el_edge = - np.pi / 8
#         ax_uv = fig.add_subplot(gs[3:, :7])
#         sub_el_uv = np.logical_and(el < np.pi / 4, el > el_edge)
#         p = ax_uv.scatter(az[sub_el_uv], el[sub_el_uv], c=mean_img[sub_el_uv], s=10 + 5 * np.abs(el[sub_el_uv]), cmap='seismic')
#
#         ax_polar = fig.add_subplot(gs[3:, 7:], polar=True)
#         # ax_polar.set_aspect('equal')
#         sub_el_polar = el < el_edge
#         # ax_polar.scatter(el[sub_el_polar] * np.cos(az[sub_el_polar]), el[sub_el_polar] * np.sin(az[sub_el_polar]))
#         ax_polar.scatter(az[sub_el_polar], el[sub_el_polar], c=mean_img[sub_el_polar], s=10 + 5 * (np.pi/4 + el[sub_el_polar]), cmap='seismic')
#
#
#         fig.tight_layout()
#
#         # plt.savefig(os.path.join('..', 'figures', ''))
#
#         plt.show()
import logging
from typing import Union

import h5py
import numpy as np
from matplotlib import pyplot as plt

import util

log = logging.getLogger(__name__)

def extract_event_triggered_snippets(filepath: str, valid_phase_ids: Union[list, tuple], snippet_range: int = 120, ):
    with h5py.File(filepath, 'r') as f:
        log.info(f'Extract snippets for {filepath}')
        coords = f['vertex_coords'][:].squeeze()
        ca_events = f['events'][:]

        all_snippets = []
        for roi_id, roi_events in enumerate(ca_events):
            roi_event_count = roi_events.sum()
            log.info(f'{roi_event_count} events for ROI {roi_id}')

            total_snippet_count = 0
            snippets = np.zeros((roi_event_count, snippet_range, coords.shape[0]), dtype=bool)
            for phase_id in valid_phase_ids:
                phase = f'phase{phase_id}'
                phase_stimulus = f[f'{phase}/ddp_vertex_states'][:]
                ca_start_idx = f[phase].attrs['ca_start_frame']
                ca_end_idx = f[phase].attrs['ca_end_frame']
                phase_events = roi_events[ca_start_idx:ca_end_idx]

                event_idcs = np.where(phase_events)[0]
                if event_idcs.shape[0] == 0:
                    log.info(f'No events for ROI {roi_id} in {phase}')
                    continue
                log.info(f'{event_idcs.shape[0]} events for ROI {roi_id} in {phase}')

                for event_idx in event_idcs:
                    if event_idx < snippet_range:
                        continue

                    snippets[total_snippet_count, :, :] = phase_stimulus[(event_idx - snippet_range):event_idx, :]

                    total_snippet_count += 1

            log.info(f'{total_snippet_count} snippets in total')
            all_snippets.append(snippets[:total_snippet_count])

    log.info('Write snippets to file')
    with h5py.File(filepath, 'a') as f:
        grp_name = f'snippets_{snippet_range}'
        if grp_name in f:
            del f[grp_name]

        grp = f.create_group(grp_name)
        for roi_id, snippets in enumerate(all_snippets):
            grp.create_dataset(f'roi_{roi_id}', data=snippets)


def match_display_frames_to_imaging_frames(filepath):
    log.info(f'Extract frame transitions')
    with h5py.File(filepath, 'a') as f:

        for key in f.keys():
            if 'phase' not in key or 'ddp_vertex_states' not in f[key]:
                log.info(f'Skip {key}')
                continue

            grp = f[key]

            log.info(f'Calculate transitions for {key}')

            display_frames = grp['ddp_vertex_states'][:]
            display_times = grp.attrs['start_time'] + grp['ddp_binary_noise_u_time'][:].squeeze()
            transitions = np.sum(np.abs(np.diff(display_frames, axis=0)), axis=1) > 0
            transition_times = display_times[1:][transitions]

            log.info(f'Write transitions for {key}')

            if 'frame_transitions' in grp:
                del grp['frame_transitions']
            grp['frame_transitions'] = transitions

            if 'frame_transition_times' in grp:
                del grp['frame_transition_times']
            grp['frame_transition_times'] = transition_times

            log.info(f'Match frames for {key}')
            dff = f['selected_dff'][:]
            imaging_times = f['frame_time'][:]
            sub = np.logical_and(imaging_times >= display_times[0], imaging_times < display_times[-1])
            imaging_times = imaging_times[sub]
            dff = dff[:,sub]

            d_f = np.zeros((imaging_times.shape[0], display_frames.shape[1]))
            for i, t in enumerate(imaging_times):
                closest_idx = np.argmin(np.abs(display_times-t))
                d_f[i] = display_frames[closest_idx]

            log.info(f'Write subset data for {key}')

            if 'dff' in grp:
                del grp['dff']
            grp['dff'] = dff

            if 'dff_t' in grp:
                del grp['dff_t']
            grp['dff_t'] = imaging_times

            if 'dff_matched_display_frames' in grp:
                del grp['dff_matched_display_frames']
            grp['dff_matched_display_frames'] = d_f


def calculate_signal_weighted_frames(filepath):
    with h5py.File(filepath, 'r') as f:
        coords = f['vertex_coords'][:].squeeze()
        az, el, _ = util.cart2sph(*coords.T)
        p2 = f['phase2']
        dff = p2['dff'][:]
        t = p2['dff_t'][:]

        for roi_id, roi in enumerate(dff):
            name = f'signal_weighted_responses_roi{roi_id}'

            roin = (roi - roi.min()) / (roi.max() - roi.min())
            frames = p2['dff_matched_display_frames'][:]

            w_frames = np.zeros(frames.shape)
            for i, w in enumerate(roin):
                w_frames[i] = frames[i] * w

            mean_img = np.mean(w_frames, axis=0)

            gs = plt.GridSpec(15,15)
            fig = plt.figure(figsize=(18, 10), num=name)

            ax_dff = fig.add_subplot(gs[:5,:])
            ax_dff.set_title(name)
            ax_dff.plot(t-t[0], roi)
            ax_dff.plot(t-t[0], roin)

            ax_scatter = fig.add_subplot(gs[5:,:])
            p = ax_scatter.scatter(az, el, c=mean_img, s=2 + 5 * np.abs(el), cmap='seismic')
            fig.colorbar(p)
            fig.tight_layout()
            log.info(f'Save figure for roi{roi_id}')
            # plt.show()
            fig.savefig(os.path.join('', '2021-12-14_Pilot', 'figures', f'{name}.png'), format='png')
            plt.close()


def fit_3component_gmm(t, roi, ax=None, yoffset=0.):

    # plt.figure(figsize=(22, 4))
    # Normalize
    roin = (roi - roi.min()) / (roi.max() - roi.min())

    if ax is not None:
        ax.plot(t, yoffset+roin, linestyle='-', linewidth=1.)

    # Get labels
    np.random.seed(1)
    gmm = mixture.GaussianMixture(n_components=3)
    gmm.fit(roin[:, np.newaxis])
    labels = gmm.predict(roin[:, np.newaxis])

    # Change to negative range [-1, -3]
    labels = -labels - 1

    # Calculate cluster means and sort by mean val
    cluster_means = sorted([(np.mean(roin[labels == -1]), -1),
                            (np.mean(roin[labels == -2]), -2),
                            (np.mean(roin[labels == -3]), -3)])
    # Re-assign (positive integer) labels based on mean val
    for new_id, (m, old_id) in enumerate(cluster_means):
        labels[labels == old_id] = new_id

    if ax is not None:
        ax.scatter(t, yoffset+roin, c=labels, s=5)

    # MERGE bottom two clusters
    labels[labels == 0] = 1

    # Cluster means
    if ax is not None:
        ax.hlines([yoffset+m for m, _ in cluster_means], t[0], t[-1], color='black', alpha=.4)

    # Get events
    events = np.diff(labels) > 0.
    if ax is not None:
        ax.vlines(t[1:][events], yoffset+roin[1:][events], yoffset+np.ones(events.sum()), color='red', alpha=.4, linewidth=1.)

    return labels, events