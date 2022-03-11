import sys
import logging

from ca_prep import opts, util
from ca_prep.preprocessing import image_processing, signal_processing
from definitions import *

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-40s : %(levelname)-10s : %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

log = logging.getLogger(__name__)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        log.error('Please specify a processing command.')
        quit()

    if len(sys.argv) < 3:
        log.error('Processing requires a folder path.')
        quit()

    root_path = util.get_path(sys.argv[2])

    if root_path is None:
        log.error('Invalid root path')
        quit()

    # Load config
    # util.load_configuration('../config.yaml')

    # Get cmd options
    opts.OVERWRITE = any([alt in sys.argv for alt in OPT_OVERWRITE])
    opts.PLOT = any([alt in sys.argv for alt in OPT_PLOT])

    if CMD_EXCTRACT_ROIS in sys.argv:
        util.apply_to_all_recording_folders(root_path, image_processing.extract_rois)

    if CMD_PROCESS_SIGNALS in sys.argv:
        util.apply_to_all_recording_folders(root_path, signal_processing.process_signals)

    # if CMD_FILTER_ROIS in sys.argv[1:]:
    #     util.apply_to_all_recording_folders(root_path, signal_processing.filter_rois, max_dff_thresh=1.2)
    #     # event_detection.filter_rois(filepath, max_dff_thresh=1.2)
    #
    # if CMD_DETECT_EVENTS in sys.argv[1:]:
    #     recording_path = os.path.join(root_path, 'rec_2021-12-14-16-33-22')
    #     signal_processing.detect_events(recording_path)
    #     # util.apply_to_all_recording_folders(root_path, signal.detect_events)
    #
    # if 'extract_snippets' in sys.argv[1:]:
    #     extract_event_triggered_snippets(filepath, [2])
    #
    # if 'match_display_frames_to_imaging_frames' in sys.argv[1:]:
    #     match_display_frames_to_imaging_frames(filepath)
    #
    # if 'plot' in sys.argv[1:]:
    #     # Plot STA
    #     min_event_num = 4
    #     with h5py.File(filepath, 'r') as f:
    #         # Calculate coordinates and appropriate 2d dot sizes
    #         coords = f['vertex_coords'][:].squeeze()
    #         az, el, r = util.cart2sph(*coords.T)
    #         dot_sizes = 5 * (0.5 + np.abs(el))
    #
    #         # Plot phases
    #         stim_p2 = f['phase2']['ddp_vertex_states'][:]
    #         stim_p2_counts = stim_p2.sum(axis=0)
    #         # plt.figure(figsize=(17, 10))
    #         # plt.scatter(az, el, c=stim_p2.mean(axis=0), s=dot_sizes)
    #         # plt.show()
    #
    #         snippets_120 = f['snippets_120']
    #         for dataset_name in snippets_120:
    #             roi_snippets = snippets_120[dataset_name]
    #             roi_idx = int(dataset_name.split('_')[-1])
    #             if roi_snippets.shape[0] < min_event_num:
    #                 continue
    #             trace = f['selected_dff'][roi_idx]
    #             events = f['events'][roi_idx]
    #
    #             log.info(f'Plot ROI {roi_idx} with {events.sum()} events')
    #
    #             fig = plt.figure(figsize=(17, 10))
    #             gs = plt.GridSpec(2,4)
    #
    #             ax1 = plt.subplot(gs[0,:])
    #             ax1.plot(trace, color='black')
    #             ax1.plot(events, color='red')
    #
    #             ax2 = plt.subplot(gs[1,:])
    #
    #             sct = ax2.scatter(az, el, s=dot_sizes)
    #             ax2.set_ylim(-np.pi/2, np.pi/2)
    #             ax2.set_xlim(-np.pi, np.pi)
    #
    #             plt.show()
    #
    # if 'generate_map_videos' in sys.argv[1:]:
    #
    #     # Plot STA
    #     min_event_num = 4
    #     with h5py.File(filepath, 'r') as f:
    #         # Calculate coordinates and appropriate 2d dot sizes
    #         coords = f['vertex_coords'][:].squeeze()
    #         az, el, r = cart2sph(*coords.T)
    #         dot_sizes = 5 * (0.5 + np.abs(el))
    #
    #         # Plot phases
    #         stim_p2 = f['phase2']['ddp_vertex_states'][:]
    #         stim_p2_counts = stim_p2.sum(axis=0)
    #
    #         snippets_120 = f['snippets_120']
    #         for dataset_name in snippets_120:
    #             roi_snippets = snippets_120[dataset_name]
    #             roi_idx = int(dataset_name.split('_')[-1])
    #             if roi_snippets.shape[0] < min_event_num:
    #                 continue
    #             trace = f['selected_dff'][roi_idx]
    #             events = f['events'][roi_idx]
    #
    #             log.info(f'Plot ROI {roi_idx} with {events.sum()} events')
    #
    #             fig = plt.figure(figsize=(17, 10))
    #             gs = plt.GridSpec(2,4)
    #
    #             ax1 = plt.subplot(gs[0,:])
    #             ax1.plot(trace, color='black')
    #             ax1.plot(events, color='red')
    #
    #             ax2 = plt.subplot(gs[1,:])
    #             # snippet_trace = np.mean(roi_snippets[:], axis=0)
    #
    #             sct = ax2.scatter(az, el, s=dot_sizes)
    #             ax2.set_ylim(-np.pi/2, np.pi/2)
    #             ax2.set_xlim(-np.pi, np.pi)
    #
    #             # for snippet in snippet_trace:
    #             snippet = roi_snippets[:].mean(axis=0)


    #
    #
    # coords = f['vertex_coords'][:].squeeze()
    #
    #
    # plt.figure()
    # plt.imshow(f['stdImg'][:])
    # plt.imshow(f['ROImask'][:], alpha=0.1)
    # plt.show()
    #
    # plt.figure()
    # for i, (trace, events) in enumerate(zip(f['DFF'], f['events'])):
    #     plt.plot(i + trace, color='black')
    #     plt.plot(i + events / 2 , color='red', alpha=0.2)
    # plt.show()
    #
    #
    # plt.figure()
    # states = f['phase2/ddp_vertex_states'][0]
    # ax = plt.subplot(projection='3d')
    # ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=states, s=3., cmap='gray')
    # plt.show()