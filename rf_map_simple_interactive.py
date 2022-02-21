import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import matplotlib
matplotlib.use('QtAgg')


filepath = '../20220207_Texture_displacement/fish1_rec2_60um_02072022_laser19_gain_658_mag2_jf7_6dpf_2fps_zeissobj_sphere/Preprocessed_data.hdf5'
data = h5.File(filepath, 'r')

large_dots = False
phase = data['phase23']
raw_stimulus = np.asarray(phase['ddp_vertex_states']).astype(np.float16)
average_stimulus = raw_stimulus.mean(axis=0)
roi_traces: np.ndarray = phase['zscore'][:]

vertexCoords = np.array(phase['vertex_coords']).squeeze()

current_idx = 0
saved_presses = []

zscore_threshold: float = 2.
zscores_above_thresh_count: int = 500

zscore_above_threshold: np.ndarray
filtered_rois: np.ndarray
roi_idcs_above_threshold: np.ndarray


def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)
    elev = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                           # phi
    return az,elev


def apply_thresholds():
    global roi_idcs_above_threshold, roi_traces, zscore_above_threshold, zscore_threshold
    zscore_above_threshold = (roi_traces > zscore_threshold).astype(np.float32)
    roi_idcs_above_threshold = np.where(np.sum(zscore_above_threshold, axis=1) > zscores_above_thresh_count)[0]
    # zscore_above_threshold = zscore_above_threshold[np.sum(zscore_above_threshold, axis=1) > zscores_above_thresh_count, :]

    print(f'Applied thresholds. Z-Score: {zscore_threshold:.2f} Count above Z-Score threshold: {zscores_above_thresh_count} '
          f'/ Selected ROIs: {roi_idcs_above_threshold.shape[0]}')
    print(roi_idcs_above_threshold)

    if zscore_above_threshold.shape[0] == 0:
        print('WARNING: no ROIs passed filter criteria')


def on_press(event):
    global current_idx, saved_presses, zscore_threshold, zscores_above_thresh_count

    idx_was = current_idx
    new_thresholds = False
    sys.stdout.flush()
    if event.key == 'a':
        try:
            rel_indices = roi_idcs_above_threshold - current_idx
            current_idx = range(roi_traces.shape[0])[roi_idcs_above_threshold[np.where(rel_indices < 0)[0][-1]]]
        except:
            pass
    elif event.key == 'd':
        try:
            rel_indices = roi_idcs_above_threshold - current_idx
            current_idx = range(roi_traces.shape[0])[roi_idcs_above_threshold[np.where(rel_indices > 0)[0][0]]]
        except:
            pass
    elif event.key == 'enter':
        try:
            idx = int(''.join(saved_presses))
        except:
            pass
        else:
            current_idx = idx
        finally:
            saved_presses = []
    elif event.key in ['up', 'down', 'left', 'right']:
        if event.key == 'up':
            zscore_threshold += 0.1
        elif event.key == 'down':
            zscore_threshold -= 0.1
        elif event.key == 'right':
            zscores_above_thresh_count += 10
        elif event.key == 'left':
            zscores_above_thresh_count -= 10

        apply_thresholds()
        new_thresholds = True
    else:
        try:
            val = int(event.key)
            if val in range(10):
                saved_presses.append(event.key)
        except:
            pass

    if current_idx != idx_was or new_thresholds:
        plot(current_idx)


def plot(idx):
    print(f'Plot {idx}')

    # Frame data
    average_frame = zscore_above_threshold[idx, :-1].dot(raw_stimulus) / zscore_above_threshold[idx, :-1].sum() - average_stimulus
    ax1.clear()
    ax1.set_title(f'ROI {idx}')
    if large_dots:
        s = 700
    else:
        s = 200
    scatter = ax1.scatter(azi[elv < np.pi/4], elv[elv < np.pi/4], s, average_frame[elv < np.pi/4], alpha=0.9)
    ax1.set_ylim(-np.pi/2-0.15, np.pi/4+0.15)
    cbar.update_normal(scatter)

    # ROI trace
    ax2.clear()
    ax2.plot(roi_traces[idx])
    ax2.hlines(zscore_threshold, 0, roi_traces.shape[1], color='red', label=f'Z-Score threshold: {zscore_threshold:.2f}')
    ax2.legend()

    # Plot used frames on timeline in relation to zscore_thresh
    ax3.clear()
    ax3.plot(zscore_above_threshold[idx], label=f'Count above threshold: {int(zscore_above_threshold[idx].sum()):d}/{zscores_above_thresh_count}')
    ax3.legend()

    fig.canvas.draw()


if __name__ == '__main__':
    azi, elv = cart2sph(*vertexCoords.T)

    midline = 6

    fig = plt.figure(figsize=(18,4), num=filepath)
    gs = plt.GridSpec(2, 14)

    ax1 = fig.add_subplot(gs[:,:midline])
    scatter = ax1.scatter([], [])
    cax = fig.add_subplot(gs[:, midline])
    cbar = plt.colorbar(mappable=scatter, cax=cax)

    ax2 = fig.add_subplot(gs[0,midline+1:])

    ax3 = fig.add_subplot(gs[1,midline+1:])

    fig.canvas.mpl_connect('key_press_event', on_press)
    apply_thresholds()
    plot(current_idx)

    fig.tight_layout()
    plt.show()
