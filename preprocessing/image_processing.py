"""Image processing functions for

Original image segmentation by Yue Zhang"""

import copy
import cv2
import h5py as h5
import logging

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage as ndi
from scipy.signal import find_peaks
import shutil
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border, watershed
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_local
from skimage import io
from skimage.registration import phase_cross_correlation

import config
import opts
import util
from definitions import *

log = logging.getLogger(__name__)


def extract_rois(recording_folder: str):
    log.info(f'Process image data in {recording_folder}')

    ca_filename = util.get_tif_filename(recording_folder)
    if ca_filename is None:
        log.warning(f'Skip processing. No Ca file found in folder {recording_folder}')
        return False
    ca_filepath = os.path.join(recording_folder, ca_filename)

    log.info(f'Using Ca file {ca_filepath}')

    # Check paths
    disp_filepath = os.path.join(recording_folder, config.DISPLAY_FILENAME)
    if not os.path.exists(disp_filepath):
        log.warning(f'{config.DISPLAY_FILENAME} not found')
        return False

    io_filepath = os.path.join(recording_folder, config.IO_FILENAME)
    if not os.path.exists(io_filepath):
        log.warning(f'{config.IO_FILENAME} not found')
        return False

    out_filepath = os.path.join(recording_folder, PATH_FN_PREPROCESSED)
    if os.path.exists(out_filepath) and not opts.OVERWRITE:
        log.warning(f'Skip {recording_folder}. Output file already exists. '
                    f'Use overwrite option "{OPT_OVERWRITE}"')
        return False

    log.info(f'Copy file {disp_filepath} to {out_filepath} for final output')
    shutil.copyfile(disp_filepath, out_filepath)

    log.info('Run registration')
    # Do registration and segmentation
    ca_data = CalciumImagingData(ca_filepath, recording_folder)
    ca_data.registration()
    ca_frame_indices, ca_frame_times = get_ca_frame_timing(recording_folder)
    log.info('Run segmentation')
    rois = RoiSelector(ca_data.std_image)
    raw_calcium_signals = ca_data.extract_calcium_signals(rois.mask_labels)

    log.info(f'Write to output file at {out_filepath}')
    with h5.File(out_filepath, 'a') as out_file:
        outputs = {REGISTERED_FRAMES: ca_data.registered_frames,
                   STD_IMAGE: ca_data.std_image,
                   RAW_F: raw_calcium_signals,
                   ROI_MASK: rois.mask_labels,
                   FRAME_TIME: ca_frame_times}

        for dataset_name, data in outputs.items():
            util.create_dataset(out_file, dataset_name, data)


class CalciumImagingData:
    def __init__(self, cafn, recording_folder, **kwargs):
        self.raw_frames = np.array(io.imread(cafn))
        self.recording_folder = recording_folder

        self.binsize = kwargs.get('binsize')
        if self.binsize is None:
            self.binsize = 500

        self.stepsize = kwargs.get('stepsize')
        if self.stepsize is None:
            self.stepsize = 250

        self.std_image = None
        self.registered_frames = None

    def registration(self, binsize=None, stepsize=None):
        if binsize is not None:
            self.binsize = binsize
        if stepsize is not None:
            self.stepsize = stepsize
        image_template = np.mean(self.raw_frames[:self.binsize], axis=0)
        i = self.stepsize
        regTifImg = copy.copy(self.raw_frames)
        old_drift = None
        while i < self.raw_frames.shape[0]:
            moving_template = np.mean(self.raw_frames[i:i + self.binsize], axis=0)
            image_drift = phase_cross_correlation(image_template, moving_template)[0]
            if old_drift is None:
                old_drift = image_drift
            for o in range(min(self.stepsize, self.raw_frames.shape[0] - i)):
                itershift = np.vstack(
                    [np.eye(2), image_drift[::-1] * (1 - o / self.stepsize) + old_drift[::-1] * o / self.stepsize]).T
                regTifImg[i + o] = cv2.warpAffine(self.raw_frames[i + o], itershift, tuple(image_template.shape))
            old_drift = image_drift
            i += self.stepsize
        self.registered_frames = regTifImg
        self.std_image = np.std(self.registered_frames, axis=0)

        # Plot result
        fig_name = 'Registration_STD_image'
        fig, ax = plt.subplots(1, 2, figsize=(16,8), num=fig_name)
        ax[0].set_title('Raw STD image')
        ax[0].imshow(np.std(self.raw_frames, axis=0))
        ax[1].set_title('Registered STD image')
        ax[1].imshow(self.std_image)
        fig.tight_layout()
        plt.savefig(os.path.join(self.recording_folder, f'{fig_name}.png'), format='png')

        if opts.PLOT:
            plt.show()

    def extract_calcium_signals(self, roi_masks: np.ndarray) -> np.ndarray:
        raw_calcium_signals = []
        for i in np.unique(roi_masks):
            if i == 0:
                continue

            raw_calcium_signals.append(self.registered_frames[:, roi_masks == i].sum(axis=1))
        log.info("ROI trace extraction: Done")

        return np.array(raw_calcium_signals)


def get_ca_frame_timing(recording_path):
    with h5.File(os.path.join(recording_path, config.IO_FILENAME), 'r') as mirInfo:
        mirror_position = np.squeeze(mirInfo[config.Y_MIRROR_SIGNAL])
        mirror_time = np.squeeze(mirInfo[f'{config.Y_MIRROR_SIGNAL}{config.TIME_POSTFIX}'])

    peak_prominence = (mirror_position.max() - mirror_position.min()) / 4
    peak_idcs, _ = find_peaks(mirror_position, prominence=peak_prominence)
    trough_idcs, _ = find_peaks(-mirror_position, prominence=peak_prominence)

    # Find first trough
    first_peak = peak_idcs[0]
    first_trough = trough_idcs[trough_idcs < first_peak][-1]
    # Discard all before first trough
    trough_idcs = trough_idcs[first_trough <= trough_idcs]

    # Use midpoint between troughs and peaks as frame index
    if trough_idcs.shape[0] == peak_idcs.shape[0]:
        trough_to_peak_frame = (peak_idcs + trough_idcs) // 2
        peak_to_trough_frames = (peak_idcs[:-1] + trough_idcs[1:]) // 2
    else:
        # Otherwise (trough_idcs.shape[0] > peak_idcs.shape[0]) has to be true, because trough always comes first
        trough_to_peak_frame = (peak_idcs + trough_idcs[:-1]) // 2
        peak_to_trough_frames = (peak_idcs + trough_idcs[1:]) // 2
    frame_idcs = np.sort(np.concatenate([trough_to_peak_frame, peak_to_trough_frames]))

    # Get corresponding times
    frame_times = mirror_time[frame_idcs]

    # Plot frame time detection results
    fig_name = 'Y_mirror_frame_detection'
    fig, ax = plt.subplots(1, 3, figsize=(18, 4), num=fig_name)
    markersize = 3.
    for a in ax:
        a.plot(mirror_time, mirror_position)
        a.plot(mirror_time[peak_idcs], mirror_position[peak_idcs], 'o', markersize=markersize)
        a.plot(mirror_time[trough_idcs], mirror_position[trough_idcs], 'o', markersize=markersize)
        a.plot(mirror_time[frame_idcs], mirror_position[frame_idcs], 'o', markersize=markersize)
    ax[0].set_xlim(mirror_time[frame_idcs][0]-10, mirror_time[frame_idcs][0]+10)
    ax[2].set_xlim(mirror_time[frame_idcs][-1]-10, mirror_time[frame_idcs][-1]+10)
    fig.tight_layout()
    plt.savefig(os.path.join(recording_path, f'{fig_name}.pdf'), format='pdf')

    if opts.PLOT:
        plt.show()

    return frame_idcs, frame_times


class RoiSelector:
    def __init__(self, img, **kwargs):
        self.raw = img

        self.hp_filter_sigma_x = kwargs.get('hp_filter_sigma_x')
        if self.hp_filter_sigma_x is None:
            self.hp_filter_sigma_x = .1

        self.local_thresh_kernel_size = kwargs.get('local_thresh_kernel_size')
        if self.local_thresh_kernel_size is None:
            self.local_thresh_kernel_size = 31

        self.smooth_sigma_x = kwargs.get('smooth_sigma_x')
        if self.smooth_sigma_x is None:
            self.smooth_sigma_x = 3

        self.binary_thresh = kwargs.get('binary_thresh')
        if self.binary_thresh is None:
            self.binary_thresh = .5

        self.bg_kernel_size = kwargs.get('bg_kernel_size')
        if self.bg_kernel_size is None:
            self.bg_kernel_size = 5

        self.fg_kernel_size = kwargs.get('fg_kernel_size')
        if self.fg_kernel_size is None:
            self.fg_kernel_size = 2

        self.min_size = kwargs.get('min_size')
        if self.min_size is None:
            self.min_size = 50

        self.max_size = kwargs.get('max_size')
        if self.max_size is None:
            self.max_size = 500

        self.conn = np.ones((3, 3,))

        self.hpRaw = cv2.GaussianBlur(self.raw, (0, 0), self.hp_filter_sigma_x)
        self.bpRaw = self.hpRaw / threshold_local(self.hpRaw, self.local_thresh_kernel_size)
        rnorm = lambda x: (x - x.min()) / (x.max() - x.min())
        self.smoothed = cv2.GaussianBlur(rnorm(self.bpRaw), (0, 0), self.smooth_sigma_x)

        bwIm = (self.smoothed > (np.mean(self.smoothed) + self.binary_thresh * np.std(self.smoothed)))
        self.binarized = remove_small_objects(bwIm, self.min_size, connectivity=4).astype(np.uint8)

        # Calculate BG marker
        bgDilateKer = np.ones((self.bg_kernel_size,) * 2, np.uint8)
        self.bgMarker = clear_border(cv2.dilate(self.binarized, bgDilateKer, 1) > 0)

        # Calculate FG marker
        fgDilateKer = np.ones((self.fg_kernel_size,) * 2, np.uint8)
        max_coord = peak_local_max(self.smoothed, footprint=self.conn, indices=False, exclude_border=0)
        # peak_mask = np.zeros_like(img2, dtype=bool)
        # >> > peak_mask[tuple(peak_idx.T)] = True
        self.fgMarker = clear_border(cv2.dilate(max_coord.astype(np.uint8), fgDilateKer)) > 0

        # Calculate watershed
        self.distanceMap = ndi.distance_transform_edt(self.bgMarker)
        markers = ndi.label(self.fgMarker)[0]
        self.rawlabel = watershed(-self.distanceMap, markers, mask=self.bgMarker, watershed_line=True)

        # Filter for size
        temp_val, temp_idx = np.unique(self.rawlabel, return_inverse=True)
        num_of_val = np.bincount(temp_idx)
        exclude_val = temp_val[
            np.bitwise_or(num_of_val <= np.array(self.min_size), num_of_val >= np.array(self.max_size))]
        self.mask_labels = copy.copy(self.rawlabel)
        self.mask_labels[np.isin(self.rawlabel, exclude_val)] = 0

        # Order labels
        sortedLabel = copy.copy(self.mask_labels)
        for i, v in enumerate(np.unique(self.mask_labels)):
            sortedLabel[self.mask_labels == v] = i
        self.mask_labels = sortedLabel
