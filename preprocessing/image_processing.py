import copy
import cv2
import h5py as h5
import logging
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
import util
from definitions import *

log = logging.getLogger(__name__)


def run(recording_folder, io_filename='Io.hdf5', disp_filename='Display.hdf5', **kwargs):
    # Iterate over all folders
    log.info(f'Process image data in {recording_folder}')

    ca_filename = util.get_tif_filename(recording_folder)
    if ca_filename is None:
        log.warning(f'Skip processing. No Ca file found in folder {recording_folder}')
        return False
    ca_filepath = os.path.join(recording_folder, ca_filename)

    log.info(f'Using Ca file {ca_filepath}')

    # Check paths
    disp_filepath = os.path.join(recording_folder, disp_filename)
    if not os.path.exists(disp_filepath):
        log.warning(f'{disp_filename} not found')
        return False

    io_filepath = os.path.join(recording_folder, io_filename)
    if not os.path.exists(io_filepath):
        log.warning(f'{io_filename} not found')
        return False

    out_name = '.'.join(ca_filename.split('.')[:-1])
    out_filepath = os.path.join(recording_folder, f'{out_name}.output.hdf5')
    if os.path.exists(out_filepath) and not kwargs[ARG_OVERWRITE]:
        log.warning(f'Skip {recording_folder}. Output file already exists. '
                    f'Use overwrite option "{OPT_OVERWRITE}"')
        return False

    log.info(f'Copy file {disp_filepath} to {out_filepath} for final output')
    shutil.copyfile(disp_filepath, out_filepath)

    log.info('Run registration')
    # Do registration and segmentation
    ca_data = CalciumImagingData(ca_filepath)
    ca_data.registration()
    ca_frame_indices, ca_frame_times = get_ca_frame_timing(io_filepath, ca_data.raw_frames.shape[0])
    rois = RoiSelector(ca_data.std_image)
    raw_calcium_signals = ca_data.extract_calcium_signals(rois.mask_labels)

    with h5.File(out_filepath, 'r') as f:
        disp_start_time = f['phase1'].attrs['start_time']
        dff = calculate_dffs(disp_start_time, ca_frame_times, ca_frame_indices, raw_calcium_signals)

    log.info(f'Write to output file at {out_filepath}')
    with h5.File(out_filepath, 'a') as out_file:
        outputs = {REGISTERED_FRAMES: ca_data.registered_frames,
                   STD_IMAGE: ca_data.std_image,
                   DFF: dff,
                   RAW_F: raw_calcium_signals,
                   ROI_MASK: rois.mask_labels,
                   FRAME_TIME: ca_frame_times}

        for key, val in outputs.items():
            if key not in out_file.keys():
                dset = out_file.create_dataset(key, val.shape)
            else:
                dset = out_file[key]
            dset[:] = val

        for key in out_file.keys():
            if 'phase' not in key:
                continue

            frame_start_idx = np.argmin(np.abs(out_file[f'{key}/ddp_time'][0] - ca_frame_times))
            out_file[key].attrs['ca_start_frame'] = frame_start_idx
            frame_end_idx = np.argmin(np.abs(out_file[f'{key}/ddp_time'][-1] - ca_frame_times))
            out_file[key].attrs['ca_end_frame'] = frame_end_idx


class CalciumImagingData:
    def __init__(self, cafn, **kwargs):
        self.raw_frames = np.array(io.imread(cafn))

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

    def extract_calcium_signals(self, roi_masks: np.ndarray) -> np.ndarray:
        raw_calcium_signals = []
        for i in np.unique(roi_masks):
            if i == 0:
                continue

            raw_calcium_signals.append(self.registered_frames[:, roi_masks == i].sum(axis=1))
        log.info("ROI trace extraction: Done")

        return np.array(raw_calcium_signals)


def calculate_dffs(stimulus_start_time, ca_frame_times, ca_frame_indices, raw_calcium_signals):
    caframetimeIdx = np.argmax(ca_frame_times[ca_frame_times < stimulus_start_time])
    bl_end_frame = ca_frame_indices[caframetimeIdx]
    dff = [(i - np.mean(i[:bl_end_frame])) / np.mean(i[:bl_end_frame]) for i in raw_calcium_signals]

    return np.array(dff)


def get_ca_frame_timing(timeInfoH5Fn, ca_frame_count):
    with h5.File(timeInfoH5Fn, 'r') as mirInfo:
        mirror_position = np.squeeze(mirInfo[config.Y_MIRROR_SIGNAL])
        mirror_time = np.squeeze(mirInfo[f'{config.Y_MIRROR_SIGNAL}{config.TIME_POSTFIX}'])

    mirror_pos_diff = np.diff(mirror_position)
    temp_ind, _ = find_peaks(mirror_pos_diff, prominence=np.std(mirror_pos_diff))
    peak_ind, _ = find_peaks(mirror_position[temp_ind[0]:], prominence=np.std(mirror_pos_diff))
    valley_ind, _ = find_peaks(-mirror_position[temp_ind[0]:], prominence=np.std(mirror_pos_diff))
    valley_ind = np.hstack([1, valley_ind])
    ca_frame_indices = temp_ind[0] + np.unique(np.concatenate([peak_ind, valley_ind], axis=0)) - 1

    # Only use timepoints for there are frames (recording may run longer than ca imaging)
    ca_frame_indices = ca_frame_indices[:ca_frame_count]
    ca_frame_times = mirror_time[ca_frame_indices]

    return ca_frame_indices, ca_frame_times


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
