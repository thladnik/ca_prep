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
import sys

import util

log = logging.getLogger(__name__)


def run(recording_folder, io_filename='Io.hdf5', disp_filename='Display.hdf5'):

    # Iterate over all folders
    log.info(f'Process image data in {recording_folder}')

    ca_filepath = util.get_tif_filepath(recording_folder)
    if ca_filepath is None:
        log.warning('Skip processing. No Ca file found')
        return False

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

    out_filepath = os.path.join(recording_folder, f'{ca_filepath[:-4]}.output.hdf5')
    if os.path.exists(out_filepath) and '--overwrite' not in sys.argv[1:]:
        log.info(f'Output file {out_filepath} already exists. No overwrite option. Skip')
        return False

    log.info(f'Copy file {disp_filepath} to {out_filepath} for final output')
    shutil.copyfile(disp_filepath, out_filepath)

    log.info('Read calcium and synchronization data')
    ca_data = CaImgPrep(ca_filepath)
    ca_data.appendTimeInfo(io_filepath)

    log.info(f'Write to output file at {out_filepath}')
    with h5.File(out_filepath, 'a') as out_file:
        disp_start_time = out_file['phase1'].attrs['start_time']
        ca_data.calcDff(disp_start_time)
        outputDstName = {'dff': np.array(ca_data.dff),
                         'raw_f': np.array(ca_data.rawCaTrace),
                         'std_image': ca_data.stdImg,
                         'roi_mask': ca_data.ROImask,
                         'frame_time': ca_data.ca_frame_time}

        for key, val in outputDstName.items():
            if key not in out_file.keys():
                dset = out_file.create_dataset(key, val.shape)
            else:
                dset = out_file[key]
            dset[:] = val

        for i in out_file.keys():
            if 'phase' in i:
                frame_start_idx = np.argmin(np.abs(out_file[f'{i}/ddp_time'][0] - ca_data.ca_frame_time))
                out_file[i].attrs['ca_start_frame'] = frame_start_idx
                frame_end_idx = np.argmin(np.abs(out_file[f'{i}/ddp_time'][-1] - ca_data.ca_frame_time))
                out_file[i].attrs['ca_end_frame'] = frame_end_idx


class CaImgPrep:
    def __init__(self, cafn, **kwargs):
        self.rawImg = np.array(io.imread(cafn))
        self._param = {
            "binsize": 500,
            "stepsize": 250,
            "hpfiltSig": .1,
            "localThreKerSize": 31,
            "smoothSig": 3,
            "binaryThre": 0.5,
            "bgKerSize": 5,
            "fgKerSize": 2,
            "minSizeLim": 50,
            "maxSizeLim": 500
        }
        self._param.update(kwargs)
        self.__dict__.update(self._param)
        self.parse()

    def parse(self):
        self.registration()
        self.segmentation()
        self.applyROImask()

    def registration(self, binsize=None, stepsize=None):
        if binsize is not None:
            self.binsize = binsize
        if stepsize is not None:
            self.stepsize = stepsize
        templateImg = np.mean(self.rawImg[:self.binsize], axis=0)
        i = self.stepsize
        regTifImg = copy.copy(self.rawImg)
        oldDrift = None
        while i < self.rawImg.shape[0]:
            movingImg = np.mean(self.rawImg[i:i + self.binsize], axis=0)
            imgDrift = phase_cross_correlation(templateImg, movingImg)[0]
            if oldDrift is None:
                oldDrift = imgDrift
            for o in range(min(self.stepsize, self.rawImg.shape[0] - i)):
                itershift = np.vstack(
                    [np.eye(2), imgDrift[::-1] * (1 - o / self.stepsize) + oldDrift[::-1] * o / self.stepsize]).T
                regTifImg[i + o] = cv2.warpAffine(self.rawImg[i + o], itershift, tuple(templateImg.shape))
                # printProgressBar(i+o,self.rawImg.shape[0],prefix='Motion correction: ')
            oldDrift = imgDrift
            i += self.stepsize
        self.regImg = regTifImg
        self.stdImg = np.std(self.regImg, axis=0)

    def segmentation(self, **kwargs):
        if self.stdImg is not None:
            self._param.update(kwargs)
            self.ROIselector = ROIselector(self.stdImg, **self._param)
            self.ROImask = self.ROIselector.labels
        log.info('Segmentation: Done')

    def applyROImask(self):
        self.rawCaTrace = []
        for i in np.unique(self.ROImask):
            if i > 0:
                self.rawCaTrace.append(self.regImg[:, self.ROImask == i].sum(axis=1))
        log.info("ROI trace extraction: Done")

    def appendTimeInfo(self, timeInfoH5Fn):
        with h5.File(timeInfoH5Fn, 'r') as mirInfo:
            mirTime = np.squeeze(mirInfo['y_mirror_in_time'])
            mirPos = np.squeeze(mirInfo['y_mirror_in'])

        diffMirPos = np.diff(mirPos)
        temp_ind, _ = find_peaks(diffMirPos, prominence=np.std(diffMirPos))
        peak_ind, _ = find_peaks(mirPos[temp_ind[0]:], prominence=np.std(diffMirPos))
        valley_ind, _ = find_peaks(-mirPos[temp_ind[0]:], prominence=np.std(diffMirPos))
        valley_ind = np.hstack([1, valley_ind])
        ca_frame_idx = temp_ind[0] + np.unique(np.concatenate([peak_ind, valley_ind], axis=0)) - 1

        # Only use timepoints for there are frames (recording may run longer than ca imaging)
        ca_frame_idx = ca_frame_idx[:self.rawImg.shape[0]]
        ca_frame_time = mirTime[ca_frame_idx]
        self.mirTime = mirTime
        self.mirPos = mirPos
        self.ca_frame_idx = ca_frame_idx
        self.ca_frame_time = ca_frame_time

    def calcDff(self, stiStTime):
        caframetimeIdx = np.argmax(self.ca_frame_time[self.ca_frame_time < stiStTime])
        blEndFrame = self.ca_frame_idx[caframetimeIdx]
        self.dff = [(i - np.mean(i[:blEndFrame])) / np.mean(i[:blEndFrame]) for i in self.rawCaTrace]
        self.stiStTime = stiStTime
        self.stiStFrame = blEndFrame


rnorm = lambda x: (x - x.min()) / (x.max() - x.min())


class ROIselector:
    def __init__(self, img, **kwargs):
        self.raw = img
        self._param = {
            "hpfiltSig": .1,
            "localThreKerSize": 31,
            "smoothSig": 3,
            "binaryThre": 0.5,
            "bgKerSize": 5,
            "fgKerSize": 2,
            "minSizeLim": 50,
            "maxSizeLim": 500
        }
        self.conn = np.ones((3, 3,))
        self._param.update(kwargs)
        self.__dict__.update(self._param)
        self.parse()

    def parse(self):
        self.bp()
        self.binarization()
        self.compute_bgMarker()
        self.compute_fgMarker()
        self.watershed()
        self.sizeFilter()
        self.orderlabel()

    def bp(self):
        self.hpRaw = cv2.GaussianBlur(self.raw, (0, 0), self.hpfiltSig)
        self.bpRaw = self.hpRaw / threshold_local(self.hpRaw, self.localThreKerSize)
        self.smoothed = cv2.GaussianBlur(rnorm(self.bpRaw), (0, 0), self.smoothSig)

    def binarization(self):
        bwIm = (self.smoothed > (np.mean(self.smoothed) + self.binaryThre * np.std(self.smoothed)))
        self.binarized = remove_small_objects(bwIm, self.minSizeLim, connectivity=4).astype(np.uint8)

    def compute_bgMarker(self):
        bgDilateKer = np.ones((self.bgKerSize,) * 2, np.uint8)
        self.bgMarker = clear_border(cv2.dilate(self.binarized, bgDilateKer, 1) > 0)

    def compute_fgMarker(self):
        fgDilateKer = np.ones((self.fgKerSize,) * 2, np.uint8)
        maxCoord = peak_local_max(self.smoothed, footprint=self.conn, indices=False, exclude_border=0)
        self.fgMarker = clear_border(cv2.dilate(maxCoord.astype(np.uint8), fgDilateKer)) > 0

    def watershed(self):
        self.distanceMap = ndi.distance_transform_edt(self.bgMarker)
        markers = ndi.label(self.fgMarker)[0]
        self.rawlabel = watershed(-self.distanceMap, markers, mask=self.bgMarker, watershed_line=True)

    def sizeFilter(self):
        temp_val, temp_idx = np.unique(self.rawlabel, return_inverse=True)
        num_of_val = np.bincount(temp_idx)
        exclude_val = temp_val[
            np.bitwise_or(num_of_val <= np.array(self.minSizeLim), num_of_val >= np.array(self.maxSizeLim))]
        self.labels = copy.copy(self.rawlabel)
        self.labels[np.isin(self.rawlabel, exclude_val)] = 0

    def orderlabel(self):
        sortedLabel = copy.copy(self.labels)
        for i, v in enumerate(np.unique(self.labels)):
            sortedLabel[self.labels == v] = i
        self.labels = sortedLabel
