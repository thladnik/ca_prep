import logging
import os
from typing import List, Union, Callable

import h5py.h5i
import numpy as np
import pandas as pd
import yaml

from ca_prep import config
from definitions import *

log = logging.getLogger(__name__)


def create_dataset(f: h5py.File, dataset_name: str, data: np.ndarray):
    log.debug(f'Create dataset {dataset_name} in {f}')
    if dataset_name in f:
        del f[dataset_name]
    f.create_dataset(dataset_name, data=data)


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def sph2cart(theta, phi, r):
    rcos_theta = r * np.cos(phi)
    x = rcos_theta * np.cos(theta)
    y = rcos_theta * np.sin(theta)
    z = r * np.sin(phi)
    return np.array([x, y, z])


def get_phase_start_points(f):
    return [f[k].attrs['ca_start_frame'] for k in f.keys() if k.startswith('phase')]


def get_path_segments(path: str) -> Union[List[str], None]:
    if '\\' in path:
        return path.split('\\')
    if '/' in path:
        return path.split('/')
    return None


def get_path(path: Union[str, list]) -> Union[str, None]:
    if not isinstance(path, (str, list)):
        log.error(f'Wrong path type provided in {get_path.__qualname__}')
        raise TypeError

    if isinstance(path, str):
        path = get_path_segments(path)
    if path is None:
        return path
    return os.path.join(*path)


def get_tif_filename(path: str) -> Union[str, None]:
    for name in os.listdir(path):
        if name.lower().endswith('.tif'):
            return name
    return None


def get_processing_filepath(path: str) -> Union[str, None]:
    if os.path.exists(os.path.join(path, PATH_FN_PREPROCESSED)):
        return os.path.join(path, PATH_FN_PREPROCESSED)
    return None


def apply_to_all_recording_folders(root_path: str, fun: Callable, *args, **kwargs):

    log.info(f'Fetch recordings in root {root_path}')
    recordings = [os.path.join(root_path, fld) for fld in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, fld))]

    for recording_folder in recordings:
        log.info(f'Apply {fun.__qualname__} to recording {recording_folder}')
        fun(recording_folder, *args, **kwargs)


def load_configuration(config_filepath: str):
    with open(config_filepath, 'r') as f:
        _config = yaml.safe_load(f)
        log.info(f'Use configuration {_config}')
        config.__dict__.update(_config)


def calculate_repeatability(df: pd.DataFrame, use_key: str = ZSCORE) -> float:

    if df.shape[0] == 3:
        # Calculate Pearson correlation coefficients for 3 repeats
        min_len = min([d.shape[0] for d in df[use_key].array])

        ar = np.stack([d[:min_len] for d in df[use_key].array])
        ccef = np.corrcoef(ar)
        ccef1, ccef2, ccef3 = ccef[0, 1], ccef[0, 2], ccef[1, 2]
        mccef = np.mean([ccef1, ccef2, ccef3])

        return mccef

    else:
        return 0


def calculate_mean_dff(df: pd.DataFrame, use_key: str = ZSCORE) -> np.ndarray:

    min_len = min([d.shape[0] for d in df[use_key].array])
    ar = np.stack([d[:min_len] for d in df[use_key].array])

    return np.mean(ar, axis=0)


def get_time(df: pd.DataFrame, time_key: str = 'var_param_time') -> np.ndarray:

    min_len = min([d.shape[0] for d in df.dff.array])
    return df[time_key].array[:min_len]
