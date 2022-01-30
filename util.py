import logging
import os
from typing import List, Union, Callable
import numpy as np
import yaml

import config

log = logging.getLogger(__name__)


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
    return [f[f'phase{i + 1}'].attrs['ca_start_frame'] for i in range(8)]


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
    for name in os.listdir(path):
        if name.endswith('.output.hdf5'):
            return os.path.join(path, name)
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
