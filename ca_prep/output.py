import h5py
import pandas as pd

from definitions import *


def get_df_of_phases(filepath: str) -> pd.DataFrame:
    """Read an HDF5 file at filepath and return a dataframe with a row for each phase/roi combination"""

    with h5py.File(filepath, 'r') as f:
        data = {}
        for item in f.values():
            if not isinstance(item, h5py.Group):
                continue

            # if item.attrs['visual_name'] != 'SingleDotRotatingAroundAxis':
            #     continue

            # Set up dictionary with lists
            if len(data) == 0:

                data['roi'] = []
                data['phase'] = []

                for attr_name in item.attrs.keys():
                    data[attr_name] = []

                for dset_name in item.keys():
                    data[dset_name] = []

            # Append data
            roi_count = item[DFF].shape[0]

            # Phase IDs
            data['phase'].extend([int(item.name.replace('phase', ''))] * roi_count)

            # Attributes
            for attr_name, attr_value in item.attrs.items():
                data[attr_name].extend([attr_value] * roi_count)

            # Datasets
            for dset_name, dset_data in item.items():
                # Skip DFF and ZSCORE, because those get unpacked according to roi ids
                if dset_name in (DFF, ZSCORE):
                    continue

                data[dset_name].extend([dset_data[:]] * roi_count)

            # Unpack zscores and dffs
            for i, (d, z) in enumerate(zip(item[DFF], item[ZSCORE])):

                data['roi'].append(i)

                # Append respective imaging traces for rois
                data[DFF].append(d)
                data[ZSCORE].append(z)

    # Process
    return pd.DataFrame(data=data)


if __name__ == '__main__':
    pass