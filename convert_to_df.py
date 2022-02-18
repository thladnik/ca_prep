import h5py
import pandas as pd
import seaborn as ns
import nixio
import numpy as np


recording_name = 'rec3'
with h5py.File(f'../Giulia_Exp/{recording_name}/Preprocessed_data.hdf5', 'r') as f_in, \
        nixio.File.open('test.nix', mode=nixio.FileMode.Overwrite) as f_out:

    block = f_out.create_block(recording_name, 'imaging_layer')

    img_time = f_in['frame_time'][:]
    d = f_in['registered_frames'][:]
    d1 = (d - d.min()) / (d.max() - d.min())
    d2 = np.array(d1 * 2 ** 8, dtype=np.uint8)

    img_frames = block.create_data_array('registered_imaging_frames', 'nix.image_series.gray',
                                         dtype=nixio.DataType.UInt8, data=d2, label='Fluorescence')
    img_frames.append_range_dimension(img_time, 'Time', unit='s')

    roi_mask = f_in['roi_mask'][:].astype(np.uint32)
    DFF = f_in['dff'][:]
    positions = []
    extents = []
    for roi_i in np.unique(roi_mask)[1:]:
        mask_idcs = np.where(roi_i == roi_mask)
        xmin, ymin = mask_idcs[1].min(), mask_idcs[0].min()
        xext, yext = mask_idcs[1].max() - xmin, mask_idcs[0].max() - ymin,

        # roi_position = block.create_data_array('roi_positions', 'nix.positions', [xmin, ymin, 0])
        # roi_extents = block.create_data_array('roi_extents', 'nix.extents', [xext, yext, imaging_time.shape[0]])
        tag = block.create_tag(f'roi_{roi_i}', 'nix.roi', position=[xmin, ymin, 0])
        tag.extent = [xext, yext, img_time.shape[0]]
        tag.references.append(img_frames)

        dff = block.create_data_array(f'roi_{roi_i}_dff', 'nix.range', data=DFF[roi_i-1])
        dff.append_range_dimension(img_time, 'Time', unit='s')

        tag.references.append(dff)

    #
    # roi_fluor = block.create_multi_tag('roi_fluorescence', '')
#
#     for grp_name, grp_data in f_in.items():
#         if isinstance(grp_data, h5py.Group):
#             max_i = grp_data['ddp_time'].shape[0]-1
#             data = {}
#             for dataset_name, dataset in grp_data.items():
#                 if isinstance(dataset, h5py.Dataset):
#                     dataset_data = dataset[:].squeeze()
#                     print(dataset, dataset_data.shape)
#                     if dataset_data.ndim == 1:
#                         data[dataset_name] = dataset_data[:max_i]
#                     else:
#                         data[dataset_name] = [d for d in dataset_data[:max_i]]
#
#             data['phase'] = grp_name
#             data['visual'] = grp_data.attrs['visual_name']
#
#             df = pd.DataFrame(data)
#             Dfs.append(df)
#
# Df = pd.concat(Dfs, ignore_index=True)
