
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#


import os
import glob
import numpy as np
import torch
from torch.utils import data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LoadData(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 dtype=torch.float32):

        super().__init__()

        self.ds_path = os.path.join(dataset_dir, ds_name)
        datasets = glob.glob(self.ds_path+'/*.pt')
        self.ds = self.load(datasets)

        frame_names = np.load(os.path.join(dataset_dir,ds_name, 'frame_names.npz'))['frame_names']
        self.frame_names = [os.path.join(dataset_dir, fname) for fname in frame_names]
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])

        self.sbjs = np.unique(self.frame_sbjs)
        self.obj_info = np.load(os.path.join(dataset_dir, 'obj_info.npy'), allow_pickle=True).item()
        self.sbj_info = np.load(os.path.join(dataset_dir, 'sbj_info.npy'), allow_pickle=True).item()

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)

    def load(self,datasets):
        loaded = {}
        for d in datasets:
            k = os.path.basename(d).split('_')[0]
            loaded[k] = torch.load(d)
        return loaded

    def load_idx(self,idx, source=None):

        if source is None:
            source = self.ds

        out = {}
        for k, v in source.items():
            if isinstance(v,dict):
                out[k] = self.load_idx(idx, v)
            else:
                out[k] = v[idx]

        return out

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, idx):

        data_out = self.load_idx(idx)
        data_out['idx'] = torch.from_numpy(np.array(idx, dtype=np.int32))
        return data_out

if __name__=='__main__':

    data_path = 'PATH_TO_PROCESSED_DATA/grab_processed'
    ds = LoadData(data_path, ds_name='val')

    bs = 256
    dataloader = data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)

