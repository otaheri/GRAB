
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


import numpy as np
import torch
import logging
from copy import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)

def params2torch(params, dtype = torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}

def prepare_params(params, frame_mask, dtype = np.float32):
    return {k: v[frame_mask].astype(dtype) for k, v in params.items()}

def DotDict(in_dict):

    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def append2dict(source, data):
    for k in data.keys():
        if isinstance(data[k], list):
            source[k] += data[k].astype(np.float32)
        else:
            source[k].append(data[k].astype(np.float32))

def np2torch(item):
    out = {}
    for k, v in item.items():
        if v ==[]:
            continue
        if isinstance(v, list):
            try:
                out[k] = torch.from_numpy(np.concatenate(v))
            except:
                out[k] = torch.from_numpy(np.array(v))
        elif isinstance(v, dict):
            out[k] = np2torch(v)
        else:
            out[k] = torch.from_numpy(v)
    return out

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array

def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

def makelogger(log_dir,mode='w'):

    makepath(log_dir, isfile=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    fh = logging.FileHandler('%s'%log_dir, mode=mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def euler(rots, order='xyz', units='deg'):

    rots = np.asarray(rots)
    single_val = False if len(rots.shape)>1 else True
    rots = rots.reshape(-1,3)
    rotmats = []

    for xyz in rots:
        if units == 'deg':
            xyz = np.radians(xyz)
        r = np.eye(3)
        for theta, axis in zip(xyz,order):
            c = np.cos(theta)
            s = np.sin(theta)
            if axis=='x':
                r = np.dot(np.array([[1,0,0],[0,c,-s],[0,s,c]]), r)
            if axis=='y':
                r = np.dot(np.array([[c,0,s],[0,1,0],[-s,0,c]]), r)
            if axis=='z':
                r = np.dot(np.array([[c,-s,0],[s,c,0],[0,0,1]]), r)
        rotmats.append(r)
    rotmats = np.stack(rotmats).astype(np.float32)
    if single_val:
        return rotmats[0]
    else:
        return rotmats

def create_video(path, fps=30,name='movie'):
    import os
    import subprocess

    src = os.path.join(path,'%*.png')
    movie_path = os.path.join(path,'%s.mp4'%name)
    i = 0
    while os.path.isfile(movie_path):
        movie_path = os.path.join(path,'%s_%02d.mp4'%(name,i))
        i +=1

    cmd = 'ffmpeg -f image2 -r %d -i %s -b:v 6400k -pix_fmt yuv420p %s' % (fps, src, movie_path)

    subprocess.call(cmd.split(' '))
    while not os.path.exists(movie_path):
        continue

# mapping the contact ids to each body part in smplx        
contact_ids={'Body': 1,
             'L_Thigh': 2,
             'R_Thigh': 3,
             'Spine': 4,
             'L_Calf': 5,
             'R_Calf': 6,
             'Spine1': 7,
             'L_Foot': 8,
             'R_Foot': 9,
             'Spine2': 10,
             'L_Toes': 11,
             'R_Toes': 12,
             'Neck': 13,
             'L_Shoulder': 14,
             'R_Shoulder': 15,
             'Head': 16,
             'L_UpperArm': 17,
             'R_UpperArm': 18,
             'L_ForeArm': 19,
             'R_ForeArm': 20,
             'L_Hand': 21,
             'R_Hand': 22,
             'Jaw': 23,
             'L_Eye': 24,
             'R_Eye': 25,
             'L_Index1': 26,
             'L_Index2': 27,
             'L_Index3': 28,
             'L_Middle1': 29,
             'L_Middle2': 30,
             'L_Middle3': 31,
             'L_Pinky1': 32,
             'L_Pinky2': 33,
             'L_Pinky3': 34,
             'L_Ring1': 35,
             'L_Ring2': 36,
             'L_Ring3': 37,
             'L_Thumb1': 38,
             'L_Thumb2': 39,
             'L_Thumb3': 40,
             'R_Index1': 41,
             'R_Index2': 42,
             'R_Index3': 43,
             'R_Middle1': 44,
             'R_Middle2': 45,
             'R_Middle3': 46,
             'R_Pinky1': 47,
             'R_Pinky2': 48,
             'R_Pinky3': 49,
             'R_Ring1': 50,
             'R_Ring2': 51,
             'R_Ring3': 52,
             'R_Thumb1': 53,
             'R_Thumb2': 54,
             'R_Thumb3': 55}
