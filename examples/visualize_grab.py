
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
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import torch
import os, glob
import smplx
import argparse
from tqdm import tqdm

from tools.objectmodel import ObjectModel
from tools.meshviewer import Mesh, MeshViewer, points2sphere, colors
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import to_cpu
from tools.utils import euler
from tools.cfg_parser import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_sequences(cfg):

    grab_path = cfg.grab_path

    all_seqs = glob.glob(grab_path + '/*/*eat*.npz')

    mv = MeshViewer(offscreen=False)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([-.5, -4., 1.5])
    mv.update_camera_pose(camera_pose)

    choice = np.random.choice(len(all_seqs), 10, replace=False)
    for i in tqdm(choice):
        vis_sequence(cfg,all_seqs[i], mv)
    mv.close_viewer()


def vis_sequence(cfg,sequence, mv):

        seq_data = parse_npz(sequence)
        n_comps = seq_data['n_comps']
        gender = seq_data['gender']

        T = seq_data.n_frames

        sbj_mesh = os.path.join(grab_path, '..', seq_data.body.vtemp)
        sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

        sbj_m = smplx.create(model_path=cfg.model_path,
                             model_type='smplx',
                             gender=gender,
                             num_pca_comps=n_comps,
                             v_template=sbj_vtemp,
                             batch_size=T)

        sbj_parms = params2torch(seq_data.body.params)
        verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)


        obj_mesh = os.path.join(grab_path, '..', seq_data.object.object_mesh)
        obj_mesh = Mesh(filename=obj_mesh)
        obj_vtemp = np.array(obj_mesh.vertices)
        obj_m = ObjectModel(v_template=obj_vtemp,
                            batch_size=T)
        obj_parms = params2torch(seq_data.object.params)
        verts_obj = to_cpu(obj_m(**obj_parms).vertices)

        table_mesh = os.path.join(grab_path, '..', seq_data.table.table_mesh)
        table_mesh = Mesh(filename=table_mesh)
        table_vtemp = np.array(table_mesh.vertices)
        table_m = ObjectModel(v_template=table_vtemp,
                            batch_size=T)
        table_parms = params2torch(seq_data.table.params)
        verts_table = to_cpu(table_m(**table_parms).vertices)

        skip_frame = 4
        for frame in range(0,T, skip_frame):
            o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
            o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['object'][frame] > 0)

            s_mesh = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
            s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame] > 0)

            t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])

            mv.set_static_meshes([o_mesh, s_mesh, t_mesh])


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='GRAB-visualize')

    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')

    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing smplx models')

    args = parser.parse_args()

    grab_path = args.grab_path
    model_path = args.model_path

    # grab_path = 'PATH_TO_DOWNLOADED_GRAB_DATA/grab'
    # model_path = 'PATH_TO_DOWNLOADED_MODELS_FROM_SMPLX_WEBSITE/'

    cfg = {
        'grab_path': grab_path,
        'model_path': model_path,
    }

    cfg = Config(**cfg)
    visualize_sequences(cfg)

