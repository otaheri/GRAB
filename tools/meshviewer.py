# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
#

import numpy as np
import trimesh
import pyrender
from pyrender.light import DirectionalLight
from pyrender.node import Node
from PIL import Image
from .utils import euler


class Mesh(trimesh.Trimesh):

    def __init__(self,
                 filename=None,
                 vertices=None,
                 faces=None,
                 vc=None,
                 fc=None,
                 vscale=None,
                 process = False,
                 visual = None,
                 wireframe=False,
                 smooth = False,
                 **kwargs):

        self.wireframe = wireframe
        self.smooth = smooth

        if filename is not None:
            mesh = trimesh.load(filename, process = process)
            vertices = mesh.vertices
            faces= mesh.faces
            visual = mesh.visual
        if vscale is not None:
            vertices = vertices*vscale

        if faces is None:
            mesh = points2sphere(vertices)
            vertices = mesh.vertices
            faces = mesh.faces
            visual = mesh.visual

        super(Mesh, self).__init__(vertices=vertices, faces=faces, process=process, visual=visual)

        if vc is not None:
            self.set_vertex_colors(vc)
        if fc is not None:
            self.set_face_colors(fc)

    def rot_verts(self, vertices, rxyz):
        return np.array(vertices * rxyz.T)

    def colors_like(self,color, array, ids):

        color = np.array(color)

        if color.max() <= 1.:
            color = color * 255
        color = color.astype(np.int8)

        n_color = color.shape[0]
        n_ids = ids.shape[0]

        new_color = np.array(array)
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color

        return new_color

    def set_vertex_colors(self,vc, vertex_ids = None):

        all_ids = np.arange(self.vertices.shape[0])
        if vertex_ids is None:
            vertex_ids = all_ids

        vertex_ids = all_ids[vertex_ids]
        new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
        self.visual.vertex_colors[:] = new_vc

    def set_face_colors(self,fc, face_ids = None):

        if face_ids is None:
            face_ids = np.arange(self.faces.shape[0])

        new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
        self.visual.face_colors[:] = new_fc

    @staticmethod
    def concatenate_meshes(meshes):
        return trimesh.util.concatenate(meshes)


class MeshViewer(object):

    def __init__(self,
                 width=1200,
                 height=800,
                 bg_color = [0.0, 0.0, 0.0, 1.0],
                 offscreen = False,
                 registered_keys=None):
        super(MeshViewer, self).__init__()

        if registered_keys is None:
            registered_keys = dict()

        self.bg_color = bg_color
        self.offscreen = offscreen
        self.scene = pyrender.Scene(bg_color=bg_color,
                                    ambient_light=(0.3, 0.3, 0.3),
                                    name = 'scene')

        self.aspect_ratio = float(width) / height
        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=self.aspect_ratio)
        camera_pose = np.eye(4)
        camera_pose[:3,:3] = euler([80,-15,0], 'xzx')
        camera_pose[:3, 3] = np.array([-.5, -2., 1.5])

        self.cam = pyrender.Node(name = 'camera', camera=pc, matrix=camera_pose)

        self.scene.add_node(self.cam)

        if self.offscreen:
            light = Node(light=DirectionalLight(color=np.ones(3), intensity=3.0),
                          matrix=camera_pose)
            self.scene.add_node(light)
            self.viewer = pyrender.OffscreenRenderer(width, height)
        else:
            self.viewer = pyrender.Viewer(self.scene,
                                          use_raymond_lighting=True,
                                          viewport_size=(width, height),
                                          cull_faces=False,
                                          run_in_thread=True,
                                          registered_keys=registered_keys)

        for i, node in enumerate(self.scene.get_nodes()):
            if node.name is None:
                node.name = 'Req%d'%i


    def is_active(self):
        return self.viewer.is_active

    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def set_background_color(self, bg_color=[1., 1., 1.]):
        self.scene.bg_color = bg_color

    def to_pymesh(self, mesh):

        wireframe = mesh.wireframe if hasattr(mesh, 'wireframe') else False
        smooth = mesh.smooth if hasattr(mesh, 'smooth') else False
        return  pyrender.Mesh.from_trimesh(mesh, wireframe=wireframe, smooth=smooth)

    def update_camera_pose(self, pose):
        if self.offscreen:
            self.scene.set_pose(self.cam, pose=pose)
        else:
            self.viewer._default_camera_pose[:] = pose

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3),
                                                    intensity=1.0),
                    matrix=matrix
                ))

        return nodes


    def set_meshes(self, meshes =[], set_type = 'static'):

        if not self.offscreen:
            self.viewer.render_lock.acquire()

        for node in self.scene.get_nodes():
            if node.name is None:
                continue
            if 'static' in set_type and 'mesh' in node.name:
                self.scene.remove_node(node)
            elif 'dynamic' in node.name:
                self.scene.remove_node(node)

        for i, mesh in enumerate(meshes):
            mesh = self.to_pymesh(mesh)
            self.scene.add(mesh, name='%s_mesh_%d'%(set_type,i))

        if not self.offscreen:
            self.viewer.render_lock.release()

    def set_static_meshes(self, meshes =[]):
        self.set_meshes(meshes=meshes, set_type='static')

    def set_dynamic_meshes(self, meshes =[]):
        self.set_meshes(meshes=meshes, set_type='dynamic')

    def save_snapshot(self, save_path):
        if not self.offscreen:
            print('We do not support rendering in Interactive mode!')
            return
        color, depth = self.viewer.render(self.scene)
        img = Image.fromarray(color)
        img.save(save_path)


def points2sphere(points, radius = .001, vc = [0., 0., 1.], count = [5,5]):

    points = points.reshape(-1,3)
    n_points = points.shape[0]

    spheres = []
    for p in range(n_points):
        sphs = trimesh.creation.uv_sphere(radius=radius, count = count)
        sphs.apply_translation(points[p])
        sphs = Mesh(vertices=sphs.vertices, faces=sphs.faces, vc=vc)

        spheres.append(sphs)

    spheres = Mesh.concatenate_meshes(spheres)
    return spheres

colors = {
    'pink': [1.00, 0.75, 0.80],
    'purple': [0.63, 0.13, 0.94],
    'red': [1.0, 0.0, 0.0],
    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [1.00, 0.25, 0.25],
    'blue': [.0, .0, 1.],
    'white': [1., 1., 1.],
    'orange': [1.00, 0.65, 0.00],
    'grey': [0.75, 0.75, 0.75],
    'black': [0., 0., 0.],
}