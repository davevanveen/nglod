# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import time

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import moviepy.editor as mpy
from scipy.spatial.transform import Rotation as R
import pyexr

from lib.models import *
from lib.options import parse_options
import mrcfile
import mcubes
import trimesh


def nglod_normalize(V):
    # Normalize mesh
    V_max = np.max(V, axis=0)
    V_min = np.min(V, axis=0)
    V_center = (V_max + V_min) / 2.
    V = V - V_center

    # Find the max distance to origin
    max_dist = np.sqrt(np.max(np.sum(V**2, axis=-1)))
    V_scale = 1. / max_dist
    V *= V_scale
    # return V_scale, V_center
    return V, V_scale, -V_center*V_scale


# def normalize(coords):
#     cmean = np.mean(coords, axis=0, keepdims=True)
#     coords -= cmean
#     coord_max = np.amax(coords)
#     coord_min = np.amin(coords)
#     coords = (coords - coord_min) / (coord_max - coord_min)
#     coords -= 0.5
#     coords *= 2.
# 
#     scale = 2 / (coord_max - coord_min)
#     offset = -2 * (cmean + coord_min) / (coord_max - coord_min) - 1
#     return coords, scale, offset


def normalize(coords, scaling=0.9):
    coords = np.array(coords).copy()
    cmean = np.mean(coords, axis=0, keepdims=True)
    coords -= cmean
    coord_max = np.amax(coords)
    coord_min = np.amin(coords)
    coords = (coords - coord_min) / (coord_max - coord_min)
    coords -= 0.5
    coords *= scaling

    scale = scaling / (coord_max - coord_min)
    offset = -scaling * (cmean + coord_min) / (coord_max - coord_min) - 0.5*scaling
    return coords, scale, offset


if __name__ == '__main__':

    # Parse
    parser = parse_options(return_parser=True)

    #parser.add_argument('--gpu', type=int, default=0, help='gpu id')

    app_group = parser.add_argument_group('app')
    app_group.add_argument('--img-dir', type=str, default='_results/render_app/imgs',
                           help='Directory to output the rendered images')
    app_group.add_argument('--render-2d', action='store_true',
                           help='Render in 2D instead of 3D')
    app_group.add_argument('--exr', action='store_true',
                           help='Write to EXR')
    app_group.add_argument('--r360', action='store_true',
                           help='Render a sequence of spinning images.')
    app_group.add_argument('--rsphere', action='store_true',
                           help='Render around a sphere.')
    app_group.add_argument('--nb-poses', type=int, default=64,
                           help='Number of poses to render for sphere rendering.')
    app_group.add_argument('--cam-radius', type=float, default=4.0,
                           help='Camera radius to use for sphere rendering.')
    app_group.add_argument('--disable-aa', action='store_true',
                           help='Disable anti aliasing.')
    app_group.add_argument('--out_file', type=str, default=None,
                           help='Export model to C++ compatible format.')
    app_group.add_argument('--export', type=str, default=None,
                           help='Export model to C++ compatible format.')
    app_group.add_argument('--rotate', type=float, default=None,
                           help='Rotation in degrees.')
    app_group.add_argument('--depth', type=float, default=0.0,
                           help='Depth of 2D slice.')
    app_group.add_argument('--shape', type=str,
                           help='shape name')
    app_group.add_argument('--mrc', action='store_true',
                           help='save fft mrc instead of mesh')


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Pick device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Get model name
    if args.pretrained is not None:
        name = args.pretrained.split('/')[-1].split('.')[0]
    else:
        assert False and "No network weights specified!"

    net = globals()[args.net](args)
    if args.jit:
        net = torch.jit.script(net)

    net.load_state_dict(torch.load(args.pretrained))

    net.to(device)
    net.eval()

    print("Total number of parameters: {}".format(sum(p.numel() for p in net.parameters())))

    if args.lod is not None:
        net.lod = args.lod

    if args.mrc:
        if 'lucy' in args.shape or 'thai' in args.shape:
            N = 512
        else:
            N = 384
        x = torch.arange(-N//2, N//2) / N
        x = x.float()

    else:
        N = 512
        x = torch.linspace(-1, 1, N)

    x, y, z = torch.meshgrid(x, x, x)
    all_coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).cuda()

    sdf_values = np.zeros((N**3, 1))
    bsize = int(128**2)
    for i in tqdm(range(int(N**3 / bsize))):
        coords = all_coords[i*bsize:(i+1)*bsize, :]
        sdf_values[i*bsize:(i+1)*bsize] = net(coords).cpu().detach().numpy()

    sdf_values = sdf_values.reshape(N, N, N)

    if args.mrc:
        sdf_ft = np.abs(np.fft.fftshift(np.fft.fftn(sdf_values)))
        sdf_ft = sdf_ft / np.max(sdf_ft)*1000
        sdf_ft = np.clip(sdf_ft, 0, 1)**(1/3)
        with mrcfile.new_mmap(f'./results/spectra/{args.out_file}.mrc', overwrite=True, shape=(N, N, N), mrc_mode=2) as mrc:
            mrc.data[:] = sdf_ft
        exit()

    vertices, triangles = mcubes.marching_cubes(-sdf_values, 0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    # mesh.vertices -= mesh.bounding_box.centroid
    # mesh.vertices /= np.max(mesh.bounding_box.extents / 2)
    # mesh.vertices, _, _ = normalize(mesh.vertices)
    mesh.vertices = 2 * (mesh.vertices / N - 0.5) + 1/N

    # load NGLOD and undo scaling
    # first undo scaling that was applied to the mesh during NGLOD training
    if 'sphere' in args.shape:
        mesh.vertices = mesh.vertices * 0.25 / np.mean(np.linalg.norm(mesh.vertices, axis=-1))
        mesh.export(f"results/{args.out_file}.obj")
        exit()

    # align with bacon GT mesh
    def bb_normalize(mesh):
        scale = 1/np.max(mesh.bounding_box.extents / 2)
        offset = -mesh.bounding_box.centroid * scale
        v = mesh.vertices.copy() * scale + offset
        return v, scale, offset

    if 'lucy' in args.shape:
        gt_mesh = trimesh.load(f'/home/lindell/workspace/mfintegration/results/outputs/shapes/gt_nglod_{args.shape}.obj')
        # undo nglod scaling and rotate
        _, scale, offset = nglod_normalize(gt_mesh.vertices.copy())
        mesh.vertices = (mesh.vertices - offset) / scale
        
        mesh.vertices = mesh.vertices @ trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])[:3, :3]
        gt_mesh.vertices = gt_mesh.vertices @ trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])[:3, :3]

        gt_mesh.vertices, scale, offset = bb_normalize(gt_mesh)
        mesh.vertices = mesh.vertices * scale + offset

        gt_mesh = trimesh.load(f'/home/lindell/workspace/mfintegration/results/outputs/shapes/gt_{args.shape}.obj')
        gt_mesh.vertices, scale, offset = bb_normalize(gt_mesh)
        mesh.vertices = (mesh.vertices - offset) / scale

    elif 'dragon' in args.shape:
        gt_mesh = trimesh.load(f'/home/lindell/workspace/mfintegration/results/outputs/shapes/gt_nglod_{args.shape}.obj')
        # undo nglod scaling
        _, scale, offset = nglod_normalize(gt_mesh.vertices.copy())
        mesh.vertices = (mesh.vertices - offset) / scale

    else:
        gt_mesh = trimesh.load(f'/home/lindell/workspace/mfintegration/results/outputs/shapes/gt_{args.shape}.obj')

        _, scale, offset = nglod_normalize(gt_mesh.vertices)
        mesh.vertices = (mesh.vertices - offset) / scale

    # then apply the scaling that we used for the bacon/siren models
    gt_mesh = trimesh.load(f'/home/lindell/workspace/mfintegration/results/outputs/shapes/gt_{args.shape}.xyz')
    gt_mesh.vertices, scale, offset = normalize(gt_mesh.vertices)
    mesh.vertices = mesh.vertices * scale + offset
    mesh.export(f"results/{args.out_file}.obj")
