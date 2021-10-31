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
#import mrcfile
import mcubes


if __name__ == '__main__':

    # Parse
    parser = parse_options(return_parser=True)

    #app_group.add_argument('--out_file', type=str, default=None,
    parser.add_argument('--out_file', type=str, default=None,
                           help='Export model to C++ compatible format.')
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
    vertices, triangles = mcubes.marching_cubes(sdf_values, 0)
    mcubes.export_mesh(vertices, triangles, args.out_file, "shape")

    # with mrcfile.new_mmap(f'results/{args.out_file}.mrc', overwrite=True, shape=(N, N, N), mrc_mode=2) as mrc:
    #     mrc.data[:] = -sdf_values
