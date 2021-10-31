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

import os

import torch
from torch.utils.data import Dataset

import numpy as np
import mesh2sdf

from lib.torchgp import load_obj, point_sample, sample_surface, compute_sdf, normalize
from lib.PsDebugger import PsDebugger

from lib.utils import PerfTimer, setparam

class SphereDataset(Dataset):
    """Base class for spherical sdf dataset"""

    def __init__(self, 
        args=None, 
        dataset_path = None,
        raw_obj_path = None,
        sample_mode = None,
        get_normals = None,
        seed = None,
        num_samples = None,
        trim = None,
        sample_tex = None,
        num_pts = None,
	radius = .5
    ):
        self.args = args
        self.dataset_path = setparam(args, dataset_path, 'dataset_path')
        self.raw_obj_path = setparam(args, raw_obj_path, 'raw_obj_path')
        self.sample_mode = setparam(args, sample_mode, 'sample_mode')
        self.get_normals = setparam(args, get_normals, 'get_normals')
        self.num_samples = setparam(args, num_samples, 'num_samples')
        self.trim = setparam(args, trim, 'trim')
        self.sample_tex = setparam(args, sample_tex, 'sample_tex')
        
        self.num_pts = num_pts
	self.res_bin = num_pts ** (1/3)
	if self.res_bin % 1 != 0:
	    raise ValueError('num_pts must be a cubed value, i.e. res_bin ** 3 = num_pts')
	self.bins, self.bin_dx = get_bins(self.res_bin)
	self.radius = radius

        #if self.sample_tex:
        #    out = load_obj(self.dataset_path, load_materials=True)
        #    self.V, self.F, self.texv, self.texf, self.mats = out
        #else:
        #    self.V, self.F = load_obj(self.dataset_path)

        #self.V, self.F = normalize(self.V, self.F)
        #self.mesh = self.V[self.F]
        #self.resample()

    def resample(self):
        """Resample SDF samples."""

        raise NotImplementedError
        
        self.nrm = None
        if self.get_normals:
            self.pts, self.nrm = sample_surface(self.V, self.F, self.num_samples*5)
            self.nrm = self.nrm.cpu()
        else:
            self.pts = point_sample(self.V, self.F, self.sample_mode, self.num_samples)

        self.d = compute_sdf(self.V.cuda(), self.F.cuda(), self.pts.cuda())   

        self.d = self.d[...,None]
        self.d = self.d.cpu()
        self.pts = self.pts.cpu()

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        
        rand_indcs = np.random.choice(self.num_pts, size=self.num_pts,
                                                    replace=False)
        coords = self.bins[rand_indcs] + self.bin_dx * \
                 np.random.uniform(0, 1, size=(self.num_pts, 3))

        # define value of each sampled coordinate
        # d = dist from center = x^2 + y^2 + z^2 --> sdf = d - r
        sdf = np.sqrt(np.sum(np.square(coords), axis=1)) - self.radius

        
        # self.pts: points on mesh, i.e. coords
        # self.d: sdf values at that point
        # TODO: figure out which shape each should be in via MeshDataset()
        if self.get_normals:
            #return self.pts[idx], self.d[idx], self.nrm[idx]
	    return coords, sdf, sdf
        elif self.sample_tex:
            #return self.pts[idx], self.d[idx], self.rgb[idx]
            raise NotImplementedError 
        else:
            #return self.pts[idx], self.d[idx]
            return coords, sdf
            
    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.pts.size()[0]

    def num_shapes(self):
        """Return length of dataset (number of _mesh models_)."""

        return 1

def get_bins(num_bins):
    ''' define 3d coordinate offsets for all bins given a resolution '''
    
    num_bins += 1 # if want num_bins bins, add 1 b/c linspace is inclusive

    # define coordinates and differential 
    x_bins = np.linspace(-1, 1, num_bins)[None, :num_bins-1]
    bin_dx = x_bins[0,1] - x_bins[0,0]
    y_bins = x_bins
    z_bins = x_bins
   
    x_bins, y_bins, z_bins = np.meshgrid(x_bins, y_bins, z_bins)

    bins = np.concatenate((x_bins.flatten()[:,None], y_bins.flatten()[:,None],
                           z_bins.flatten()[:,None]), axis=-1)
    return bins, bin_dx
