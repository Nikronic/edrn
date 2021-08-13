import os, sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import MLP
from utils import SupervisedMeshGrid
import utils
import visualizations

import numpy as np

import matplotlib as mpl
mpl.use('Agg')

import json
import argparse

sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore
import MeshFEM, mesh  # type: ignore
from ipopt_helpers import initializeTensorProductSimulator, problemObjectWrapper, initializeIpoptProblem  # type: ignore

parser=argparse.ArgumentParser()
parser.add_argument('--expp', help='Path to the pretrained weights (result will be saved in "images" with same path)', 
                    default='fourfeat_supervised/3900276.pt')
args=parser.parse_args()
experiment_path = args.expp
sys.stderr.write('Loading pretrained weight with experiment ID and path: {}\n'.format(experiment_path))

# PyTorch related global variables
torch.autograd.set_detect_anomaly(False)

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = False
save_density = False
interpolation = False
fixed_scale = True
clamp_output = True  # TODO

problem_path = 'problems/2d/mbb_beam.json'  # TODO
gt_path = 'logs/densities/gt/3879060/voxlfem_1500x500_2000_MbbBeamSeed88_Vol0.3_Filt__gt.npy'  # TODO
# max resolution assumption - all other solutions are coarser than this
max_resolution = [1500, 500]  # TODO 

mrconfprint = 'Testing pretrained model in: {} '.format(max_resolution)
sys.stderr.write(mrconfprint)

# hyper parameter of positional encoding in NeRF
embedding_size = 256

# deep learning modules
nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=embedding_size,
                 scale=1., hidden_act=nn.ReLU(), output_act=None)
# In original nerf, they use sigmoid as the last layer. We remove it as we have constrainted sigmoid!                 
model = nerf_model

# load pretrained weights
weights_path = 'logs/weights/{}'.format(experiment_path)
images_path = weights_path.replace('weights', 'images')[:-3]+'/'
utils.load_weights(model, weights_path)
if torch.cuda.is_available():
    model.cuda()

sys.stderr.write('scale: {}, fourier embedding size:{}\n'.format(model.scale, embedding_size))

# loss
criterion = nn.MSELoss(reduction='sum')
sys.stderr.write('Deep learning model config: {}\n'.format(model))
sys.stderr.write('Critertion {} \n'.format(criterion))

# hyperparameters of the problem 
with open(problem_path, 'r') as j:
     configs = json.loads(j.read())

problem_name = configs['problem_name']
MATERIAL_PATH = configs['MATERIAL_PATH']
BC_PATH = configs['BC_PATH']
orderFEM = configs['orderFEM']
domainCorners = configs['domainCorners']
gridDimensions = configs['gridDimensions']
E0 = configs['E0']
Emin = configs['Emin']
SIMPExponent = configs['SIMPExponent']
maxVolume = torch.tensor(configs['maxVolume'])
seed = configs['seed']
if torch.cuda.is_available():
    maxVolume = maxVolume.cuda()
sys.stderr.write('VoxelFEM problem configs: {}\n'.format(configs))

if max_resolution is None:
    max_resolution = gridDimensions

# reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# query for max resolution
domain = np.array([[0., 1.],[0., 1.]])
dataset = SupervisedMeshGrid(sidelen=max_resolution, domain=domain, gt_path=gt_path, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
model_input, gt_densities = next(iter(dataloader))
if torch.cuda.is_available():
    model_input = model_input.cuda()
    gt_densities = gt_densities.cuda()

with torch.no_grad():
    model.eval()
    density = model(model_input)
    density = density.permute(0, 3, 1, 2)
    if clamp_output:
        density = torch.clamp(density, min=0., max=1.)
    if density.shape != gt_densities.shape:
        gt_densities = torch.nn.functional.interpolate(gt_densities, size=density.shape[2:])
    supervised_loss = criterion(density, gt_densities)


# visualization
density = density.cpu()
grid_title = ''.join(str(i)+'x' for i in max_resolution)[:-1]
maxVolume_np = maxVolume.detach().cpu().numpy()
title = 'testPretrained_s{}_{}_Vol{}_Clmp-{}'.format(model.scale, grid_title, maxVolume_np, clamp_output)
visualizations.density_vis(density, supervised_loss, max_resolution, title, True, visualize, True,
                            binary_loss=-1, path=images_path)
sys.stderr.write('Test image saved to: {}\n'.format(images_path))

# utils.save_densities(density, gridDimensions, title, save_density, True, path='logs/densities/fourfeat_multires/')
