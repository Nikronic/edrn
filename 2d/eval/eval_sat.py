import os, sys
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import MLP
import utils
import fem
import visualizations

import numpy as np

import matplotlib as mpl
mpl.use('Agg')

sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore
import MeshFEM, mesh  # type: ignore
from ipopt_helpers import initializeTensorProductSimulator, problemObjectWrapper, initializeIpoptProblem  # type: ignore


parser=argparse.ArgumentParser()
parser.add_argument('--expp', help='Path to the pretrained weights (result will be saved in "images" with same path)', 
                    default='fourfeat_cl/4199517.pt')
args=parser.parse_args()
experiment_path = args.expp
sys.stderr.write('Loading pretrained weight with experiment ID and path: {}\n'.format(experiment_path))

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = False
save_density = False
fixed_scale = True

problem_path = 'problems/2d/mbb_beam.json'  # TODO
test_resolution = [300, 100]  # TODO
max_resolution = [300, 100]  # TODO
interpolate = False  # TODO

# deep learning modules
nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=256,
                 scale=1., hidden_act=nn.ReLU(), output_act=None)
model = nerf_model

# load pretrained weights
weights_path = 'logs/weights/{}'.format(experiment_path)
images_path = weights_path.replace('weights', 'images')[:-3]+'/'

####################################################################
images_path = 'tmp/'
####################################################################

utils.load_weights(model, weights_path)
if torch.cuda.is_available():
    model.cuda()

mrconfprint = 'Testing pretrained model in: {} \n'.format(max_resolution)
mrconfprint += 'Interpolation: {}'.format(interpolate)
sys.stderr.write(mrconfprint)

# hyperparameters of the problem 
with open(problem_path, 'r') as j:
     configs = json.loads(j.read())

problem_name = configs['problem_name']
MATERIAL_PATH = configs['MATERIAL_PATH']
BC_PATH = configs['BC_PATH']
orderFEM = configs['orderFEM']
domainCorners = configs['domainCorners']
if test_resolution is None:
    test_resolution = configs['gridDimensions']
E0 = configs['E0']
Emin = configs['Emin']
SIMPExponent = configs['SIMPExponent']
maxVolume = torch.tensor(configs['maxVolume'])
seed = configs['seed']
sys.stderr.write('VoxelFEM problem configs: {}\n'.format(configs))

# reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# query for max resolution
domain = np.array([[0., 300.],[0., 100.]])
dataset = utils.MeshGrid(sidelen=test_resolution, domain=domain, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
x = next(iter(dataloader))
# prepare D, B, C, A as the four corners of reconstructing i(x) from I(x) where I(x)=i(x).cumsum(...) 
step_x, step_y = 1 / (np.array(test_resolution).astype(np.float32) - 1)
D = x + 0.5 * torch.tensor([+step_x, +step_y])  # bottom right
B = x + 0.5 * torch.tensor([+step_x, -step_y])  # top right
C = x + 0.5 * torch.tensor([-step_x, +step_y])  # bottom left
A = x + 0.5 * torch.tensor([-step_x, -step_y])  # top left
if torch.cuda.is_available():
    x = x.cuda()
    D, B, C, A = D.cuda(), B.cuda(), C.cuda(), A.cuda()
    maxVolume = maxVolume.cuda()

with torch.no_grad():
    model.eval()
    sum_D = model(D)
    sum_B = model(B)
    sum_C = model(C)
    sum_A = model(A)

    density = sum_D - sum_B - sum_C + sum_A
    density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                            mode='constrained_sigmoid')

# now query for test resolution
if interpolate:
    test_resolution = max_resolution
    density = density.permute(0, 3, 1, 2)
    density = torch.nn.functional.interpolate(density, size=tuple(test_resolution), mode='bilinear', align_corners=False)

# topopt (via VoxelFEM-Optimization-Problem)
constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
uniformDensity = maxVolume
tps = initializeTensorProductSimulator(
    orderFEM, domainCorners, test_resolution, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH
)                                                                                  
objective = pyVoxelFEM.ComplianceObjective(tps)                                    
top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, [])
if torch.cuda.is_available():
    maxVolume = maxVolume.cuda()
voxelfem_engine = fem.VoxelFEMFunction.apply
density = density.cpu()
compliance_loss = voxelfem_engine(density.flatten(), top)
sys.stderr.write('Compliance loss: {} and volume constraint={}\n'.format(compliance_loss, density.mean()))
binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top, loss_engine=voxelfem_engine)

# visualization
grid_title = ''.join(str(i)+'x' for i in test_resolution)[:-1]
maxVolume_np = maxVolume.detach().cpu().numpy()
expp_title = args.expp[args.expp.rfind('/')+1:-3]
title = 'testPretrained-{}_s{}_{}_Vol{}_intpol-{}_'.format(expp_title, model.scale, grid_title, maxVolume_np, interpolate)
visualizations.density_vis(density, compliance_loss, test_resolution, title, True, visualize, True,
                            binary_loss=binary_compliance_loss, path=images_path)
sys.stderr.write('Test image saved to: {}\n'.format(images_path))
