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
                    default='fourfeat_cl_new/4427682.pt')
args=parser.parse_args()
experiment_path = args.expp
sys.stderr.write('Loading pretrained weight with experiment ID and path: {}\n'.format(experiment_path))


# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = False
save_density = False
fixed_scale = True

problem_path = 'problems/2d/mbb_beam.json'  # TODO
volume_constraint_satisfier = 'maxed_barrier'
test_resolution = [300, 100]  # TODO
max_resolution = [300, 100]  # TODO
interpolate = False  # TODO

mrconfprint = 'Testing pretrained model in: {} \n'.format(max_resolution)
mrconfprint += 'Interpolation: {}'.format(interpolate)
sys.stderr.write(mrconfprint)

# deep learning modules
is_volume_constraint_satisfier_hard = fem.type_of_volume_constaint_satisfier(mode=volume_constraint_satisfier)
if is_volume_constraint_satisfier_hard:
    nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=1024,
                     scale=1, hidden_act=nn.ReLU(), output_act=None)
else:
    nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=1024,
                     scale=1, hidden_act=nn.ReLU(), output_act=nn.Sigmoid())     
model = nerf_model

# load pretrained weights
weights_path = 'logs/weights/{}'.format(experiment_path)
images_path = weights_path.replace('weights', 'images')[:-3]+'/'

####################################################################
# images_path = 'tmp/'
####################################################################

utils.load_weights(model, weights_path)
if torch.cuda.is_available():
    model.cuda()

# hyperparameters of the problem 
with open(problem_path, 'r') as j:
     configs = json.loads(j.read())

problem_name = configs['problem_name']
MATERIAL_PATH = configs['MATERIAL_PATH']
BC_PATH = configs['BC_PATH']
orderFEM = configs['orderFEM']
domainCorners = configs['domainCorners']
test_resolution = test_resolution
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
domain = np.array([[0., 1.],[0., 1.]])
if not interpolate:
    test_resolution = max_resolution
dataset = utils.MeshGrid(sidelen=test_resolution, domain=domain, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
model_input = next(iter(dataloader))
if torch.cuda.is_available():
    model_input = model_input.cuda()
    maxVolume = maxVolume.cuda()

with torch.no_grad():
    model.eval()
    density = model(model_input)
    if is_volume_constraint_satisfier_hard:
        density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                mode=volume_constraint_satisfier)
    else: 
        density = torch.clamp(density, min=0., max=1.)

# now query for test resolution
if interpolate:
    test_resolution = max_resolution
    density = density.permute(0, 3, 1, 2)
    density = torch.nn.functional.interpolate(density, size=tuple(test_resolution), mode='bilinear', align_corners=True)

# topopt (via VoxelFEM-Optimization-Problem)
constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
uniformDensity = maxVolume
tps = initializeTensorProductSimulator(
    orderFEM, domainCorners, test_resolution, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH)
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

import matplotlib.pyplot as plt
ratio = test_resolution[0] / test_resolution[1]
if ratio == 1.0:  
    ratio = ratio * 3.5
same_size=True
if same_size:
    plt.figure(figsize=(ratio * 5, ratio//ratio * 5 + 1))
else:
    plt.figure(figsize=(test_resolution[0]/10, test_resolution[1]/10+1))
pred_density = -density.view(test_resolution).detach().cpu().numpy()[:, :].T
plt.imshow(pred_density, cmap='gray')
plt.axis('off')
plt.savefig(images_path + title + '.svg', bbox_inches='tight')

sys.stderr.write('Test image saved to: {}\n'.format(images_path))
exit()
visualizations.density_vis(density, compliance_loss, test_resolution, title, True, visualize, True,
                            binary_loss=binary_compliance_loss, path=images_path)
sys.stderr.write('Test image saved to: {}\n'.format(images_path))
