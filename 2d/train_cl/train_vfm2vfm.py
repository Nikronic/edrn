import os, sys
import json
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import MLP
from fem import ground_truth_topopt
import multires_utils
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
parser.add_argument('--jid', help='Slurm job id to name experiment dirs with it')
args=parser.parse_args()

# folders
log_base_path = 'logs/'
log_image_path = '{}images/nn2vfm/'.format(log_base_path)
log_loss_path =  '{}loss/nn2vfm/'.format(log_base_path)
log_weights_path =  '{}weights/nn2vfm/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, None, 
                                                experiment_id=args.jid)
log_image_path = '{}images/nn2vfm/{}'.format(log_base_path, append_path)
log_loss_path =  '{}loss/nn2vfm/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path: {}\n'.format(log_image_path, log_loss_path))

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_density = False

# problem and its resolutions
problem_path = 'problems/2d/mbb_beam.json'  # TODO
experiment_path = 'gt/3928276/voxelfem_750x250_1500_MbbBeamS88_Vol0.3_F[125, 2, 1, 1]_gt.npy'  # TODO
is_sat = False  # TODO 
dnn_resolution = [750, 250]  # TODO
voxelfem_resolution = [1500, 500]  # TODO
voxelfem_max_iter = 150  # TODO
interpolate = True  # TODO

# load density field
density_path = 'logs/densities/{}'.format(experiment_path)
density = np.load(density_path)
density = -torch.from_numpy(density.transpose(1, 0)).float().unsqueeze(0).unsqueeze(0)
sys.stderr.write('Loaded densities with experiment ID and path: {} loaded.\n'.format(experiment_path))

mrconfprint = 'Use saved denisties as init for VoxelFEM at : {}\n'.format(voxelfem_resolution)
mrconfprint += 'Interpolation: {}\n'.format(interpolate)
sys.stderr.write(mrconfprint)

# hyperparameters of the problem 
with open(problem_path, 'r') as j:
     configs = json.loads(j.read())

problem_name = configs['problem_name']
MATERIAL_PATH = configs['MATERIAL_PATH']
BC_PATH = configs['BC_PATH']
orderFEM = configs['orderFEM']
domainCorners = configs['domainCorners']
if voxelfem_resolution is None:
    voxelfem_resolution = configs['gridDimensions']
E0 = configs['E0']
Emin = configs['Emin']
SIMPExponent = configs['SIMPExponent']
maxVolume = torch.tensor(configs['maxVolume'])
seed = configs['seed']
sys.stderr.write('VoxelFEM problem configs: {}\n\n'.format(configs))

# reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# record full runtime
start_time = time.perf_counter()

# now query for test resolution
if interpolate:
    dnn_resolution = voxelfem_resolution
    density = torch.nn.functional.interpolate(density, size=tuple(dnn_resolution), mode='bilinear', align_corners=False)
binary_density = (density > 0.5) * 1.

# topopt (via VoxelFEM-Optimization-Problem)
sys.stderr.write('Start --> Testing DNN at desired resolution... \n')
constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
uniformDensity = maxVolume
tps = initializeTensorProductSimulator(
    orderFEM, domainCorners, dnn_resolution, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH
)                                                                                  
objective = pyVoxelFEM.ComplianceObjective(tps)                                    
top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, [])
voxelfem_engine = fem.VoxelFEMFunction.apply
compliance_loss = voxelfem_engine(density.flatten(), top)
sys.stderr.write('Compliance loss: {} and volume constraint={}\n'.format(compliance_loss, density.mean()))
binary_compliance_loss = voxelfem_engine(binary_density.flatten(), top)
sys.stderr.write('Binary Compliance loss: {} and binary volume constraint={}\n'.format(binary_compliance_loss, binary_density.mean()))

# visualization
grid_title = ''.join(str(i)+'x' for i in dnn_resolution)[:-1]
maxVolume_np = maxVolume.detach().cpu().numpy()
expp_title = experiment_path[experiment_path.find('/')+1: experiment_path.rfind('/')]
title = 'testSaved-{}_{}_Vol{}_intpol-{}_'.format(expp_title, grid_title, maxVolume_np, interpolate)
visualizations.density_vis(density, compliance_loss, dnn_resolution, title, False, visualize, True,
                           binary_loss=binary_compliance_loss, path=log_image_path)
sys.stderr.write('End --> Testing DNN at desired resolution. \n ######################### \n')

sys.stderr.write('Start --> Testing VoxelFEM at desired resolution... \n')
# train VoxelFEM initilized with result of deep neural network at `voxelfem_resolution`
vfem_density, vfem_loss, vfem_binary_loss = ground_truth_topopt(MATERIAL_PATH=MATERIAL_PATH, BC_PATH=BC_PATH, 
                                                                orderFEM=orderFEM, domainCorners=domainCorners,
                                                                gridDimensions=dnn_resolution, SIMPExponent=SIMPExponent,
                                                                maxVolume=maxVolume_np, adaptive_filtering=None,
                                                                max_iter=voxelfem_max_iter, init=density)
sys.stderr.write('Compliance loss: {} and volume constraint={}\n'.format(vfem_loss, vfem_density.mean()))
title = 'testVoxelFEM-{}_{}_Vol{}_intpol-{}_'.format(expp_title, grid_title, maxVolume_np, interpolate)
visualizations.density_vis(vfem_density, vfem_loss, dnn_resolution, title, False, visualize, True,
                            binary_loss=vfem_binary_loss, path=log_image_path)
sys.stderr.write('End --> Testing VoxelFEM at desired resolution. \n')

# ending reports
sys.stderr.write('Test images saved to: {}\n'.format(log_image_path))
execution_time = time.perf_counter() - start_time
sys.stderr.write('\nOverall runtime: {}\n'.format(execution_time))
