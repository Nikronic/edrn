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

def GPU_OFF():
    return False
torch.cuda.is_available = GPU_OFF

parser=argparse.ArgumentParser()
parser.add_argument('--jid', help='Slurm job id to name experiment dirs with it')
args=parser.parse_args()

# folders
log_base_path = 'logs/'
log_image_path = '{}images/nn2vfm/'.format(log_base_path)
log_loss_path =  '{}loss/nn2vfm/'.format(log_base_path)
log_densities_path = '{}densities/nn2vfm/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, log_densities_path, None, 
                                                experiment_id=args.jid)
log_image_path = '{}images/nn2vfm/{}'.format(log_base_path, append_path)
log_loss_path =  '{}loss/nn2vfm/{}'.format(log_base_path, append_path)
log_densities_path = '{}densities/nn2vfm/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path: {}\n'.format(log_image_path, log_loss_path))

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_density = True

# problem and its resolutions
problem_path = 'problems/2d/mbb_beam.json'  # TODO
experiment_path = 'fourfeat_cl_new/4427683.pt'  # TODO
volume_constraint_satisfier = 'maxed_barrier'
optim = 'OC'  # TODO
dnn_resolution = [300, 100]  # TODO
voxelfem_resolution = [900,300]  # TODO
voxelfem_max_iter = 1500  # TODO
interpolate = False  # TODO
is_sat = False  # TODO

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
utils.load_weights(model, weights_path)
sys.stderr.write('Pretrained weight with experiment ID and path: {} loaded.\n'.format(experiment_path))
mrconfprint = 'Use pretrained model as init for VoxelFEM at : {}\n'.format(voxelfem_resolution)
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

# query for max resolution
if is_sat:
    domain = np.array([[0., 300.],[0., 100.]])
else:
    domain = np.array([[0., 1.],[0., 1.]])
if not interpolate:
    dnn_resolution = voxelfem_resolution
dataset = utils.MeshGrid(sidelen=dnn_resolution, domain=domain, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
x = next(iter(dataloader))
if is_sat:
    # prepare D, B, C, A as the four corners of reconstructing i(x) from I(x) where I(x)=i(x).cumsum(...) 
    step_x, step_y = 1 / (np.array(dnn_resolution).astype(np.float32) - 1)
    D = x + 0.5 * torch.tensor([+step_x, +step_y])  # bottom right
    B = x + 0.5 * torch.tensor([+step_x, -step_y])  # top right
    C = x + 0.5 * torch.tensor([-step_x, +step_y])  # bottom left
    A = x + 0.5 * torch.tensor([-step_x, -step_y])  # top left

with torch.no_grad():
    model.eval()
    if is_sat:
        sum_D = model(D)
        sum_B = model(B)
        sum_C = model(C)
        sum_A = model(A)

        density = sum_D - sum_B - sum_C + sum_A
    else:
        density = model(x)
        if is_volume_constraint_satisfier_hard:
            density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                    mode=volume_constraint_satisfier, )
        else: 
            density = torch.clamp(density, min=0., max=1.)

# now query for test resolution
if interpolate:
    dnn_resolution = voxelfem_resolution
    density = torch.nn.functional.interpolate(density, size=tuple(dnn_resolution), mode='nearest')
    if is_volume_constraint_satisfier_hard:
        density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                mode=volume_constraint_satisfier)
    else: 
        density = torch.clamp(density, min=0., max=1.)

# topopt (via VoxelFEM-Optimization-Problem)
sys.stderr.write('Start --> Testing DNN at desired resolution... \n')
maxVolume = maxVolume.detach().cpu().numpy()
density = density.cpu().flatten()

gt_tps, gt_loss, binary_gt_loss, gt_loss_array = fem.ground_truth_topopt(MATERIAL_PATH, BC_PATH, orderFEM, domainCorners,
                                                          dnn_resolution, SIMPExponent, maxVolume, use_multigrid=False,
                                                          init=density, optimizer=optim, 
                                                          multigrid_levels=None, adaptive_filtering=None, 
                                                          max_iter=0, obj_history=True)
sys.stderr.write('Final step => Compliance loss {:.6f}, Binary Compliance loss {:.6f}, '.format(gt_loss, binary_gt_loss))
sys.stderr.write('Volume constraint={:.6f}\n'.format(density.mean()))

# visualization
grid_title = ''.join(str(i)+'x' for i in dnn_resolution)[:-1]
expp_title = experiment_path[experiment_path.rfind('/')+1:-3]
title = 'testPretrained-{}_s{}_{}_Vol{}_intpol-{}_'.format(expp_title, model.scale, grid_title, maxVolume, interpolate)
visualizations.density_vis(density, gt_loss, dnn_resolution, title, True, visualize, True,
                            binary_loss=binary_gt_loss, path=log_image_path)
sys.stderr.write('End --> Testing DNN at desired resolution. \n ######################### \n')

sys.stderr.write('Start --> Testing VoxelFEM at desired resolution... \n')
# train VoxelFEM initilized with result of deep neural network at `voxelfem_resolution`
gt_tps, gt_loss, binary_gt_loss, gt_loss_array = fem.ground_truth_topopt(MATERIAL_PATH, BC_PATH, orderFEM, domainCorners,
                                                          dnn_resolution, SIMPExponent, maxVolume, use_multigrid=False,
                                                          init=density, optimizer=optim, multigrid_levels=None,
                                                          adaptive_filtering=None,  max_iter=voxelfem_max_iter,
                                                          obj_history=True, title=title,
                                                          log_image_path=log_image_path, log_densities_path=log_densities_path)
sys.stderr.write('Final step => Compliance loss {:.6f}, Binary Compliance loss {:.6f}, '.format(gt_loss, binary_gt_loss))
sys.stderr.write('Volume constraint={}\n'.format(gt_tps.mean()))

######################
error = ((gt_tps.astype(np.float32) - density.numpy()) ** 2).mean()
sys.stderr.write('\nL2 norm of init density and final density: {}\n'.format(error))
######################

# vis and saving densities
title = 'testVoxelFEM-{}_{}_Vol{}_intpol-{}_'.format(expp_title, grid_title, maxVolume, interpolate)
same_size_vis = False
if domainCorners[1][1] / domainCorners[1][0] >= 3:
    same_size_vis = True

visualizations.density_vis(gt_tps, gt_loss, dnn_resolution, title, False, visualize, same_size_vis,
                               binary_loss=binary_gt_loss, path=log_image_path)
visualizations.loss_vis(gt_loss_array, title, visualize, path=log_loss_path, 
                        ylim=np.max(gt_loss_array)+0.1*np.max(gt_loss_array))
utils.save_densities(gt_tps, dnn_resolution, title, save_density, False, path=log_densities_path)
sys.stderr.write('End --> Testing VoxelFEM at desired resolution. \n')

# ending reports
sys.stderr.write('Test images saved to: {}\n'.format(log_image_path))
execution_time = time.perf_counter() - start_time
sys.stderr.write('\nOverall runtime: {}\n'.format(execution_time))

# load density
# z = -np.load('logs/densities/nn2vfm/exp25/testVoxelFEM-4427678_150x50_Vol[0.3]_intpol-False__gt.npy').T.flatten()


# vis
import matplotlib.pyplot as plt

z1 = -np.load('logs/densities/nn2vfm/exp26/testVoxelFEM-4427683_675x225_Vol[0.3]_intpol-False__gt.npy').T.flatten()
zn = -np.load('logs/densities/nn2vfm/4939420/testVoxelFEM-4427683_900x300_Vol[0.3]_intpol-False__gt.npy').T.flatten()
z1 = z1.reshape(voxelfem_resolution)
zn = zn.reshape(voxelfem_resolution)
z = (zn - z1)**2
d = density.numpy().reshape(voxelfem_resolution)
zf = (zn - d)**2
plt.imshow(np.log(z.T), cmap='gray')
plt.imshow(np.log(zf.T), cmap='gray')
plt.savefig('tmp/t.png')
