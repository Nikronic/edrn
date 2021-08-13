import os, sys
import torch
import torch.nn as nn
import torch_optimizer
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import MLP
from fem import VoxelFEMFunction
from utils import MeshGrid
import fem
import utils
import filtering
import multires_utils
import visualizations

import numpy as np

import matplotlib as mpl
mpl.use('Agg')

from tqdm import tqdm
import json, time
import argparse
import itertools
import copy

sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore
import MeshFEM, mesh  # type: ignore
from ipopt_helpers import initializeTensorProductSimulator, problemObjectWrapper, initializeIpoptProblem  # type: ignore


parser=argparse.ArgumentParser()
parser.add_argument('--jid', help='Slurm job id to name experiment dirs with it')
args=parser.parse_args()
experiment_id = args.jid
sys.stderr.write('Experiment ID: {}\n'.format(experiment_id))

# global setting
if experiment_id is None:
    sys.stderr.write = print

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = True
save_density = False
fixed_scale = True

# record runtime
start_time = time.perf_counter()

problem_path = 'problems/2d/mbb_beam.json'  # TODO
gridDimensions = [150, 50]  # TODO
max_resolution = [150, 50]  # TODO

# multires hyperparameters
volume_constraint_satisfier = 'linear'
# using filtering as post processing after each iteration (e.g. does not affect if combined with constraint satisfaction)
## (0<...<1, _, True|False) means (no update, _, usage) filters respectively
adaptive_filtering_configs = {}
adaptive_filtering_configs['projection_filter'] = False
adaptive_filtering_configs['beta_init'] = 1
adaptive_filtering_configs['beta_interval'] = 0.1
adaptive_filtering_configs['beta_scaler'] = -1
adaptive_filtering_configs['smoothing_filter'] = False
adaptive_filtering_configs['radius_init'] = 1
adaptive_filtering_configs['radius_interval'] = 0.1
adaptive_filtering_configs['radius_scaler'] = -1
adaptive_filtering_configs['gaussian_filter'] = False
adaptive_filtering_configs['sigma_init'] = 1
adaptive_filtering_configs['sigma_interval'] = 0.1
adaptive_filtering_configs['sigma_scaler'] = -1
adaptive_filtering_configs['cone_filter'] = False
adaptive_filtering_configs['cone_init'] = 1
adaptive_filtering_configs['cone_interval'] = 0.1
adaptive_filtering_configs['cone_scaler'] = -1

is_volume_constraint_satisfier_hard = fem.type_of_volume_constaint_satisfier(mode=volume_constraint_satisfier)
# hyper parameter of positional encoding in NeRF
scale = [4.0]  # TODO
learning_rate = 0.01  # TODO
n_neurons = 128  # TODO
n_layers = 4  # TODO
embedding_size = 256  # TODO

resolutions = multires_utils.prepare_resolutions(interval=0, start=0, end=1, order='ftc', repeat_res=1)[:-1]
epoch_sizes = multires_utils.prepare_epoch_sizes(n_resolutions=len(resolutions), # TODO
                                                 start=800, end=1500, 
                                                 mode='constant', constant_value=150)
mrconfprint = 'adaptive filtering configs: {} \n'.format(adaptive_filtering_configs)
mrconfprint += 'Volume constraint satisfier: {} (hard: {})\n'.format(volume_constraint_satisfier,
                                                                     is_volume_constraint_satisfier_hard)
sys.stderr.write(mrconfprint)
sys.stderr.write('scale: {}, fourier embedding size: {}\n'.format(scale, embedding_size))

# create experiments folders for each run  
log_base_path = 'logs/'
log_image_path = '{}images/nroc/'.format(log_base_path)
log_loss_path =  '{}loss/nroc/'.format(log_base_path)
log_weights_path =  '{}weights/nroc/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, None, 
                                                experiment_id=args.jid)
log_image_path = '{}images/nroc/{}'.format(log_base_path, append_path)
log_loss_path =  '{}loss/nroc/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path: {}\n'.format(log_image_path, log_loss_path))


# hyperparameters of the topopt problem
with open(problem_path, 'r') as j:
     configs = json.loads(j.read())
problem_name = configs['problem_name']
MATERIAL_PATH = configs['MATERIAL_PATH']
BC_PATH = configs['BC_PATH']
orderFEM = configs['orderFEM']
domainCorners = configs['domainCorners']
if gridDimensions is None:
    gridDimensions = configs['gridDimensions']
else:
    configs['gridDimensions'] = gridDimensions
E0 = configs['E0']
Emin = configs['Emin']
SIMPExponent = configs['SIMPExponent']
maxVolume = torch.tensor(configs['maxVolume'])
if adaptive_filtering_configs is None:
    adaptive_filtering_configs = configs['adaptive_filtering']
seed = configs['seed']
sys.stderr.write('VoxelFEM problem configs: {}\n'.format(configs))

gridDimensions_ = copy.deepcopy(gridDimensions)
if max_resolution is None:
    max_resolution = gridDimensions

# reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    maxVolume = maxVolume.cuda()

domain = np.array([[0., 1.],[0., 1.]])
sys.stderr.write('Domain: {}\n'.format(domain))

# deep learning modules
projection = nn.Sigmoid()
if is_volume_constraint_satisfier_hard:
    nerf_model = MLP(in_features=2, out_features=1, n_neurons=n_neurons, n_layers=n_layers, residuals=None,
                     embedding_size=embedding_size, scale=scale[0], hidden_act=nn.ReLU(), output_act=None)
else:
    nerf_model = MLP(in_features=2, out_features=1, n_neurons=n_neurons, n_layers=n_layers, residuals=None, 
                     embedding_size=embedding_size, scale=scale[0], hidden_act=nn.ReLU(), output_act=projection)
model = nerf_model
if torch.cuda.is_available():
    model.cuda()

# apply homogenization
fem.homogeneous_init(model=model, constant=maxVolume.item(), projection=projection)
sys.stderr.write('Deep learning model config: {}\n'.format(model))

# filtering
projection_filter = filtering.ProjectionFilter(beta=adaptive_filtering_configs['beta_init'], normalized=True)
smoothing_filter = filtering.SmoothingFilter(radius=adaptive_filtering_configs['radius_init'])
gauss_smoothing_filter = filtering.GaussianSmoothingFilter(sigma=adaptive_filtering_configs['sigma_init'])
filters = [projection_filter, smoothing_filter, gauss_smoothing_filter]
 
# training
batch_size = 1
actual_steps = 0  # overall number of iterations over all resolutions
compliance_loss_array = []


# training
model.train()

if torch.cuda.is_available():
    maxVolume = maxVolume.cuda()

# deep learning modules
dataset = MeshGrid(sidelen=gridDimensions, domain=domain, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
model_input = next(iter(dataloader))
# we dont want to update input in nerf so dont enable grads here
if torch.cuda.is_available():
    model_input = model_input.cuda()

# optim
optim = fem.OC(m=learning_rate, params=model.parameters(), model=model, model_input=model_input, max_volume=maxVolume)

# topopt (via VoxelFEM)
constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
uniformDensity = maxVolume
tps = initializeTensorProductSimulator(
    orderFEM, domainCorners, gridDimensions, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH)                                                                                  
objective = pyVoxelFEM.ComplianceObjective(tps)                                    
top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, []) 
# instantiate autograd.Function for VoxelFEM engine
voxelfem_engine = VoxelFEMFunction.apply

# save loss values for plotting
compliance_loss_array_res = []

# reset adaptive filtering
filtering.reset_adaptive_filtering(filters=filters, configs=adaptive_filtering_configs)

# training of xPhys
for step in tqdm(range(epoch_sizes[0]), desc='Training: '):

    def closure():
        optim.zero_grad()

        # aka x
        density = model(model_input)
        density = density.view(gridDimensions)

        # aka xPhys
        if is_volume_constraint_satisfier_hard:
            density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                    mode=volume_constraint_satisfier, projection=projection_filter)
        else: 
            density = torch.clamp(density, min=0., max=1.)
        
        # adaptive filtering
        if adaptive_filtering_configs is not None:
            density = filtering.apply_filters_group(x=density, filters=filters, configs=adaptive_filtering_configs)
            filtering.update_adaptive_filtering(iteration=step, filters=filters, configs=adaptive_filtering_configs)

        # compliance for predicted xPhys
        if torch.cuda.is_available():
            density = density.cpu()
        compliance_loss = voxelfem_engine(density.flatten(), top)
        if torch.cuda.is_available():
            compliance_loss.cuda()

        # populate objective gradients for OC step
        compliance_loss.backward(retain_graph=True)  # retain graph for constraints jacobian population
        optim.populate_objective_grads()
        optim.zero_grad()
        
        # for 'soft' volume constraint 
        # TODO: #62 make it adaptive, first use small values then after 100, 200 iterations, use the larger recommended value
        if not is_volume_constraint_satisfier_hard:
            volume_loss = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume,
                                                        compliance_loss=compliance_loss, 
                                                        scaler_mode='adaptive', constant=None,
                                                        mode=volume_constraint_satisfier)
            sys.stderr.write('\n{} with mode: {} with constant: {} -> v-loss={} \n'.format(volume_constraint_satisfier,
                                                                        'adaptive', 'None', volume_loss.clone().detach().item()))
        
        # populate constraints jacobians for OC step
        volume_loss.backward()
        optim.populate_constraints_jacobian()  # TODO: the grads of volume are already scaled given method above!, so no need for lagrange
        optim.zero_grad()

        # save loss values for plotting
        global actual_steps
        actual_steps += 1
        compliance_loss_array_res.append(compliance_loss.detach().item())
        sys.stderr.write('Total Steps: {:d}, Resolution Steps: {:d}, Compliance loss {:0.6f}'.format(actual_steps, step, compliance_loss))
        return compliance_loss

    optim.step(closure)

compliance_loss_array.extend(compliance_loss_array_res)

# test model
with torch.no_grad():
    model.eval()
    density = model(model_input)
    if is_volume_constraint_satisfier_hard:
        density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                mode=volume_constraint_satisfier, projection=projection_filter)
    else:
        density = torch.clamp(density, min=0., max=1.)

    # loss of conversion to binary by thresholding
    binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top,
                                                                    loss_engine=voxelfem_engine)
    if torch.cuda.is_available():
        density = density.cpu()
    compliance_loss = voxelfem_engine(density.flatten(), top)

    maxVolume_np = maxVolume.detach().cpu().numpy()
    grid_title = ''.join(str(i)+'x' for i in gridDimensions)[:-1]
    title = 'FF(HC'+str(is_volume_constraint_satisfier_hard)+')_s'+str(scale)+'_'+grid_title+'x'+str(step)
    title +=  problem_name+'_Vol'+str(maxVolume_np)
    visualizations.density_vis(density, compliance_loss, max_resolution, title, True, visualize, True,
                                binary_loss=binary_compliance_loss, path=log_image_path)

# recording run time
execution_time = time.perf_counter() - start_time
sys.stderr.write('Overall runtime: {}\n'.format(execution_time))

# now query for max resolution after training finished
test_resolution = max_resolution
dataset = MeshGrid(sidelen=test_resolution, domain=domain, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
model_input = next(iter(dataloader))
if torch.cuda.is_available():
    model_input = model_input.cuda()

# topopt (via VoxelFEM-Optimization-Problem)
constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
uniformDensity = maxVolume
SIMPExponent = 3
tps = initializeTensorProductSimulator(
    orderFEM, domainCorners, test_resolution, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH)
objective = pyVoxelFEM.ComplianceObjective(tps)                                    
top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, []) 
if torch.cuda.is_available():
    maxVolume = maxVolume.cuda()
voxelfem_engine = VoxelFEMFunction.apply

with torch.no_grad():
    model.eval()
    density = model(model_input)
    if is_volume_constraint_satisfier_hard:
        density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                mode=volume_constraint_satisfier, projection=projection_filter)
    else:
        density = torch.clamp(density, min=0., max=1.)

# loss of conversion to binary by thresholding
binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top,
                                                              loss_engine=voxelfem_engine)
if torch.cuda.is_available():
    density = density.cpu()
compliance_loss = voxelfem_engine(density.flatten(), top)
maxVolume_np = maxVolume.detach().cpu().numpy()
grid_title = ''.join(str(i)+'x' for i in test_resolution)[:-1]
adaptive_filtering_configs_title = ''.join(str(i)+';' for i in adaptive_filtering_configs.values())[:-1]
title = 'FF(HC'+str(is_volume_constraint_satisfier_hard)+'_test)_s'+str(scale)+'_'+grid_title+'_'+str(actual_steps)
title +=  problem_name+'_Vol'+str(maxVolume_np)+'_F'+adaptive_filtering_configs_title
visualizations.density_vis(density, compliance_loss, max_resolution, title, True, visualize, True,
                           binary_loss=binary_compliance_loss, path=log_image_path)
title = 'FF(HC'+str(is_volume_constraint_satisfier_hard)+'_overall)_s'+str(scale)+'_'+grid_title+'_'+str(actual_steps)
title +=  problem_name+'_Vol'+str(maxVolume_np)+'_F'+adaptive_filtering_configs_title
compliance_loss_array.append(compliance_loss)
title = visualizations.loss_vis(compliance_loss_array, title, True, path=log_loss_path,
                                ylim=np.max(compliance_loss_array_res))

utils.save_weights(model, append_path[:-1] if args.jid is None else args.jid, save_model, path=log_weights_path)

# utils.save_densities(density, gridDimensions, title, save_density, True, path='logs/densities/fourfeat_multires/')
