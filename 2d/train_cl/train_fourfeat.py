import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import MLP
from fem import VoxelFEMFunction
import fem
from utils import MeshGrid
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

# PyTorch related global variables
torch.autograd.set_detect_anomaly(False)

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = True
save_density = False
interpolation = False
fixed_scale = True
fixed_resolution = True

problem_path = 'problems/2d/mbb_beam.json'  # TODO
gridDimensions = [300, 100]  # TODO
max_resolution = [300, 100]  # TODO

# record runtime
start_time = time.perf_counter()

# multires hyperparameters for single res
volume_constraint_satisfier = 'constrained_sigmoid'  # TODO
# using filtering as post processing after each iteration (e.g. does not affect if combined with constraint satisfaction)
## (0<...<1, _, True|False) means (no update, _, usage) filters respectiveadaptive_filtering_configs
adaptive_filtering_configs = {}  # TODO
adaptive_filtering_configs['projection_filter'] = False
adaptive_filtering_configs['beta_init'] = 1  ##################### TODO
adaptive_filtering_configs['beta_interval'] = 0.1  ################## TODO
adaptive_filtering_configs['beta_scaler'] = 1  ################## TODO
adaptive_filtering_configs['smoothing_filter'] = False
adaptive_filtering_configs['radius_init'] = 1
adaptive_filtering_configs['radius_interval'] = 0.1
adaptive_filtering_configs['radius_scaler'] = -1
adaptive_filtering_configs['gaussian_filter'] = True
adaptive_filtering_configs['sigma_init'] = 1  ###################### TODO
adaptive_filtering_configs['sigma_interval'] = 0.1  ################ TODO
adaptive_filtering_configs['sigma_scaler'] = 1  #################### TODO

is_volume_constraint_satisfier_hard = fem.type_of_volume_constaint_satisfier(mode=volume_constraint_satisfier)
weight_decay = 0
use_scheduler = False
forgetting_weights = 'orthogonal'  # TODO
forgetting_activations = None  # TODO
rate = 0.7  # TODO
res_order = 'ftc'
repeat_res = 10  # TODO
epoch_mode = 'constant'
# hyper parameter of positional encoding in NeRF
if not fixed_scale:
    interval_scale = 0.5
    scale = np.arange(60) * interval_scale + interval_scale
else:
    scale = [30.0]  # TODO
resolutions = multires_utils.prepare_resolutions(interval=0, start=0, end=1, order=res_order, repeat_res=repeat_res)  # TODO
if (forgetting_weights is None) and (forgetting_activations is None):  # no forgetting case
    resolutions = resolutions[:-1]
    rate = 0.0
    repeat_res = 1
epoch_sizes = multires_utils.prepare_epoch_sizes(n_resolutions=len(resolutions), # TODO
                                                 start=300, end=2500, 
                                                 mode=epoch_mode, constant_value=600)
mrconfprint = 'resolution order: {}, epoch mode: {}\n'.format(res_order, epoch_mode)
mrconfprint += 'forgetting_weights: {}, forgetting_activations: {}, rate: {}\n'.format(forgetting_weights,
                                                                                       forgetting_activations,
                                                                                       rate)
mrconfprint += 'repeat resolutions: {} times \n'.format(repeat_res)
mrconfprint += 'adaptive filtering configs: {} \n'.format(adaptive_filtering_configs)
mrconfprint += 'Volume constraint satisfier: {} (hard: {})\n'.format(volume_constraint_satisfier,
                                                                     is_volume_constraint_satisfier_hard)
sys.stderr.write(mrconfprint)

# create experiments folders for each run  
log_base_path = 'logs/'
log_image_path = '{}images/fourfeat/'.format(log_base_path)
log_loss_path =  '{}loss/fourfeat/'.format(log_base_path)
log_weights_path =  '{}weights/fourfeat/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, None, 
                                                experiment_id=args.jid)
log_image_path = '{}images/fourfeat/{}'.format(log_base_path, append_path)
log_loss_path =  '{}loss/fourfeat/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path: {}\n'.format(log_image_path, log_loss_path))

embedding_size = 256
sys.stderr.write('scale: {}, fourier embedding size: {}\n'.format(scale, embedding_size))

with open(problem_path, 'r') as j:
     configs = json.loads(j.read())

# hyperparameters of the problem 
problem_name = configs['problem_name']
MATERIAL_PATH = configs['MATERIAL_PATH']
BC_PATH = configs['BC_PATH']
orderFEM = configs['orderFEM']
domainCorners = configs['domainCorners']
if gridDimensions is None:
    gridDimensions = configs['gridDimensions']
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
if is_volume_constraint_satisfier_hard:
    nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=embedding_size,
                     scale=scale[0], hidden_act=nn.ReLU(), output_act=None)
else:
    nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=embedding_size,
                     scale=scale[0], hidden_act=nn.ReLU(), output_act=nn.Sigmoid())
model = nerf_model
if torch.cuda.is_available():
    model.cuda()

sys.stderr.write('Deep learning model config: {}\n'.format(model))

# filtering
projection_filter = filtering.ProjectionFilter(beta=adaptive_filtering_configs['beta_init'])
smoothing_filter = filtering.SmoothingFilter(radius=adaptive_filtering_configs['radius_init'])
gauss_smoothing_filter = filtering.GaussianSmoothingFilter(sigma=adaptive_filtering_configs['sigma_init'])
filters = [projection_filter, smoothing_filter, gauss_smoothing_filter]

if weight_decay > 0:
    learning_rate = 1e-3
else:
    learning_rate = 1e-4

optim = torch.optim.Adam(lr=learning_rate, params=itertools.chain(list(model.parameters())), weight_decay=weight_decay)
# reduce on plateau
scheduler = None
if use_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='min', patience=20)
sys.stderr.write('DL optim: {}, LR scheduler: {}\n'.format(optim, scheduler))
sys.stderr.write('L2 Regularization: {}\n'.format(weight_decay))

# training
batch_size = 1
actual_steps = 0  # overall number of iterations over all resolutions
compliance_loss_array = []

for idx, res in enumerate(resolutions):
    for s in scale:

        model.train()

        gridDimensions = tuple(np.array(gridDimensions_) + res * np.array(domainCorners[1]))  # type: ignore
        sys.stderr.write('New resolution within loop: {}\n'.format(gridDimensions))

        if torch.cuda.is_available():
            maxVolume = maxVolume.cuda()

        # deep learning modules
        dataset = MeshGrid(sidelen=gridDimensions, domain=domain, flatten=False)
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
        model_input = next(iter(dataloader))

        # we dont want to update input in nerf so dont enable grads here
        if torch.cuda.is_available():
            model_input = model_input.cuda()

        # topopt (via VoxelFEM-Optimization-Problem)
        constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
        uniformDensity = maxVolume
        tps = initializeTensorProductSimulator(
            orderFEM, domainCorners, gridDimensions, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH
        )                                                                                  
        objective = pyVoxelFEM.ComplianceObjective(tps)                                    
        top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, []) 

        # instantiate autograd.Function for VoxelFEM engine
        voxelfem_engine = VoxelFEMFunction.apply

        # save loss values for plotting
        compliance_loss_array_res = []

        # apply 'fogetting' for weights
        if forgetting_weights is not None:
            multires_utils.forget_weights(model=model, rate=rate, mode=forgetting_weights,
                                          n_neurons=256, embedding_size=embedding_size)
            sys.stderr.write('Weight forgetting has been applied. \n')
        
        # apply 'forgetting' for activations
        if forgetting_activations is not None:
            model.register_gated_activations(model_input, rate=rate)
            sys.stderr.write('Activation forgetting has been applied. \n')

        # reset adaptive filtering
        filtering.reset_adaptive_filtering(filters=filters, configs=adaptive_filtering_configs)

        # training of xPhys
        for step in tqdm(range(epoch_sizes[idx]), desc='Training: '):

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

                global actual_steps
                actual_steps += 1

                # for 'soft' volume constraint 
                if not is_volume_constraint_satisfier_hard:
                    volume_loss = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume,
                                                                compliance_loss=compliance_loss, scaler_mode='clip', constant=500,
                                                                mode=volume_constraint_satisfier)
                    sys.stderr.write('\n{} with mode: {} with constant: {} -> v-loss={}\n'.format(volume_constraint_satisfier,
                                                                                'clip', 500, volume_loss.clone().detach().item()))
                    compliance_loss = compliance_loss + volume_loss

                compliance_loss.backward()

                # reduce LR if no reach plateau
                if use_scheduler:
                    scheduler.step(compliance_loss)

                # save loss values for plotting
                compliance_loss_array_res.append(compliance_loss.detach().item())
                sys.stderr.write("Total Steps: %d, Resolution Steps: %d, Compliance loss %0.6f" % (actual_steps, step, compliance_loss))

                return compliance_loss

            optim.step(closure)

        compliance_loss_array.extend(compliance_loss_array_res)

        # test model with FORGETTING ENABLED
        density = model(model_input)
        if is_volume_constraint_satisfier_hard:
            density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                    mode=volume_constraint_satisfier, projection=projection_filter)
        else: 
            density = torch.clamp(density, min=0., max=1.)

        # loss of conversion to binary by thresholding
        binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top,
                                                                      loss_engine=voxelfem_engine)

        # visualization and saving model
        grid_title = ''.join(str(i)+'x' for i in gridDimensions)[:-1]
        adaptive_filtering_configs_title = ''.join(str(i)+';' for i in adaptive_filtering_configs.values())[:-1]
        maxVolume_np = maxVolume.detach().cpu().numpy()
        forgetting_title = 'None'
        if forgetting_weights is not None:
            forgetting_title = 'forget_W{}'.format(forgetting_weights)
        if forgetting_activations is not None:
            forgetting_title = 'forget_A{}'.format(forgetting_activations)

        title = 'FF(wFgt_HC'+str(is_volume_constraint_satisfier_hard)+')_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(idx+1)+'x'+str(actual_steps)
        title +=  '_'+problem_name+'_Vol'+str(maxVolume_np)+'_F'+adaptive_filtering_configs_title
        title = visualizations.loss_vis(compliance_loss_array_res, title, True, path=log_loss_path)
        visualizations.density_vis(density, compliance_loss_array_res[-1], gridDimensions, title, True, visualize, True,
                                   binary_loss=binary_compliance_loss, path=log_image_path)

        # test model with FORGETTING DISABLED
        if forgetting_activations is not None:
            model.eval()        
  
        # test model with weight FORGETTING DISABLED (INVALID operation! omitted)
        if forgetting_weights is not None:
            sys.stderr.write('Disabling forgetting and density plot have been omitted (does not make sense) \n')
            sys.stderr.write('Instead, we will plot densities queried for max resolution.\n')

        density = model(model_input)
        if is_volume_constraint_satisfier_hard:
            density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                    mode=volume_constraint_satisfier, projection=projection_filter)
        else:
            density = torch.clamp(density, min=0., max=1.)

        # loss of conversion to binary by thresholding
        binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top,
                                                                      loss_engine=voxelfem_engine)

        if forgetting_activations is not None:
            title = 'FF(woFgt)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(idx+1)+'x'+str(actual_steps)
            title +=  '_'+problem_name+'_Vol'+str(maxVolume_np)+'_F'+adaptive_filtering_configs_title
            title = visualizations.loss_vis(compliance_loss_array_res, title, True, path=log_loss_path)
            visualizations.density_vis(density, compliance_loss_array_res[-1], gridDimensions, title, True, visualize, True,
                                       binary_loss=binary_compliance_loss, path=log_image_path)
        
        if forgetting_weights is not None:
            # now query for max resolution
            test_resolution = max_resolution
            dataset = MeshGrid(sidelen=test_resolution, domain=domain, flatten=False)
            dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
            model_input = next(iter(dataloader))
            if torch.cuda.is_available():
                model_input = model_input.cuda()

            # topopt (via VoxelFEM-Optimization-Problem)
            constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
            uniformDensity = maxVolume
            tps = initializeTensorProductSimulator(
                orderFEM, domainCorners, test_resolution, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH
            )                                                                                  
            objective = pyVoxelFEM.ComplianceObjective(tps)                                    
            top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, []) 
            # maxVolume = torch.tensor(maxVolume)
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

            title = 'FF(wFgt_HC'+str(is_volume_constraint_satisfier_hard)+'_max)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(idx+1)+'x'+str(actual_steps)
            title +=  '_'+problem_name+'_Vol'+str(maxVolume_np)+'_F'+adaptive_filtering_configs_title
            visualizations.density_vis(density, compliance_loss, max_resolution, title, True, visualize, True,
                                       binary_loss=binary_compliance_loss, path=log_image_path)


# recording run time
execution_time = time.perf_counter() - start_time
sys.stderr.write('Overall runtime: {}\n'.format(execution_time))

# now query for max resolution
test_resolution = max_resolution
dataset = MeshGrid(sidelen=test_resolution, domain=domain, flatten=False)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
model_input = next(iter(dataloader))
if torch.cuda.is_available():
    model_input = model_input.cuda()

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
voxelfem_engine = VoxelFEMFunction.apply

# test model without activation FORGETTING ENABLED
if forgetting_activations is not None:
    pass

# test model without weight FORGETTING ENABLED
if forgetting_weights is not None:
    pass

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
title = 'FF(woFgt_HC'+str(is_volume_constraint_satisfier_hard)+'_test)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(actual_steps)
title +=  '_'+problem_name+'_Vol'+str(maxVolume_np)+'_F'+adaptive_filtering_configs_title
visualizations.density_vis(density, compliance_loss, max_resolution, title, True, visualize, True,
                           binary_loss=binary_compliance_loss, path=log_image_path)
title = 'FF(woFgt_HC'+str(is_volume_constraint_satisfier_hard)+'_overall)_s'+str(scale)+'_'+forgetting_title+str(rate)+'_'+grid_title+'_'+str(actual_steps)
title +=  '_'+problem_name+'_Vol'+str(maxVolume_np)+'_F'+adaptive_filtering_configs_title
compliance_loss_array.append(compliance_loss)
title = visualizations.loss_vis(compliance_loss_array, title, True, path=log_loss_path)

utils.save_weights(model, append_path[:-1] if args.jid is None else args.jid, save_model, path=log_weights_path)

# utils.save_densities(density, gridDimensions, title, save_density, True, path='logs/densities/fourfeat_multires/')
