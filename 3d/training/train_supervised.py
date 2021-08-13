import os, sys
import torch
import torch.nn as nn
import torch_optimizer
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import MLP
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
import argparse, psutil
import itertools
import copy

# def GPU_OFF():
#     return False
# torch.cuda.is_available = GPU_OFF

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
save_density = True
fixed_scale = True

# record runtime
start_time = time.perf_counter()

# ground truth
gt_path = 'logs/densities/gt/4564333/4564333_voxelfem_optim-OC_80x40x20_5000_bridgeS88_Vol0.4_iter4999_gt.mesh'
gridDimensions = [80, 40, 20]  # TODO
max_resolution = gridDimensions
domainCorners = [[0, 0, 0], [4, 2, 1]]
# gt = utils.load_ct(path=gt_path, shape=gridDimensions, interpolate_size=[128, 128])
gt = utils.load_mesh(path=gt_path, shape=gridDimensions)

# multires hyperparameters
volume_constraint_satisfier = 'maxed_barrier'
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

is_volume_constraint_satisfier_hard = fem.type_of_volume_constaint_satisfier(mode=volume_constraint_satisfier)
embedding_size = 256  # TODO
n_neurons = 128  # TODO
n_layers  = 4  # TODO
learning_rate = 3e-4  # TODO
weight_decay = 0.0
use_scheduler = None
epoch_mode = 'constant'

# hyper parameter of positional encoding in NeRF
scale = [0.005]  # TODO
resolutions = multires_utils.prepare_resolutions(interval=0, start=0, end=1, order='ftc', repeat_res=1)[:-1]  # TODO
epoch_sizes = multires_utils.prepare_epoch_sizes(n_resolutions=len(resolutions), # TODO
                                                 start=800, end=1500, 
                                                 mode=epoch_mode, constant_value=5000)
mrconfprint = 'epoch mode: {}, '.format(epoch_mode)
mrconfprint += 'adaptive filtering configs: {} \n'.format(adaptive_filtering_configs)
mrconfprint += 'Volume constraint satisfier: {} (hard: {})\n'.format(volume_constraint_satisfier,
                                                                     is_volume_constraint_satisfier_hard)
sys.stderr.write(mrconfprint)
sys.stderr.write('scale: {}, fourier embedding size: {}\n'.format(scale, embedding_size))

# create experiments folders for each run  
log_base_path = 'logs/'
log_image_path = '{}images/supervised/'.format(log_base_path)
log_loss_path =  '{}loss/supervised/'.format(log_base_path)
log_weights_path = '{}weights/supervised/'.format(log_base_path)
log_densities_path = '{}densities/supervised/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, log_densities_path,
                                                log_weights_path, None, experiment_id=args.jid)
log_image_path = '{}images/supervised/{}'.format(log_base_path, append_path)
log_loss_path =  '{}loss/supervised/{}'.format(log_base_path, append_path)
log_weights_path =  '{}weights/supervised/{}'.format(log_base_path, append_path)
log_densities_path = '{}densities/supervised/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path: {}\n'.format(log_image_path, log_loss_path))

gridDimensions_ = copy.deepcopy(gridDimensions)
if max_resolution is None:
    max_resolution = gridDimensions

# reproducibility
seed = 88
torch.manual_seed(seed)
np.random.seed(seed)

# domain
domain = np.array([[0., 1.], [0., 1.], [0., 1.]])
sys.stderr.write('Domain: {}\n'.format(domain))

# deep learning modules
projection_filter = filtering.ProjectionFilter(beta=adaptive_filtering_configs['beta_init'], normalized=True)
if is_volume_constraint_satisfier_hard:
    nerf_model = MLP(in_features=3, out_features=1, n_neurons=n_neurons, n_layers=n_layers, embedding_size=embedding_size,
                     scale=scale[0], hidden_act=nn.ReLU(), output_act=None)
else:
    nerf_model = MLP(in_features=3, out_features=1, n_neurons=n_neurons, n_layers=n_layers, embedding_size=embedding_size,
                     scale=scale[0], hidden_act=nn.ReLU(), output_act=nn.Sigmoid())
model = nerf_model
if torch.cuda.is_available():
    model.cuda()

# filtering
smoothing_filter = filtering.SmoothingFilter(radius=adaptive_filtering_configs['radius_init'])
gauss_smoothing_filter = filtering.GaussianSmoothingFilter(sigma=adaptive_filtering_configs['sigma_init'])
filters = [projection_filter, smoothing_filter, gauss_smoothing_filter]

# optim
optim = torch.optim.Adam(lr=learning_rate, params=model.parameters(), weight_decay=weight_decay)
mse_criterion = nn.MSELoss(reduction='sum')

# reduce on plateau
scheduler = None
if use_scheduler == 'reduce_on_plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='min', patience=10)
if use_scheduler == 'multi_step_lr':
    milestones_step = 100
    milestones = [i*milestones_step for i in range(1, 4)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=milestones, gamma=0.3)
sys.stderr.write('DL optim: {}, LR scheduler: {}\n'.format(optim, scheduler))
sys.stderr.write('L2 Regularization: {}\n'.format(weight_decay))

# training
batch_size = 1
actual_steps = 0  # overall number of iterations over all resolutions
mse_loss_array = []

for idx, res in enumerate(resolutions):
    for s in scale:

        model.train()

        gridDimensions = tuple(np.array(gridDimensions_) + res * np.array(domainCorners[1]))  # type: ignore
        sys.stderr.write('New resolution within multires loop: {}\n'.format(gridDimensions))

        # deep learning modules
        dataset = MeshGrid(sidelen=gridDimensions, domain=domain, flatten=False)
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
        model_input = next(iter(dataloader))

        # we dont want to update input in nerf so dont enable grads here
        if torch.cuda.is_available():
            model_input = model_input.cuda()
            gt = gt.cuda()

        # reset adaptive filtering
        filtering.reset_adaptive_filtering(filters=filters, configs=adaptive_filtering_configs)

        # save loss values for plotting
        mse_loss_array_res = []

        ckp_step = epoch_sizes[idx] // 5
        # training of xPhys
        for step in tqdm(range(epoch_sizes[idx]), desc='Training: '):

            def closure():
                optim.zero_grad()

                # aka x
                density = model(model_input)
                density = density.view(gridDimensions)

                # aka xPhys
                if is_volume_constraint_satisfier_hard:
                    pass
                    # density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                    #                                         mode=volume_constraint_satisfier, projection=projection_filter)
                else: 
                    density = torch.clamp(density, min=0., max=1.)
                
                # adaptive filtering
                if adaptive_filtering_configs is not None:
                    density = filtering.apply_filters_group(x=density, filters=filters, configs=adaptive_filtering_configs)
                    filtering.update_adaptive_filtering(iteration=step, filters=filters, configs=adaptive_filtering_configs)

                # compliance for predicted xPhys
                mse_loss = mse_criterion(density, gt)

                global actual_steps
                actual_steps += 1

                mse_loss.backward()

                # save loss values for plotting
                mse_loss = mse_loss.detach().item()
                mse_loss_array_res.append(mse_loss)
                
                if experiment_id is None:
                    tqdm.write('Total Steps: {:d}, Resolution Steps: {:d}, MSE loss {:.6f}'.format(actual_steps,
                                                                                                     step, mse_loss))
                else:
                    sys.stderr.write('Total Steps: {:d}, Resolution Steps: {:d}, MSE loss {:.6f}\n'.format(actual_steps,
                                                                                                     step, mse_loss))
                return mse_loss

            optim.step(closure)

            # reduce LR if no reach plateau
            if use_scheduler is not None:
                if use_scheduler == 'reduce_lr_on_plateau':
                    scheduler.step(mse_loss_array_res[-1])
                else:
                    scheduler.step()

        mse_loss_array.extend(mse_loss_array_res)

        # test model with for res idx
        with torch.no_grad():
            density = model(model_input)
            density = density.view(gridDimensions)
            if is_volume_constraint_satisfier_hard:
                pass
                # density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                #                                         mode=volume_constraint_satisfier, projection=projection_filter)
            else: 
                density = torch.clamp(density, min=0., max=1.)

            # visualization and saving model
            grid_title = ''.join(str(i)+'x' for i in gridDimensions)[:-1]
            title = str(experiment_id)+'_FF(HC'+str(is_volume_constraint_satisfier_hard)+')_s'+str(scale)+'_'+grid_title+'_'+str(idx+1)+'x'+str(actual_steps)
            title = visualizations.loss_vis(mse_loss_array_res, title, True, path=log_loss_path,
                                            ylim=np.max(mse_loss_array_res) if np.max(mse_loss_array_res) < 100000 else 100000)

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

with torch.no_grad():
    model.eval()
    density = model(model_input)
    density = density.view(gridDimensions)
    if is_volume_constraint_satisfier_hard:
        pass
        # density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
        #                                         mode=volume_constraint_satisfier, projection=projection_filter)
    else:
        density = torch.clamp(density, min=0., max=1.)

mse_loss = mse_criterion(density, gt)
maxVolume_np = density.detach().cpu().numpy().mean()
grid_title = ''.join(str(i)+'x' for i in test_resolution)[:-1]
title = str(experiment_id)+'_FF(HC'+str(is_volume_constraint_satisfier_hard)+'_test)_s'+str(scale)+'_'+grid_title+'_'+str(actual_steps)
title += '_Vol'+str(maxVolume_np)
utils.save_for_interactive_vis(density, gridDimensions, title, visualize, path=log_image_path)
title = title.replace('test', 'overall')
mse_loss_array.append(mse_loss)
title = visualizations.loss_vis(mse_loss_array, title, True, path=log_loss_path,
                                ylim=np.max(mse_loss_array) if np.max(mse_loss_array) < 100000 else 100000)
utils.save_weights(model, append_path[:-1] if args.jid is None else args.jid, save_model, path=log_weights_path)
