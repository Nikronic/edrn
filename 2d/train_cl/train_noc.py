# NOC: Non-Overlapping Coordinates
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import MLP
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

# use_gpu = True
# def f():
#     return use_gpu
# torch.cuda.is_available = f

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
fixed_resolution = False

problem_path = 'problems/2d/mbb_beam.json'  # TODO
gt_paths = [
            'logs/densities/gt/3917043/voxelfem_150x50_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            'logs/densities/gt/3917043/voxelfem_210x70_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            # 'logs/densities/gt/3917043/voxelfem_300x100_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            'logs/densities/gt/3917043/voxelfem_375x125_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            'logs/densities/gt/3917043/voxelfem_435x145_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            'logs/densities/gt/3917044/voxelfem_510x170_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            # 'logs/densities/gt/3917046/voxelfem_600x200_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            'logs/densities/gt/3917046/voxelfem_705x235_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            # 'logs/densities/gt/3917047/voxelfem_810x270_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            # 'logs/densities/gt/3917048/voxelfem_990x330_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            # 'logs/densities/gt/3917049/voxelfem_1110x370_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            # 'logs/densities/gt/3917051/voxelfem_1230x410_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            # 'logs/densities/gt/3917053/voxelfem_1365x455_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            # 'logs/densities/gt/3917056/voxelfem_1500x500_1500_MbbBeamSeed88_Vol0.3_Filt__gt.npy',
            ]
resolutions = [
                [150, 50],
                [210, 70],
                # [300, 100],
                [375, 125],
                [435, 145],
                [510, 170],
                # [600, 200],
                [705, 235],
                # [810, 270],
                # [990, 330],
                # [1110, 370],
                # [1230, 410],
                # [1365, 455],
                # [1500, 500],
                ] 
max_resolution = None  # TODO  # if ``None``, then uses each resolutions' coordinates

if max_resolution is None:
    sys.stderr.write('Method: "ADAPTIVE" coordinates for each resolution.\n')
else:
    sys.stderr.write('Method: "MAX={}" coordinates for each "MASKED" resolution.\n'.format(max_resolution))
sys.stderr.write('Resolutions: {}\n\n'.format(resolutions))

# hyperparameters
epoch_size = 3000  # TODO

clamp_output = False  # TODO
repeat_res = 1  # TODO
weight_decay = 0
use_scheduler = False
# hyper parameter of positional encoding in NeRF
if not fixed_scale:
    interval_scale = 0.5
    scale = np.arange(60) * interval_scale + interval_scale
else:
    scale = [15.0]  # TODO

# create experiments folders for each run  
log_base_path = 'logs/'
log_image_path = '{}images/fourfeat_supervised/'.format(log_base_path)
log_loss_path =  '{}loss/fourfeat_supervised/'.format(log_base_path)
log_weights_path =  '{}weights/fourfeat_supervised/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, None, None, 
                                                experiment_id=args.jid)
log_image_path = '{}images/fourfeat_supervised/{}'.format(log_base_path, append_path)
log_loss_path =  '{}loss/fourfeat_supervised/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path:{}\n'.format(log_image_path, log_loss_path))

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
E0 = configs['E0']
Emin = configs['Emin']
SIMPExponent = configs['SIMPExponent']
maxVolume = torch.tensor(configs['maxVolume'])
seed = configs['seed']
sys.stderr.write('VoxelFEM problem configs: {}\n'.format(configs))

# reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

domain = np.array([[0., 1.],[0., 1.]])
sys.stderr.write('Domain: {}\n'.format(domain))

# deep learning modules
nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=embedding_size,
                    scale=scale[0], hidden_act=nn.ReLU(), output_act=nn.Sigmoid())
model = nerf_model
if torch.cuda.is_available():
    model.cuda()
sys.stderr.write('Deep learning model config: {}\n'.format(model))

if weight_decay > 0:
    learning_rate = 1e-3
else:
    learning_rate = 1e-4

optim = torch.optim.Adam(lr=learning_rate, params=model.parameters(), weight_decay=weight_decay)
# loss
criterion = nn.MSELoss(reduction='mean')
# reduce on plateau
scheduler = None
if use_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='min', patience=20)
sys.stderr.write('Criterion: {} with reduction: {}\n'.format(criterion, criterion.reduction))
sys.stderr.write('DL optim: {}, LR scheduler: {}\n'.format(optim, scheduler))
sys.stderr.write('L2 Regularization: {}\n'.format(weight_decay))

# training
batch_size = 1
actual_steps = 0  # overall number of iterations over all resolutions
supervised_loss_array = []

# record runtime
start_time = time.perf_counter()

# prepare data
if max_resolution is not None:
    dataset = utils.MeshGrid(sidelen=max_resolution, domain=domain, flatten=False)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
    model_input = next(iter(dataloader))
    if torch.cuda.is_available():
        model_input = model_input.cuda()

# prepare gt data
gt_coords_list = []
gt_densities_list = []
gt_mask_list = []
for idx, res in enumerate(resolutions):
    # densities
    dataset = utils.SupervisedMeshGrid(sidelen=res, domain=domain, gt_path=gt_paths[idx], flatten=False)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
    gt_coords, gt_densities = next(iter(dataloader))

    if torch.cuda.is_available():
        gt_densities = gt_densities.cuda()
        if max_resolution is None:
            gt_coords_list.append(gt_coords.cuda())
        else:
            del gt_coords
    gt_densities_list.append(gt_densities)

    if max_resolution is not None:
        # masks
        xs = res[1]
        xl = max_resolution[1]
        scale_factor = xl // xs
        mask_range = [(k * scale_factor * scale_factor * xs, (k+1) * scale_factor * scale_factor * xs - (scale_factor-1) * xs * scale_factor, scale_factor) for k in range(xs * domainCorners[1][0])]
        mask = []
        for k in range(len(mask_range)):
            for j in range(*mask_range[k]):
                mask.append(j)
        gt_mask_list.append(mask)


# training
for step in tqdm(range(epoch_size), desc='Training: '):

    def closure():
        global actual_steps
        actual_steps += 1

        # overall supervised loss over n resolution as the weighted sum with equal weights
        supervised_loss = 0 

        optim.zero_grad()

        if max_resolution is not None:
            density = model(model_input)
            # clamp to prevent negative values (MSE cannot handle it solo)
            if clamp_output:
                density = torch.clamp(density, min=0., max=1.)
            density = density.permute(0, 3, 1, 2).view(max_resolution)

        # iterate over resolutions
        for idx, res in enumerate(resolutions):
            if max_resolution is not None:
                gt_densities = gt_densities_list[idx]
                mask = gt_mask_list[idx]

                masked_density = density.flatten()[mask].view(gt_densities.shape)
                supervised_loss = supervised_loss + criterion(masked_density, gt_densities)
            else:
                gt_densities = gt_densities_list[idx]
                gt_coords = gt_coords_list[idx]

                density = model(gt_coords)
                # clamp to prevent negative values (MSE cannot handle it solo)
                if clamp_output:
                    density = torch.clamp(density, min=0., max=1.)
                density = density.permute(0, 3, 1, 2).view(gt_densities.shape)

                supervised_loss = supervised_loss + criterion(density, gt_densities)

        # reduce LR if no reach plateau
        if use_scheduler:
            scheduler.step(supervised_loss)

        supervised_loss.backward()

        # save loss values for plotting
        supervised_loss_array.append(supervised_loss.detach().item())
        sys.stderr.write("Total Steps: %d, Resolution Steps: %d, Compliance loss %0.6f" % (actual_steps, step, supervised_loss))

        return supervised_loss

    optim.step(closure)

    if ((step+1) % (epoch_size//3) == 0) or (step == epoch_size-1):
        if max_resolution is not None:
            resolutions_ = [max_resolution]
        else:
            resolutions_ = resolutions

        for idx, res in enumerate(resolutions_):
            with torch.no_grad():
                if max_resolution is None:
                    model_input = gt_coords_list[idx]

                density = model(model_input)
                if clamp_output:
                    density = torch.clamp(density, min=0., max=1.)

                # visualization and saving model
                grid_title = ''.join(str(i)+'x' for i in resolutions_[idx])[:-1]
                maxVolume_np = maxVolume.detach().cpu().numpy()
                title = ''
                if clamp_output:
                    title = 'fourFeat_clmp_s'
                title = title+str(scale)+'_'+grid_title+'_'+'x'+str(actual_steps)+'_'+problem_name+'_Vol'+str(maxVolume_np)
                title = visualizations.loss_vis(supervised_loss_array, title, True, path=log_loss_path)
                visualizations.density_vis(density, supervised_loss_array[-1], resolutions_[idx], title, True, visualize, True,
                                            binary_loss=None, path=log_image_path)

utils.save_weights(model, append_path[:-1] if args.jid is None else args.jid, save_model, path=log_weights_path)
# utils.save_densities(density, gridDimensions, title, save_density, True, path='logs/densities/fourfeat_multires/')
