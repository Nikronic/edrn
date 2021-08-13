import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from networks import MLP
from fem import physical_density
from utils import SupervisedMeshGrid
import utils
import multires_utils
import visualizations

import numpy as np


import matplotlib as mpl
mpl.use('Agg')

from tqdm import tqdm
import json
import os, sys
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
clamp_output = True  # TODO

problem_path = 'problems/2d/mbb_beam.json'  # TODO
gt_path = 'logs/densities/gt/3879060/voxlfem_1500x500_2000_MbbBeamSeed88_Vol0.3_Filt__gt.npy'  # TODO
# max resolution assumption - all other solutions are coarser than this
max_resolution = [1500, 500]  # TODO 


# multires hyperparameters 
use_scheduler = False
res_order = 'ftc'
epoch_mode = 'constant'
resolutions = multires_utils.prepare_resolutions(interval=0, start=0, end=1, order=res_order)  # TODO
epoch_sizes = multires_utils.prepare_epoch_sizes(n_resolutions=len(resolutions), # TODO
                                                 start=300, end=2000, 
                                                 mode=epoch_mode, constant_value=20000)
sys.stderr.write('resolution order: {}, epoch mode: {}, clamp: {}\n'.format(res_order, epoch_mode, clamp_output))

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

# hyper parameter of positional encoding in NeRF
if not fixed_scale:
    interval_scale = 0.5
    scale = np.arange(60) * interval_scale + interval_scale
else:
    scale = [50.0]  # TODO

embedding_size = 256
sys.stderr.write('scale: {}, fourier embedding size:{}\n'.format(scale, embedding_size))

with open(problem_path, 'r') as j:
     configs = json.loads(j.read())

# hyperparameters of the problem 
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
sys.stderr.write('VoxelFEM problem configs: {}\n'.format(configs))

gridDimensions_ = copy.deepcopy(gridDimensions)

if max_resolution is None:
    max_resolution = gridDimensions
else:
    gridDimensions_ = max_resolution

# reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    maxVolume = maxVolume.cuda()

domain = np.array([[0., 1.],[0., 1.]])
sys.stderr.write('Domain: {}\n'.format(domain))

# deep learning modules
# coordinate to density
nerf_model = MLP(in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=embedding_size,
                 scale=scale[0], hidden_act=nn.ReLU(), output_act=None)
              
model = nerf_model
sys.stderr.write('Deep learning model config: {}\n'.format(model))

if torch.cuda.is_available():
    model.cuda()

optim = torch.optim.Adam(lr=1e-4, params=itertools.chain(list(model.parameters())))
# loss
criterion = nn.MSELoss(reduction='sum')
# reduce on plateau
scheduler = None
if use_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='min', patience=20)
sys.stderr.write('DL optim: {}, Criterion: {}, LR scheduler: {}\n'.format(optim, criterion, scheduler))

# training
batch_size = 1
actual_steps = 0  # overall number of iterations over all resolutions
supervised_loss_array = []

for idx, res in enumerate(resolutions):
    for s in scale:

        gridDimensions = tuple(np.array(gridDimensions_) + res * np.array(domainCorners[1]))
        sys.stderr.write('New resolution within multires loop: {}\n'.format(gridDimensions))

        if torch.cuda.is_available():
            maxVolume = maxVolume.cuda()

        # deep learning modules
        dataset = SupervisedMeshGrid(sidelen=gridDimensions, domain=domain, gt_path=gt_path, flatten=False)
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
        model_input, gt_densities = next(iter(dataloader))

        # we dont want to update input in nerf so dont enable grads here
        if torch.cuda.is_available():
            model_input = model_input.cuda()
            gt_densities = gt_densities.cuda()

        # save loss values for plotting
        supervised_loss_array_res = []

        # training of xPhys
        for step in tqdm(range(epoch_sizes[idx]), desc='Training: '):

            def closure():
                optim.zero_grad()
                # aka x
                density = model(model_input)
                # clamp to prevent negative values (MSE cannot handle it solo)
                if clamp_output:
                    density = torch.clamp(density, min=0., max=1.)
                density = density.permute(0, 3, 1, 2)

                supervised_loss = criterion(density, gt_densities)

                global actual_steps
                actual_steps += 1

                # reduce LR if no reach plateau
                if use_scheduler:
                    scheduler.step(supervised_loss)

                supervised_loss.backward()

                # save loss values for plotting
                supervised_loss_array_res.append(supervised_loss.detach().item())
                sys.stderr.write("Total Steps: %d, Resolution Steps: %d, Compliance loss %0.6f" % (actual_steps, step, supervised_loss))

                return supervised_loss

            optim.step(closure)

        supervised_loss_array.extend(supervised_loss_array_res)

        density = model(model_input)
        density = physical_density(density, maxVolume)

        # visualization and saving model
        grid_title = ''.join(str(i)+'x' for i in gridDimensions)[:-1]
        model_title = model.__class__.__name__
        optim_title = optim.__class__.__name__
        maxVolume_np = maxVolume.detach().cpu().numpy()
        title = ''
        if clamp_output:
            title = 'fourFeat_clmp_s'
        title = title+str(scale)+'_'+grid_title+'_'+str(idx+1)+'x'+str(actual_steps)+'_'+model_title+'_'+optim_title+'_'+problem_name+'_Vol'+str(maxVolume_np)

        title = visualizations.loss_vis(supervised_loss_array_res, title, True, path=log_loss_path)
        visualizations.density_vis(density, supervised_loss_array_res[-1], gridDimensions, title, True, visualize, True,
                                   binary_loss=None, path=log_image_path)

utils.save_weights(model, append_path[:-1] if args.jid is None else args.jid, save_model, path=log_weights_path)
# utils.save_densities(density, gridDimensions, title, save_density, True, path='logs/densities/fourfeat_multires/')
