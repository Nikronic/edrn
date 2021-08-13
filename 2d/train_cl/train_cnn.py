import os, sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from networks import CNNModel
from fem import VoxelFEMFunction
from utils import NormalLatent
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

# PyTorch related global variables
torch.autograd.set_detect_anomaly(False)

# global variables that control visualizing/saving problem domain, losses, etc.
visualize = True
save_model = True
save_density = False
fixed_scale = True

# record runtime
start_time = time.perf_counter()


problem_path = 'problems/2d/roof.json'  # TODO
gridDimensions = [180, 180]  # TODO
latent_size = 128  # TODO

# multires hyperparameters
volume_constraint_satisfier = 'constrained_sigmoid'
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
adaptive_filtering_configs['cone_filter'] = True  # only used in CNN
adaptive_filtering_configs['cone_init'] = 2
adaptive_filtering_configs['cone_interval'] = 0.1
adaptive_filtering_configs['cone_scaler'] = -1

is_volume_constraint_satisfier_hard = fem.type_of_volume_constaint_satisfier(mode=volume_constraint_satisfier)
res_order = 'ftc'
repeat_res = 1  # TODO
epoch_mode = 'constant'
# hyper parameter of positional encoding in NeRF
SIMPExponent = 3
resolutions = multires_utils.prepare_resolutions(interval=0, start=0, end=1, order=res_order, repeat_res=repeat_res)  # TODO
epoch_sizes = multires_utils.prepare_epoch_sizes(n_resolutions=len(resolutions), # TODO
                                                 start=800, end=1500, 
                                                 mode=epoch_mode, constant_value=1000)  # 1000: match ours
mrconfprint = 'resolution order: {}, epoch mode: {}, '.format(res_order, epoch_mode)
mrconfprint += 'repeat resolutions: {} times \n'.format(repeat_res)
mrconfprint += 'adaptive filtering configs: {} \n'.format(adaptive_filtering_configs)
mrconfprint += 'Volume constraint satisfier: {} (hard: {})\n'.format(volume_constraint_satisfier,
                                                                     is_volume_constraint_satisfier_hard)
mrconfprint += 'Input latent size: {}\n'.format(latent_size)
sys.stderr.write(mrconfprint)

# create experiments folders for each run  
log_base_path = 'logs/'
log_image_path = '{}images/cnn/'.format(log_base_path)
log_loss_path =  '{}loss/cnn/'.format(log_base_path)
log_weights_path =  '{}weights/cnn/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, None, 
                                                experiment_id=args.jid)
log_image_path = '{}images/cnn/{}'.format(log_base_path, append_path)
log_loss_path =  '{}loss/cnn/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path: {}\n'.format(log_image_path, log_loss_path))

# load BCs
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
else:
    configs['gridDimensions'] = gridDimensions
E0 = configs['E0']
Emin = configs['Emin']
if SIMPExponent is None:
    SIMPExponent = configs['SIMPExponent']
else:
    configs['SIMPExponent'] = SIMPExponent
maxVolume = torch.tensor(configs['maxVolume'])
if adaptive_filtering_configs is None:
    adaptive_filtering_configs = configs['adaptive_filtering']
seed = configs['seed']
sys.stderr.write('VoxelFEM problem configs: {}\n'.format(configs))

gridDimensions_ = copy.deepcopy(gridDimensions)

# reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    maxVolume = maxVolume.cuda()

# deep learning modules
n_elements = np.prod(gridDimensions)
batch_size = 1
dataset = NormalLatent(latent_size, std=1, mean=0)
dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
model_input = next(iter(dataloader))

# deep learning modules
if is_volume_constraint_satisfier_hard:
    cnn_model = CNNModel(gridDimension=gridDimensions, latent_size=latent_size)
else:
    raise ValueError('CNN model only works with hard volume constraint!\n')
model = cnn_model
if torch.cuda.is_available():
    model_input = model_input.cuda().requires_grad_(True)
    model.cuda()
else:
    model_input = model_input.requires_grad_(True)

projection_filter = filtering.ProjectionFilter(beta=adaptive_filtering_configs['beta_init'], normalized=True)
smoothing_filter = filtering.SmoothingFilter(radius=adaptive_filtering_configs['radius_init'])
cone_filter = filtering.ConeFilter(radius=adaptive_filtering_configs['cone_init'])
gauss_smoothing_filter = filtering.GaussianSmoothingFilter(sigma=adaptive_filtering_configs['sigma_init'])
filters = [projection_filter, smoothing_filter, gauss_smoothing_filter, cone_filter]

# create ground truth and set voxelFEM engine based on model output
with torch.no_grad():
    new_shape = model(model_input).shape[1:]
    gridDimensions = list(new_shape)

# topopt (via VoxelFEM-Optimization-Problem)
constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
uniformDensity = maxVolume
tps = initializeTensorProductSimulator(
    orderFEM, domainCorners, gridDimensions, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH)
objective = pyVoxelFEM.ComplianceObjective(tps)
top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, [])

# instantiate autograd.Function for VoxelFEM engine
voxelfem_engine = VoxelFEMFunction.apply

# optim
optim = torch.optim.LBFGS(lr=1, params=itertools.chain(list(model.parameters()),[model_input]),
                          history_size=10, line_search_fn='strong_wolfe', max_eval=5, max_iter=5)
# optim = torch.optim.Adam(params=itertools.chain(list(model.parameters()),[model_input]))

# training
actual_steps = 0
compliance_loss_array = []

# training of xPhys
for step in tqdm(range(epoch_sizes[0]), desc='Training: '):

    def closure():
        if torch.is_grad_enabled():
            optim.zero_grad()
        # aka x
        density = model(model_input)
        # aka xPhys
        density = fem.satisfy_volume_constraint(density=density, max_volume=maxVolume, compliance_loss=None,
                                                mode=volume_constraint_satisfier)
        if torch.is_grad_enabled():
            # smoothing
            if adaptive_filtering_configs is not None:
                density = density.squeeze(0)
                density = filtering.apply_filters_group(x=density, filters=filters, configs=adaptive_filtering_configs)
                filtering.update_adaptive_filtering(iteration=step, filters=filters, configs=adaptive_filtering_configs)
                density = density.unsqueeze(0)
        # compliance for predicted xPhys
        if torch.cuda.is_available():
            density = density.cpu()
        compliance_loss = voxelfem_engine(density.flatten(), top)
        if torch.cuda.is_available() and torch.is_grad_enabled():
            compliance_loss.cuda()

        if torch.is_grad_enabled():
            global actual_steps
            actual_steps += 1

            compliance_loss.backward()

        # save loss values for plotting
        # compliance_loss_array.append(compliance_loss.detach().item())
        # print("Optim Step %d, Line Search Step %d, Compliance loss %0.6f" % (step , actual_steps, compliance_loss))
        # sys.stderr.write('Optim Step %d, Compliance loss %0.6f'.format(actual_steps, compliance_loss))

        return compliance_loss

    optim.step(closure)

    # save loss values for plotting
    with torch.no_grad():
        compliance_loss = closure().detach().item()
    compliance_loss_array.append(compliance_loss)
    sys.stderr.write('Optim Step {:d}, Compliance loss {:.6f}'.format(actual_steps, compliance_loss))

# recording run time
execution_time = time.perf_counter() - start_time
sys.stderr.write('Overall runtime: {}\n'.format(execution_time))

# test network
with torch.no_grad():
    density = model(model_input)
    density = fem.physical_density(density, maxVolume)
    density = density.detach().cpu()
    # loss of conversion to binary by thresholding
    compliance_loss = voxelfem_engine(density.flatten(), top)
    binary_compliance_loss = utils.compute_binary_compliance_loss(density=density, top=top,
                                                                loss_engine=voxelfem_engine)

# visualization and saving model
maxVolume_np = maxVolume.detach().cpu().numpy()
grid_title = ''.join(str(i)+'x' for i in gridDimensions)[:-1]
adaptive_filtering_configs_title = ''.join(str(i)+';' for i in adaptive_filtering_configs.values())[:-1]
title = 'CNN(HC'+str(is_volume_constraint_satisfier_hard)+'_test)_'+grid_title+'_'+str(actual_steps)
title +=  problem_name+'_Vol'+str(maxVolume_np)
visualizations.density_vis(density, compliance_loss, gridDimensions, title, True, visualize, True,
                           binary_loss=binary_compliance_loss, path=log_image_path)
title = 'CNN(HC'+str(is_volume_constraint_satisfier_hard)+'_overall)_'+grid_title+'_'+str(actual_steps)
title +=  problem_name+'_Vol'+str(maxVolume_np)
title = visualizations.loss_vis(compliance_loss_array, title, True, path=log_loss_path,
                                ylim=np.max(compliance_loss_array))

utils.save_weights(model, append_path[:-1] if args.jid is None else args.jid, save_model, path=log_weights_path)
