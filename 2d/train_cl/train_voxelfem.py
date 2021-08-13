import sys, os
import numpy as np

import matplotlib as mpl
mpl.use('Agg')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import visualizations
import utils
import fem
import multires_utils

import time, copy
import argparse, json
from tqdm import tqdm

sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore
import MeshFEM, mesh # type: ignore
from ipopt_helpers import initializeTensorProductSimulator, problemObjectWrapper, initializeIpoptProblem  # type: ignore

parser=argparse.ArgumentParser()
parser.add_argument('--jid', help='Slurm job id to name experiment dirs with it')
args=parser.parse_args()
experiment_id = args.jid
sys.stderr.write('Experiment ID: {}\n'.format(experiment_id))

if experiment_id is None:
    sys.stderr.write = print

visualize = True
save_density = True

# record full runtime
start_time = time.perf_counter()

problem_path = 'problems/2d/mbb_beam.json'  # TODO
gridDimensions = [300, 100]  # TODO
max_iter = 1500  # TODO
optim = 'LBFGS'  # TODO
adaptive_filtering = [1, 1, 1, 1]  # TODO
sys.stderr.write('adaptive filtering configs: {}\n'.format(adaptive_filtering))

# create experiments folders for each run  
log_base_path = 'logs/'
log_image_path = '{}images/gt/'.format(log_base_path)
log_loss_path = '{}loss/gt/'.format(log_base_path)
log_densities_path = '{}densities/gt/'.format(log_base_path)
append_path = multires_utils.mkdir_multires_exp(log_image_path, log_loss_path, log_densities_path, None,
                                                experiment_id=args.jid)
log_image_path = '{}images/gt/{}'.format(log_base_path, append_path)
log_loss_path = '{}loss/gt/{}'.format(log_base_path, append_path)
log_densities_path = '{}densities/gt/{}'.format(log_base_path, append_path)
sys.stderr.write('image path: {}, loss path: {}\n'.format(log_image_path, log_loss_path))

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
maxVolume = configs['maxVolume'][0]
if adaptive_filtering is None:
    adaptive_filtering = configs['adaptive_filtering'][:-2]  # dont use gaussian smoothing
seed = configs['seed']
sys.stderr.write('VoxelFEM problem configs: {}\n'.format(configs))

# reproducibility
np.random.seed(seed)

# vis title
grid_title = ''.join(str(i)+'x' for i in gridDimensions)[:-1]
title = str(experiment_id)+'_voxelfem_'+'optim-'+optim+'_'+grid_title+'_'+str(max_iter)+'_'+problem_name+'_Vol'+str(maxVolume)

# solve
gt_tps, gt_loss, binary_gt_loss, gt_loss_array = fem.ground_truth_topopt(MATERIAL_PATH, BC_PATH, orderFEM, domainCorners,
                                                          gridDimensions, SIMPExponent, maxVolume, use_multigrid=False,
                                                          init=None, optimizer=optim, multigrid_levels=None,
                                                          adaptive_filtering=adaptive_filtering, 
                                                          max_iter=max_iter, obj_history=True, title=title,
                                                          log_image_path=log_image_path, log_densities_path=log_densities_path)
sys.stderr.write('Final step => Compliance loss {:.6f}, Volume constraint {:.6f} \n'.format(gt_loss, gt_tps.mean()))

# visualization and saving model
same_size_vis = False
if domainCorners[1][1] / domainCorners[1][0] >= 3:
    same_size_vis = True
visualizations.loss_vis(gt_loss_array, title, visualize, path=log_loss_path, 
                        ylim=np.max(gt_loss_array)+0.1*np.max(gt_loss_array))
visualizations.density_vis(gt_tps, gt_loss, gridDimensions, title, False, visualize, same_size_vis,
                               binary_loss=binary_gt_loss, path=log_image_path)
utils.save_densities(gt_tps, gridDimensions, title, save_density, False, path=log_densities_path)
execution_time = time.perf_counter() - start_time
sys.stderr.write('\nOverall runtime: {}\n'.format(execution_time))
