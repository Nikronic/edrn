import torch
import networks
from torch.utils.data import Dataset

import numpy as np

import os, sys
from datetime import datetime

import os, sys
sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore
import MeshFEM, mesh  # type: ignore


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def get_mgrid(sidelen, domain, flatten=True):
    '''
    Generates a grid of nodes of elements in given ``domain`` range with ``sidelen`` nodes of that dim

    :param sidelen:  a 2D/3D tuple of number of nodes
    :param domain: a tuple of list of ranges of each dim corresponding to sidelen
    :param flatten: whether or not flatten the final grid (-1, 2/3)
    :return:
    '''

    sidelen = np.array(sidelen)
    tensors = []
    for d in range(len(sidelen)):
        tensors.append(torch.linspace(domain[d, 0], domain[d, 1], steps=sidelen[d]))
    tensors = tuple(tensors)
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    if flatten:
        mgrid = mgrid.reshape(-1, len(sidelen))
    return mgrid


class MeshGrid(Dataset):
    def __init__(self, sidelen, domain, flatten=True):
        """
        Generates a mesh grid matrix of equally distant coordinates

        :param sidelen: Grid dimensions (number of nodes along each dimension)
        :param domain: Domain boundry
        :param flatten: whether or not flatten the final grid (-1, 2 or 3)
        :return: Meshgrid of coordinates (elements, 2 or 3)
        """
        super().__init__()
        self.sidelen = sidelen
        self.domain = domain
        self.flatten = flatten

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return get_mgrid(self.sidelen, self.domain, self.flatten)


class SupervisedMeshGrid(Dataset):
    def __init__(self, sidelen, domain, gt_path, flatten=True):
        """
        Generates a mesh grid matrix of equally distant coordinates for a ground truth target with same grid size

        :param sidelen: Grid dimensions (number of nodes along each dimension)
        :param domain: Domain boundry
        :param gt_path: Path to the .npy saved ground truth densities of the same shape
        :param flatten: whether or not flatten the final grid (-1, 2 or 3)
        :return: Meshgrid of coordinates (elements, 2 or 3)
        """
        super().__init__()
        self.sidelen = sidelen
        self.domain = domain
        self.flatten = flatten
        self.gt_path = gt_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        
        # get saved ground truth
        gt_densities = np.load(self.gt_path).astype(np.float32)
        gt_densities = torch.as_tensor(gt_densities)
        gt_densities = gt_densities.permute(1, 0).unsqueeze(0)

        return get_mgrid(self.sidelen, self.domain, self.flatten), -gt_densities


class RandomField(Dataset):
    def __init__(self, latent, std=0.1, mean=0):
        """
        Generates a latent vector distributed from random normal

        :param latent: Latent vector size based on number of elements
        :param std: std of gaussian noise
        :param mean: mean of gaussian noise
        :return: A random tensor with size of latent
        """
        super().__init__()
        self.latent = latent
        self.std = std
        self.mean = mean

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        # latent size with one feature for each element in latent space
        return torch.randn(self.latent, 1) * self.std + self.mean


class NormalLatent(Dataset):
    def __init__(self, latent_size, std=1, mean=0):
        """
        Generates a latent vector distributed from random normal

        :param latent: Latent vector size based
        :param std: std of gaussian noise
        :param mean: mean of gaussian noise
        :return: A random tensor with size of latent
        """
        super().__init__()
        self.latent_size = latent_size
        self.std = std
        self.mean = mean

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return torch.normal(mean=self.mean, std=self.std, size=(self.latent_size, ))


# Reference: https://github.com/jacobkimmel/pytorch_modelsize
class SizeEstimator(object):

    def __init__(self, model, input_size=(1,1,32,32), bits=32):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.input_size = input_size
        self.bits = bits
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = list(self.model.modules())
        sizes = []
        
        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''
        input_ = torch.FloatTensor(*self.input_size).requires_grad_(True)
        mods = list(self.model.modules())
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        '''Calculate total number of bits to store `model` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        '''Calculate bits to store forward and backward pass'''
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s))*self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits*2)
        return

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self.input_size))*self.bits
        return

    def estimate_size(self):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total/8)/(1024**2)
        return total_megabytes, total


def count_parameters(model, trainable=True):
    """
    Counts the number of trainable parameters in a model

    :param model: Model to be processes
    :param trainable: Wether to only count trainable parameters
    """
    
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


# see issue #20: register_buffer is bugged in pytorch!
def save_weights(model, title, save=False, path=None):
    if path is None:
        path = 'tmp/'
    
    if save:
        if isinstance(model, networks.CNNModel):
            d = {
                'model_state_dict': model.state_dict()
            }
        if isinstance(model, networks.MLP):
            d = {
                'scale': model.scale,
                'B': model.B,
                'model_state_dict': model.state_dict()
            }
        torch.save(d, path + title + '.pt')


def load_weights(model, path):
    if torch.cuda.is_available():
        d = torch.load(path)
    else:
        d = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(d['model_state_dict'])
    model.B = d['B']
    model.scale = d['scale']
    sys.stderr.write('Weights, scale, and B  loaded.\n')
    

def save_densities(density, gridDimensions, title, save=False, prediciton=True, path=None):
    if path is None:
        path = 'tmp/'

    if save:
        if prediciton:
            if os.path.isfile(path + title + '_pred.npy'):
                title += str(int(datetime.timestamp(datetime.now())))
            if len(gridDimensions) == 2:
                with open(path + title + '_pred.npy', 'wb') as f:
                    np.save(f, -density.view(gridDimensions).detach().cpu().numpy()[:, :].T)
            else:
                tps = density
                mfw = mesh.MSHFieldWriter(path + title + '_pred.mesh', *tps.getMesh())
                mfw.addField('density', tps.getDensities())

        else:
            if len(gridDimensions) == 2:
                with open(path + title + '_gt.npy', 'wb') as f:
                    np.save(f, -density.reshape(gridDimensions[0], gridDimensions[1]).T)
            else:
                tps = density
                mfw = mesh.MSHFieldWriter(path + title + '_gt.mesh', *tps.getMesh())
                mfw.addField('density', tps.getDensities())


def compute_binary_compliance_loss(density, loss_engine, top):
    voxelfem_engine = loss_engine
    if voxelfem_engine is None:
        density_binary = (density > 0.5) * 1.
        top.setVars(density_binary.astype(np.float64))
        binary_compliance_loss = 2.0 * top.evaluateObjective()
        sys.stderr.write('Compliance loss of binary densities for "{}": {}, b-vol={:.7f}\n'.format(density_binary.shape,
                                                                                                   binary_compliance_loss,
                                                                                                   density_binary.mean()))        
        top.setVars(density)
    else:
        density_binary = (density > 0.5).float() * 1.
        if torch.cuda.is_available():
            density_binary = density_binary.cpu()
        binary_compliance_loss = voxelfem_engine(density_binary.flatten(), top)
        sys.stderr.write('Compliance loss of binary densities for "{}": {}, b-vol={:.7f}\n'.format(density_binary.shape,
                                                                                                   binary_compliance_loss.detach().numpy(),
                                                                                                   density_binary.mean().numpy()))
    
    return binary_compliance_loss
