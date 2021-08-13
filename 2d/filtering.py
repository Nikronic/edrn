import os, sys
import torch
import torch.nn as nn
import kornia
import numpy as np

import utils
sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore

import matplotlib as mpl
mpl.use('Agg')


def sigmoid5x(x):  # TODO: send it to ``filtering``
    """
    following projection introduced in NTopo 
    """
    return torch.sigmoid(5 * x)


def projection_filter(x, beta, normalized=False):
    beta = torch.tensor([float(beta)])
    if torch.cuda.is_available():
        beta = beta.cuda()
    if normalized:
        return 0.5 * (torch.tanh(0.5 * beta) + torch.tanh(beta * (x))) / torch.tanh(0.5 * beta)
    else:
        return 0.5 * torch.tanh(beta * (x)) + 0.5


def smoothing_filter(x, radius):    
    return kornia.box_blur(input=x.unsqueeze(0).unsqueeze(0), kernel_size=(radius*2+1, radius*2+1), 
                           border_type='reflect', normalized=True).squeeze(0).squeeze(0)


def gaussian_filter(x, sigma):
    kernel_size = np.floor(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size - 1
    kernel_size = int(kernel_size)
    return kornia.gaussian_blur2d(input=x.unsqueeze(0).unsqueeze(0), kernel_size=(kernel_size, kernel_size),
                                  sigma=(sigma, sigma), border_type='reflect').squeeze(0).squeeze(0)



class ProjectionFilter(nn.Module):
    """
    A projection filter as a binarizer filter where higher ``beta`` pushed the filter more binary (step function)
      that computes ``0.5 * tanh(beta * x) + 0.5`` or its ``normalized`` variant.
    """
    def __init__(self, beta=1, normalized=False):
        super().__init__()
        self.normalized = normalized
        self.beta = torch.tensor([float(beta)])
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()

    def forward(self, x):
        beta = self.beta
        if self.normalized:
            return 0.5 * (torch.tanh(0.5 * beta) + torch.tanh(beta * (x))) / torch.tanh(0.5 * beta)
        else:
            return 0.5 * torch.tanh(beta * (x)) + 0.5
        
    def update_params(self, scaler):
        self.beta = self.beta * scaler
    
    def reset_params(self, beta=1):
        self.beta = torch.tensor([float(beta)])
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()


class SigmoidAlphaX(nn.Module):
    """
    A projection filter as a binarizer filter where higher ``alpha`` pushed the filter more binary (step function)
      that computes ``torch.sigmoid(alpha * x)``
    """
    def __init__(self, alpha=5.):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        alpha = self.alpha
        return torch.sigmoid(alpha * x)
        
    def update_params(self, scaler):
        self.alpha = self.alpha * scaler
    
    def reset_params(self, alpha=1):
        self.alpha = alpha


class SmoothingFilter(nn.Module):
    """
    A box filter as a smoothing filter with ``reflect`` padding and ``kernel_size = radius * 2 + 1``
    """
    def __init__(self, radius=1):
        super().__init__()
        self.radius = radius

    def forward(self, x):
        return kornia.box_blur(input=x.unsqueeze(0).unsqueeze(0), kernel_size=(self.radius*2+1, self.radius*2+1), 
                               border_type='reflect', normalized=True).squeeze(0).squeeze(0)

    def update_params(self, scaler):
        self.radius = self.radius * scaler
    
    def reset_params(self, radius=1):
        self.radius = radius


class VFMGaussianSmoothingFilter:
    """
    A Gaussian filter as a smoothing filter with ``reflect`` padding and corresponding ``kernel_size = floor(6 * sigma)``.
    """
    def __init__(self, sigma=1, kernel_size=5, shape=[300, 100]):
        # super().__init__()
        self.python_filt = pyVoxelFEM.PythonFilter()

        self.sigma = sigma
        self.kernel_size = kernel_size
        self.shape = shape
        if sigma == 1:
            self.kernel_size = 5

        self.grads = None

    def apply(self, inDensities, outDensities):
        x = inDensities
        x = x.reshape(self.shape)
        x = torch.from_numpy(x).float()
        x.requires_grad_(True)
        density = kornia.gaussian_blur2d(input=x.unsqueeze(0).unsqueeze(0), kernel_size=(self.kernel_size, self.kernel_size),
                                         sigma=(self.sigma, self.sigma), border_type='reflect').squeeze(0).squeeze(0)
        grads = utils.gradient(density, x)
        self.grads = grads
        density = density.detach().numpy().astype(np.float64).flatten()
        outDensities[:] = density
        # return density.flatten()

    def backprop(self, dJ_dout, x, dJ_din):
        if self.grads is None:
            raise Exception('Grads are not accumulated yet!')
        dJ_dout = self.grads.detach().flatten().numpy()
        dJ_din[:] = dJ_dout
        # return self.grads.detach().flatten().numpy()
    
    def update_params(self, scaler):
        self.sigma = self.sigma * scaler
        self.kernel_size = self.compute_kernel_size(self.sigma)
    
    def reset_params(self, sigma=1):
        self.sigma = sigma
        self.kernel_size = self.compute_kernel_size(sigma)
        
    def compute_kernel_size(self, sigma):
        kernel_size = np.floor(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size - 1
        return int(kernel_size)


class GaussianSmoothingFilter(nn.Module):
    """
    A Gaussian filter as a smoothing filter with ``reflect`` padding and corresponding ``kernel_size = floor(6 * sigma)``.
    """
    def __init__(self, sigma=1):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = self.compute_kernel_size(sigma)
        if sigma == 1:
            self.kernel_size = 5

    def forward(self, x):
        density = kornia.gaussian_blur2d(input=x.unsqueeze(0).unsqueeze(0), kernel_size=(self.kernel_size, self.kernel_size),
                                         sigma=(self.sigma, self.sigma), border_type='reflect').squeeze(0).squeeze(0)
        return density

    def update_params(self, scaler):
        self.sigma = self.sigma * scaler
        self.kernel_size = self.compute_kernel_size(self.sigma)
    
    def reset_params(self, sigma=1):
        self.sigma = sigma
        self.kernel_size = self.compute_kernel_size(sigma)
        
    def compute_kernel_size(self, sigma):
        kernel_size = np.floor(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size - 1
        return int(kernel_size)


class ConeFilter(nn.Module):
    """
    A linear filter as a smoothing filter with ``reflect`` padding and ``kernel_size = radius * 2 + 1``
    """
    def __init__(self, radius=2):
        super().__init__()
        self.radius = radius
        self.kernel = self.make_kernel(radius=radius)

    def forward(self, x):
        return kornia.filters.filter2D(input=x.unsqueeze(0).unsqueeze(0), kernel=self.kernel.unsqueeze(0), 
                               border_type='reflect', normalized=True).squeeze(0).squeeze(0)

    def update_params(self, scaler):
        self.radius = self.radius * scaler
    
    def reset_params(self, radius=2):
        self.radius = radius

    def make_kernel(self, radius=2):
        """
        Creates a cone kernel regarding given radius

        :param radius: radius of the output kernel (default=2 in CNN)
        :return: A cone kernel with size of (radius*2+1)
        """
        domain = torch.tensor(list(np.arange(-radius, radius+1))).float()
        x, y = torch.meshgrid(domain, domain)
        zero_tensor = torch.tensor([0.])
        if torch.cuda.is_available():
            domain = domain.cuda()
            x, y = x.cuda(), y.cuda()
            zero_tensor = zero_tensor.cuda()
        distance = x**2 + y**2
        kernel = torch.maximum(zero_tensor, radius - torch.sqrt(distance))
        return kernel

### Utility methods

def apply_filters_group(x, filters, configs):
    """
    Applies a group of ``filters`` on given input ``x`` w.r.t. given ``configs``

    :param x: Input `x` (2D)
    :param filters: A list of filters which have ``forward`` method implemented
    :param configs: A dicitonary of configs for whether or not to use a filter
    :return: Filtered input ``x`` 
    """
    projection_filter = configs['projection_filter']
    smoothing_filter = configs['smoothing_filter']
    gaussian_filter = configs['gaussian_filter']
    cone_filter = configs['cone_filter']

    for filt in filters: 
        if isinstance(filt, ProjectionFilter):
            if projection_filter:
                x = filt(x)
        if isinstance(filt, SmoothingFilter):
            if smoothing_filter:
                x = filt(x)
        if isinstance(filt, GaussianSmoothingFilter):
            if gaussian_filter:
                x = filt(x)
        if isinstance(filt, ConeFilter):
            if cone_filter:
                x = filt(x)
    return x

def update_adaptive_filtering(iteration, filters, configs):
    """
    Updates ``*_init`` value of each given filter in ``filters`` by ``*_scaler`` every ``*_interval`` iteration given by
      the ``configs`` dictionary (inplace operation).

    :param iteration: Current iteration to compare with ``*_interval``
    :param filters: A list of filters which have ``__call__`` implemented
    :param configs: A dicitonary of configs which contains ``*_interval`` and ``*_scaler``
    :return: None
    """
    beta_interval = configs['beta_interval']
    beta_scaler = configs['beta_scaler']
    radius_interval = configs['radius_interval']
    radius_scaler = configs['radius_scaler']
    sigma_interval = configs['sigma_interval']
    sigma_scaler = configs['sigma_scaler']
    cone_interval = configs['cone_interval']
    cone_scaler = configs['cone_scaler']

    for filt in filters:
        if isinstance(filt, ProjectionFilter):            
            if (iteration % beta_interval) == 0 and (iteration != 0):
                filt.update_params(scaler=beta_scaler)
                if beta_scaler != 1:
                    sys.stderr.write(" Update -> Projection Filter       (beta={:0.2f})\n".format(filt.beta.item()))
        if isinstance(filt, SmoothingFilter):
            if (iteration % radius_interval) == 0 and (iteration != 0):
                filt.update_params(scaler=radius_scaler)
                if radius_scaler != 1:
                    sys.stderr.write(" Update -> Smoothing Filter       (radius={:0.2f})\n".format(filt.radius))
        if isinstance(filt, GaussianSmoothingFilter):
            if (iteration % sigma_interval) == 0 and (iteration != 0):
                filt.update_params(scaler=sigma_scaler)
                if sigma_scaler != 1:
                    sys.stderr.write(" Update -> Gaussian Smoothing Filter       (sigma={:0.2f})\n".format(filt.sigma))
        if isinstance(filt, ConeFilter):
            if (iteration % cone_interval) == 0 and (iteration != 0):
                filt.update_params(scaler=cone_scaler)
                if cone_scaler != 1:
                    sys.stderr.write(" Update -> Cone Smoothing Filter       (radius={:0.2f})\n".format(filt.radius))

def reset_adaptive_filtering(filters, configs):
    """
    Reset filters to their initial state provided with ``configs`` dictionary

    :param filters: A list of filters object with method ``reset_params(args)``
    :param configs: A dictionary of configs for adaptive filtering (only ``*_init`` values are needed)
    :return: None
    """

    beta_init = configs['beta_init']
    radius_init = configs['radius_init']
    sigma_init = configs['sigma_init']
    cone_init = configs['cone_init']

    for filt in filters:
        if isinstance(filt, ProjectionFilter):
            filt.reset_params(beta=beta_init)
        if isinstance(filt, SmoothingFilter):
            filt.reset_params(radius=radius_init)
        if isinstance(filt, GaussianSmoothingFilter):
            filt.reset_params(sigma=sigma_init)
        if isinstance(filt, ConeFilter):
            filt.reset_params(sigma=cone_init)
    sys.stderr.write('Adaptive filtering has been reset to their defaults. \n')
