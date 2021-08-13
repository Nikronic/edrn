import torch
import numpy as np
import kornia
import torch.autograd as autograd
from torch.optim.optimizer import Optimizer, required

from functools import partial
import sys, os, time
from copy import deepcopy

import filtering
import utils

sys.path.append(os.getcwd()+'/VoxelFEM/python')
sys.path.append(os.getcwd()+'/VoxelFEM/python/helpers')
import pyVoxelFEM  # type: ignore
import MeshFEM, mesh  # type: ignore
from ipopt_helpers import initializeTensorProductSimulator, problemObjectWrapper, initializeIpoptProblem  # type: ignore


def ground_truth_topopt(MATERIAL_PATH, BC_PATH, orderFEM, 
                        domainCorners, gridDimensions, SIMPExponent,
                        maxVolume, optimizer, multigrid_levels,
                        use_multigrid=False, adaptive_filtering=[1, 1, 1, 1],
                        max_iter=1500, init=None, obj_history=False, **kwargs):

    # Visualization/saving
    title=kwargs['title'] if 'title' in kwargs else None
    log_image_path = kwargs['log_image_path'] if 'log_image_path' in kwargs else None
    log_densities_path = kwargs['log_densities_path'] if 'log_densities_path' in kwargs else None
    
    E0 = 1
    Emin = 1e-9

    constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]

    ##############
    python_filt = pyVoxelFEM.PythonFilter()

    def compute_kernel_size(sigma):
        kernel_size = np.floor(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size - 1
        return int(kernel_size)
    shape = [300, 100]
    sigma = 4.0  # TODO
    def customFilterApply(inDensities, outDensities):
        kernel_size = compute_kernel_size(sigma=sigma)
        # kernel_size = 3
        x = inDensities
        x = x.reshape(shape)
        x = torch.from_numpy(x).float()
        density = kornia.gaussian_blur2d(input=x.unsqueeze(0).unsqueeze(0), kernel_size=(kernel_size, kernel_size),
                                            sigma=(sigma, sigma), border_type='reflect').squeeze(0).squeeze(0)
        density = density.detach().numpy().astype(np.float64).flatten()
        outDensities[:] = density

    def customFilterBackprop(dJ_dout, x, dJ_din):
        kernel_size = compute_kernel_size(sigma=sigma)
        # kernel_size = 3
        x = x.reshape(shape)
        x = torch.from_numpy(x).float()
        x.requires_grad_(True)
        density = kornia.gaussian_blur2d(input=x.unsqueeze(0).unsqueeze(0), kernel_size=(kernel_size, kernel_size),
                                            sigma=(sigma, sigma), border_type='reflect').squeeze(0).squeeze(0)
        grads = utils.gradient(density, x)
        density = density.detach().numpy().astype(np.float64).flatten()
        dJ_din[:] = dJ_dout * grads.detach().flatten().numpy()

    python_filt.apply_cb = customFilterApply
    python_filt.backprop_cb = customFilterBackprop
    ##############

    if max_iter == 0:  # evaluating compliance of the density (no training)
        filters = []
    else:
        filters = [
            python_filt,
            pyVoxelFEM.ProjectionFilter(),
            ]
    uniformDensity = maxVolume
    tps = initializeTensorProductSimulator(
        orderFEM, domainCorners, gridDimensions, uniformDensity, E0, Emin, SIMPExponent, MATERIAL_PATH, BC_PATH)
    if use_multigrid:
        objective = pyVoxelFEM.MultigridComplianceObjective(tps.multigridSolver(multigrid_levels))
    else:
        objective = pyVoxelFEM.ComplianceObjective(tps)                                    
    top = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, filters)
    top_binary = pyVoxelFEM.TopologyOptimizationProblem(tps, objective, constraints, [])
    nonLinearProblem, problemObj = initializeIpoptProblem(top)

    # for filt in filters:
    #     if isinstance(filt, pyVoxelFEM.SmoothingFilter):
    #         filt.radius = 5

    # add adaptive filtering
    if adaptive_filtering is not None:
        problemObj.beta_interval, problemObj.beta_scaler, problemObj.radius_interval, problemObj.radius_scaler = adaptive_filtering
    
    if init is None:
        x0 = tps.getDensities()
    else:
        init = init.numpy().astype(np.float64)
        top.setVars(init.flatten())
        x0 = tps.getDensities()
    
    if use_multigrid:
        # Configure multigrid objective
        objective.tol = 1e-7
        objective.mgIterations = 2
        objective.fullMultigrid = True

    if optimizer == 'OC':        
        oco = pyVoxelFEM.OCOptimizer(top)
        top.setVars(tps.getDensities())
        iter_start_time = 0
        for idx in range(max_iter):
            z = top.getDensities()
            iter_time = time.perf_counter() - iter_start_time
            objective_value = 2.0 * top.evaluateObjective()
            problemObj.history.objective.append(objective_value)
            sys.stderr.write('Total Steps: {:d}, Runtime: {:.1f}, Compliance loss {:.6f}\n'.format(idx, iter_time, objective_value))
            iter_start_time = time.perf_counter()
            oco.step()
            zz = top.getDensities()
            error = ((z-zz)**2).mean()
            if error < 1e-7:
                break

    elif optimizer == 'LBFGS':
        nonLinearProblem.addOption('print_level', 0)
        nonLinearProblem.addOption(b'sb', b'yes')
        nonLinearProblem.addOption('max_iter', max_iter)
        nonLinearProblem.addOption('tol', 1e-7)
        if max_iter != 0:
            x0, _ = nonLinearProblem.solve(x0)
    else:
        raise ValueError('Optimizer {} is unknown or not implemented.'.format(optimizer))
    x0 = tps.getDensities()
    binary_objective = utils.compute_binary_compliance_loss(density=x0, loss_engine=None, top=top_binary)

    if obj_history is False:
        return (tps if len(orderFEM) == 3 else tps.getDensities(), 
                2.0 * top.evaluateObjective(), binary_objective)
    else:
        return (tps if len(orderFEM) == 3 else tps.getDensities(),
                2.0 * top.evaluateObjective(), binary_objective, problemObj.history.objective)


class VoxelFEMFunction(autograd.Function):
    @staticmethod
    def forward(ctx, densities: torch.Tensor, top):  # type: ignore
        """

        :param ctx: pytorch context manager
        :param densities: predicted densities (xPhys)
        :param top: topology optimization object instantiated from ``pyVoxelFEM.TopologyOptimizationProblem``
        """  

        # nlp.solve for 0 iterations = top.__objective(densities) 
        ## where densities were updated using ``top.setVars(densities)``
        top.setVars(densities.numpy().astype(np.float64))
        output_objective = 2.0 * top.evaluateObjective()
        
        # already accumulated from ``top.setVars(densities)``
        output_gradient = top.evaluateObjectiveGradient().astype(np.float32)
        output_gradient = torch.from_numpy(output_gradient)
        ctx.save_for_backward(output_gradient) 

        return torch.tensor(output_objective).float()

    @staticmethod
    def backward(ctx, grad_output):
        output_gradient = ctx.saved_tensors[0]
        return (output_gradient * grad_output), None


@torch.no_grad()
def optimality_criteria(x, max_volume, top, move=0.2, ctol=1e-6, **kwargs):
    """
    optimality Criteria Method
    """

    verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    top.setVars(x.clone().detach().cpu().numpy().astype(np.float32).flatten())
    x = x.flatten().to(device)
    dj = torch.as_tensor(top.evaluateObjectiveGradient().astype(np.float32), device=device)
    dc = torch.as_tensor(top.evaluateConstraintsJacobian()[0].astype(np.float32), device=device)
    zero_tensor = torch.zeros((1, ), device=device)
    one_tensor = torch.ones((1, ), device=device)

    lambda_min = 1.
    lambda_max = 2.
    
    def stepped_vars_for_lambda(lambda_):
        x_new = (x * (dj / (dc * lambda_ + 1e-7)).sqrt()).maximum(x - move).maximum(zero_tensor).minimum(x + move).minimum(one_tensor)
        x_new[torch.isnan(x_new)] = zero_tensor
        return x_new
    
    def constraint_eval(lambda_):
        current_volume = torch.zeros_like(max_volume).fill_(torch.mean(stepped_vars_for_lambda(lambda_)))
        # eps = 1e-7
        # return torch.maximum(-torch.log(1 + max_volume + eps - current_volume), zero_tensor)
        return 1.0 - current_volume / max_volume
    
    while (constraint_eval(lambda_min) > 0):  # current vol > max vol
        lambda_max = lambda_min
        lambda_min /= 2
    while (constraint_eval(lambda_max) < 0):  # current vol < max vol
        lambda_min = lambda_max 
        lambda_max *= 2
    
    lambda_mid = 0.5 * (lambda_min + lambda_max)
    vol = constraint_eval(lambda_mid)
    while (torch.abs(vol) > ctol):
        if (vol < 0): lambda_min = lambda_mid
        if (vol > 0): lambda_max = lambda_mid
        lambda_mid = 0.5 * (lambda_min + lambda_max)
        vol = constraint_eval(lambda_mid)
    
    # top.setVars(stepped_vars_for_lambda(lambda_mid).cpu().numpy().astype(np.float64))

    if verbose:
        sys.stderr.write('objective={}, constraint={}, lambda estimate={} \n'.format(top.evaluateObjective() * 2,
                                                                                     top.evaluateConstraints()[0], 
                                                                                     lambda_mid))

    return stepped_vars_for_lambda(lambda_mid).view(x.shape), top.evaluateObjective() * 2


class OC(Optimizer):
    """
    Optimality Criterion Method (OC) over neural network parameters as the design parameters
    """

    def __init__(self, params, model, model_input, max_volume=required, m=required, ctol=1e-6, **kwargs):
        if m is not required and m < 0.0:
            raise ValueError("Invalid move: {}".format(m))    
        
        defaults = dict(max_volume=max_volume, m=m, ctol=ctol)
        super(OC, self).__init__(params, defaults)

        self.model = model
        self.model_input = model_input
        self.verbose = kwargs['verbose'] if 'verbose' in kwargs else False

    def __setstate__(self, state):
        super(OC, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('nesterov', False)
    
    @torch.no_grad()
    def populate_objective_grads(self):
        objective_grads = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    objective_grads.append(-p.grad.clone())  # negative?

        self.objective_grads = objective_grads  # dj in VoxelFEM OC

    @torch.no_grad()
    def populate_constraints_jacobian(self):
        constraints_jacobians = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    value = -p.grad.clone()  # negative?
                    # if value == 0.:
                    #     value.fill_(-1e-4)
                    constraints_jacobians.append(value)

        self.constraints_jacobians = constraints_jacobians  # dc in VoxelFEM OC
    
    @torch.no_grad()
    def clone_model_params(self, model):
        model_clone = deepcopy(model)
        
        params_with_grad = []
        for p in model_clone.parameters():
            # if p.grad is not None:
            params_with_grad.append(p)
        return model_clone, params_with_grad

    @torch.no_grad()
    def optimality_criteria(self, params, max_volume, move=0.2, ctol=1e-6, **kwargs):
        """
        optimality Criteria Method
        """

        dj = self.objective_grads
        dc = self.constraints_jacobians

        lambda_min = 1.
        lambda_max = 2.
        
        def stepped_vars_for_lambda(lambda_):
            # find best lambda on a cloned model (prevent original model from changing)
            model_clone, params_clone = self.clone_model_params(self.model)
            for i, param in enumerate(params_clone):
                param_clone = param.clone()
                # param.mul_((dj[i] / (dc[i] * lambda_)).sqrt()).maximum(param - move).minimum(param + move)
                param.mul_((dj[i] / (dc[i] * lambda_)).sqrt()).mul_(move)
                # param[:] = param.maximum(param_clone - move).minimum(param_clone + move)
                nan_mask = torch.isnan(param)
                param[nan_mask] = param_clone[nan_mask]
            
            x_new = model_clone(self.model_input)
            return x_new
        
        def constraint_eval(lambda_):
            current_volume = torch.zeros_like(max_volume).fill_(torch.mean(stepped_vars_for_lambda(lambda_)))
            return 1.0 - current_volume / max_volume
        
        while (constraint_eval(lambda_min) > 0): 
            lambda_max = lambda_min
            lambda_min /= 2
        while (constraint_eval(lambda_max) < 0): 
            lambda_min = lambda_max 
            lambda_max *= 2
        
        lambda_mid = 0.5 * (lambda_min + lambda_max)
        vol = constraint_eval(lambda_mid)
        while (torch.abs(vol) > ctol):
            if (vol < 0): lambda_min = lambda_mid
            if (vol > 0): lambda_max = lambda_mid
            lambda_mid = 0.5 * (lambda_min + lambda_max)
            vol = constraint_eval(lambda_mid)

        # return stepped_vars_for_lambda(lambda_mid).view(x.shape)

        # update orinial model with best lambda found
        for i, param in enumerate(params):
            param_clone = param.clone()
            # param.mul_((dj[i] / (dc[i] * lambda_mid)).sqrt())
            param.mul_((dj[i] / (dc[i] * lambda_mid)).sqrt()).mul_(move)
            # param[:] = param.maximum(param_clone - move).minimum(param_clone + move)
            # param.mul_(param * (dj[i] / (dc[i] + 1e-4)).sqrt()).maximum(param - move).minimum(param + move)
            nan_mask = torch.isnan(param)
            param[nan_mask] = param_clone[nan_mask]


    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        :param closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            m = group['m']
            ctol = group['ctol']
            max_volume = group['max_volume']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)

            self.optimality_criteria(params=params_with_grad, max_volume=max_volume, move=m, ctol=ctol)

        return loss


class FindRootFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, y, average, lower_bound, upper_bound, tolerance=1e-12, max_iterations=128, projection=None):
        """
        Implicitly solve f(x,y)=0 for y(x) using binary search where f = lambda x, y: projection(x + y).mean() - average
        Assumes that y is a scalar and f(x,y) is monotonic in y.

        :param ctx: pytorch context manager
        """

        step = 0
        while (step < max_iterations) and (upper_bound - lower_bound >= tolerance):
            y = 0.5 * (lower_bound + upper_bound)
            if (projection(x + y).mean() - average) > 0:
                upper_bound = y
            else:
                lower_bound = y
            step = step + 1
        
        y = 0.5 * (lower_bound + upper_bound)

        if torch.cuda.is_available():
            y = y.clone().detach().cuda().requires_grad_(True)
            x = x.clone().detach().cuda().requires_grad_(True)

            average = torch.tensor([average]).cuda()
        else:
            y = y.clone().detach().requires_grad_(True)
            x = x.clone().detach().requires_grad_(True)

            average = torch.tensor([average]).float()

        # pytorch enforces no grad in forward and backward
        with torch.set_grad_enabled(True):
            f = projection(x + y).mean() - average
        dfdx = autograd.grad(f, x)[0].detach()
        with torch.set_grad_enabled(True):  # TODO: currently retain_graph does not work
            f = projection(x + y).mean() - average
        dfdy = autograd.grad(f, y)[0].detach()

        ctx.save_for_backward(x.detach(), y.detach(), average, dfdx, dfdy)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y, average, dfdx, dfdy = ctx.saved_tensors

        # we are looking for 
        ## dfdx = -1
        ## dfdy = 2*y where y is from forward = 1.4142
        ## grad = - dfdx / dfdy
        return -dfdx / dfdy * grad_output, None, None, None, None, None, None, None


# instantiate above class for following method 
find_root = FindRootFunction.apply


def physical_density(x, maxVolume):
    """
    Computes physical densities x from predicted logit densities x_hat (input ``x``)

    :param x: Input density logits (unconstrainted)
    :param maxVolume: Maximum amount of volume
    :return: Constrainted ``x`` which satisfies volume constraint given by ``maxVolume`` with same shape as ``x``
    """

    x = sigmoid_with_constrained_mean(x, maxVolume)
    return x


def sigmoid_with_constrained_mean(x, average, projection=torch.sigmoid):
    """
    Satisfy reduction constraint by pushing average of input x toward input argument ``average``
    In this method, sigmoid of input is satisfied.

    :param x: Constrained input tensor 
    :param average: The constaint value
    :param projection: Function as the projection of values to binary (``torch.sigmoid`` here)
    :return: Satisfied version of input ``x``
    """

    # f = lambda x, y: torch.sigmoid(x + y).mean() - average
    lower_bound = logit(average) - torch.max(x)
    upper_bound = logit(average) - torch.min(x)
    y = 0.5 * (lower_bound + upper_bound)
    b = find_root(x, y, average, lower_bound, upper_bound, 1e-12, 128, projection)
    return projection(x + b)


def projection_filter_with_constrained_mean(x, average, projection=None):
    """
    Satisfy reduction constraint by pushing average of input x toward input argument ``average``
    In this method, sigmoid of input is satisfied.

    :param x: Constrained input tensor 
    :param average: The constaint value
    :param beta: Beta hyperparameter of projection filter (higher ``beta``, closer to step function)
    :param projection: Function as the projection of values to binary (``filtering.ProjectionFilter()`` here)
        use ``ProjectionFilter(beta, normalized=False)`` to prevent positive definitness error
    :return: Satisfied version of input ``x``
    """

    if projection is None:
        filtering.ProjectionFilter(beta=1)

    lower_bound = logit(average) - torch.max(x)
    upper_bound = logit(average) - torch.min(x)
    y = 0.5 * (lower_bound + upper_bound)
    b = find_root(x, y, average, lower_bound, upper_bound, 1e-12, 128, projection)
    return projection(x + b)


def logit(p):
    p = torch.clamp(p, 0, 1)
    return torch.log(p) - torch.log1p(-p)


# wrapper around all volume constraint satisfaction methods
def satisfy_volume_constraint(density, max_volume, compliance_loss=None, 
                              mode='constrained_sigmoid', scaler_mode='clip', constant=500., **kwargs):
    """
    Soft/Hard methods to satisfy volume constraint during training

    :return: A tuple of (density, volume_loss)
    """
    
    # even though density is now on CPU, but because of following two line, operation will happen on GPU (does not matter actually)
    current_volume = torch.zeros_like(max_volume).fill_(torch.mean(density))
    zero_tensor = torch.zeros_like(max_volume)

    if mode == 'constrained_sigmoid':
        # google method
        return sigmoid_with_constrained_mean(x=density, average=max_volume, projection=torch.sigmoid)

    elif mode == 'constrained_projection':
        # default voxelfem binarization method: recommended (even in case of default values)
        projection = kwargs['projection'] if 'projection' in kwargs else None
        density = projection_filter_with_constrained_mean(x=density, average=max_volume, projection=projection)
        return density
        
    elif mode == 'add_mean':
        # enforces volume constaint **equality** by computing difference between current volume and desired volume
        volume_loss = torch.abs(current_volume - max_volume)
        scaler = compute_volume_loss_scaler(compliance_loss=compliance_loss, volume_loss=volume_loss,
                                            mode=scaler_mode, constant=constant)
        return volume_loss * scaler

    elif mode == 'one_sided_max':
        # enforces volume constraint **inequality** ``max(V - V_max)^2``
        volume_loss = torch.maximum(current_volume - max_volume, zero_tensor) ** 2
        scaler = compute_volume_loss_scaler(compliance_loss=compliance_loss, volume_loss=volume_loss,
                                            mode=scaler_mode, constant=constant)
        return volume_loss * scaler

    elif mode == 'maxed_barrier':
        # enforces volume constraint **inequality** ``max(-log(1 + V_max + eps - x), 0)``
        eps = 1e-7
        volume_loss = torch.maximum(-torch.log(1 + max_volume + eps - current_volume), zero_tensor)
        scaler = compute_volume_loss_scaler(compliance_loss=compliance_loss, volume_loss=volume_loss,
                                            mode=scaler_mode, constant=constant)
        return volume_loss * scaler

    elif mode == 'thresholded_barrier':
        # enforces volume constraint **inequality** ``min(log(a / (V_max - V), 0)^2`` where ``a`` is activation threshold
        eps = 1e-7
        a = 1 + max_volume + eps - current_volume if current_volume <= max_volume else 1.
        volume_loss = torch.log(a / (1 + max_volume + eps - current_volume)) ** 2
        scaler = compute_volume_loss_scaler(compliance_loss=compliance_loss, volume_loss=volume_loss,
                                            mode=scaler_mode, constant=constant)
        return volume_loss * scaler
    
    elif mode == 'linear':
        volume_loss = 1.0 - current_volume / max_volume
        scaler = compute_volume_loss_scaler(compliance_loss=compliance_loss, volume_loss=volume_loss,
                                            mode=scaler_mode, constant=constant)
        # return volume_loss * scaler
        return volume_loss


def compute_volume_loss_scaler(compliance_loss, volume_loss, mode='clip', constant=500.):
    """
    As volume constraint loss is much smaller in the begining, we add a scaler as a weight to increase/decrease its value


    :param compliance_loss: Compliance loss for given density
    :param volume_loss: Volume loss for given density

    :param mode: A heuristic that changes ``scaler``
    :param constant: Used for ``mode='clip'`` and ``None`` for other ``mode``s

    :return: A scaler as the weight for `volume_loss` in weighted sum `compliance_loss + scaler * volume_loss`
    """
    with torch.no_grad():
        scaler = compliance_loss / volume_loss

        if mode == 'clip':
            if scaler >= constant:
                scaler = torch.clamp_max(scaler, max=constant)
                return scaler
            else:
                return scaler
        elif mode == 'equalize':
            return scaler
        elif mode == 'adaptive':
            scaler = compliance_loss / (3. * volume_loss)
            return scaler


def type_of_volume_constaint_satisfier(mode):
    """
    Says mode is hard or not i.e. change the density directly or add a loss term respectively

    """
    if mode == 'constrained_sigmoid': return True
    elif mode == 'constrained_projection': return True
    elif mode == 'add_mean': return False
    elif mode == 'one_sided_max': return False
    elif mode == 'maxed_barrier': return False
    elif mode == 'thresholded_barrier': return False
    elif mode == 'linear': return False
    else: raise ValueError('The mode "{}" does not exist'.format(mode))


def homogeneous_init(model, constant, projection):
    """
    Ensures the first output of model is a homogeneous field by zeroing out weights and initializing bias
      with a `constant`. This function is inplace.
    
    :param model: A `Module` model
    :param constant: A float scalar
    :param projection: The projection filtering used on raw output of MLP
    :return: None
    """

    def apply_homogeneous_init(m, constant):
            """
            Zero outs weights of last layer and initialize biases with `constant` value
            Used to ensure first output of neural network is homogeneous density field.

            :param m: Module m (rec: use module.apply(this method))
            """

            zx = torch.linspace(-10., 10., 10000)
            zy = projection(zx)
            tol = 1e-7
            mask = torch.abs(zy - constant) < tol
            while mask.sum() == 0:
                mask = torch.abs(zy - constant) < tol
                tol = tol * 10.
            constant = zx[mask][0].numpy().item()
            del zx, zy, mask

            classname = m.__class__.__name__
            if (classname.find('Linear') != -1):
                if ((m.weight.shape.__contains__(1)) or m.weight.shape.__contains__(2)):
                    torch.nn.init.normal_(m.weight, 0.0, 0.0001)
                    torch.nn.init.constant_(m.bias, constant)

    model.apply(partial(apply_homogeneous_init, constant=constant))
    sys.stderr.write('Homogenization has been applied on model with constant value: {}\n'.format(constant))
