import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from mpl_toolkits.axes_grid1 import ImageGrid

from multires_utils import prepare_resolutions

import pandas as pd
import cv2

import os
import glob
from datetime import datetime


def memory_usage_vis(csv_file, names=None):
    if names is None:
        names=['Dimensions', 'CPU-Mem', 'GPU-Mem']
    dataset = pd.read_csv(csv_file, header=None, names=names, index_col=0)
    plt.figure()
    dataset.plot.bar(figsize=(11, 4), align='edge', width=0.5, rot=0)
    plt.title('Memory Consumption')
    plt.xlabel('Dimensions')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.savefig(csv_file[:-3]+'jpg')


def loss_vis(compliance_loss_array, title, save=True, path=None, **kwargs):
    if path is None:
        path = 'tmp/'
    title_ = title
    if os.path.isfile(path + title + '.png'):
        title_ += str(int(datetime.timestamp(datetime.now())))
    if save:
        ylim = kwargs['ylim'] if 'ylim' in kwargs.keys() else 5000.0

        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(14, 10))
        plt.plot(np.arange(0, len(compliance_loss_array)), compliance_loss_array, label='compliance loss')
        plt.title('Compliance')
        plt.xlabel('Iteration')
        plt.ylabel('Compliance Loss')
        plt.ylim(0, ylim)
        plt.suptitle(title, fontsize=18)
        plt.savefig(path + title_ + '.png')
        plt.close()
    return title_

def density_vis(density, loss, gridDimensions, title, prediction=True, save=True, same_size=True, binary_loss=None, path=None):
    plt.rcParams.update({'font.size': 18})

    ratio = gridDimensions[0] / gridDimensions[1]
    # for including titles properly (not related to solution)
    if ratio == 1.0:  
        ratio = ratio * 3.5

    if path is None:
        path = 'tmp/'

    if prediction:
        if same_size:
            plt.figure(figsize=(ratio * 5, ratio//ratio * 5 + 1))
        else:
            plt.figure(figsize=(gridDimensions[0]/10, gridDimensions[1]/10+1))
        pred_density = -density.view(gridDimensions).detach().cpu().numpy()[:, :].T
        plt.imshow(pred_density, cmap='gray')
        if binary_loss is not None:
            plt.title('Prediction (loss={:.3f}, b-loss={:.3f}, vol={:.3f})'.format(loss, binary_loss, -pred_density.mean())) 
        else:   
            plt.title('Prediction (loss={:.3f}, vol={:.3f})'.format(loss, -pred_density.mean()))
        plt.suptitle('Prediction ('+title+')', fontsize=13)

        if save:
            plt.suptitle(title, fontsize=18)
            title_ = title
            if os.path.isfile(path + title + '.png'):
                title_ += str(int(datetime.timestamp(datetime.now())))
            plt.savefig(path + title_ + '.png')

    else:
        if same_size:
            plt.figure(figsize=(ratio * 5, ratio//ratio * 5 + 1))
        else:
            plt.figure(figsize=(gridDimensions[0]/10, gridDimensions[1]/10+1))
        
        plt.imshow(-density.reshape(gridDimensions[0], gridDimensions[1]).T, cmap='gray')
        
        if save:
            if binary_loss is not None:
                plt.title('{} (loss={:.3f}, b-loss={:.3f})'.format(title, loss, binary_loss))
            else:
                plt.title('{} (loss={:.3f})'.format(title, loss))
            title_ = title
            if os.path.isfile(path + title + '_gt.png'):
                title_ += str(int(datetime.timestamp(datetime.now())))
            plt.savefig(path + title_ + '_gt.png')
    
    plt.close()


def pred_gt_density_vis(pred, gt, gridDimensions, pred_loss, gt_loss, title, save=True, path=None):
    density = pred
    gt_densities = gt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pred_density = -density.view(gridDimensions).detach().cpu().numpy()[:, :].T
    axes[0].imshow(pred_density, cmap='gray')
    axes[0].set_title('Prediction (loss={:.3f}, vol={:.3f})'.format(pred_loss, -pred_density.mean()))
    axes[1].imshow(-gt_densities.reshape(gridDimensions[0], gridDimensions[1]).T, cmap='gray')
    axes[1].set_title('Ground Truth (loss={:.3f})'.format(gt_loss)) 
    fig.suptitle('VoxelFEM vs CNN VoxelFEM ('+title+')', fontsize=13)

    if path is None:
        path = 'logs/images/cnn/vanilla'

    if save:
        fig.suptitle(title, fontsize=18)
        if os.path.isfile(path + title + '.png'):
            title += str(int(datetime.timestamp(datetime.now())))
        plt.savefig(path + title + '.png')


def n_column_image_grid(title, image_list=None, path=None, patterns=None):
    if (image_list is None) and (path is None):
        raise Exception('"image_list" and "path" both cannot be "None"')

    if patterns is None:
        raise Exception('Provide patterns for file names (relative path)')
    
    if image_list is None:
        path = ''

    def key_for_sort(string: str):
        idx = string.find('_s') + 2  # scale value
        scale_str = string[idx: idx + 4]
        try:
            scale = float(scale_str)  # from 10.0 to 99.9
        except ValueError:
            scale = float(scale_str[:-1])  # from 0.1 to 9.9
        return scale

    images = []
    for p in patterns:
        images_path = glob.glob(p)
        images_path.sort(key=key_for_sort)
        images_col = []
        for j in images_path:
            images_col.append(cv2.imread(j, 0))
        images.append(images_col)
    
    cols = len(images[0])
    rows = len(patterns)

    factor = 1
    rows = rows * factor
    cols = cols // factor
    fig = plt.figure(figsize=(rows*5, cols*5))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(cols, rows),
                    axes_pad=0.1,
                    )
    
    images = np.array(images)
    images = list(images.transpose(1, 0, 2, 3).reshape(-1, images.shape[-2], images.shape[-1]))

    for ax, im in zip(grid, images):
        ax.tick_params(left=False, labelleft=False) #remove ticks
        ax.tick_params(bottom=False, labelbottom=False) #remove ticks
        ax.set_frame_on(True)
        ax.imshow(im[5:-240, 120:-120], cmap='gray')

    plt.savefig(path+title+'_.png')
    plt.close()

def plot_n_large_scale_fourfeat_outputs(title, patterns=None):
    patterns = [r'tmp/fourfeat_vscale/*150x50*MbbBeamSeed88*.png',
                r'tmp/fourfeat_vscale/*300x100*MbbBeamSeed88*.png']
    n_column_image_grid(title='fourfeat_150x50_VS_300x100_S[0.5-30]_MbbBeamSeed88', 
                        image_list=None, path='', patterns=patterns)

def upsampling_comparison_HARDCODED():  # TODO: make it general ez
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 13))

    # prepare x-axis
    resolution_steps = prepare_resolutions(5, 0, 81, 'ctf', 1)[:-1]
    base_resolution = [300, 100]
    domainCorners = [[0, 0], [3, 1]]
    resolutions = []
    for res in resolution_steps:
        value = tuple(np.array(base_resolution) + res * np.array(domainCorners[1]))
        resolutions.append('{}x{}'.format(value[0], value[1]))

    plt.clf()
    # prepare y-axis
    IDs = [4427678, 4427680, 4427682, 4427683, 4427685, 4427687, 1111111]
    IDs_scale = {4427678: 2, 4427680: 4, 4427682: 6, 4427683: 7, 4427685: 9,  4427687: 10, 1111111: 'SIMP'}
    for ID in IDs:
        with open('tmp/{}.log'.format(ID), 'r') as log:
            values = log.readlines()
            fc_string = values[1][5:-2].split(' ')
            fc_values = np.array([float(v) for v in fc_string if v != ''])
            # if ID == 1111111:  # TODO: currently interpolate values in between, until i get exact values.
            # fc_values = fc_values / np.max(fc_values)
            plt.plot(resolutions, fc_values, '-D', markevery=[np.argmin(fc_values[1:])+1], label='scale={}'.format(IDs_scale[ID]))

            # fc_string = values[1][5:-2]

    plt.ylim(top=320)
    # plt.ylim((0,1))
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.title('Comparison of upsampling power in our method and SIMP for increasing resolutions and scales',
            pad=25, fontsize=23)
    plt.tick_params(axis='both', pad=6)
    plt.xlabel('Resolutions', fontsize=20, labelpad=18)
    plt.xticks(rotation=90)
    plt.tick_params(axis='y', labelsize=14)
    plt.ylabel('Compliance value', fontsize=20, labelpad=18)
    plt.savefig('tmp/binary.png')
    plt.close()
