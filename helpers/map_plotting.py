import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from PIL import Image
import random
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.optimize import brentq, curve_fit, minimize_scalar
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter
from matplotlib import colors as mcolors
import matplotlib as mpl
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
import math
from tqdm import tqdm
from pathlib import Path

from .wiring_efficiency_utils import *

def sample_and_plot(distribution, num_samples, sample_idx, ori_map=None, full=False):

    M = distribution.shape[-1]

    # Convert distribution to PyTorch tensor and flatten for sampling
    dist_tensor = torch.tensor(distribution.flatten(), dtype=torch.float)
    
    # Sample S locations from the distribution
    indices = torch.multinomial(dist_tensor, num_samples, replacement=True)

    if full:
        indices = torch.where(dist_tensor>0)[0]
        num_samples = indices.shape[0]
    
    # Convert flat indices back to 2D indices
    y, x = np.unravel_index(indices.numpy(), (M, M))
    
    # Get the center coordinates
    center_x = sample_idx % M
    center_y = sample_idx // M

    x = x / M
    y = y / M
    center_x = center_x / M
    center_y = center_y / M

    
    # Display the original image with HSV colormap and save it
    plt.imshow(ori_map.cpu(), cmap='hsv')
    plt.axis('off')
    plt.savefig('figures/root_exports/original_image.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Load the saved image using PIL
    image = Image.open('figures/root_exports/original_image.png')
    image_np = np.array(image)
    
    # Pad the image before blurring to avoid losing corners
    pad_size = 25  # The same size as the Gaussian kernel
    padded_image_np = cv2.copyMakeBorder(image_np, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    
    # Apply Gaussian blur to the padded image
    blurred_padded_image = cv2.GaussianBlur(padded_image_np, (5,5), 0)
    
    # Remove the padding after blurring
    blurred_image = blurred_padded_image[pad_size:-pad_size, pad_size:-pad_size]
    
    # Convert the blurred image to a tensor
    blurred_image_tensor = TF.to_tensor(Image.fromarray(blurred_image))
    
    # Add batch dimension and convert to float
    blurred_image_tensor = blurred_image_tensor.unsqueeze(0).float()
    
    # Remove the batch dimension
    blurred_image_tensor = blurred_image_tensor.squeeze(0)
    
    # Convert tensor to numpy array for plotting
    blurred_image_tensor = blurred_image_tensor.permute(1, 2, 0).numpy()
    
    # Display the upsampled blurred image
    plt.figure(figsize=(5, 5))
    #plt.imshow(blurred_image_tensor, alpha=0.15)
    plt.axis('off')

    k = blurred_image_tensor.shape[0]
    # Add scatter to the sampled points with random scatter
    x_scatter = x + np.random.randn(num_samples) * 7e-3  # Add random scatter to x coordinates
    x_scatter = np.clip(x_scatter, 0, 1) * k
    y_scatter = y + np.random.randn(num_samples) * 7e-3  # Add random scatter to y coordinates
    y_scatter = np.clip(y_scatter, 0, 1) * k

    colors = [blurred_image_tensor[int(y), int(x)] for x, y in zip(np.round(x_scatter), np.round(y_scatter))]
    
    # Add scatter to the sampled points and draw lines from center to each point
    for i in range(len(x)):
        plt.plot([center_x*k, x_scatter[i]], [center_y*k, y_scatter[i]], color='black', linestyle='-', linewidth=1, alpha=0.2, zorder=1)  # More transparent lines

    #plt.xlim(130,200)
    #plt.ylim(165 ,230)
    plt.scatter(x_scatter, y_scatter, color=colors, s=200, alpha=0.8, zorder=2, edgecolors=None)  # Add transparency to the sampled points
    
    plt.scatter(center_x*k, center_y*k, color='white', s=400, zorder=3, edgecolors=None)  # Plot the center
    plt.scatter(center_x*k, center_y*k, color='black', s=200, zorder=4)  # Plot the center
    plt.axis('off')
    plt.savefig('figures/root_exports/samples.svg', bbox_inches='tight', pad_inches=0)
    plt.close()

    resized_ori_map = F.interpolate(ori_map[None,None], blurred_image_tensor.shape[0])[0,0]
    sampled_oris = [resized_ori_map[int(y), int(x)] for x, y in zip(np.round(x_scatter), np.round(y_scatter))]

    plt.hist(sampled_oris, bins=13)
    plt.axis('off')
    plt.savefig('figures/root_exports/ori_hist.svg', bbox_inches='tight', pad_inches=0)
    plt.close()


def animate(array, n_frames=None, cmap="viridis", interval=300):
    if n_frames is None:
        n_frames = array.shape[0]

    # Convert torch.Tensor → numpy if needed
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()

    vmin, vmax = array.min(), array.max()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6,6))
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])  # fill entire figure, no margins

    # Transparent figure background (optional)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("black")  # optional: set to match your data background

    im = ax.imshow(array[0], animated=True, cmap=cmap, vmin=vmin, vmax=vmax)

    def update(frame):
        im.set_array(array[frame])
        return (im,)

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=True, repeat=True
    )

    plt.close(fig)
    return HTML(anim.to_jshtml())


def show_map(model, network, random_sample=None):

    plt.figure(figsize=(12, 14))
    titles = [
        "Current Input", "Afferent Weights", "Current Aff Response", "Inhibitory weights",
        "Lateral correlations", "Current Response", "Current Response Histogram",
        "Orientation Map", "Orientation Histogram", "LRE", "Fourier domain", "Mean Histogram",
        "Reconstruction", "Thresholds", "Excitatory weights", "Mean Frs"
    ]

    # Displaying the model's current input
    img = model.current_input[0, 0].detach().cpu()
    #c = model.rf_size // 2
    #img = img[c:-c,c:-c]
    plt.subplot(4, 4, 1)
    plt.imshow(img, cmap=cm.Greys)
    plt.title(titles[0])

    reco_input = network['activ'](network['model'](model.current_response))[0,0].detach().cpu()
    # nn reconstruction
    plt.subplot(4, 4, 14)
    plt.imshow(reco_input)
    plt.title('reco')

    # Afferent weights of a random sample
    aff_weights = model.get_aff_weights()[random_sample, 0].detach().cpu().clone()
    aff_weights[0,0] = 0
    plt.subplot(4, 4, 12)
    plt.imshow(aff_weights)
    plt.title(titles[1])

    # Afferent weights of a random sample
    net_afferent = (
        (model.current_afferent - model.aff_baseline) * model.aff_gain
    )[0, 0].detach().cpu()
    net_afferent_bar = net_afferent + 0
    net_afferent_bar[0,0] = 0
    plt.subplot(4, 4, 2)
    plt.imshow(net_afferent_bar)
    plt.title(titles[2])

    # Lateral correlations of the random sample
    plt.subplot(4, 4, 4)
    #plotvar = model.long_interactions[random_sample, 0]#* model.eye[random_sample, 0]
    inh = model.inh
    plotvar = inh[random_sample, 0].detach().cpu().clone()
    plotvar[0,0] = 0
    plt.imshow(plotvar)
    plt.title(titles[3])

    # Lateral weights excitation of the random sample
    plt.subplot(4, 4, 5)
    plotvar = model.lateral_correlations[random_sample, 0].detach().cpu()
    plt.imshow(plotvar)
    plt.title(titles[4])

    # Model's current response
    plt.subplot(4, 4, 6)
    plt.imshow(model.current_response[0, 0].detach().cpu(), cmap=cm.Greys)
    plt.title(titles[5])

    # Histogram of the current response
    plt.subplot(4, 4, 7)
    hist = model.current_response.flatten().detach().cpu().numpy()
    plt.hist(hist[hist > 0], range=(0,1))
    plt.title(titles[6])

    # Generate and display orientation and phase maps
    M = int(np.sqrt(model.afferent_weights.shape[0]))  # Assuming MxM grid for reshaping
    ori_map = compute_orientation_maps(model, model.current_input.shape[-1], device=model.device)[0].cpu()
    #ori_map = detect_orientation_map_from_aff_weights(model.get_aff_weights())['pref'].view(M,M).cpu()
    
    # Orientation map
    plt.subplot(4, 4, 9)
    plt.imshow(ori_map, cmap='hsv')
    plt.title(titles[7])

    # Orientation histogram
    plt.subplot(4, 4, 10)
    hist_map = ori_map.flatten()
    plt.hist(hist_map.detach().cpu().numpy(), bins=10)
    plt.title(titles[8])

    # Retinotopic Bias
    plt.subplot(4, 4, 11)
    _,ring,_ = get_typical_dist_fourier(ori_map, 0)
    plt.imshow(ring.cpu(), cmap=cm.Greys)
    plt.title(titles[10])

    plt.subplot(4, 4, 8)
    avg_hist = model.avg_hist.detach().cpu()
    plt.stairs(avg_hist, torch.linspace(0, 1, avg_hist.numel() + 1), fill=True)
    plt.title(titles[11])

    # thresholds
    #thresholds[0,0] = 0
    plt.subplot(4, 4, 3)
    plt.imshow(model.thresholds.view(M,M).detach().cpu())
    plt.title('thresh')

    exc = model.s_exc if not model.microcolumnar else model.l_exc
    plt.subplot(4, 4, 13)
    plt.imshow(exc[random_sample,0].view(M,M).detach().cpu())
    plt.title(titles[-7])

    plt.subplot(4,4,15)
    response_trace = model.response_tracker[:model.iterations].sum([1,2,3]).detach().cpu()
    plt.plot(response_trace, color='black')
    plt.title('mean act')

    plt.subplot(4,4,16)
    plt.imshow(model.lateral_correlations[random_sample,0].detach().cpu())
    plt.title('exc_corr')


    print('Net Afferent Max: {:.3f}, Net Afferent Min: {:.3f}'. format(net_afferent.max(), net_afferent.min()))
    print('L4 Thresholds Max: {:.3f}, L4 Thresholds Min: {:.3f}'. format(model.thresholds.max(), model.thresholds.min()))
    print('Mean current response: {:.3f}'.format(model.current_response.mean()))
    loss = torch.mean((reco_input - img)**2)
    print('Reco loss: {:.3f}%'.format(loss))


    plt.show()


def show_map_l3(model, network, random_sample=None):

    plt.figure(figsize=(12, 14))
    titles = [
        "Current Input", "Afferent Weights", "Current Aff Response", "Inhibitory weights",
        "Lateral correlations", "Current Response", "Current Response Histogram",
        "Orientation Map", "Orientation Histogram", "LRE", "Fourier domain", "Mean Histogram",
        "Reconstruction", "Thresholds", "Excitatory weights", "Mean Frs"
    ]

    # Displaying the model's current input
    img = model.current_input[0, 0].detach().cpu()
    #c = model.rf_size // 2
    #img = img[c:-c,c:-c]
    plt.subplot(4, 4, 1)
    plt.imshow(img, cmap=cm.Greys)
    plt.title(titles[0])

    reco_input = network['activ'](network['model'](model.current_response_l3))[0,0].detach().cpu()
    plt.subplot(4, 4, 14)
    plt.imshow(reco_input)
    plt.title('reco')

    # Afferent weights of a random sample
    aff_weights = model.get_aff_weights_l3()[random_sample, 0].detach().cpu().clone()
    aff_weights[0,0] = 0
    plt.subplot(4, 4, 12)
    plt.imshow(aff_weights)
    plt.title(titles[1])

    if False:
        # Afferent weights of a random sample
        net_afferent = model.current_afferent_l3[0,0].detach().cpu() - model.thresholds_l3[0,0].detach().cpu()
        net_afferent_bar = net_afferent + 0
        net_afferent_bar[0,0] = 0
        plt.subplot(4, 4, 2)
        plt.imshow(net_afferent_bar)
        plt.title(titles[2])

        # Lateral correlations of the random sample
        plt.subplot(4, 4, 4)
        #plotvar = model.long_interactions[random_sample, 0]#* model.eye[random_sample, 0]
        inh = model.inh_l3
        plotvar = inh[random_sample, 0]
        plotvar[0,0] = 0
        plt.imshow(plotvar.detach().cpu())
        plt.title(titles[3])

    # Lateral weights excitation of the random sample
    plt.subplot(4, 4, 5)
    plotvar = model.lateral_correlations_l3[random_sample, 0].detach().cpu()
    plt.imshow(plotvar)
    plt.title(titles[4])

    # Model's current response
    plt.subplot(4, 4, 6)
    plt.imshow(model.current_response_l3[0, 0].detach().cpu(), cmap=cm.Greys)
    plt.title(titles[5])

    # Histogram of the current response
    plt.subplot(4, 4, 7)
    hist = model.current_response_l3.flatten().detach().cpu().numpy()
    plt.hist(hist[hist > 0], range=(0,1))
    plt.title(titles[6])

    M = int(np.sqrt(model.afferent_weights.shape[0]))  # Assuming MxM grid for reshaping

    exc = model.global_exc_l3
    plt.subplot(4, 4, 13)
    plt.imshow(exc[random_sample,0].view(M,M).detach().cpu())
    plt.title(titles[-7])

    if model.microcolumnar:
        exc = model.l_exc_l4_l3
        plt.subplot(4, 4, 2)
        plt.imshow(exc[random_sample,0].view(M,M).detach().cpu())
        plt.title(titles[-7])

    ori_map = compute_orientation_maps(
        model, model.current_input.shape[-1], device=model.device
    )[1].detach().cpu()
    # -------------------------------------------------------------------------------
    
    # Orientation map
    plt.subplot(4, 4, 9)
    plt.imshow(ori_map, cmap='hsv')
    plt.title(titles[7])

    # Orientation histogram
    plt.subplot(4, 4, 10)
    hist_map = ori_map.flatten().numpy()
    plt.hist(hist_map, bins=10)
    plt.title(titles[8])

    # Retinotopic Bias
    plt.subplot(4, 4, 11)
    _,ring,_ = get_typical_dist_fourier(ori_map, 0)
    plt.imshow(ring.cpu(), cmap=cm.Greys)
    plt.title(titles[10])

    plt.subplot(4, 4, 8)
    avg_hist_l3 = model.avg_hist_l3.detach().cpu()
    plt.stairs(
        avg_hist_l3,
        torch.linspace(0, 1, avg_hist_l3.numel() + 1),
        fill=True,
    )
    plt.title(titles[11])

    # thresholds
    #thresholds[0,0] = 0
    plt.subplot(4, 4, 3)
    plt.imshow(model.thresholds_l3.view(M,M).detach().cpu())
    plt.title('thresh')

    plt.subplot(4,4,15)
    response_trace_l3 = (
        model.response_tracker_l3[:model.iterations]
        .sum([1,2,3])
        .detach()
        .cpu()
    )
    plt.plot(response_trace_l3, color='black')
    plt.title('mean act')

    # ---- Create L4 coordinate maps (retinotopy field) --------------------------
    coords = torch.linspace(-1, 1, M, device=model.device)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    
    xx = xx.view(1,1,M,M)
    yy = yy.view(1,1,M,M)
    
    # ---- Extract local L4 patches for each L3 neuron ---------------------------
    x_patch = extract_patches(xx, model.rf_grids_l3)
    y_patch = extract_patches(yy, model.rf_grids_l3)
    
    w3 = model.get_aff_weights_l3()
    
    # normalize weights inside each RF
    w3 = w3 / (w3.sum([1,2,3], keepdim=True) + 1e-8)
    
    # ---- Weighted spatial average ---------------------------------------------
    cx = (x_patch * w3).sum([1,2,3])
    cy = (y_patch * w3).sum([1,2,3])
    
    l3_centers = torch.stack([cx, cy], dim=1)

    plt.subplot(4, 4, 16)

    grid_x = l3_centers[:,0].reshape(M, M).detach().cpu()
    grid_y = l3_centers[:,1].reshape(M, M).detach().cpu()
    
    for i in range(M):
        plt.plot(grid_x[i,:], grid_y[i,:], 'k-', linewidth=0.5)
    for j in range(M):
        plt.plot(grid_x[:,j], grid_y[:,j], 'k-', linewidth=0.5)
    
    plt.gca().set_aspect('equal')
    plt.xticks([])
    plt.yticks([])


    plt.show()


def plot_orientation_histogram(
    orientation_map,
    *,
    ax=None,
    bins=10,
    font_size=18,
    show=True,
):
    """Plot orientation coverage using the established HSV bar style."""

    values = np.asarray(torch.as_tensor(orientation_map).detach().cpu()).ravel()
    counts, edges = np.histogram(values, bins=bins, range=(0, np.pi))
    proportions = counts / max(1, counts.sum())
    colors = cm.get_cmap("hsv")(np.linspace(0, 1, bins, endpoint=False))
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    for index in range(bins):
        ax.bar(
            edges[index],
            proportions[index],
            width=edges[index + 1] - edges[index],
            align="edge",
            color=colors[index],
            edgecolor="none",
        )
    ax.set_xticks(
        (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi),
        labels=(0, 45, 90, 135, 180),
    )
    ax.set_xlabel("orientation (°)", fontsize=font_size)
    ax.set_ylabel("proportion", fontsize=font_size)
    ax.tick_params(labelsize=font_size * 0.75)
    ax.spines[["top", "right"]].set_visible(False)
    if owns_figure:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_retinotopy_grid(
    topography,
    *,
    ax=None,
    max_lines=15,
    show_points=False,
    show=True,
):
    """Plot a precomputed cortical retinotopy in fishnet style."""

    topography = torch.as_tensor(topography).detach().cpu()
    if topography.ndim != 3 or topography.shape[-1] != 2:
        raise ValueError("topography must have shape (side, side, 2)")
    step = max(1, topography.shape[0] // max_lines)
    sampled = topography[::step, ::step]
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    ax.add_collection(LineCollection(sampled.numpy(), linewidth=0.8, color="black"))
    ax.add_collection(
        LineCollection(
            sampled.permute(1, 0, 2).numpy(), linewidth=0.8, color="black"
        )
    )
    if show_points:
        ax.scatter(sampled[..., 0], sampled[..., 1], s=4, color="black", zorder=2)
    ax.autoscale()
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_axis_off()
    if owns_figure:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_receptive_field_mosaic(
    receptive_fields,
    *,
    ax=None,
    cmap="Greys",
    show=True,
):
    """Display a square mosaic of neighbouring ON-channel receptive fields."""

    fields = torch.as_tensor(receptive_fields).detach().cpu().float()
    if fields.ndim == 4:
        fields = fields[:, 0]
    if fields.ndim != 3:
        raise ValueError("receptive_fields must have shape (count, channels, y, x)")
    grid_side = int(math.ceil(math.sqrt(len(fields))))
    field_side = fields.shape[-1]
    mosaic = torch.zeros(grid_side * field_side, grid_side * field_side)
    for index, field in enumerate(fields):
        row, column = divmod(index, grid_side)
        mosaic[
            row * field_side : (row + 1) * field_side,
            column * field_side : (column + 1) * field_side,
        ] = field
    lower = float(mosaic.min())
    upper = float(mosaic.max().clamp_min(lower + 1e-11))
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    ax.imshow(mosaic, cmap=cmap, vmin=lower, vmax=upper)
    ax.set_axis_off()
    if owns_figure:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_absolute_phases(rfs,target_channel=0,figpath=None,ori_map=None):

    # exctracting useful params
    rfs = rfs.clone().cpu()
    aff_units = rfs.shape[-1]
    sheet_units = int(np.sqrt(rfs.shape[0]))
    channels = 1
    
    # making a meshgrid to localise any points within the aff cf
    rng = torch.arange(aff_units) - aff_units//2
    coordinates = torch.meshgrid(rng,rng)
    coordinates = torch.stack(coordinates)[None]
    
    # averaging over all locations to detect the greatest intensity
    rfs = rfs.view(-1,channels,aff_units,aff_units)[:,target_channel][:,None]
    rfs = rfs.repeat(1,2,1,1)
    c = (coordinates * rfs)
    c = c.sum([2,3]) * 2

    # organising everything into a grid and plotting the centre of mass of each point
    rng = torch.arange(sheet_units)
    topography = torch.meshgrid(rng,rng)
    topography = torch.stack(topography)
    topography = topography.reshape(2,-1)
    topography = (topography.T.float() + c).T

    plt.figure(figsize=(10,10))
    if ori_map is not None:
        plt.scatter(topography[0],topography[1], s=50, c=ori_map.flatten() / np.pi, cmap='hsv', edgecolor='black')
    else:
        plt.scatter(topography[0],topography[1], s=10, color='black')

    # plotting the lines of the grid
    topography = topography.T.view(sheet_units,sheet_units,2)
    segs1 = topography
    segs2 = segs1.permute(1,0,2)
    plt.gca().add_collection(LineCollection(segs1, linewidth=1, color='black', zorder=-1))
    plt.gca().add_collection(LineCollection(segs2, linewidth=1, color='black', zorder=-1))

    #plt.ylim(0, 40)
    #plt.xlim(0, 40)

    if figpath:
        plt.savefig(figpath)

    plt.show()

    return topography.view(-1,2)


def plot_mexican_hat(Z, r=None):
    # Get dimensions
    h, w = Z.shape
    
    # Generate X and Y grid centered at 0 (so circle mask is symmetric)
    x = np.linspace(-(w-1)/2, (w-1)/2, w)
    y = np.linspace(-(h-1)/2, (h-1)/2, h)
    X, Y = np.meshgrid(x, y)
    
    # Create a mask for outside the circle
    if r is not None:
        mask = np.sqrt(X**2 + Y**2) > r
        Z = Z.clone()
        Z[mask] = np.nan  # masked region

    # Set up figure and 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # Colormap (blue for negative, red for positive)
    cmap = cm.coolwarm
    norm = plt.Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))

    # Plot surface (NaNs are simply skipped -> background shows through)
    surf = ax.plot_surface(X, Y, Z, facecolors=cmap(norm(Z)),
                           rstride=1, cstride=1, antialiased=True)

    # Remove axes for a clean look
    ax.set_axis_off()
    
    # Set camera view
    ax.view_init(elev=20, azim=80)

    plt.tight_layout()
    plt.show()


def plot_tensor_grid(tensor, s, figpath=None):
    """
    Plots the first s^2 slices from the last dimension of a tensor of shape (N, N, N^2)
    in an s x s grid of subplots with minimal spacing and no axes/labels.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (N, N, N^2)
        s (int): Number of rows and columns in the subplot grid
    """
    N = tensor.shape[0]
    assert tensor.shape[1] == N, "Tensor must be cubic in first two dims"
    assert tensor.shape[2] >= s**2, "Tensor's last dimension must have at least s^2 elements"
    
    # Select first s^2 slices along the last dimension
    slices = tensor[:, :, :s**2]  # shape: (N, N, s^2)
        
    # Create figure with tight layout
    fig, axes = plt.subplots(s, s, figsize=(10,10))
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    for i in range(s**2):
        axes[i].imshow(slices[:, :, i], cmap='Greys')
        axes[i].axis('off')  # remove axes
    
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    if figpath:
        plt.savefig(figpath, dpi=300)
    
    plt.show()

    
def plot_umap_with_angles_3d(
    code_tracker,
    variable,
    n_components=3,
    n_neighbors=15,
    min_dist=0.3,
    metric="euclidean",
    random_state=42,
    fs=22,
    figpath=None
):
    """
    Plot UMAP embedding of codes with HSV color based on `variable`.
    Axes, grids, ticks, panes, and labels are completely removed.
    """

    assert n_components in (2, 3), "n_components must be 2 or 3"

    # Convert codes to 2D array [N, D]
    codes = torch.cat([c.view(1, -1) for c in code_tracker], dim=0)
    codes = codes.detach().cpu().numpy()

    variable = np.asarray(variable, dtype=float)
    variable = variable - variable.min()
    variable = variable / (variable.max() + 1e-11)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    embedding = reducer.fit_transform(codes)

    # HSV → RGB
    hsv_colors = np.zeros((len(variable), 3))
    hsv_colors[:, 0] = variable
    hsv_colors[:, 1] = 1.0
    hsv_colors[:, 2] = 1.0
    rgb_colors = mcolors.hsv_to_rgb(hsv_colors)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(embedding[:, 0], embedding[:, 1], c=rgb_colors, s=20)

        # Remove everything
        ax.set_axis_off()
        ax.set_aspect("equal")
        plt.tight_layout()
        return

    # -------- 3D --------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=rgb_colors,
        s=50
    )

    # Remove absolutely everything
    ax.set_axis_off()
    ax.grid(False)

    # Extra safety: remove panes if backend still draws them
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_alpha(0)

    ax.view_init(elev=90, azim=-270)
    plt.tight_layout()

    if figpath:
        plt.savefig(figpath, dpi=300)

    plt.show()
    return embedding


def plot_precomputed_umap_with_angles_3d(
    embedding,
    variable,
    *,
    title=None,
    point_size=20,
    ax=None,
    figpath=None,
    show=True,
):
    """Plot an existing 2D/3D embedding with the repository's HSV styling."""

    embedding = np.asarray(embedding)
    if embedding.ndim != 2 or embedding.shape[1] not in (2, 3):
        raise ValueError("embedding must have shape (samples, 2) or (samples, 3)")
    variable = np.asarray(variable, dtype=float).reshape(-1)
    if len(variable) != len(embedding):
        raise ValueError("variable and embedding must contain the same samples")
    variable = variable - variable.min()
    variable = variable / (variable.max() + 1e-11)

    hsv_colors = np.zeros((len(variable), 3))
    hsv_colors[:, 0] = variable
    hsv_colors[:, 1] = 1.0
    hsv_colors[:, 2] = 1.0
    rgb_colors = mcolors.hsv_to_rgb(hsv_colors)

    owns_figure = ax is None
    if owns_figure:
        if embedding.shape[1] == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    if embedding.shape[1] == 2:
        ax.scatter(embedding[:, 0], embedding[:, 1], c=rgb_colors, s=point_size)
        ax.set_axis_off()
        ax.set_aspect("equal")
    else:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=rgb_colors,
            s=point_size,
        )
        ax.set_axis_off()
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0))
            axis.line.set_alpha(0)
        ax.view_init(elev=90, azim=-270)

    if title:
        ax.set_title(title)
    if owns_figure:
        fig.tight_layout()
    if figpath:
        fig.savefig(figpath, dpi=300)
    if show:
        plt.show()
    return fig, ax


def collect_and_plot_grating_umap_3d(
    model,
    crop_size,
    device=None,
    n=10000,
    wavelength=6,
    include_noise=False,
    fit_umap_on_noise=False,
    pca_compress=None,
    use_l3=False,
    clean_noise_gamma=0.0,
    noisy_noise_gamma=0.05,
    n_neighbors=15,
    min_dist=0.3,
    metric="euclidean",
    random_state=0,
):
    """
    Collect grating responses and plot 3D UMAP embeddings with Plotly.

    The first plot always fits UMAP on clean responses and plots clean responses.
    When include_noise=True, a second plot shows noisy responses, either transformed
    through the clean fit or fitted independently on noisy responses.

    When pca_compress is a positive integer, responses are compressed to that many
    PCA dimensions before UMAP. The clean PCA fit is reused when noisy responses
    are transformed through the clean-fit UMAP. If UMAP is fitted independently
    on noisy responses, PCA is also fitted independently on the noisy responses.
    Pass pca_compress=None to send flattened responses directly to UMAP.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if pca_compress is not None:
        if isinstance(pca_compress, (bool, np.bool_)) or not isinstance(
            pca_compress, (int, np.integer)
        ):
            raise TypeError("pca_compress must be a positive integer or None.")
        if pca_compress <= 0:
            raise ValueError("pca_compress must be a positive integer or None.")

    if device is None:
        device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")

    def _flatten_codes(codes):
        if torch.is_tensor(codes):
            codes = codes.detach().cpu().numpy()
        else:
            codes = np.asarray(codes)
        return codes.reshape(codes.shape[0], -1)

    def _fit_pca(codes, label):
        if pca_compress is None:
            return codes, None
        max_components = min(codes.shape)
        if pca_compress > max_components:
            raise ValueError(
                f"pca_compress={pca_compress} exceeds the maximum possible "
                f"{label} PCA dimensions ({max_components})."
            )
        reducer = PCA(n_components=int(pca_compress), random_state=random_state)
        return reducer.fit_transform(codes), reducer

    def _response():
        attr = "current_response_l3" if use_l3 else "current_response"
        if not hasattr(model, attr):
            raise AttributeError(f"model has no {attr}; run with use_l3=False or enable L2/3.")
        return getattr(model, attr).detach().cpu().clone()

    def _scatter_trace(embedding, orientations, name, showscale=True):
        orientations = np.asarray(orientations, dtype=float).reshape(-1) % 180
        return go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=orientations,
                colorscale="HSV",
                cmin=0,
                cmax=180,
                colorbar=dict(title="orientation (mod 180)") if showscale else None,
                showscale=showscale,
                opacity=0.85,
            ),
            name=name,
            text=[f"ori={o:.1f}" for o in orientations],
            hovertemplate="%{text}<br>x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}",
        )

    def _plot_embeddings(plots, title):
        if len(plots) == 1:
            fig = go.Figure()
            fig.add_trace(_scatter_trace(plots[0]["embedding"], plots[0]["orientations"], plots[0]["name"]))
            fig.update_layout(
                title=title,
                scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3"),
                width=800,
                height=700,
            )
            fig.show()
            return fig

        fig = make_subplots(
            rows=1,
            cols=len(plots),
            specs=[[{"type": "scene"} for _ in plots]],
            subplot_titles=[p["title"] for p in plots],
            horizontal_spacing=0.02,
        )

        for col, plot in enumerate(plots, start=1):
            fig.add_trace(
                _scatter_trace(
                    plot["embedding"],
                    plot["orientations"],
                    plot["name"],
                    showscale=(col == len(plots)),
                ),
                row=1,
                col=col,
            )

        scene_layout = dict(xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3")
        layout = {"title": title, "width": 1400, "height": 700}
        for idx in range(1, len(plots) + 1):
            layout["scene" if idx == 1 else f"scene{idx}"] = scene_layout
        fig.update_layout(
            **layout,
        )
        fig.show()
        return fig

    clean_codes = []
    noisy_codes = []
    angles = []

    for _ in tqdm(range(n)):
        angle = np.random.uniform(0, 360)
        phase = np.random.uniform(0, 360)
        angles.append(angle % 180)
        img = generate_sinusoidal_grating(crop_size, wavelength, angle, phase).unsqueeze(0).to(device) * 1.3

        with torch.no_grad():
            model(img, adaptation=False, noise_gamma=clean_noise_gamma, layer_3=use_l3)
            clean_codes.append(_response())

            if include_noise:
                model(img, adaptation=False, noise_gamma=noisy_noise_gamma, layer_3=use_l3)
                noisy_codes.append(_response())

    clean_codes = torch.cat(clean_codes, dim=0)
    clean_codes_np = _flatten_codes(clean_codes)
    clean_umap_input, clean_pca = _fit_pca(clean_codes_np, "clean")
    angles_np = np.asarray(angles, dtype=float)

    clean_reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=3,
        metric=metric,
        random_state=random_state,
    )
    clean_embedding = clean_reducer.fit_transform(clean_umap_input)
    layer_name = "L3" if use_l3 else "L4"
    pca_title = "" if pca_compress is None else f" after PCA({pca_compress})"
    clean_title = f"{layer_name} clean codes, UMAP fit on clean{pca_title}"

    result = {
        "angles": angles_np,
        "pca_compress": pca_compress,
        "clean_codes": clean_codes,
        "clean_umap_input": clean_umap_input,
        "clean_pca": clean_pca,
        "clean_embedding": clean_embedding,
        "clean_reducer": clean_reducer,
    }

    plots = [{
        "embedding": clean_embedding,
        "orientations": angles_np,
        "title": clean_title,
        "name": "clean",
    }]

    if include_noise:
        noisy_codes = torch.cat(noisy_codes, dim=0)
        noisy_codes_np = _flatten_codes(noisy_codes)

        if fit_umap_on_noise:
            noisy_umap_input, noisy_pca = _fit_pca(noisy_codes_np, "noisy")
            noisy_reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=3,
                metric=metric,
                random_state=random_state,
            )
            noisy_embedding = noisy_reducer.fit_transform(noisy_umap_input)
            noisy_title = f"{layer_name} noisy codes, UMAP fit on noisy{pca_title}"
        else:
            noisy_pca = clean_pca
            noisy_umap_input = (
                noisy_codes_np
                if clean_pca is None
                else clean_pca.transform(noisy_codes_np)
            )
            noisy_reducer = clean_reducer
            noisy_embedding = clean_reducer.transform(noisy_umap_input)
            noisy_title = (
                f"{layer_name} noisy codes, transformed onto clean-fit UMAP{pca_title}"
            )

        plots.append({
            "embedding": noisy_embedding,
            "orientations": angles_np,
            "title": noisy_title,
            "name": "noisy",
        })
        result.update({
            "noisy_codes": noisy_codes,
            "noisy_umap_input": noisy_umap_input,
            "noisy_pca": noisy_pca,
            "noisy_embedding": noisy_embedding,
            "noisy_reducer": noisy_reducer,
        })

    fig = _plot_embeddings(plots, f"{layer_name} grating-response UMAP")
    result["fig"] = fig
    result["clean_fig"] = fig
    if include_noise:
        result["noisy_fig"] = fig

    return result


    
def analyze_orientation_periodicity(
    ori_map: torch.Tensor,
    max_scale_mm: float,
    n_bins: int = 50,
    fs: int = 36,
    figpath=None
):
    """
    Analyze hidden spatial periodicity in an orientation map.

    Parameters
    ----------
    ori_map : torch.Tensor
        NxN tensor with values in [0, pi)
    max_scale_mm : float
        Physical size of the map (e.g., 1.0 for 1 mm)
    n_bins : int
        Number of radial bins for averaging
    fs : int
        Font size for plotting

    Returns
    -------
    radii_mm : np.ndarray
        Radial distances (mm)
    autocorr_radial : np.ndarray
        Radially averaged autocorrelation
    freq_centers : np.ndarray
        Radially averaged spatial frequencies (cycles/mm)
    power_radial : np.ndarray
        Radially averaged power spectrum
    """

    # ---------- Safety checks ----------
    assert ori_map.ndim == 2 and ori_map.shape[0] == ori_map.shape[1], \
        "ori_map must be square NxN"
    assert ori_map.min() >= 0 and ori_map.max() <= np.pi + 1e-6, \
        "Orientation values must be in [0, pi)"

    N = ori_map.shape[0]
    dx = max_scale_mm / N  # mm per pixel

    # ---------- Handle circular orientation ----------
    ori_complex = torch.exp(2j * ori_map)
    ori_complex = ori_complex - ori_complex.mean()  # remove mean

    # ---------- 2D autocorrelation ----------
    fft_map = torch.fft.fft2(ori_complex)
    power = fft_map * torch.conj(fft_map)
    autocorr = torch.fft.ifft2(power).real
    autocorr = torch.fft.fftshift(autocorr)
    autocorr /= autocorr.max()

    # ---------- 2D power spectrum ----------
    power_spectrum = torch.abs(fft_map) ** 2
    power_spectrum = torch.fft.fftshift(power_spectrum)

    # ---------- Radial coordinates ----------
    coords = torch.arange(-N//2, N//2)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)
    rr_mm = rr.numpy() * dx

    # ---------- Radial bins ----------
    r_max = rr_mm.max()
    bins = np.linspace(0, r_max, n_bins + 1)

    def radial_average(field, rr_array):
        field = field.numpy()
        radial_mean = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (rr_array >= bins[i]) & (rr_array < bins[i + 1])
            if np.any(mask):
                radial_mean[i] = field[mask].mean()
        return radial_mean

    # ---------- Compute radially averaged autocorr ----------
    autocorr_radial = radial_average(autocorr, rr_mm)

    # ---------- Compute radially averaged power spectrum using frequency bins ----------
    freqs = np.fft.fftfreq(N, d=dx)  # cycles/mm
    fx, fy = np.meshgrid(freqs, freqs, indexing="ij")
    f_r = np.sqrt(fx**2 + fy**2)

    f_max = f_r.max()
    bins_f = np.linspace(0, f_max, n_bins + 1)
    power_spectrum_np = power_spectrum.numpy()
    power_radial = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (f_r >= bins_f[i]) & (f_r < bins_f[i+1])
        if np.any(mask):
            power_radial[i] = power_spectrum_np[mask].mean()
    freq_centers = 0.5 * (bins_f[:-1] + bins_f[1:])

    # ---------- Axes for autocorr ----------
    radii_mm = 0.5 * (bins[:-1] + bins[1:])

    # ---------- Plot ----------
    plt.figure(figsize=(6.5,5))
    ax = plt.gca()

    # Autocorrelation
    ax.spines[['top', 'right']].set_visible(False)
    ax.plot(radii_mm[:N//9], autocorr_radial[:N//9], lw=3, color='black')
    ax.set_xlabel("distance (mm)", fontsize=fs)
    ax.set_ylabel("mean autocorr.", fontsize=fs)
    ax.tick_params(labelsize=fs*0.8)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_ylim(-0.4, 1.1)

    plt.tight_layout()

    if figpath:
        plt.savefig(figpath)
    
    plt.show()

    return radii_mm, autocorr_radial, freq_centers, power_radial


def plot_orientation_periodicity_profile(
    radius,
    autocorrelation,
    *,
    ax=None,
    x_label="distance (mm)",
    font_size=18,
    ylim=None,
    label=None,
    color="black",
    line_style="-",
    figpath=None,
    show=True,
):
    """Plot a precomputed axial-orientation autocorrelation profile."""

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(6.5, 5))
    else:
        fig = ax.figure
    ax.plot(
        radius,
        autocorrelation,
        lw=3,
        color=color,
        linestyle=line_style,
        label=label,
    )
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel("mean autocorr.", fontsize=font_size)
    ax.tick_params(labelsize=font_size * 0.8)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.spines[["top", "right"]].set_visible(False)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if label:
        ax.legend(frameon=False)
    if owns_figure:
        fig.tight_layout()
    if figpath:
        fig.savefig(figpath)
    if show:
        plt.show()
    return fig, ax


    
def plot_sparse_tensor(
    tensor: torch.Tensor,
    highlight_idx: int,
    figpath=None,
    *,
    ax=None,
    show=True,
    marker_size=300,
    edge_linewidth=2,
):
    """
    Scatter plot of a sparse NxN tensor.
    
    Parameters
    ----------
    tensor : torch.Tensor
        NxN tensor containing only 0s and 1s
    highlight_idx : int
        Linear index in [0, N^2 - 1] to highlight
    """
    assert tensor.dim() == 2 and tensor.shape[0] == tensor.shape[1], "Tensor must be NxN"
    
    N = tensor.shape[0]
    assert 0 <= highlight_idx < N * N, "highlight_idx out of range"

    # Get coordinates of ones
    rows, cols = torch.nonzero(tensor, as_tuple=True)

    # Convert linear index to (row, col)
    hi_row = highlight_idx % N
    hi_col = highlight_idx // N

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(10,10))
    else:
        fig = ax.figure

    # Plot ones as black filled points
    ax.scatter(
        cols,
        rows,
        facecolors=(0, 0, 0, 0.),
        s=marker_size,
        edgecolor='black',
        linewidth=edge_linewidth,
    )

    # Highlight selected unit with larger grey circle
    #ax.scatter(hi_col, hi_row, c='grey', s=400, edgecolors='black')

    # Formatting
    ax.set_aspect('equal')
    ax.invert_yaxis()  # matrix-style orientation
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove top/right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if figpath:
        fig.savefig(figpath, dpi=300)

    if show:
        plt.show()
    return fig, ax


def plot_patchiness_profile(
    frequency,
    mean_profile,
    std_profile,
    profile_count,
    *,
    ax=None,
    max_freq_cyc_per_mm=10.0,
    band="sem",
    alpha=0.25,
    linewidth=4.0,
    font_size=18,
    figpath=None,
    show=True,
):
    """Plot a precomputed Van-Hooser-style aligned Fourier profile."""

    frequency = np.asarray(frequency)
    mean_profile = np.asarray(mean_profile)
    std_profile = np.asarray(std_profile)
    keep = (frequency >= 0) & (frequency <= max_freq_cyc_per_mm)
    if band.lower() == "sem":
        spread = std_profile / np.sqrt(max(1, int(profile_count)))
    elif band.lower() == "std":
        spread = std_profile
    else:
        raise ValueError("band must be 'sem' or 'std'.")

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    x = frequency[keep]
    y = mean_profile[keep]
    spread = spread[keep]
    ax.fill_between(x, y - spread, y + spread, alpha=alpha, color="0.7")
    ax.plot(x, y, color="black", linewidth=linewidth)
    ax.set_xlabel("spatial freq. (cycles/mm)", fontsize=font_size)
    ax.set_ylabel("mean Fourier coef.", fontsize=font_size)
    ax.tick_params(labelsize=font_size * 0.8)
    ax.spines[["top", "right"]].set_visible(False)
    if owns_figure:
        fig.tight_layout()
    if figpath:
        fig.savefig(figpath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def analyze_patchiness(connection_fields: torch.Tensor,
                       pixel_size_mm: float = 0.022,
                       mask_radius_mm: float = 0.5,
                       figpath=None,
                       max_freq_cyc_per_mm: float = 10.0,
                       band: str = "sem",          # "std" or "sem"
                       alpha: float = 0.25,
                       linewidth: float = 4.0):
    """
    Plot mean normalized Fourier amplitude vs spatial frequency (cycles/mm)
    with a soft grey uncertainty band (mean ± std or mean ± SEM).

    Notes:
    - Uses signed fft frequencies but plots only the non-negative half (freq >= 0),
      which avoids the common plotting artifact from abs(freq).
    - Uncertainty convention in many papers: mean ± SEM (default), or mean ± SD.

    Parameters
    ----------
    connection_fields : torch.Tensor
        Shape (B, N, N) or (B, 1, N, N).
    pixel_size_mm : float
        Physical size of one pixel in mm.
    mask_radius_mm : float
        (Currently unused in your code; kept for API compatibility.)
    figpath : str or None
        If provided, save figure here.
    max_freq_cyc_per_mm : float
        Max spatial frequency to show.
    band : {"sem","std"}
        Uncertainty band type.
    alpha : float
        Transparency of uncertainty band.
    linewidth : float
        Width of mean curve.
    """
    # Handle input shape
    if connection_fields.dim() == 4:
        connection_fields = connection_fields.squeeze(1)  # (B, N, N)
    if connection_fields.dim() != 3:
        raise ValueError("Expected (B, N, N) or (B, 1, N, N).")

    B, N, _ = connection_fields.shape
    device = connection_fields.device
    center_idx = N // 2

    # --- Frequency axis: signed, shifted; then plot only freq >= 0 ---
    freqs = torch.fft.fftshift(torch.fft.fftfreq(N, d=pixel_size_mm)).cpu().numpy()  # (N,)
    keep = (freqs >= 0) & (freqs <= max_freq_cyc_per_mm)

    # --- Your Gaussian mask (as in your code) ---
    # Assumes get_gaussian returns tensor shaped like (1,1,N,N) or similar; you used [0,0]
    gmask = get_gaussian(N, N/2)[0, 0].to(device)
    gmask = gmask / (gmask.max() + 1e-12)

    profiles = []

    for i in range(B):
        field = connection_fields[i].float()  # (N, N)
        field_masked = field * gmask

        if field_masked.sum().item() < 10:
            continue

        # Principal axis via PCA
        coords = torch.nonzero(field_masked, as_tuple=False).float()  # (K, 2): row, col
        if coords.shape[0] < 2:
            angle_deg = 0.0
        else:
            coords_centered = coords - torch.tensor([center_idx, center_idx], device=device).float()
            cov = torch.cov(coords_centered.T)
            eigvecs = torch.linalg.eigh(cov).eigenvectors
            principal_dir = eigvecs[:, -1]

            # coords are (row=y, col=x). atan2(y, x) => atan2(principal_dir[0], principal_dir[1])
            angle_rad = torch.atan2(principal_dir[0], principal_dir[1])
            angle_deg = (-angle_rad * 180.0 / np.pi).item()

        # Rotate to align principal axis horizontally
        field_rot = rotate(
            field_masked.unsqueeze(0).unsqueeze(0),  # (1,1,N,N)
            angle=angle_deg,
            interpolation=InterpolationMode.NEAREST,
            expand=False,
            fill=0
        ).squeeze(0).squeeze(0)  # (N, N)

        # 2D FFT
        fft = torch.fft.fftshift(torch.fft.fft2(field_rot))
        amplitude = torch.abs(fft)

        dc = amplitude[center_idx, center_idx]
        if dc == 0:
            continue

        amplitude_normalized = amplitude / dc
        horiz_profile = amplitude_normalized[center_idx, :].detach().cpu().numpy()  # (N,)
        profiles.append(horiz_profile)

    if len(profiles) == 0:
        print("No valid connection fields after processing.")
        return

    profiles = np.stack(profiles, axis=0)  # (M, N)
    mean_profile = profiles.mean(axis=0)

    # Uncertainty band
    std_profile = profiles.std(axis=0, ddof=1) if profiles.shape[0] > 1 else np.zeros_like(mean_profile)
    if band.lower() == "sem":
        denom = np.sqrt(profiles.shape[0]) if profiles.shape[0] > 0 else 1.0
        spread = std_profile / denom
        band_label = "Mean ± SEM"
    elif band.lower() == "std":
        spread = std_profile
        band_label = "Mean ± SD"
    else:
        raise ValueError("band must be 'sem' or 'std'.")

    # Plot
    plt.figure(figsize=(8, 6))

    x = freqs[keep]
    y = mean_profile[keep]
    s = spread[keep]

    # Grey uncertainty band (convention)
    plt.fill_between(x, y - s, y + s, alpha=alpha, label=band_label)

    # Mean curve
    plt.plot(x, y, color="black", linewidth=linewidth, label="Mean")

    fs = 36
    plt.xlabel("spatial freq. (cycles/mm)", fontsize=fs)
    plt.ylabel("mean Fourier coef.", fontsize=fs)
    plt.xticks(fontsize=fs * 0.8)
    plt.yticks(fontsize=fs * 0.8)
    plt.tight_layout()
    plt.gca().spines[["top", "right"]].set_visible(False)

    # Optional: legend (often kept small / off in figure panels)
    # plt.legend(frameon=False, fontsize=fs * 0.6)

    if figpath:
        plt.savefig(figpath, dpi=300, bbox_inches="tight")

    plt.show()



def fit_and_plot(
    x, y,
    num_fit_points=100,
    font_size=25,
    axis_labels=("X-axis", "Y-axis"),
    legend_labels=None,
    color=None,
    integer_axes=(False, False),
    x_ticks=None,
    y_ticks=None,
    global_fit=False,
    fit_func="exp",            # "exp", "offset_exp", "shifted_power", "hyperbolic", "one_minus_exp", "log"
    b_grid=None,               # used for offset_exp and one_minus_exp
    mask_eps=1e-8,             # mask threshold for exp-log fit
    xlim=None,
    ylim=None,
    fit_flag=True,
    scatter=True,
    log_x0=None,               # optional fixed x0; otherwise x0 is fitted
    log_delta=1e-3,            # minimum fitted (min(x)-x0) as a fraction of x span
    marker_mode=False,
    s=100,
    cmap='inferno',
    rep_axis="auto",
    show_rep_spread=True,
    rep_spread="sem",
    legend_on_points=False,
    rep_alpha=0.2,
    figpath=None,
    show=True,
    legend_loc=None,
    omit_second_from_fit=False,
    fit_exclude_indices=None,
    ax=None,
):
    """
    Fits one of:
      - fit_func="exp":          y = a * exp(-b x)
      - fit_func="offset_exp":   y = c + a*exp(-b x)
      - fit_func="shifted_power":y = A*(1 + (x-x_min)/tau)^(-p)
      - fit_func="hyperbolic":    y = c + a/(1 + (x-x_min)/tau)
      - fit_func="one_minus_exp":y = c + a*(1 - exp(-b x))
      - fit_func="log":          y = c + a*log(x - x0)  (x0 fitted below min(x))

    global_fit:
      - False: fit per curve
      - True : fit once across all points, broadcast params to all curves

    fit_exclude_indices:
      - Optional point indices omitted from fitting but retained in the plot.

    omit_second_from_fit:
      - If True, omit point index 1 from fitting while retaining it in the plot.

    Returns:
      - exp: (a, b)
      - offset_exp/hyperbolic/one_minus_exp: (c0, a, b)
      - shifted_power: (A, tau, p)
      - log: (c0, a, x0)

    y may be:
      - [points]
      - [curves, points]
      - [reps, curves, points] for repeated runs, or [curves, points, reps].
        Repeated data is plotted as the mean with optional SEM/STD spread.
    """

    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y).float()

    y_reps = None
    y_spread = None

    if y.dim() == 1:
        y = y[None, :]
    elif y.dim() == 2:
        pass
    elif y.dim() == 3:
        point_count = x.shape[-1] if x.dim() > 0 else y.shape[-1]

        if rep_axis == "auto":
            if x.dim() == 2 and tuple(y.shape[1:]) == tuple(x.shape):
                inferred_rep_axis = 0
            elif y.shape[-1] == point_count:
                inferred_rep_axis = 0
            elif y.shape[1] == point_count:
                inferred_rep_axis = 2
            else:
                inferred_rep_axis = 0
        else:
            inferred_rep_axis = int(rep_axis)

        if inferred_rep_axis < 0:
            inferred_rep_axis = y.dim() + inferred_rep_axis
        if inferred_rep_axis not in (0, 1, 2):
            raise ValueError("rep_axis must be 'auto' or one of 0, 1, 2 for 3D y data.")

        y_reps = y.movedim(inferred_rep_axis, 0)
        y = y_reps.mean(dim=0)

        if rep_spread == "sem":
            denom = max(y_reps.shape[0], 1) ** 0.5
            y_spread = y_reps.std(dim=0, unbiased=y_reps.shape[0] > 1) / denom
        elif rep_spread == "std":
            y_spread = y_reps.std(dim=0, unbiased=y_reps.shape[0] > 1)
        elif rep_spread == "2std":
            y_spread = 2.0 * y_reps.std(dim=0, unbiased=y_reps.shape[0] > 1)
        elif rep_spread in (None, False, "none"):
            y_spread = None
        else:
            raise ValueError("rep_spread must be 'sem', 'std', '2std', or 'none'.")
    else:
        raise ValueError("fit_and_plot expects y with 1, 2, or 3 dimensions.")

    if y.dim() != 2:
        raise ValueError("After repeat reduction, y must have shape [curves, points].")

    c_curves, p = y.shape

    if x.dim() == 1 and x.numel() != p:
        raise ValueError(f"x has {x.numel()} points but y has {p}; slice x/y to matching point axes.")
    if x.dim() == 2 and tuple(x.shape) != tuple(y.shape):
        raise ValueError(f"2D x must match y shape after repeat reduction. Got x={tuple(x.shape)}, y={tuple(y.shape)}.")

    fit_func = str(fit_func).lower().strip()
    if fit_func not in {
        "exp", "offset_exp", "shifted_power", "hyperbolic", "one_minus_exp", "log"
    }:
        raise ValueError(
            "fit_func must be one of: 'exp', 'offset_exp', 'shifted_power', "
            f"'hyperbolic', 'one_minus_exp', 'log'. Got: {fit_func}"
        )

    # params to return (some unused depending on fit_func)
    a = torch.zeros(c_curves, dtype=torch.float32)
    b = torch.zeros(c_curves, dtype=torch.float32)
    c0 = torch.zeros(c_curves, dtype=torch.float32)
    x0 = torch.zeros(c_curves, dtype=torch.float32)  # used for log fit
    power = torch.zeros(c_curves, dtype=torch.float32)
    power_xmin = torch.zeros(c_curves, dtype=torch.float32)

    colors = sns.color_palette(cmap, n_colors=c_curves)
    marker_list = ["o", "s", "^", "D", "v", "P", "X", "*"]        

    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=(7, 6))
    else:
        plt.sca(ax)

    if legend_labels is None:
        legend_labels = [''] * c_curves

    def _get_xi(i):
        return x if x.dim() == 1 else x[i]

    def _get_fit_xy(i):
        xi = _get_xi(i)
        yi = y[i]
        if fit_exclude_indices is None and not omit_second_from_fit:
            return xi, yi

        keep = torch.ones(xi.numel(), dtype=torch.bool, device=xi.device)
        excluded_indices = [] if fit_exclude_indices is None else list(fit_exclude_indices)
        if omit_second_from_fit:
            excluded_indices.append(1)
        excluded = torch.as_tensor(excluded_indices, dtype=torch.long, device=xi.device).reshape(-1)
        excluded = excluded[(excluded >= -xi.numel()) & (excluded < xi.numel())]
        excluded = excluded.remainder(xi.numel())
        keep[excluded] = False
        return xi[keep], yi[keep]

    def _fit_exp(xi, yi):
        """
        Fit y = a * exp(-b x), allowing sign flips via sign(a) and growth via sign(b).
        Uses log(|y|) with a near-zero mask.
        """
        m = torch.abs(yi) > mask_eps
        xi_m = xi[m]
        yi_m = yi[m]
        if xi_m.numel() < 2:
            a_i = yi_m.mean() if yi_m.numel() > 0 else torch.tensor(0.0, device=yi.device)
            b_i = torch.tensor(0.0, device=yi.device)
            return a_i, b_i

        yi_abs = torch.abs(yi_m).clamp_min(1e-12)

        x_min = xi_m.min()
        x_max = xi_m.max()
        delta_x = (x_max - x_min).clamp_min(1e-12)
        x_norm = (xi_m - x_min) / delta_x

        Y = torch.log(yi_abs).unsqueeze(1)
        X = torch.cat([torch.ones_like(x_norm).unsqueeze(1), -x_norm.unsqueeze(1)], dim=1)

        sol = torch.linalg.lstsq(X, Y).solution.squeeze()
        log_a_norm, b_norm = sol[0], sol[1]
        a_norm = torch.exp(log_a_norm)

        b_i = b_norm / delta_x
        a_pos = a_norm * torch.exp(b_norm * x_min / delta_x)

        s = torch.sign(yi_m.mean())
        if s == 0:
            s = torch.tensor(1.0, device=yi.device)
        a_i = s * a_pos
        return a_i, b_i

    def _fit_one_minus_exp(xi, yi):
        """
        Fit y = c + a*(1 - exp(-b x)) by grid-searching b and solving (c,a) via LS.
        """
        m = torch.isfinite(xi) & torch.isfinite(yi)
        xi_m = xi[m]
        yi_m = yi[m]
        if xi_m.numel() < 2:
            return (yi_m.mean() if yi_m.numel() > 0 else torch.tensor(0.0, device=yi.device),
                    torch.tensor(0.0, device=yi.device),
                    torch.tensor(0.0, device=yi.device))

        if b_grid is None:
            xr = (xi_m.max() - xi_m.min()).clamp_min(1e-12)
            b_vals = torch.logspace(-3, 3, steps=121, device=yi.device) / xr
        else:
            b_vals = torch.as_tensor(b_grid, dtype=torch.float32, device=yi.device)

        ones = torch.ones_like(xi_m)
        best_sse = None
        best_c, best_a, best_b = None, None, None

        for bi in b_vals:
            phi = 1.0 - torch.exp(-bi * xi_m)
            A = torch.stack([ones, phi], dim=1)

            sol = torch.linalg.lstsq(A, yi_m.unsqueeze(1)).solution.squeeze()
            ci, ai = sol[0], sol[1]

            y_hat = ci + ai * phi
            sse = torch.sum((y_hat - yi_m) ** 2)

            if best_sse is None or sse < best_sse:
                best_sse = sse
                best_c, best_a, best_b = ci, ai, bi

        return best_c, best_a, best_b

    def _fit_offset_exp(xi, yi):
        """Fit y = c + a*exp(-b x) in the original y-space."""
        m = torch.isfinite(xi) & torch.isfinite(yi)
        xi_m = xi[m]
        yi_m = yi[m]
        if xi_m.numel() < 2:
            return (yi_m.mean() if yi_m.numel() > 0 else torch.tensor(0.0, device=yi.device),
                    torch.tensor(0.0, device=yi.device),
                    torch.tensor(0.0, device=yi.device))

        if b_grid is None:
            xr = (xi_m.max() - xi_m.min()).clamp_min(1e-12)
            b_vals = torch.logspace(-3, 3, steps=121, device=yi.device) / xr
        else:
            b_vals = torch.as_tensor(b_grid, dtype=torch.float32, device=yi.device)

        ones = torch.ones_like(xi_m)
        best_sse = None
        best_c, best_a, best_b = None, None, None

        for bi in b_vals:
            phi = torch.exp(-bi * xi_m)
            A = torch.stack([ones, phi], dim=1)
            sol = torch.linalg.lstsq(A, yi_m.unsqueeze(1)).solution.squeeze()
            ci, ai = sol[0], sol[1]

            y_hat = ci + ai * phi
            sse = torch.sum((y_hat - yi_m) ** 2)
            if best_sse is None or sse < best_sse:
                best_sse = sse
                best_c, best_a, best_b = ci, ai, bi

        return best_c, best_a, best_b

    def _fit_shifted_power(xi, yi):
        """Fit y = A*(1 + (x-x_min)/tau)^(-p) in the original y-space."""
        m = torch.isfinite(xi) & torch.isfinite(yi)
        xi_m = xi[m]
        yi_m = yi[m]
        if xi_m.numel() < 2:
            amplitude = yi_m.mean() if yi_m.numel() > 0 else torch.tensor(0.0, device=yi.device)
            return (amplitude,
                    torch.tensor(1.0, device=yi.device),
                    torch.tensor(1.0, device=yi.device),
                    xi_m.min() if xi_m.numel() > 0 else torch.tensor(0.0, device=yi.device))

        xmin_i = xi_m.min()
        xr = (xi_m.max() - xmin_i).clamp_min(1e-12)
        z = xi_m - xmin_i

        # For fixed tau and p, A has a closed-form least-squares solution.
        # Search the two shape parameters on a broad log grid, then refine
        # locally twice. This keeps the implementation Torch-only and stable.
        log_tau_lo = torch.log(xr * 1e-3)
        log_tau_hi = torch.log(xr * 1e3)
        log_p_lo = torch.log(torch.tensor(1e-3, dtype=yi.dtype, device=yi.device))
        log_p_hi = torch.log(torch.tensor(10.0, dtype=yi.dtype, device=yi.device))

        best_a = best_tau = best_p = None
        for steps in (121, 41, 41):
            log_tau = torch.linspace(log_tau_lo, log_tau_hi, steps=steps, device=yi.device)
            log_p = torch.linspace(log_p_lo, log_p_hi, steps=steps, device=yi.device)
            tau_vals = torch.exp(log_tau)[:, None, None]
            p_vals = torch.exp(log_p)[None, :, None]
            phi = (1.0 + z[None, None, :] / tau_vals).pow(-p_vals)
            amplitude = ((phi * yi_m).sum(dim=-1) /
                         phi.square().sum(dim=-1).clamp_min(1e-12))
            residual = amplitude[:, :, None] * phi - yi_m
            sse = residual.square().sum(dim=-1)
            flat_index = torch.argmin(sse)
            tau_index = flat_index // steps
            p_index = flat_index % steps

            best_a = amplitude[tau_index, p_index]
            best_tau = torch.exp(log_tau[tau_index])
            best_p = torch.exp(log_p[p_index])

            tau_step = (log_tau_hi - log_tau_lo) / max(steps - 1, 1)
            p_step = (log_p_hi - log_p_lo) / max(steps - 1, 1)
            log_tau_lo = torch.log(best_tau) - tau_step
            log_tau_hi = torch.log(best_tau) + tau_step
            log_p_lo = torch.log(best_p) - p_step
            log_p_hi = torch.log(best_p) + p_step

        return best_a, best_tau, best_p, xmin_i

    def _fit_hyperbolic(xi, yi):
        """Fit y = c + a/(1 + (x-x_min)/tau), with x_min fixed."""
        m = torch.isfinite(xi) & torch.isfinite(yi)
        xi_m = xi[m]
        yi_m = yi[m]
        if xi_m.numel() < 2:
            level = (
                yi_m.mean()
                if yi_m.numel() > 0
                else torch.tensor(0.0, device=yi.device)
            )
            return (
                level,
                torch.tensor(0.0, device=yi.device),
                torch.tensor(1.0, device=yi.device),
                xi_m.min() if xi_m.numel() > 0 else torch.tensor(0.0, device=yi.device),
            )

        work_dtype = torch.float64
        x_work = xi_m.to(dtype=work_dtype)
        y_work = yi_m.to(dtype=work_dtype)
        xmin_i = x_work.min()
        z = x_work - xmin_i
        span = z.max().clamp_min(1e-12)
        log_tau_lo = torch.log(span * 1e-8)
        log_tau_hi = torch.log(span * 1e8)
        best_log_tau = None

        # For each tau, the plateau c and signed amplitude a are an exact
        # two-column linear least-squares problem. Search only the remaining
        # positive scale parameter in log space, then refine locally.
        for steps in (241, 81, 81):
            log_tau = torch.linspace(
                log_tau_lo,
                log_tau_hi,
                steps=steps,
                dtype=work_dtype,
                device=x_work.device,
            )
            tau = torch.exp(log_tau)
            phi = 1.0 / (1.0 + z[None, :] / tau[:, None])
            phi_centered = phi - phi.mean(dim=1, keepdim=True)
            y_centered = y_work - y_work.mean()
            amplitudes = (
                (phi_centered * y_centered[None, :]).sum(dim=1)
                / phi_centered.square().sum(dim=1).clamp_min(1e-24)
            )
            plateaus = y_work.mean() - amplitudes * phi.mean(dim=1)
            residual = plateaus[:, None] + amplitudes[:, None] * phi - y_work
            best_index = torch.argmin(residual.square().sum(dim=1))
            best_log_tau = log_tau[best_index]

            step_size = (log_tau_hi - log_tau_lo) / max(steps - 1, 1)
            log_tau_lo = best_log_tau - step_size
            log_tau_hi = best_log_tau + step_size

        best_tau = torch.exp(best_log_tau)
        best_phi = 1.0 / (1.0 + z / best_tau)
        design = torch.stack([torch.ones_like(best_phi), best_phi], dim=1)
        solution = torch.linalg.lstsq(
            design, y_work.unsqueeze(1)
        ).solution.squeeze()
        plateau, amplitude = solution[0], solution[1]
        return (
            plateau.to(dtype=yi.dtype),
            amplitude.to(dtype=yi.dtype),
            best_tau.to(dtype=yi.dtype),
            xmin_i.to(dtype=yi.dtype),
        )

    def _fit_log(xi, yi):
        """
        Fit y = c + a*log(x - x0), constrained to x0 < min(x).

        For each candidate x0, c and a are profiled out by linear least
        squares. The remaining one-dimensional search is performed over
        log(min(x) - x0), so the positivity constraint is always satisfied.
        Passing log_x0 retains an explicitly fixed horizontal offset.
        """
        m = torch.isfinite(xi) & torch.isfinite(yi)
        xi_m = xi[m]
        yi_m = yi[m]
        if xi_m.numel() < 2:
            return (yi_m.mean() if yi_m.numel() > 0 else torch.tensor(0.0, device=yi.device),
                    torch.tensor(0.0, device=yi.device),
                    torch.tensor(0.0, device=yi.device))

        work_dtype = torch.float64
        x_work = xi_m.to(dtype=work_dtype)
        y_work = yi_m.to(dtype=work_dtype)
        xmin = x_work.min()

        if log_x0 is None:
            span = (x_work.max() - xmin).clamp_min(1e-12)
            min_fraction = float(log_delta)
            if not np.isfinite(min_fraction) or min_fraction <= 0:
                raise ValueError("log_delta must be a finite positive number.")

            log_distance_lo = torch.log(span * min_fraction)
            log_distance_hi = torch.log(span * 1e6)
            y_centered = y_work - y_work.mean()
            best_log_distance = None

            for steps in (241, 81, 81):
                log_distances = torch.linspace(
                    log_distance_lo,
                    log_distance_hi,
                    steps=steps,
                    dtype=work_dtype,
                    device=x_work.device,
                )
                distances = torch.exp(log_distances)
                candidate_x0 = xmin - distances
                phi = torch.log(x_work[None, :] - candidate_x0[:, None])
                phi_centered = phi - phi.mean(dim=1, keepdim=True)
                slopes = (
                    (phi_centered * y_centered[None, :]).sum(dim=1)
                    / phi_centered.square().sum(dim=1).clamp_min(1e-24)
                )
                intercepts = y_work.mean() - slopes * phi.mean(dim=1)
                residual = intercepts[:, None] + slopes[:, None] * phi - y_work
                best_index = torch.argmin(residual.square().sum(dim=1))
                best_log_distance = log_distances[best_index]

                step_size = (
                    (log_distance_hi - log_distance_lo) / max(steps - 1, 1)
                )
                log_distance_lo = best_log_distance - step_size
                log_distance_hi = best_log_distance + step_size

            x0_work = xmin - torch.exp(best_log_distance)
        else:
            x0_work = torch.tensor(
                float(log_x0), dtype=work_dtype, device=x_work.device
            )
            if not bool(x0_work < xmin):
                raise ValueError(
                    f"log_x0 must be below the smallest fitted x ({float(xmin):.6g})."
                )

        z = x_work - x0_work
        phi = torch.log(z)

        A = torch.stack([torch.ones_like(phi), phi], dim=1)  # columns: c, a
        sol = torch.linalg.lstsq(A, y_work.unsqueeze(1)).solution.squeeze()
        ci = sol[0].to(dtype=yi.dtype)
        ai = sol[1].to(dtype=yi.dtype)
        x0_i = x0_work.to(dtype=yi.dtype)
        return ci, ai, x0_i

    # ---- fitting: per-curve or global ----
    if fit_flag:
        if global_fit:
            xs, ys = [], []
            for i in range(c_curves):
                xi_fit, yi_fit = _get_fit_xy(i)
                xs.append(xi_fit.reshape(-1))
                ys.append(yi_fit.reshape(-1))
            x_all = torch.cat(xs, dim=0)
            y_all = torch.cat(ys, dim=0)

            if fit_func == "exp":
                a_s, b_s = _fit_exp(x_all, y_all)
                a[:] = a_s
                b[:] = b_s
            elif fit_func == "offset_exp":
                c_s, a_s, b_s = _fit_offset_exp(x_all, y_all)
                c0[:] = c_s
                a[:] = a_s
                b[:] = b_s
            elif fit_func == "shifted_power":
                a_s, tau_s, p_s, xmin_s = _fit_shifted_power(x_all, y_all)
                a[:] = a_s
                b[:] = tau_s
                power[:] = p_s
                power_xmin[:] = xmin_s
            elif fit_func == "hyperbolic":
                c_s, a_s, tau_s, xmin_s = _fit_hyperbolic(x_all, y_all)
                c0[:] = c_s
                a[:] = a_s
                b[:] = tau_s
                power_xmin[:] = xmin_s
            elif fit_func == "one_minus_exp":
                c_s, a_s, b_s = _fit_one_minus_exp(x_all, y_all)
                c0[:] = c_s
                a[:] = a_s
                b[:] = b_s
            elif fit_func == "log":
                c_s, a_s, x0_s = _fit_log(x_all, y_all)
                c0[:] = c_s
                a[:] = a_s
                x0[:] = x0_s

        else:
            for i in range(c_curves):
                xi, yi = _get_fit_xy(i)

                if fit_func == "exp":
                    ai, bi = _fit_exp(xi, yi)
                    a[i], b[i] = ai, bi
                elif fit_func == "offset_exp":
                    ci, ai, bi = _fit_offset_exp(xi, yi)
                    c0[i], a[i], b[i] = ci, ai, bi
                elif fit_func == "shifted_power":
                    ai, taui, pi, xmini = _fit_shifted_power(xi, yi)
                    a[i], b[i], power[i], power_xmin[i] = ai, taui, pi, xmini
                elif fit_func == "hyperbolic":
                    ci, ai, taui, xmini = _fit_hyperbolic(xi, yi)
                    c0[i], a[i], b[i], power_xmin[i] = ci, ai, taui, xmini
                elif fit_func == "one_minus_exp":
                    ci, ai, bi = _fit_one_minus_exp(xi, yi)
                    c0[i], a[i], b[i] = ci, ai, bi
                elif fit_func == "log":
                    ci, ai, x0i = _fit_log(xi, yi)
                    c0[i], a[i], x0[i] = ci, ai, x0i

    # ---- plotting ----
    for i in range(c_curves):
        xi = _get_xi(i)
        yi = y[i]

        marker_i = 'o'
        if marker_mode:
            marker_i = marker_list[i % len(marker_list)]

        if color is not None:

            if type(color) is list:
                
                color_i = color[i]

            else:

                color_i = color
        else:

            color_i = colors[i]
        
        if scatter:
            if show_rep_spread and y_spread is not None:
                plt.errorbar(
                    xi.detach().cpu().numpy(),
                    yi.detach().cpu().numpy(),
                    yerr=y_spread[i].detach().cpu().numpy(),
                    fmt=marker_i,
                    color=color_i,
                    markersize=max((float(s) ** 0.5), 1.0),
                    linewidth=0,
                    elinewidth=1.5,
                    capsize=0,
                    label=legend_labels[i] if (legend_on_points or global_fit) else None,
                    alpha=1.0,
                )
            else:
                plt.scatter(
                    xi.detach().cpu().numpy(),
                    yi.detach().cpu().numpy(),
                    color=color_i,
                    marker=marker_i,
                    s=s,
                    linewidths=2 if marker_mode else 0,
                    label=legend_labels[i] if (legend_on_points or global_fit) else None,
                    edgecolor='black'
                )
            
        else:
            plt.plot(
                xi.detach().cpu().numpy(),
                yi.detach().cpu().numpy(),
                color=color_i,
                marker=marker_i if marker_mode else None,
                linewidth=2
            )
            if show_rep_spread and y_spread is not None:
                spread = y_spread[i]
                plt.fill_between(
                    xi.detach().cpu().numpy(),
                    (yi - spread).detach().cpu().numpy(),
                    (yi + spread).detach().cpu().numpy(),
                    color=color_i,
                    alpha=rep_alpha,
                    linewidth=0,
                )

        if global_fit:
            x_fit = torch.linspace(x.min(), x.max(), num_fit_points, device=xi.device)
        else:
            x_fit = torch.linspace(xi.min(), xi.max(), num_fit_points, device=xi.device)

        if fit_func == "exp":
            y_fit = a[i] * torch.exp(-b[i] * x_fit)
        elif fit_func == "offset_exp":
            y_fit = c0[i] + a[i] * torch.exp(-b[i] * x_fit)
        elif fit_func == "shifted_power":
            scaled_x = 1.0 + (x_fit - power_xmin[i]) / b[i]
            y_fit = a[i] * scaled_x.clamp_min(1e-12).pow(-power[i])
        elif fit_func == "hyperbolic":
            scaled_x = 1.0 + (x_fit - power_xmin[i]) / b[i]
            y_fit = c0[i] + a[i] / scaled_x.clamp_min(1e-12)
        elif fit_func == "one_minus_exp":
            y_fit = c0[i] + a[i] * (1.0 - torch.exp(-b[i] * x_fit))
        elif fit_func == "log":
            z = (x_fit - x0[i]).clamp_min(1e-12)
            y_fit = c0[i] + a[i] * torch.log(z)

        if fit_flag and (not global_fit or i==0):

            plt.plot(
                x_fit.detach().cpu().numpy(),
                y_fit.detach().cpu().numpy(),
                color=colors[i] if not global_fit else 'black',
                linewidth=2,
                label=None if legend_on_points else (legend_labels[i] if not global_fit else None)
            )

    # ---- axes/ticks (your original) ----
    plt.xlabel(axis_labels[0], fontsize=font_size)
    plt.ylabel(axis_labels[1], fontsize=font_size)
    plt.xticks(fontsize=font_size * 0.8)
    plt.yticks(fontsize=font_size * 0.8)

    if x_ticks is None:
        x_ticks_vals = torch.linspace(x.min(), x.max(), 4)
        if integer_axes[0]:
            x_ticks_vals = torch.round(x_ticks_vals)
        x_labels = [str(int(val)) if integer_axes[0] else f"{val:.2g}"
                    for val in x_ticks_vals.detach().cpu().numpy()]
    else:
        x_ticks_vals = torch.tensor(x_ticks, dtype=torch.float32)
        x_labels = [str(int(val)) if float(val).is_integer() else f"{float(val):.2g}" for val in x_ticks_vals]

    if y_ticks is None:
        y_ticks_vals = torch.linspace(y.min(), y.max(), 4)
        if integer_axes[1]:
            y_ticks_vals = torch.round(y_ticks_vals)
        y_labels = [str(int(val)) if integer_axes[1] else f"{val:.2g}"
                    for val in y_ticks_vals.detach().cpu().numpy()]
    else:
        y_ticks_vals = torch.tensor(y_ticks, dtype=torch.float32)
        y_labels = [str(int(val)) if float(val).is_integer() else f"{float(val):.2g}" for val in y_ticks_vals]

    plt.xticks(x_ticks_vals.detach().cpu().numpy(), x_labels)
    plt.yticks(y_ticks_vals.detach().cpu().numpy(), y_labels)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if legend_labels[0] != '':
        plt.legend(frameon=False, fontsize=font_size * 0.8, loc=legend_loc)

    if owns_figure:
        plt.tight_layout()

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    if figpath:
        plt.savefig(figpath)

    if show and owns_figure:
        plt.show()

    # ---- return params ----
    if fit_func == "exp":
        return a, b
    if fit_func in {"offset_exp", "hyperbolic"}:
        return c0, a, b
    if fit_func == "shifted_power":
        return a, b, power
    if fit_func == "one_minus_exp":
        return c0, a, b
    if fit_func == "log":
        return c0, a, x0


def plot_fidelity_vs_dimensionality_from_fits(
    neighborhood_x,
    dimensionality,
    fidelity,
    fidelity_fit,
    dimensionality_fit,
    axis_labels=("complexity", "fidelity"),
    legend_labels=None,
    x_ticks=None,
    y_ticks=None,
    xlim=None,
    ylim=None,
    num_fit_points=200,
    font_size=25,
    s=100,
    cmap="inferno",
    rep_spread="std",
    average_derived_curve=False,
    average_curve_label="mean derived curve",
    figpath=None,
    show=True,
    ax=None,
):
    """Plot the analytical fidelity--dimensionality relation implied by two fits.

    Both source fits must use ``fit_func="shifted_power"``:

        y = A * (1 + (x - x_min) / tau) ** (-p)

    Neighborhood size is eliminated analytically; this function does not fit
    the fidelity--dimensionality points again.
    """
    neighborhood_x = torch.as_tensor(neighborhood_x, dtype=torch.float32)
    dimensionality = torch.as_tensor(dimensionality, dtype=torch.float32)
    fidelity = torch.as_tensor(fidelity, dtype=torch.float32)

    def _mean_and_spread(values):
        if values.dim() == 2:
            return values, None
        if values.dim() != 3:
            raise ValueError("Expected [curve, point] or [rep, curve, point] data.")
        mean = values.mean(dim=0)
        if rep_spread == "std":
            spread = values.std(dim=0, unbiased=values.shape[0] > 1)
        elif rep_spread == "2std":
            spread = 2.0 * values.std(dim=0, unbiased=values.shape[0] > 1)
        elif rep_spread == "sem":
            spread = values.std(dim=0, unbiased=values.shape[0] > 1) / max(values.shape[0], 1) ** 0.5
        elif rep_spread in (None, False, "none"):
            spread = None
        else:
            raise ValueError("rep_spread must be 'sem', 'std', '2std', or 'none'.")
        return mean, spread

    dim_mean, dim_spread = _mean_and_spread(dimensionality)
    fid_mean, fid_spread = _mean_and_spread(fidelity)
    if dim_mean.shape != fid_mean.shape:
        raise ValueError(
            f"Dimensionality and fidelity shapes must match; got "
            f"{tuple(dim_mean.shape)} and {tuple(fid_mean.shape)}."
        )

    acc_A, acc_tau, acc_p = [torch.as_tensor(v, dtype=torch.float32) for v in fidelity_fit]
    dim_A, dim_tau, dim_p = [torch.as_tensor(v, dtype=torch.float32) for v in dimensionality_fit]
    n_curves = dim_mean.shape[0]
    for name, values in {
        "fidelity A": acc_A,
        "fidelity tau": acc_tau,
        "fidelity p": acc_p,
        "dimensionality A": dim_A,
        "dimensionality tau": dim_tau,
        "dimensionality p": dim_p,
    }.items():
        if values.numel() != n_curves:
            raise ValueError(f"{name} has {values.numel()} values; expected {n_curves}.")

    if legend_labels is None:
        legend_labels = [""] * n_curves
    colors = sns.color_palette(cmap, n_colors=n_curves)
    x_model = torch.linspace(
        neighborhood_x.min(), neighborhood_x.max(), num_fit_points
    )
    neighborhood_min = neighborhood_x.min()

    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=(7, 6))
    else:
        plt.sca(ax)
    derived_curves = []
    for i in range(n_curves):
        # Use D(x) only to select the observed neighborhood-size range, then
        # evaluate the analytically eliminated relationship F(D).
        dim_fit = dim_A[i] * (
            1.0 + (x_model - neighborhood_min) / dim_tau[i]
        ).pow(-dim_p[i])
        fidelity_fit_curve = acc_A[i] * (
            1.0
            + (dim_tau[i] / acc_tau[i])
            * ((dim_A[i] / dim_fit).pow(1.0 / dim_p[i]) - 1.0)
        ).pow(-acc_p[i])
        derived_curves.append((dim_fit, fidelity_fit_curve))

        xerr = None if dim_spread is None else dim_spread[i].detach().cpu().numpy()
        yerr = None if fid_spread is None else fid_spread[i].detach().cpu().numpy()
        plt.errorbar(
            dim_mean[i].detach().cpu().numpy(),
            fid_mean[i].detach().cpu().numpy(),
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=colors[i],
            markersize=max(float(s) ** 0.5, 1.0),
            linewidth=0,
            elinewidth=1.5,
            capsize=0,
            alpha=0.9,
            label=legend_labels[i] if average_derived_curve else None,
        )
        if not average_derived_curve:
            plt.plot(
                dim_fit.detach().cpu().numpy(),
                fidelity_fit_curve.detach().cpu().numpy(),
                color=colors[i],
                linewidth=2,
                label=legend_labels[i],
            )

    if average_derived_curve:
        # Average predictions pointwise without extrapolating any component
        # curve beyond the complexity interval generated by its source fit.
        common_dim = torch.linspace(
            min(curve[0].min().item() for curve in derived_curves),
            max(curve[0].max().item() for curve in derived_curves),
            num_fit_points,
        )
        prediction_sum = torch.zeros_like(common_dim)
        prediction_count = torch.zeros_like(common_dim)
        for dim_fit, fidelity_fit_curve in derived_curves:
            order = torch.argsort(dim_fit)
            dim_sorted = dim_fit[order]
            fidelity_sorted = fidelity_fit_curve[order]
            supported = (common_dim >= dim_sorted[0]) & (common_dim <= dim_sorted[-1])
            query = common_dim[supported]
            right = torch.searchsorted(dim_sorted, query).clamp(1, dim_sorted.numel() - 1)
            left = right - 1
            weight = ((query - dim_sorted[left]) /
                      (dim_sorted[right] - dim_sorted[left]).clamp_min(1e-12))
            interpolated = fidelity_sorted[left] + weight * (
                fidelity_sorted[right] - fidelity_sorted[left]
            )
            prediction_sum[supported] += interpolated
            prediction_count[supported] += 1

        supported = prediction_count > 0
        mean_fidelity = prediction_sum[supported] / prediction_count[supported]
        plt.plot(
            common_dim[supported].detach().cpu().numpy(),
            mean_fidelity.detach().cpu().numpy(),
            color="black",
            linewidth=2.5,
            label=average_curve_label,
        )

    plt.xlabel(axis_labels[0], fontsize=font_size)
    plt.ylabel(axis_labels[1], fontsize=font_size)
    plt.xticks(fontsize=font_size * 0.8)
    plt.yticks(fontsize=font_size * 0.8)
    if x_ticks is not None:
        plt.xticks(x_ticks)
    if y_ticks is not None:
        plt.yticks(y_ticks)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    ax.spines[["top", "right"]].set_visible(False)
    if legend_labels and legend_labels[0] != "":
        plt.legend(frameon=False, fontsize=font_size * 0.8)
    if owns_figure:
        plt.tight_layout()
    if figpath:
        plt.savefig(figpath)
    if show and owns_figure:
        plt.show()
    return derived_curves


def complexity_at_saturation_fraction(
    fidelity_fit,
    dimensionality_fit,
    saturation_fraction=0.9,
):
    """Solve the mean analytical fidelity--complexity curve at a plateau fraction."""
    if not np.isscalar(saturation_fraction):
        raise TypeError("saturation_fraction must be a scalar between 0 and 1.")
    saturation_fraction = float(saturation_fraction)
    if not 0.0 < saturation_fraction < 1.0:
        raise ValueError("saturation_fraction must satisfy 0 < value < 1.")

    acc_A, acc_tau, acc_p = [
        torch.as_tensor(v, dtype=torch.float64).flatten().cpu().numpy()
        for v in fidelity_fit
    ]
    dim_A, dim_tau, dim_p = [
        torch.as_tensor(v, dtype=torch.float64).flatten().cpu().numpy()
        for v in dimensionality_fit
    ]
    lengths = {
        len(values)
        for values in (acc_A, acc_tau, acc_p, dim_A, dim_tau, dim_p)
    }
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("All fit-parameter arrays must have the same non-zero length.")

    plateau_bases = 1.0 - dim_tau / acc_tau
    if np.any(plateau_bases <= 0):
        raise ValueError("The fitted curve does not have a finite real-valued plateau.")
    plateau = np.mean(acc_A * plateau_bases ** (-acc_p))

    def mean_fidelity(complexity):
        inner = (
            1.0
            + (dim_tau / acc_tau)
            * ((dim_A / complexity) ** (1.0 / dim_p) - 1.0)
        )
        return np.mean(acc_A * inner ** (-acc_p))

    target = saturation_fraction * plateau
    lower = 1e-12
    upper = max(float(np.max(dim_A)), 1.0)
    for _ in range(100):
        if mean_fidelity(upper) >= target:
            break
        upper *= 2.0
    else:
        raise RuntimeError("Could not bracket the requested saturation fraction.")

    return float(brentq(lambda value: mean_fidelity(value) - target, lower, upper))


def saturation_log_curvature_point(fidelity_fit, dimensionality_fit):
    """Return the strongest negative bend of the mean curve in log-complexity space."""
    acc_A, acc_tau, acc_p = [
        torch.as_tensor(v, dtype=torch.float64).flatten().cpu().numpy()
        for v in fidelity_fit
    ]
    dim_A, dim_tau, dim_p = [
        torch.as_tensor(v, dtype=torch.float64).flatten().cpu().numpy()
        for v in dimensionality_fit
    ]
    lengths = {
        len(values)
        for values in (acc_A, acc_tau, acc_p, dim_A, dim_tau, dim_p)
    }
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("All fit-parameter arrays must have the same non-zero length.")

    plateau_bases = 1.0 - dim_tau / acc_tau
    if np.any(plateau_bases <= 0):
        raise ValueError("The fitted curve does not have a finite real-valued plateau.")
    plateau = float(np.mean(acc_A * plateau_bases ** (-acc_p)))

    def curve_values(complexity):
        fidelity = first_derivative = second_derivative = 0.0
        for a, tau_a, p_a, A, tau_d, p_d in zip(
            acc_A, acc_tau, acc_p, dim_A, dim_tau, dim_p
        ):
            ratio = tau_d / tau_a
            exponent = 1.0 / p_d
            coefficient = ratio * A ** exponent
            inner = 1.0 - ratio + coefficient * complexity ** (-exponent)
            component = a * inner ** (-p_a)
            component_first = (
                a * p_a * exponent * coefficient
                * complexity ** (-exponent - 1.0)
                * inner ** (-p_a - 1.0)
            )
            component_second = (
                a * p_a * exponent * coefficient
                * complexity ** (-exponent - 2.0)
                * inner ** (-p_a - 2.0)
                * (
                    -(exponent + 1.0) * inner
                    + (p_a + 1.0) * exponent * coefficient
                    * complexity ** (-exponent)
                )
            )
            fidelity += component / len(acc_A)
            first_derivative += component_first / len(acc_A)
            second_derivative += component_second / len(acc_A)
        return fidelity, first_derivative, second_derivative

    lower = 1e-12
    upper = max(float(np.max(dim_A)), 1.0)
    for _ in range(100):
        if curve_values(upper)[0] >= 0.5 * plateau:
            break
        upper *= 2.0
    else:
        raise RuntimeError("Could not bracket the half-saturation complexity.")
    half_saturation = brentq(
        lambda value: curve_values(value)[0] / plateau - 0.5,
        lower,
        upper,
    )

    def signed_curvature(log_relative_complexity):
        complexity = half_saturation * np.exp(log_relative_complexity)
        _, first_derivative, second_derivative = curve_values(complexity)
        first_log_derivative = complexity * first_derivative / plateau
        second_log_derivative = (
            complexity * first_derivative
            + complexity * complexity * second_derivative
        ) / plateau
        return second_log_derivative / (
            1.0 + first_log_derivative * first_log_derivative
        ) ** 1.5

    log_grid = np.linspace(-12.0, 12.0, 4001)
    curvature_grid = np.asarray([signed_curvature(value) for value in log_grid])
    minimum_index = int(np.argmin(curvature_grid))
    if minimum_index in (0, len(log_grid) - 1) or curvature_grid[minimum_index] >= 0:
        raise RuntimeError("Could not identify a negative-curvature saturation bend.")
    optimum = minimize_scalar(
        signed_curvature,
        bounds=(log_grid[minimum_index - 1], log_grid[minimum_index + 1]),
        method="bounded",
    )
    complexity = float(half_saturation * np.exp(optimum.x))
    fidelity = float(curve_values(complexity)[0])
    return {
        "complexity": complexity,
        "plateau_fraction": fidelity / plateau,
        "fidelity": fidelity,
        "plateau": plateau,
        "signed_log_curvature": float(optimum.fun),
        "half_saturation_complexity": float(half_saturation),
    }


def plot_fidelity_dimensionality_summary(
    neighborhood_x,
    fidelity,
    dimensionality,
    legend_labels=("Δ=1", "Δ=4", "Δ=9"),
    rep_spread="std",
    omit_second_from_dimensionality_fit=True,
    figure_paths=(
        "./figures/accuracy.svg",
        "./figures/dimensionality.svg",
        "./figures/accdimratio.svg",
    ),
    font_size=25,
    show=True,
):
    """Fit and plot fidelity, complexity, and their derived relation in one row.

    The three panels are produced by one call. Each panel is also cropped from
    the row and saved separately to ``figure_paths``. The third panel is
    derived analytically from the first two shifted-power fits; it is not fit
    again.
    """
    if len(figure_paths) != 3:
        raise ValueError("figure_paths must contain accuracy, dimensionality, and joint paths.")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fidelity_fit = fit_and_plot(
        neighborhood_x,
        fidelity,
        axis_labels=("lateral neighborhood size $N_\\mathrm{lat}$", "fidelity"),
        legend_labels=legend_labels,
        fit_func="shifted_power",
        x_ticks=(20, 60, 100),
        rep_axis=0,
        rep_spread=rep_spread,
        legend_on_points=True,
        font_size=font_size,
        show=False,
        ax=axes[0],
    )
    dimensionality_fit = fit_and_plot(
        neighborhood_x,
        dimensionality,
        axis_labels=("lateral neighborhood size $N_\\mathrm{lat}$", "complexity"),
        legend_labels=legend_labels,
        fit_func="shifted_power",
        omit_second_from_fit=omit_second_from_dimensionality_fit,
        y_ticks=(0, 1, 2),
        x_ticks=(20, 60, 100),
        rep_axis=0,
        rep_spread=rep_spread,
        legend_on_points=True,
        font_size=font_size,
        show=False,
        ax=axes[1],
    )
    derived_curves = plot_fidelity_vs_dimensionality_from_fits(
        neighborhood_x,
        dimensionality,
        fidelity,
        fidelity_fit,
        dimensionality_fit,
        axis_labels=("complexity", "fidelity"),
        legend_labels=legend_labels,
        x_ticks=(0, 1, 2),
        rep_spread=rep_spread,
        average_derived_curve=True,
        average_curve_label=None,
        font_size=font_size,
        show=False,
        ax=axes[2],
    )

    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for panel_ax, path in zip(axes, figure_paths):
        if path:
            bbox = panel_ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(path, bbox_inches=bbox.expanded(1.03, 1.04))

    if show:
        plt.show()
    return {
        "fidelity_fit": fidelity_fit,
        "dimensionality_fit": dimensionality_fit,
        "derived_curves": derived_curves,
        "figure": fig,
        "axes": axes,
    }


def plot_fidelity_vs_dimensionality_from_fits(
    neighborhood_x,
    dimensionality,
    fidelity,
    fidelity_fit,
    dimensionality_fit,
    axis_labels=("complexity", "fidelity"),
    legend_labels=None,
    x_ticks=None,
    y_ticks=None,
    xlim=None,
    ylim=None,
    num_fit_points=200,
    font_size=25,
    s=100,
    cmap="inferno",
    rep_spread="std",
    average_derived_curve=False,
    average_curve_label="mean derived curve",
    figpath=None,
    show=True,
    ax=None,
):
    """Plot the analytical fidelity--dimensionality relation implied by two fits.

    Both source fits must use ``fit_func="shifted_power"``:

        y = A * (1 + (x - x_min) / tau) ** (-p)

    Neighborhood size is eliminated analytically; this function does not fit
    the fidelity--dimensionality points again.
    """
    neighborhood_x = torch.as_tensor(neighborhood_x, dtype=torch.float32)
    dimensionality = torch.as_tensor(dimensionality, dtype=torch.float32)
    fidelity = torch.as_tensor(fidelity, dtype=torch.float32)

    def _mean_and_spread(values):
        if values.dim() == 2:
            return values, None
        if values.dim() != 3:
            raise ValueError("Expected [curve, point] or [rep, curve, point] data.")
        mean = values.mean(dim=0)
        if rep_spread == "std":
            spread = values.std(dim=0, unbiased=values.shape[0] > 1)
        elif rep_spread == "2std":
            spread = 2.0 * values.std(dim=0, unbiased=values.shape[0] > 1)
        elif rep_spread == "sem":
            spread = values.std(dim=0, unbiased=values.shape[0] > 1) / max(values.shape[0], 1) ** 0.5
        elif rep_spread in (None, False, "none"):
            spread = None
        else:
            raise ValueError("rep_spread must be 'sem', 'std', '2std', or 'none'.")
        return mean, spread

    dim_mean, dim_spread = _mean_and_spread(dimensionality)
    fid_mean, fid_spread = _mean_and_spread(fidelity)
    if dim_mean.shape != fid_mean.shape:
        raise ValueError(
            f"Dimensionality and fidelity shapes must match; got "
            f"{tuple(dim_mean.shape)} and {tuple(fid_mean.shape)}."
        )

    acc_A, acc_tau, acc_p = [torch.as_tensor(v, dtype=torch.float32) for v in fidelity_fit]
    dim_A, dim_tau, dim_p = [torch.as_tensor(v, dtype=torch.float32) for v in dimensionality_fit]
    n_curves = dim_mean.shape[0]
    for name, values in {
        "fidelity A": acc_A,
        "fidelity tau": acc_tau,
        "fidelity p": acc_p,
        "dimensionality A": dim_A,
        "dimensionality tau": dim_tau,
        "dimensionality p": dim_p,
    }.items():
        if values.numel() != n_curves:
            raise ValueError(f"{name} has {values.numel()} values; expected {n_curves}.")

    if legend_labels is None:
        legend_labels = [""] * n_curves
    colors = sns.color_palette(cmap, n_colors=n_curves)
    x_model = torch.linspace(
        neighborhood_x.min(), neighborhood_x.max(), num_fit_points
    )
    neighborhood_min = neighborhood_x.min()

    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=(7, 6))
    else:
        plt.sca(ax)
    derived_curves = []
    for i in range(n_curves):
        # Use D(x) only to select the observed neighborhood-size range, then
        # evaluate the analytically eliminated relationship F(D).
        dim_fit = dim_A[i] * (
            1.0 + (x_model - neighborhood_min) / dim_tau[i]
        ).pow(-dim_p[i])
        fidelity_fit_curve = acc_A[i] * (
            1.0
            + (dim_tau[i] / acc_tau[i])
            * ((dim_A[i] / dim_fit).pow(1.0 / dim_p[i]) - 1.0)
        ).pow(-acc_p[i])
        derived_curves.append((dim_fit, fidelity_fit_curve))

        xerr = None if dim_spread is None else dim_spread[i].detach().cpu().numpy()
        yerr = None if fid_spread is None else fid_spread[i].detach().cpu().numpy()
        plt.errorbar(
            dim_mean[i].detach().cpu().numpy(),
            fid_mean[i].detach().cpu().numpy(),
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=colors[i],
            markersize=max(float(s) ** 0.5, 1.0),
            linewidth=0,
            elinewidth=1.5,
            capsize=0,
            alpha=0.9,
            label=legend_labels[i] if average_derived_curve else None,
        )
        if not average_derived_curve:
            plt.plot(
                dim_fit.detach().cpu().numpy(),
                fidelity_fit_curve.detach().cpu().numpy(),
                color=colors[i],
                linewidth=2,
                label=legend_labels[i],
            )

    if average_derived_curve:
        # Average predictions pointwise without extrapolating any component
        # curve beyond the complexity interval generated by its source fit.
        common_dim = torch.linspace(
            min(curve[0].min().item() for curve in derived_curves),
            max(curve[0].max().item() for curve in derived_curves),
            num_fit_points,
        )
        prediction_sum = torch.zeros_like(common_dim)
        prediction_count = torch.zeros_like(common_dim)
        for dim_fit, fidelity_fit_curve in derived_curves:
            order = torch.argsort(dim_fit)
            dim_sorted = dim_fit[order]
            fidelity_sorted = fidelity_fit_curve[order]
            supported = (common_dim >= dim_sorted[0]) & (common_dim <= dim_sorted[-1])
            query = common_dim[supported]
            right = torch.searchsorted(dim_sorted, query).clamp(1, dim_sorted.numel() - 1)
            left = right - 1
            weight = ((query - dim_sorted[left]) /
                      (dim_sorted[right] - dim_sorted[left]).clamp_min(1e-12))
            interpolated = fidelity_sorted[left] + weight * (
                fidelity_sorted[right] - fidelity_sorted[left]
            )
            prediction_sum[supported] += interpolated
            prediction_count[supported] += 1

        supported = prediction_count > 0
        mean_fidelity = prediction_sum[supported] / prediction_count[supported]
        plt.plot(
            common_dim[supported].detach().cpu().numpy(),
            mean_fidelity.detach().cpu().numpy(),
            color="black",
            linewidth=2.5,
            label=average_curve_label,
        )

    plt.xlabel(axis_labels[0], fontsize=font_size)
    plt.ylabel(axis_labels[1], fontsize=font_size)
    plt.xticks(fontsize=font_size * 0.8)
    plt.yticks(fontsize=font_size * 0.8)
    if x_ticks is not None:
        plt.xticks(x_ticks)
    if y_ticks is not None:
        plt.yticks(y_ticks)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    ax.spines[["top", "right"]].set_visible(False)
    if legend_labels and legend_labels[0] != "":
        plt.legend(frameon=False, fontsize=font_size * 0.8)
    if owns_figure:
        plt.tight_layout()
    if figpath:
        plt.savefig(figpath)
    if show and owns_figure:
        plt.show()
    return derived_curves


def plot_fidelity_dimensionality_summary(
    neighborhood_x,
    fidelity,
    dimensionality,
    legend_labels=("Δ=1", "Δ=4", "Δ=9"),
    rep_spread="std",
    omit_second_from_dimensionality_fit=True,
    figure_paths=(
        "./figures/accuracy.svg",
        "./figures/dimensionality.svg",
        "./figures/accdimratio.svg",
    ),
    font_size=25,
    show=True,
):
    """Fit and plot fidelity, complexity, and their derived relation in one row.

    The three panels are produced by one call. Each panel is also cropped from
    the row and saved separately to ``figure_paths``. The third panel is
    derived analytically from the first two shifted-power fits; it is not fit
    again.
    """
    if len(figure_paths) != 3:
        raise ValueError("figure_paths must contain accuracy, dimensionality, and joint paths.")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fidelity_fit = fit_and_plot(
        neighborhood_x,
        fidelity,
        axis_labels=("lateral neighborhood size $N_\\mathrm{lat}$", "fidelity"),
        legend_labels=legend_labels,
        fit_func="shifted_power",
        x_ticks=(20, 60, 100),
        rep_axis=0,
        rep_spread=rep_spread,
        legend_on_points=True,
        font_size=font_size,
        show=False,
        ax=axes[0],
    )
    dimensionality_fit = fit_and_plot(
        neighborhood_x,
        dimensionality,
        axis_labels=("lateral neighborhood size $N_\\mathrm{lat}$", "complexity"),
        legend_labels=legend_labels,
        fit_func="shifted_power",
        omit_second_from_fit=omit_second_from_dimensionality_fit,
        y_ticks=(0, 1, 2),
        x_ticks=(20, 60, 100),
        rep_axis=0,
        rep_spread=rep_spread,
        legend_on_points=True,
        font_size=font_size,
        show=False,
        ax=axes[1],
    )
    derived_curves = plot_fidelity_vs_dimensionality_from_fits(
        neighborhood_x,
        dimensionality,
        fidelity,
        fidelity_fit,
        dimensionality_fit,
        axis_labels=("complexity", "fidelity"),
        legend_labels=legend_labels,
        x_ticks=(0, 1, 2),
        rep_spread=rep_spread,
        average_derived_curve=True,
        average_curve_label=None,
        font_size=font_size,
        show=False,
        ax=axes[2],
    )

    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for panel_ax, path in zip(axes, figure_paths):
        if path:
            bbox = panel_ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(path, bbox_inches=bbox.expanded(1.03, 1.04))

    if show:
        plt.show()
    return {
        "fidelity_fit": fidelity_fit,
        "dimensionality_fit": dimensionality_fit,
        "derived_curves": derived_curves,
        "figure": fig,
        "axes": axes,
    }


def _noise_result_path(kind, prefer_backup=True, backup_dir="./parameter_search_data/no_rep_data", current_dir="./data_l4"):
    if kind not in {"topo", "sp"}:
        raise ValueError("kind must be 'topo' or 'sp'.")

    backup_path = Path(backup_dir) / f"data_noise_{kind}.pt"
    current_path = Path(current_dir) / f"data_noise_{kind}.pt"

    if prefer_backup and backup_path.exists():
        return backup_path
    return current_path


def latest_noise_result_paths(data_dir="./data_l4"):
    """Return the newest matching salt-and-pepper/topological result pair."""
    data_dir = Path(data_dir)
    prefixes = {
        "sp": "data_noise_sp",
        "topo": "data_noise_topo",
    }
    tagged_paths = {}

    for kind, prefix in prefixes.items():
        paths_by_tag = {}
        for path in data_dir.glob(f"{prefix}*.pt"):
            suffix = path.stem[len(prefix):]
            if suffix and not suffix.startswith("_"):
                continue
            tag = suffix[1:] if suffix.startswith("_") else ""
            paths_by_tag[tag] = path
        tagged_paths[kind] = paths_by_tag

    common_tags = set(tagged_paths["sp"]) & set(tagged_paths["topo"])
    if not common_tags:
        raise FileNotFoundError(
            f"No matching data_noise_sp*.pt/data_noise_topo*.pt pair found in {data_dir}."
        )

    def _pair_timestamp(tag):
        mtimes = [tagged_paths[kind][tag].stat().st_mtime for kind in prefixes]
        return min(mtimes), max(mtimes), tag

    latest_tag = max(common_tags, key=_pair_timestamp)
    return {
        kind: tagged_paths[kind][latest_tag]
        for kind in prefixes
    }


def _load_noise_robustness_result(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    noise_rob = torch.as_tensor(data["noise_rob"]).float()

    # Old schema: [radius, noise_condition]
    # Current schema: [rep, radius, noise_condition]
    if noise_rob.dim() == 2:
        noise_rob = noise_rob.unsqueeze(0)
    if noise_rob.dim() != 3:
        raise ValueError(
            "Expected noise_rob with shape [radius, noise] or "
            f"[rep, radius, noise], got {tuple(noise_rob.shape)}"
        )

    trialvar = torch.as_tensor(data["trialvar"]).float()
    noise_conditions = torch.as_tensor(data["noise_conditions"]).float()
    return data, noise_rob, trialvar, noise_conditions


def _threshold_noise_intensity(noise_rob, noise_conditions, threshold):
    reps, radii, _ = noise_rob.shape
    out = torch.empty(reps, radii)

    for rep in range(reps):
        for radius in range(radii):
            loc = float(find_val_loc(noise_rob[rep, radius], threshold))
            loc = float(np.clip(loc, 0, len(noise_conditions) - 1))
            idx0 = int(np.floor(loc))
            idx1 = min(idx0 + 1, len(noise_conditions) - 1)
            frac = loc - idx0
            out[rep, radius] = noise_conditions[idx0] * (1 - frac) + noise_conditions[idx1] * frac

    return out


def plot_noise_robustness(
    both=False,
    robustness_threshold=0.95,
    omit_first_k=0,
    prefer_backup=True,
    topo_path=None,
    sp_path=None,
    figpath="./figures/robustness.svg",
    ylim=(0, 0.1),
    x_ticks=(2, 8, 14),
    y_ticks=(0.0, 0.05, 0.1),
    font_size=25,
    s=300,
    rep_spread="sem",
    neighborhood_divisor=7.3,
    show=True,
    ax=None,
):
    """
    Plot robustness threshold vs lateral neighborhood size.

    both=True reproduces the two-condition view with a shared/global fit.
    both=False plots the topo fit as a solid line and SP data as open circles.
    omit_first_k removes that many leading SP radius conditions only.
    """
    topo_path = Path(topo_path) if topo_path is not None else _noise_result_path("topo", prefer_backup)
    sp_path = Path(sp_path) if sp_path is not None else _noise_result_path("sp", prefer_backup)

    _, topo_noise, topo_trialvar, topo_noise_conds = _load_noise_robustness_result(topo_path)
    _, sp_noise, sp_trialvar, sp_noise_conds = _load_noise_robustness_result(sp_path)

    topo_robustness = _threshold_noise_intensity(topo_noise, topo_noise_conds, robustness_threshold)
    sp_robustness = _threshold_noise_intensity(sp_noise, sp_noise_conds, robustness_threshold)

    neighborhood_divisor = float(neighborhood_divisor)
    if not np.isfinite(neighborhood_divisor) or neighborhood_divisor <= 0:
        raise ValueError("neighborhood_divisor must be a finite positive number.")

    if isinstance(omit_first_k, bool) or not isinstance(omit_first_k, (int, np.integer)):
        raise TypeError("omit_first_k must be a non-negative integer.")
    omit_first_k = int(omit_first_k)
    if omit_first_k < 0 or omit_first_k >= sp_robustness.shape[1]:
        raise ValueError(
            "omit_first_k must satisfy 0 <= omit_first_k < "
            f"{sp_robustness.shape[1]}; got {omit_first_k}."
        )
    if both and omit_first_k:
        raise ValueError(
            "omit_first_k applies only to the SP series and therefore requires both=False."
        )
    sp_robustness = sp_robustness[:, omit_first_k:]
    sp_trialvar = sp_trialvar[omit_first_k:]

    topo_x = (
        torch.as_tensor(np.pi, dtype=torch.float32)
        * (topo_trialvar / 3) ** 2
        / neighborhood_divisor
    )
    sp_x = (
        torch.as_tensor(np.pi, dtype=torch.float32)
        * (sp_trialvar / 3) ** 2
        / neighborhood_divisor
    )

    axis_labels = (r"lateral neighborhood size $N_\mathrm{lat}$", "robustness")
    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=(7, 6))
    else:
        plt.sca(ax)

    if both:
        if topo_robustness.shape[0] != sp_robustness.shape[0]:
            raise ValueError("Expected matching repeat counts for topo and SP noise files.")

        x = torch.stack([topo_x, sp_x], dim=0)
        robustness = torch.stack([topo_robustness, sp_robustness], dim=1)

        params = fit_and_plot(
            x,
            robustness,
            axis_labels=axis_labels,
            fit_func="hyperbolic",
            global_fit=True,
            scatter=True,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            ylim=ylim,
            marker_mode=True,
            s=s,
            cmap="tab10",
            legend_labels=("macro-GCAL", "micro-GCAL"),
            fit_flag=True,
            figpath=figpath,
            color=["silver", "black"],
            rep_axis=0,
            rep_spread=rep_spread,
            font_size=font_size,
            legend_loc="upper right",
            show=False,
            ax=ax,
        )
        if show:
            plt.show()
        return params

    topo_curve = topo_robustness.mean(dim=0)
    sp_curve = sp_robustness.mean(dim=0)

    if rep_spread == "sem":
        sp_curve_spread = sp_robustness.std(
            dim=0, unbiased=sp_robustness.shape[0] > 1
        ) / max(sp_robustness.shape[0], 1) ** 0.5
    elif rep_spread == "std":
        sp_curve_spread = sp_robustness.std(
            dim=0, unbiased=sp_robustness.shape[0] > 1
        )
    elif rep_spread == "2std":
        sp_curve_spread = 2.0 * sp_robustness.std(
            dim=0, unbiased=sp_robustness.shape[0] > 1
        )
    elif rep_spread in (None, False, "none"):
        sp_curve_spread = None
    else:
        raise ValueError("rep_spread must be 'sem', 'std', '2std', or 'none'.")

    params = fit_and_plot(
        topo_x,
        topo_curve,
        axis_labels=axis_labels,
        fit_func="hyperbolic",
        global_fit=False,
        scatter=False,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        ylim=ylim,
        fit_flag=True,
        figpath=None,
        color="white",
        show=False,
        font_size=font_size,
        ax=ax,
    )

    if ax.lines:
        ax.lines[0].set_visible(False)
    if len(ax.lines) > 1:
        ax.lines[1].set_color("black")
        ax.lines[1].set_linewidth(2.5)
        ax.lines[1].set_label("macro-GCAL (scaled)")

    point_colors = plt.get_cmap("inferno")(np.linspace(0.2, 0.85, len(sp_curve)))

    if sp_curve_spread is not None:
        ax.errorbar(
            sp_x.detach().cpu().numpy(),
            sp_curve.detach().cpu().numpy(),
            yerr=sp_curve_spread.detach().cpu().numpy(),
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=0,
            zorder=3,
        )

    ax.scatter(
        sp_x.detach().cpu().numpy(),
        sp_curve.detach().cpu().numpy(),
        facecolors=point_colors,
        edgecolors=point_colors,
        marker="o",
        s=s,
        linewidths=1.5,
        label="micro-GCAL",
        zorder=4,
    )

    ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=font_size * 0.8, loc="upper left")
    if owns_figure:
        plt.tight_layout()
    if figpath:
        ax.figure.savefig(figpath)
    if show:
        plt.show()

    return params


def plot_topological_noise_robustness(
    robustness_threshold=0.95,
    prefer_backup=True,
    topo_path=None,
    figpath="./figures/topological_robustness.svg",
    ylim=(0, 0.1),
    x_ticks=(20, 50, 80),
    y_ticks=(0.0, 0.05, 0.1),
    font_size=25,
    s=300,
    rep_spread="sem",
    show=True,
    ax=None,
):
    """Plot all topological-map robustness points above their logarithmic fit."""
    topo_path = (
        Path(topo_path)
        if topo_path is not None
        else _noise_result_path("topo", prefer_backup)
    )
    _, topo_noise, topo_trialvar, topo_noise_conds = _load_noise_robustness_result(
        topo_path
    )
    topo_robustness = _threshold_noise_intensity(
        topo_noise, topo_noise_conds, robustness_threshold
    )
    topo_x = torch.as_tensor(np.pi, dtype=torch.float32) * (topo_trialvar / 3) ** 2
    topo_curve = topo_robustness.mean(dim=0)

    if rep_spread == "sem":
        spread = topo_robustness.std(
            dim=0, unbiased=topo_robustness.shape[0] > 1
        ) / max(topo_robustness.shape[0], 1) ** 0.5
    elif rep_spread == "std":
        spread = topo_robustness.std(
            dim=0, unbiased=topo_robustness.shape[0] > 1
        )
    elif rep_spread == "2std":
        spread = 2.0 * topo_robustness.std(
            dim=0, unbiased=topo_robustness.shape[0] > 1
        )
    elif rep_spread in (None, False, "none"):
        spread = None
    else:
        raise ValueError("rep_spread must be 'sem', 'std', '2std', or 'none'.")

    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=(7, 6))
    else:
        plt.sca(ax)

    params = fit_and_plot(
        topo_x,
        topo_curve,
        axis_labels=(r"lateral neighborhood size $N_\mathrm{lat}$", "robustness"),
        fit_func="hyperbolic",
        scatter=False,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        ylim=ylim,
        fit_flag=True,
        figpath=None,
        color="white",
        show=False,
        font_size=font_size,
        ax=ax,
    )
    if ax.lines:
        ax.lines[0].set_visible(False)
    if len(ax.lines) > 1:
        ax.lines[1].set_color("black")
        ax.lines[1].set_linewidth(2.5)

    if spread is not None:
        ax.errorbar(
            topo_x.detach().cpu().numpy(),
            topo_curve.detach().cpu().numpy(),
            yerr=spread.detach().cpu().numpy(),
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=0,
            zorder=3,
        )
    point_colors = plt.get_cmap("inferno")(
        np.linspace(0.2, 0.85, len(topo_curve))
    )
    ax.scatter(
        topo_x.detach().cpu().numpy(),
        topo_curve.detach().cpu().numpy(),
        facecolors=point_colors,
        edgecolors=point_colors,
        marker="o",
        s=s,
        linewidths=1.5,
        zorder=4,
    )

    ax.set_ylim(*ylim)
    if owns_figure:
        plt.tight_layout()
    if figpath:
        ax.figure.savefig(figpath)
    if show:
        plt.show()
    return params


def save_topological_noise_stability(
    topo_path,
    figpath="./figures/stability.svg",
    rep_spread="2std",
    font_size=25,
):
    """Save topological stability trajectories without displaying the figure."""
    _, topo_noise, _, noise_conditions = _load_noise_robustness_result(topo_path)
    figure, axis = plt.subplots(figsize=(7, 6))
    fit_and_plot(
        noise_conditions,
        topo_noise,
        axis_labels=("noise intensity", "stability"),
        ylim=(0.8999, 1),
        y_ticks=(0.9, 0.95, 1),
        x_ticks=(0, 0.05, 0.1),
        fit_flag=False,
        scatter=False,
        rep_axis=0,
        rep_spread=rep_spread,
        font_size=font_size,
        figpath=None,
        show=False,
        ax=axis,
    )
    figure.tight_layout()
    output_path = Path(figpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved stability figure: {output_path}")
    return output_path


def plot_wiring_efficiency_six_panel_summary(
    neighborhood_x,
    fidelity,
    dimensionality,
    topo_path,
    sp_path,
    acc_baseline=0.65,
    dim_baseline=85,
    robustness_threshold=0.95,
    omit_first_k=0,
    legend_labels=("Δ=1", "Δ=4", "Δ=9"),
    rep_spread="2std",
    omit_second_from_dimensionality_fit=True,
    figure_paths=(
        "./figures/accuracy.svg",
        "./figures/dimensionality.svg",
        "./figures/accdimratio.svg",
        "./figures/topological_robustness.svg",
        "./figures/robustness.svg",
        "./figures/efficiency.svg",
    ),
    stability_path="./figures/stability.svg",
    font_size=25,
    show=True,
):
    """Plot the six paper panels in a 3+3 layout and export each axis as SVG."""
    if len(figure_paths) != 6:
        raise ValueError("figure_paths must contain exactly six output paths.")

    neighborhood_x = torch.as_tensor(neighborhood_x, dtype=torch.float32)
    fidelity = torch.as_tensor(fidelity, dtype=torch.float32)
    dimensionality = torch.as_tensor(dimensionality, dtype=torch.float32)

    figure = plt.figure(figsize=(21, 12))
    grid = figure.add_gridspec(2, 6)
    axes = (
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
        figure.add_subplot(grid[1, 0:2]),
        figure.add_subplot(grid[1, 2:4]),
        figure.add_subplot(grid[1, 4:6]),
    )

    fidelity_fit = fit_and_plot(
        neighborhood_x,
        fidelity,
        axis_labels=(r"lateral neighborhood size $N_\mathrm{lat}$", "fidelity"),
        legend_labels=legend_labels,
        fit_func="shifted_power",
        x_ticks=(20, 50, 80),
        y_ticks=(0.2, 0.4, 0.6),
        ylim=(0.2, 0.62),
        rep_axis=0,
        rep_spread=rep_spread,
        legend_on_points=True,
        font_size=font_size,
        show=False,
        ax=axes[0],
    )
    dimensionality_fit = fit_and_plot(
        neighborhood_x,
        dimensionality,
        axis_labels=(r"lateral neighborhood size $N_\mathrm{lat}$", "complexity"),
        legend_labels=legend_labels,
        fit_func="shifted_power",
        omit_second_from_fit=omit_second_from_dimensionality_fit,
        y_ticks=(0, 0.5, 1),
        x_ticks=(20, 50, 80),
        ylim=(0, 1.3),
        rep_axis=0,
        rep_spread=rep_spread,
        legend_on_points=True,
        font_size=font_size,
        show=False,
        ax=axes[1],
    )
    saturation_point = saturation_log_curvature_point(
        fidelity_fit, dimensionality_fit
    )
    saturation_complexity = saturation_point["complexity"]
    saturation_fidelity = saturation_point["fidelity"]
    axes[2].axvline(
        saturation_complexity,
        color="black",
        linestyle="--",
        linewidth=2,
        zorder=0,
    )
    axes[2].axhline(
        saturation_fidelity,
        color="black",
        linestyle="--",
        linewidth=2,
        zorder=0,
    )
    derived_curves = plot_fidelity_vs_dimensionality_from_fits(
        neighborhood_x,
        dimensionality,
        fidelity,
        fidelity_fit,
        dimensionality_fit,
        axis_labels=("complexity", "fidelity"),
        legend_labels=legend_labels,
        x_ticks=(0, 0.5, 1),
        y_ticks=(0.2, 0.4, 0.6),
        xlim=(0, 1.3),
        ylim=(0.2, 0.7),
        rep_spread=rep_spread,
        average_derived_curve=True,
        average_curve_label=None,
        font_size=font_size,
        show=False,
        ax=axes[2],
    )
    axes[2].legend(
        frameon=False,
        fontsize=font_size * 0.8,
        loc="upper left",
    )

    topological_robustness_fit = plot_topological_noise_robustness(
        robustness_threshold=robustness_threshold,
        prefer_backup=False,
        topo_path=topo_path,
        ylim=(0, 0.1),
        rep_spread=rep_spread,
        figpath=None,
        font_size=font_size,
        show=False,
        ax=axes[3],
    )

    robustness_fit = plot_noise_robustness(
        both=False,
        robustness_threshold=robustness_threshold,
        omit_first_k=omit_first_k,
        prefer_backup=False,
        topo_path=topo_path,
        sp_path=sp_path,
        ylim=(0, 0.1),
        rep_spread=rep_spread,
        figpath=None,
        font_size=font_size,
        show=False,
        ax=axes[4],
    )

    sp_data = torch.load(sp_path, map_location="cpu", weights_only=False)
    sp_fidelity = (
        torch.as_tensor(sp_data["noise_acc"], dtype=torch.float32)[..., 0]
        - acc_baseline
    ) / (1.0 - acc_baseline)
    sp_dimensionality = (
        torch.as_tensor(sp_data["se_pca_tracker"], dtype=torch.float32)
        / dim_baseline
    )
    axes[5].axvline(
        saturation_complexity,
        color="black",
        linestyle="--",
        linewidth=2,
        zorder=0,
    )
    axes[5].axhline(
        saturation_fidelity,
        color="black",
        linestyle="--",
        linewidth=2,
        zorder=0,
    )
    efficiency_fit = fit_and_plot(
        sp_dimensionality.mean(dim=0)[omit_first_k:, None],
        sp_fidelity[:, omit_first_k:, None],
        axis_labels=("complexity", "fidelity"),
        ylim=(0.2, 0.7),
        xlim=(0, 1.3),
        x_ticks=(0, 0.5, 1),
        y_ticks=(0.2, 0.4, 0.6),
        s=400,
        fit_flag=False,
        rep_axis=0,
        rep_spread=rep_spread,
        figpath=None,
        show=False,
        ax=axes[5],
    )

    figure.tight_layout()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for panel_ax, path in zip(axes, figure_paths):
        if path:
            bbox = panel_ax.get_tightbbox(renderer).transformed(
                figure.dpi_scale_trans.inverted()
            )
            figure.savefig(path, bbox_inches=bbox.expanded(1.03, 1.04))

    saved_stability_path = None
    if stability_path:
        saved_stability_path = save_topological_noise_stability(
            topo_path,
            figpath=stability_path,
            rep_spread=rep_spread,
            font_size=font_size,
        )

    if show:
        plt.show()
    return {
        "fidelity_fit": fidelity_fit,
        "dimensionality_fit": dimensionality_fit,
        "derived_curves": derived_curves,
        "topological_robustness_fit": topological_robustness_fit,
        "robustness_fit": robustness_fit,
        "efficiency_fit": efficiency_fit,
        "saturation_complexity": saturation_complexity,
        "saturation_fidelity": saturation_fidelity,
        "saturation_plateau_fraction": saturation_point["plateau_fraction"],
        "saturation_point": saturation_point,
        "stability_path": saved_stability_path,
        "figure": figure,
        "axes": axes,
    }


def _make_grating_bank_onoff(
    W: int,
    thetas_rad: torch.Tensor,      # [K]
    freqs: torch.Tensor,           # [F] cycles per image (or per W pixels, see notes)
    phases_rad: torch.Tensor,      # [P]
    device=None,
    dtype=torch.float32,
    centered: bool = True,
) -> torch.Tensor:
    """
    Returns stimuli shaped [K, F, P, 2, W, W] for ON/OFF channels:
      L(x,y) = cos(2π f (x cosθ + y sinθ) + phase)
      ON = relu(L), OFF = relu(-L)
    """
    device = device or thetas_rad.device
    thetas_rad = thetas_rad.to(device=device, dtype=dtype)
    freqs = freqs.to(device=device, dtype=dtype)
    phases_rad = phases_rad.to(device=device, dtype=dtype)

    # grid in [-0.5, 0.5] (roughly) or [0,1)
    if centered:
        coords = torch.linspace(-(W - 1) / 2, (W - 1) / 2, W, device=device, dtype=dtype)
        coords = coords / max(W, 1)  # normalize to ~[-0.5,0.5]
    else:
        coords = torch.linspace(0, 1, W, device=device, dtype=dtype)

    yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # [W,W]

    # Expand for broadcasting
    # theta: [K,1,1,1,1]
    ct = torch.cos(thetas_rad)[:, None, None, None, None]
    st = torch.sin(thetas_rad)[:, None, None, None, None]

    # freqs: [1,F,1,1,1]
    fr = freqs[None, :, None, None, None]

    # phases: [1,1,P,1,1]
    ph = phases_rad[None, None, :, None, None]

    # project coordinate along orientation
    # proj: [K,1,1,W,W]
    proj = (xx[None, None, None, :, :] * ct) + (yy[None, None, None, :, :] * st)

    # L: [K,F,P,W,W]
    L = torch.cos(2 * math.pi * fr * proj + ph)

    on = F.relu(L)
    off = F.relu(-L)

    stim = torch.stack([on, off], dim=3)  # [K,F,P,2,W,W]
    return stim


def detect_orientation_map_from_aff_weights(
    aff: torch.Tensor,                 # [N, 2, W, W]  (channel 0=ON, 1=OFF)
    *,
    num_orientations: int = 18,         # e.g. 18 => 0..170° in 10° steps
    freqs: Optional[torch.Tensor] = None,
    num_phases: int = 8,
    rectify_output: bool = False,       # optional cortical half-wave rectification
    batch_size: int = 2048,
    return_degrees: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Approximates the paper's map readout (vector-average from grating responses),
    using the afferent ON/OFF->V1 weights as the "receptive field".

    Steps:
      1) Build ON/OFF grating bank with K orientations, F spatial freqs, P phases.
      2) For each neuron, compute linear response = sum(w * stim).
         Optionally rectify.
      3) For each orientation, take max over phase and frequency -> r(theta).
      4) Vector average with 2*theta: v = Σ r(theta) * exp(i*2*theta)
         pref = 0.5 * arg(v), selectivity = |v|, osi = |v| / (Σ r + eps)

    Returns dict with:
      - pref: [N] preferred orientation (deg in [0,180) unless return_degrees=False)
      - selectivity: [N] magnitude of vector sum (raw)
      - osi: [N] normalized selectivity in [0,1] (common)
      - r_theta: [N,K] orientation response curve after max over phase/freq
    """
    if aff.ndim != 4 or aff.shape[1] != 2 or aff.shape[2] != aff.shape[3]:
        raise ValueError(f"Expected aff shape [N,2,W,W], got {tuple(aff.shape)}")

    N, _, W, _ = aff.shape
    device = aff.device
    dtype = aff.dtype

    # orientations in [0, pi)
    K = int(num_orientations)
    thetas = torch.linspace(0, math.pi, K + 1, device=device, dtype=dtype)[:-1]  # [K]

    # spatial freqs (cycles per normalized image width ~W). Tune as needed.
    if freqs is None:
        # A small set tends to work well. You can pass your own.
        freqs = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)  # [F]
    else:
        freqs = freqs.to(device=device, dtype=dtype)

    P = int(num_phases)
    phases = torch.linspace(0, 2 * math.pi, P + 1, device=device, dtype=dtype)[:-1]  # [P]

    stim = _make_grating_bank_onoff(
        W=W,
        thetas_rad=thetas,
        freqs=freqs,
        phases_rad=phases,
        device=device,
        dtype=dtype,
        centered=True,
    )  # [K,F,P,2,W,W]

    # Flatten stim spatial+channel for fast matmul
    stim_flat = stim.reshape(K * freqs.numel() * P, 2 * W * W)  # [K*F*P, 2WW]
    stim_flat_t = stim_flat.t().contiguous()  # [2WW, K*F*P]

    r_theta_all = torch.empty((N, K), device=device, dtype=dtype)

    # Batch neurons to control memory
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        w = aff[start:end].reshape(end - start, 2 * W * W)  # [B,2WW]

        # Linear responses to all gratings: [B, K*F*P]
        resp = w @ stim_flat_t
        if rectify_output:
            resp = F.relu(resp)

        # Reshape to [B,K,F,P]
        resp = resp.view(end - start, K, freqs.numel(), P)

        # Max over phase then frequency -> [B,K]
        r_theta = resp.max(dim=3).values.max(dim=2).values
        r_theta_all[start:end] = r_theta

    # Vector average using 2*theta
    # v = Σ r(theta) * exp(i*2θ)
    exp_i2t = torch.exp(1j * (2.0 * thetas).to(torch.complex64))  # [K] complex
    r_complex = r_theta_all.to(torch.complex64)  # [N,K]
    v = (r_complex * exp_i2t[None, :]).sum(dim=1)  # [N] complex

    pref_rad = 0.5 * torch.atan2(v.imag, v.real)  # [-pi/2, pi/2]
    # map to [0, pi)
    pref_rad = torch.remainder(pref_rad, math.pi)

    selectivity = torch.abs(v).to(dtype)  # [N]
    denom = r_theta_all.sum(dim=1).clamp_min(1e-8)
    osi = (selectivity / denom).clamp(0, 1)

    if return_degrees:
        pref = pref_rad * (180.0 / math.pi)
    else:
        pref = pref_rad

    return {
        "pref": pref,                 # [N]
        "selectivity": selectivity,   # [N] raw
        "osi": osi,                   # [N] normalized
        "r_theta": r_theta_all,       # [N,K]
        "thetas": (thetas * (180.0 / math.pi) if return_degrees else thetas),  # [K]
    }


def _make_sine_grating(
    W: int,
    theta_rad: float,
    spatial_freq_cyc_per_px: float,
    phase_rad: float,
    device=None,
    dtype=None,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-(W - 1) / 2, (W - 1) / 2, W, device=device, dtype=dtype),
        torch.linspace(-(W - 1) / 2, (W - 1) / 2, W, device=device, dtype=dtype),
        indexing="ij",
    )
    x_prime = xx * math.cos(theta_rad) + yy * math.sin(theta_rad)
    return torch.sin(2.0 * math.pi * spatial_freq_cyc_per_px * x_prime + phase_rad)


def compute_orientation_maps(
    model,
    W: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    n_orientations: int = 12,
    n_phases: int = 4,
    spatial_freq_cyc_per_px: float = 0.05,
    contrast: float = 1.0,
    scaler: float = 0.3,
    rectify: bool = True,
    use_vector_sum: bool = True,
    reset_fn: Optional[str] = "reset_state",
    settle_repeats: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes two orientation preference maps at once:
      - ori_map: from model.current_response (L4-like)
      - ori_map_l3: from model.current_response_l3 (L2/3-like)

    Both returned maps are 2D tensors in radians in [0, π), ready for:
        plt.imshow(map.cpu(), cmap="hsv", vmin=0, vmax=math.pi)

    Assumes your model forward is called as model(x, adaptation=False) (as in your NeuralSheet).
    """

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device(getattr(model, "device", "cpu"))
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

    def _maybe_reset():
        if reset_fn is None:
            return
        fn = getattr(model, reset_fn, None)
        if callable(fn):
            fn()

    def _reduce_to_2d(r: torch.Tensor) -> torch.Tensor:
        r = r.to(device=device)
        if r.dim() >= 3 and r.shape[0] == 1:
            r = r[0]
        if r.dim() == 2:
            r2 = r
        elif r.dim() == 3:
            # (C,H,W) -> mean over channels
            r2 = r.mean(dim=0)
        elif r.dim() == 1:
            n = r.numel()
            s = int(round(math.sqrt(n)))
            if s * s != n:
                raise ValueError(f"1D response length {n} is not a perfect square.")
            r2 = r.view(s, s)
        else:
            raise ValueError(f"Unsupported response shape: {tuple(r.shape)}")

        if rectify:
            return torch.relu(r2)
        return r2.abs()

    # Orientations in [0, π)
    thetas = torch.linspace(0.0, math.pi, n_orientations + 1, device=device, dtype=torch.float32)[:-1]
    phases = torch.linspace(0.0, 2.0 * math.pi, n_phases + 1, device=device, dtype=torch.float32)[:-1]

    resp_maps_per_ori = []
    resp_maps_l3_per_ori = []

    for theta in thetas.tolist():
        phase_accum = None
        phase_accum_l3 = None

        for ph in phases.tolist():
            stim = _make_sine_grating(
                W=W,
                theta_rad=float(theta),
                spatial_freq_cyc_per_px=spatial_freq_cyc_per_px,
                phase_rad=float(ph),
                device=device,
                dtype=dtype,
            )
            stim = (stim + 1.0) * (contrast * scaler)

            x_4d = stim.unsqueeze(0).unsqueeze(0)  # (1,1,W,W)

            _maybe_reset()
            for _ in range(max(1, int(settle_repeats))):
                try:
                    model(x_4d, adaptation=False, layer_3=True)
                except TypeError:
                    model(x_4d, adaptation=False)

            r = model.current_response
            r_l3 = model.current_response_l3
            if not torch.is_tensor(r):
                r = torch.as_tensor(r, device=device)
            if not torch.is_tensor(r_l3):
                r_l3 = torch.as_tensor(r_l3, device=device)

            r2 = _reduce_to_2d(r)
            r2_l3 = _reduce_to_2d(r_l3)

            phase_accum = r2 if phase_accum is None else (phase_accum + r2)
            phase_accum_l3 = r2_l3 if phase_accum_l3 is None else (phase_accum_l3 + r2_l3)

        resp_maps_per_ori.append(phase_accum / float(len(phases)))
        resp_maps_l3_per_ori.append(phase_accum_l3 / float(len(phases)))

    R = torch.stack(resp_maps_per_ori, dim=0)        # (n_ori, H, W)
    R_l3 = torch.stack(resp_maps_l3_per_ori, dim=0)  # (n_ori, H, W)

    # remove orientation-mean component per pixel (helps with global DC / uniform activation)
    R = R - R.mean(dim=0, keepdim=True)
    R_l3 = R_l3 - R_l3.mean(dim=0, keepdim=True)

    def _ori_from_stack(Rstack: torch.Tensor) -> torch.Tensor:
        if use_vector_sum:
            twotheta = 2.0 * thetas
            cos2 = torch.cos(twotheta).view(-1, 1, 1).to(device)
            sin2 = torch.sin(twotheta).view(-1, 1, 1).to(device)
            vx = (Rstack * cos2).sum(dim=0)
            vy = (Rstack * sin2).sum(dim=0)
            ori = 0.5 * torch.atan2(vy, vx)
            return torch.remainder(ori, math.pi)
        else:
            idx = torch.argmax(Rstack, dim=0)
            return thetas[idx].to(device)

    ori_map = _ori_from_stack(R)
    ori_map_l3 = _ori_from_stack(R_l3)

    return ori_map, ori_map_l3
