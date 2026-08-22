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
from sklearn.decomposition import PCA
import umap
import matplotlib.cm as cm
import math
import nn_template 
from typing import Tuple, Optional, Dict


class RandomCropDataset(Dataset):
    def __init__(self, directory, crop_size):
        self.directory = directory
        self.crop_size = crop_size
        self.images = [os.path.join(directory, f) for f in sorted(os.listdir(directory))
                       if os.path.isfile(os.path.join(directory, f))]
        # Pre-filter images smaller than the crop size
        self.images = [img for img in self.images if self._image_size(img) >= crop_size]

    def _image_size(self, filepath):
        with Image.open(filepath) as img:
            return min(img.size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)  # Open with PIL
        
        # Apply random transformation
        transformed_image = self.random_transformation(image, self.crop_size)
        return transformed_image

    def random_transformation(self, image, N):
        """Return a randomly located, rotated crop containing no padding.

        The source patch is large enough that its central ``N x N`` region
        remains inside real image pixels after rotation.  Sampling the source
        patch before rotating it avoids the filled corners produced when the
        complete image is rotated on a fixed-size canvas.
        """
        min_image_side = min(image.size)
        for _ in range(128):
            rotation_degrees = random.uniform(-180, 180)
            angle = math.radians(rotation_degrees)
            rotated_extent = N * (abs(math.cos(angle)) + abs(math.sin(angle)))
            # One guard pixel keeps bilinear interpolation away from the edge.
            source_size = math.ceil(rotated_extent - 1e-12) + 1
            if source_size <= min_image_side:
                break
        else:
            # Every indexed image is at least N pixels on its shorter side.
            # This fallback is relevant only when it is too small to support a
            # non-cardinal rotation without padding.
            rotation_degrees = 0.0
            source_size = N

        top, left, height, width = transforms.RandomCrop.get_params(
            image, output_size=(source_size, source_size)
        )
        source_patch = TF.crop(image, top, left, height, width)
        rotated_patch = TF.rotate(
            source_patch,
            rotation_degrees,
            interpolation=transforms.InterpolationMode.BILINEAR,
            expand=False,
        )
        return TF.to_tensor(TF.center_crop(rotated_patch, (N, N)))

def seed_dataloader_worker(worker_id):
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloader(
        root_dir,
        crop_size,
        batch_size,
        num_workers,
        seed=None,
        dataset=None):
    if dataset is None:
        dataset = RandomCropDataset(root_dir, crop_size)
    elif dataset.crop_size != crop_size:
        raise ValueError(
            'The reused dataset crop size does not match the requested '
            f'crop size: {dataset.crop_size} != {crop_size}.'
        )
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        worker_init_fn = seed_dataloader_worker

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    return dataloader

def generate_gaussians(number_of_gaussians, size_of_gaussian, sigma, offset=0):

    size_of_gaussian = size_of_gaussian + offset*2
    # Create a grid of coordinates (x, y) for the centers
    lin_centers = torch.linspace(-size_of_gaussian / 2 + offset, size_of_gaussian / 2 - offset, number_of_gaussians)
        
    x_centers, y_centers = torch.meshgrid(lin_centers, lin_centers, indexing='ij')
    
    # Create a grid of coordinates (x, y) for a single Gaussian of size MxM
    lin_gaussian = torch.linspace(-size_of_gaussian / 2, size_of_gaussian / 2, size_of_gaussian)
    x_gaussian, y_gaussian = torch.meshgrid(lin_gaussian, lin_gaussian, indexing='ij')
    
    # Flatten the center coordinates to easily use broadcasting
    x_centers_flat = x_centers.reshape(-1, 1, 1)
    y_centers_flat = y_centers.reshape(-1, 1, 1)
    
    # Calculate the squared distance for each Gaussian center to each point in the MxM grid
    dist_squared = (x_gaussian - x_centers_flat) ** 2 + (y_gaussian - y_centers_flat) ** 2
    
    # Precompute the Gaussian denominator
    gaussian_denom = 2 * sigma ** 2
    
    # Calculate the Gaussians
    gaussians = torch.exp(-dist_squared / gaussian_denom)
    
    # Normalize each Gaussian to have a maximum value of 1
    gaussians /= gaussians.view(number_of_gaussians**2, -1).sum(1).unsqueeze(1).unsqueeze(1)
    
    # Reshape to have each Gaussian in its own channel (N*N, M, M)
    gaussians = gaussians.reshape(number_of_gaussians**2, 1, size_of_gaussian, size_of_gaussian)
    
    return gaussians


def generate_circles(number_of_circles, size_of_circles, radius=0, offset=0):

    size_of_circles = size_of_circles + offset*2
    # Create a grid of coordinates (x, y) for the centers
    lin_centers = torch.linspace(-size_of_circles / 2 + offset, size_of_circles / 2 - offset, number_of_circles)

    if number_of_circles == 1:
        lin_centers += 0.5  # Adjust the center for a single circle

    x_centers, y_centers = torch.meshgrid(lin_centers, lin_centers, indexing='ij')
    
    # Create a grid of coordinates (x, y) for a single circle of size MxM
    lin_circle = torch.linspace(-size_of_circles / 2, size_of_circles / 2, size_of_circles)
    x_circle, y_circle = torch.meshgrid(lin_circle, lin_circle, indexing='ij')
    
    # Flatten the center coordinates to easily use broadcasting
    x_centers_flat = x_centers.reshape(-1, 1, 1)
    y_centers_flat = y_centers.reshape(-1, 1, 1)
    
    # Calculate the squared distance from each circle center to each point in the MxM grid
    dist_squared = (x_circle - x_centers_flat) ** 2 + (y_circle - y_centers_flat) ** 2
    dist = torch.sqrt(dist_squared)
    
    # Calculate circle membership with smoothing
    # Define the radius band for smooth transition (0.5 pixel width)
    radius_inner = radius - 0.5
    radius_outer = radius + 0.5

    # Compute a smooth transition in pixel values across the boundary of the circle
    circles = 1 - torch.clamp((dist - radius_inner) / (radius_outer - radius_inner), 0, 1)

    # Reshape back to the format (number_of_circles^2, 1, size_of_circles, size_of_circles)
    circles = circles.reshape(number_of_circles**2, 1, size_of_circles, size_of_circles)
    
    return circles.float()

def generate_euclidean_space(number_of_spaces, size_of_spaces, offset=0):

    size_of_spaces = size_of_spaces + offset*2
    # Create a grid of coordinates (x, y) for the centers
    lin_centers = torch.linspace(-number_of_spaces / 2 + offset, number_of_spaces / 2 - offset, number_of_spaces)

    if number_of_spaces == 1:
        lin_centers += 0.5  # Adjust the center for a single space

    x_centers, y_centers = torch.meshgrid(lin_centers, lin_centers, indexing='ij')
    
    # Create a grid of coordinates (x, y) for a single circle of size MxM
    lin_circle = torch.linspace(-size_of_spaces / 2, size_of_spaces / 2, size_of_spaces)
    x_circle, y_circle = torch.meshgrid(lin_circle, lin_circle, indexing='ij')
    
    # Flatten the center coordinates to easily use broadcasting
    x_centers_flat = x_centers.reshape(-1, 1, 1)
    y_centers_flat = y_centers.reshape(-1, 1, 1)
    
    # Calculate the squared distance from each center to each point in the MxM grid
    dist_squared = (x_circle - x_centers_flat) ** 2 + (y_circle - y_centers_flat) ** 2
    dist = torch.sqrt(dist_squared)

    # Reshape back to the format (number_of_circles^2, 1, size_of_circles, size_of_circles)
    dist = dist.reshape(number_of_spaces**2, 1, size_of_spaces, size_of_spaces)
    
    return dist.float()
    

def get_detectors(gabor_size, discreteness, device='cuda'):
    orientations = torch.linspace(0, np.pi, discreteness, device=device)
    lambd = gabor_size
    sigma = gabor_size/5
    gamma = 1
    psi = torch.tensor([0, np.pi/2], device=device)

    # Create a meshgrid for Gabor function
    x, y = torch.meshgrid(torch.linspace(-gabor_size//2 + 1/2, gabor_size//2 + 1/2, gabor_size, device=device), 
                          torch.linspace(-gabor_size//2 + 1/2, gabor_size//2 + 1/2, gabor_size, device=device), indexing='ij')

    x = x.expand(discreteness, 2, gabor_size, gabor_size)
    y = y.expand(discreteness, 2, gabor_size, gabor_size)
    orientations = orientations.view(discreteness, 1, 1, 1)
    psi = psi.view(1, 2, 1, 1)

    x_theta = x * torch.cos(orientations) + y * torch.sin(orientations)
    y_theta = -x * torch.sin(orientations) + y * torch.cos(orientations)
    
    gb = torch.cos(2 * np.pi * x_theta / lambd + psi) #* torch.exp(-.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
    
    gb = gb * get_circle(gabor_size, gabor_size/2, smooth=True).cuda()
    
    gb -= gb.mean([-1,-2], keepdim=True)
    
    return gb  # (discreteness, 2, gabor_size, gabor_size)

def get_orientations(weights, discreteness=101, gabor_size=25):
    
    device = weights.device

    #weights = weights[:, 0] - weights[:, 1]
    #weights = weights[:, None]
    
    # input is (M, 1, S, S)
    M, _, S, _ = weights.shape
    detectors = get_detectors(gabor_size, discreteness).to(device)
    
    # Prepare weights and detectors for convolution
    responses = torch.zeros((M, discreteness, 2, S, S), device=device)

    # Convolution over the receptive fields with each detector
    for i in range(discreteness):
        for j in range(2):
            responses[:, i, j] = F.conv2d(weights, detectors[i:i+1, j].unsqueeze(1), padding='valid').squeeze(1)
            
            
    responses = responses.view(M, discreteness, 2, -1).max(3)[0]
                
    # Compute phase map and magnitude of responses
    magnitudes = torch.sqrt((responses**2).sum(dim=2))
    orientations = magnitudes.max(dim=1)[1]
    
    phases = responses.gather(1, orientations[:,None,None].expand(-1,-1,2))
    phase_map = torch.atan2(phases[:, :, 1], phases[:, :, 0])
    
    shifts = torch.arange(discreteness)[None] + orientations[:,None].cpu() + discreteness//2
    shifts = shifts % discreteness
    mean_tc = magnitudes.cpu().gather(1, shifts).mean(0)
    
    orientation_map = orientations.float() / discreteness * np.pi
    orientation_map = orientation_map % torch.pi
    
    # output is M
    return orientation_map, phase_map, mean_tc
    

def get_grids(W, H, kernel_size, N, jitter=0, device='cuda'):

    # Generate grid positions for each patch using broadcasting
    grid_positions_w = torch.linspace(0, W - kernel_size, N, device=device).view(-1, 1) / (W - 1) * 2 - 1
    grid_positions_h = torch.linspace(0, H - kernel_size, N, device=device).view(1, -1) / (H - 1) * 2 - 1
    
    # Compute normalized coordinates for each patch
    x = grid_positions_w + torch.linspace(0, kernel_size - 1, kernel_size, device=device).view(1, -1) / (W - 1) * 2
    y = grid_positions_h + torch.linspace(0, kernel_size - 1, kernel_size, device=device).view(-1, 1) / (H - 1) * 2
    
    # Stack and reshape to create grid
    grids_x, grids_y = torch.meshgrid(
        x.flatten(), y.flatten(), indexing='ij')
    grids = torch.stack((grids_x, grids_y), dim=-1)
    
    grids = grids.view(N, kernel_size, kernel_size, N,  2).permute(3,0,2,1,4).reshape(N*N, kernel_size, kernel_size, 2)

    if jitter:
        
        grids += torch.randn((N*N,1,1,2), device=device) * jitter


    grids = grids.clip(-1,1)

    return grids

def extract_patches(input_image, grids):
    """
    Extracts N patches of size kernel_size x kernel_size from input_image, calculating the step size automatically.
    This function now handles cases where the last patch may not fit perfectly and returns patches with dimensions matching the input.
    This version leverages CUDA for improved performance.
    """
    Nxx2 = grids.shape[0]
    # Extract patches
    patches = F.grid_sample(input_image.expand(Nxx2, -1, -1, -1), grids, mode='bilinear', align_corners=False)
    
    return patches

def init_nn(input_size, output_size, out_channels=2, device='cuda'):

    network = {}
    
    network['structure'] = [
        ('flatten', 1, input_size**2),
        ('dense', out_channels*output_size**2, input_size**2),
        ('relu', ),
        ('dense', out_channels*output_size**2, out_channels*output_size**2),
        ('relu', ),
        ('dense', out_channels*output_size**2, out_channels*output_size**2),
        #('relu', ),
        ('unflatten', out_channels, output_size)
    ]
    
    network['model'] = nn_template.Network(network['structure'], device=device)
    params_list = [list(network['model'].layers[l].parameters()) for l in range(len(network['structure']))]
    params_list = sum(params_list, [])
    
    network['optim'] = torch.optim.Adam(params_list, lr=1e-3)
    network['activ'] = torch.sigmoid
    
    return network

def init_nn_audio(input_size, output_size, device='cuda'):

    network = {}
    
    network['structure'] = [
        ('flatten', 1, input_size),
        ('dense', output_size, input_size),
        ('relu', ),
        ('dense', output_size, output_size)
    ]
    
    network['model'] = nn_template.Network(network['structure'], device=device, bias=False)
    params_list = [list(network['model'].layers[l].parameters()) for l in range(len(network['structure']))]
    params_list = sum(params_list, [])
    
    network['optim'] = torch.optim.Adam(params_list, lr=1e-3)
    network['activ'] = torch.relu
    
    return network

def nn_loss(network, true_input, reco_input):

    mse = ((true_input - reco_input)**2).mean([1,2,3])

    l2 = sum([list(network['model'].layers[l].parameters())[0].square().sum() if list(network['model'].layers[l].parameters()) else 0 for l in range(len(network['structure']))])

    loss = mse.mean() + l2 * 0e-5
    loss_std = mse.std()
    
    return loss, loss_std

# Function to compute the Laplacian Of Gaussian Operator
def get_log(size, std):
    
    distance = torch.arange(size) - size//2
    x = distance.expand(1,1,size,size)**2
    y = x.transpose(-1,-2)
    t = (x + y) / (2*std**2)
    LoG = -1/(np.pi*std**2) * (1-t) * torch.exp(-t)
    LoG = LoG - LoG.mean()
    return LoG

def oddenise(number):
    return round(number)+1 if round(number)%2==0 else round(number)
    
def evenise(number):
    return round(number)+1 if round(number)%2==1 else round(number)


def get_gaussian(size, std, yscale=1, centre_x=0, centre_y=0):
    
    distance = torch.arange(size) - size//2 - centre_x*(size//2)
    x = distance.expand(1,1,size,size)**2
    distance = torch.arange(size) - size//2 - centre_y*(size//2)
    y = (distance.expand(1,1,size,size)**2).transpose(-1,-2)*yscale
    t = (x + y) / (2*std**2)
    gaussian = torch.exp(-t)
    gaussian /= gaussian.sum()
        
    return gaussian 

def get_euclid(size, yscale=1, centre_x=0, centre_y=0):
    
    distance = torch.arange(size) - size//2 - centre_x*(size//2)
    x = distance.expand(1,1,size,size)**2
    distance = torch.arange(size) - size//2 - centre_y*(size//2)
    y = (distance.expand(1,1,size,size)**2).transpose(-1,-2)*yscale
    t = (x + y)
    euclid = torch.sqrt(t)
        
    return euclid 

def get_cartesian(size):
    
    distance = torch.arange(size) - size//2 
    x = distance.expand(1,1,size,size)
    distance = torch.arange(size) - size//2 
    y = (distance.expand(1,1,size,size)).transpose(-1,-2)

    t = torch.cat([x,y], dim=1)
        
    return t 


def get_spectral_entropy(codes):
    
    fft = (torch.fft.fft2(codes).abs()**2).mean(0, keepdim=True)
    #spectral_dist = fft / (fft.sum([1,2,3], keepdim=True) + 1e-11)
    #spectral_entropy = spectral_dist * torch.log(spectral_dist + 1e-11)
    #spectral_entropy = - spectral_entropy.sum()
        
    return fft.sum()

def count_significant_freqs(codes):
    # Assuming codes is of shape (N, H, W), where N is the batch size

    # Step 0: Perform 2D FFT on each image in the batch
    fourier_spectrum = torch.fft.fft2(codes[:,0])
    fourier_spectrum[:,0,0] = 0

    # Step 1: Calculate the magnitude of the complex numbers
    magnitudes = torch.abs(fourier_spectrum)
    
    # Step 2: Calculate the squared magnitudes (power of each component)
    power = magnitudes ** 2
    
    # Step 3: Sum to find total power for each sample in the batch
    total_power = torch.sum(power, dim=(1, 2))  # Sum over both H and W dimensions
    
    # Step 4: Flatten the powers for sorting, then sort powers in descending order for each example
    sorted_power = torch.sort(power.view(power.size(0), -1), dim=1, descending=True)[0]
    
    # Step 5: Calculate cumulative sum of the sorted powers along each batch
    cumulative_power = torch.cumsum(sorted_power, dim=1)
        
    counts = cumulative_power > (total_power[:,None] * 0.95)
        
    # The number of components needed to reach 95% of the total power
    return counts.float().sum(1).mean()


def get_typical_dist_fourier(orientations,
                              mask=1,
                              match_std=1):
    """
    Estimate typical domain spacing from an orientation map using
    the complex order parameter z = exp(i*2*theta).

    Returns:
        avg_peak     : estimated wavelength (domain spacing, in pixels)
        avg_spectrum : averaged 2D Fourier amplitude spectrum
        avg_hist     : radial histogram used to detect the ring
    """


    # Complex orientation order parameter
    z = torch.exp(2j * orientations)

    # Use real part (imag works equally well)
    spectrum_field = torch.real(z)

    # Remove DC component
    spectrum_field = spectrum_field - spectrum_field.mean()

    # Fourier transform
    af = torch.fft.fft2(spectrum_field)
    af = torch.abs(torch.fft.fftshift(af))

    # Remove central low-frequency bias
    af *= (1 - get_circle(af.shape[-1], mask)[0, 0].to(af.device))

    # Detect ring in Fourier domain
    hist, peak_interpolate = match_ring(af, match_std)

    # Convert frequency to wavelength
    avg_peak = 1.0 / peak_interpolate

    return avg_peak, af, hist


# function to find the peak of a fourier transform
def match_ring(af, match_std=1):
    
    R = af.shape[-1]
    hist = torch.zeros(R//2)
    steps = torch.fft.fftfreq(R)
    
    # use progressively bigger doughnut funtions to find the most active radius
    # which will correspond to the predominant frequency
    for r in range(R//2):

        doughnut = get_doughnut(R, r, match_std)
        prod = af * doughnut
        hist[r] = (prod).sum()

    argmax_p = hist.argmax()
    peak_interpolate = steps[argmax_p]

    # interpolate between the peak value and its two neighbours to get a more accurate estimate
    if argmax_p+1<R//2:
        base = hist[argmax_p-2]
        a,b,c = (hist[argmax_p-1]-base).abs(), (hist[argmax_p]-base).abs(), (hist[argmax_p+1]-base).abs()
        tot = a+b+c
        a /= tot
        b /= tot
        c /= tot
        peak_interpolate = a*steps[argmax_p-1] + b*steps[argmax_p] + c*steps[argmax_p+1]
            
    return hist, peak_interpolate

# Function to get a doughnut function which is 
# defined as a Gaussian Ring around a radius value with a certain STD
def get_doughnut(size, r, std):
    
    distance = torch.arange(size) - size//2
    x = distance.expand(size,size)**2
    y = x.transpose(-1,-2)
    t = torch.sqrt(x + y)
    doughnut = torch.exp(-(t - r)**2/(2*std**2))
    doughnut /= doughnut.sum()
    return doughnut

# Function to get a circle mask with ones inside and zeros outside
def get_circle(size, radius, smooth=True):
    
    distance = torch.arange(size) - size//2
    x = distance.expand(1,1,size,size)**2
    y = x.transpose(-1,-2)
    circle = torch.sqrt(x + y)
    
    # Calculate circle membership with smoothing
    # Define the radius band for smooth transition (0.5 pixel width)
    radius_inner = radius - 0.5
    radius_outer = radius + 0.5

    if smooth:
        # Compute a smooth transition in pixel values across the boundary of the circle
        circle = 1 - torch.clamp((circle - radius_inner) / (radius_outer - radius_inner), 0, 1)
        
    else:
        circle = circle < radius
    
    return circle


def get_effective_dims(code_tracker, debug=False):
    
    code_size = code_tracker.shape[-1]
    fft1 = torch.fft.fft2(code_tracker) - torch.fft.fft2(TF.rotate(code_tracker, 5))
    fft1 = torch.fft.fftshift(fft1)
    #fft1 *= ~get_circle(code_size, 1).cuda()
    fft1 *= get_circle(code_size, (code_size/2)).cuda()

    meanfft = (fft1.abs()**2).mean([0,1]).cpu()
    meanfft /= meanfft.sum()
    sorted_power = torch.argsort(meanfft.view(-1), dim=0, descending=True)
    ordered = (meanfft.view(-1)[sorted_power]).cumsum(dim=0) < .75
    cutoff = ordered.sum()
    
    plotfft = meanfft.view(-1)
    plotfft[sorted_power[cutoff:]] = 0
    plotfft = plotfft.view(code_size,code_size)

    _, peak = match_ring(plotfft, 1)
    
    if debug:
        fft1 = fft1.view(-1, code_size**2)
        fft1[:, sorted_power[cutoff:]] = 0
        fft1 = fft1.view(-1, 1, code_size, code_size)
        fft1 = torch.fft.fftshift(fft1)

        reco = torch.relu(torch.fft.ifft2(fft1).float())
        plt.imshow(codes_var[0,0].cpu())
        plt.show()
        plt.imshow(plotfft, cmap=cm.Greys_r)
        plt.show()
        plt.imshow(reco[0,0].cpu())
        plt.show()
        
        cos = cosim(codes_var, reco)

        print(cos, cutoff)
        
    return cutoff, plotfft, peak

def cosim(X, Y, weighted=False, compute_on_input_device=False):

    if not compute_on_input_device:
        X = X.cpu()
        Y = Y.cpu()
    cos = X * Y
    norm = torch.sqrt((X**2).sum([2,3], keepdim=True) * (Y**2).sum([2,3], keepdim=True))
    cos /= norm + 1e-11
    cos = cos.sum([2,3])
    
    if weighted:
        weights = X.sum([2,3])
        weights /= weights.sum()
        cos = (cos * weights).sum()
        
    else:
        cos = cos.mean()
    
    return cos    

def get_pca_dimensions(code_tracker, n_samps=1, return_components=True):

    mask = torch.isnan(code_tracker).any(dim=(-2, -1))
    # Keep only the images where no NaN exists in the last two dimensions
    code_tracker = code_tracker[~mask]

    print('measuring dimensionality, number of Nan found: ' + str(int(mask.sum())))

    w = code_tracker.shape[-1]
    h = code_tracker.shape[-2]
    
    pca = PCA(n_components=w*h, svd_solver='full')
    pca.fit(code_tracker.cpu().view(-1,w*h))
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    # Minimum number of components whose cumulative explained variance reaches
    # 95%. Counting only entries below the threshold underestimates this by one.
    eff_dim = min(
        int(np.searchsorted(cumulative_variance, 0.95, side='left')) + 1,
        len(cumulative_variance),
    )

    if not return_components:
        return eff_dim, None
    
    comp_sampled_all = torch.tensor(pca.components_)
    comp_sampled = comp_sampled_all.view(w,h,w*h)
    step_w = w // n_samps
    step_h = h // n_samps
    comp_sampled = comp_sampled[::step_w, ::step_h][:n_samps, :n_samps]
    
    comp_sampled = comp_sampled.view(n_samps, n_samps, w, h)
    #comp_sampled = comp_sampled.permute(0,2,1,3).reshape(n_samps*w, n_samps*h)
        
    return eff_dim, comp_sampled_all

def get_umap(code_tracker, window_size):
    
    codes_var = torch.cat(code_tracker, dim=0).cpu()   
    codes_var = F.unfold(codes_var, window_size, stride=window_size).permute(0,2,1).reshape(-1, 1, window_size, window_size)
    #sorting_idx = codes_var.sum([1,2,3]).sort(descending=True)[1]
    #codes_var = codes_var[sorting_idx][:2000]
    codes_var = codes_var.reshape(-1, window_size**2)
        
    size = codes_var.shape[-1]
    
    pca = PCA(n_components=6)
    pca.fit(codes_var.cpu().view(-1,window_size**2))
    
    print('variance explained: ', pca.explained_variance_ratio_.sum())
    
    reduced = pca.transform(codes_var)
    
    embedding = umap.UMAP(
                n_neighbors=1000,
                min_dist=0.8,
                metric='cosine',
                n_components=3,
                init='spectral'
             ).fit_transform(reduced)
    
    return embedding, codes_var
    

# function to plot the umap
def draw_umap(embedding, size, dims=3, title='3D projection'):
    plt.ioff()
    plt.title(title, fontsize=18)
    fig = plt.figure(figsize=(size, size))
    if dims==3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], s=10)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:,0], embedding[:,1], s=10)
    plt.show()
    return fig, ax


def select_grid_modules(code_tracker, window_size):
    
    codes_var = torch.cat(code_tracker, dim=0).cpu()
    #sorting_idx = codes_var.sum([1,2,3]).sort(descending=True)[1]
    #codes_var = codes_var[sorting_idx][2000:3000]    
    codes_var = F.unfold(codes_var, window_size, stride=window_size)
    codes_var = codes_var.permute(0,2,1).reshape(-1, 1, window_size, window_size)
    
    autocorrelograms = compute_autocorrelograms(codes_var, 3, 0.7, 0)
    
    embedding = umap.UMAP(
                n_neighbors=5,
                min_dist=0.05,
                metric='manhattan',
                n_components=2,
                init='spectral'
             ).fit_transform(autocorrelograms.view(-1, window_size**2))
    
    return embedding, autocorrelograms


def get_gratings(size, orientation, period, phase):
    """
    Generates a sinusoidal grating.
    
    Arguments:
    size : int - The size of the grating image (height and width).
    orientation : float - The orientation of the grating in degrees.
    period : int - The spatial period of the grating, in pixels.
    phase : float - The phase shift of the sinusoidal pattern, in degrees.
    
    Returns:
    torch.Tensor - A 2D tensor representing the grating pattern.
    """
    # Convert orientation and phase from degrees to radians
    orientation = torch.deg2rad(torch.tensor(orientation))
    phase = torch.deg2rad(torch.tensor(phase))
    
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    
    # Adjust the coordinates based on the center of the image
    x = x - size // 2
    y = y - size // 2
    
    # Rotate the coordinate system by the orientation
    x_rot = x * torch.cos(orientation) + y * torch.sin(orientation)
    y_rot = -x * torch.sin(orientation) + y * torch.cos(orientation)
    
    # Apply the sinusoidal function
    grating = torch.sin(2 * torch.pi * x_rot / period + phase)
    
    return grating


def compute_autocorrelograms(tiles, r, sigma, eps):
    """
    Computes spatial autocorrelograms for a batch of tiles, applies a mask around the peak,
    smooths the result with a Gaussian filter, and thresholds the autocorrelogram.

    Parameters:
    tiles (torch.Tensor): Input tensor of shape (N, 1, W, W)
    r (int): Radius for masking around the peak
    sigma (float): Standard deviation for Gaussian smoothing
    eps (float): Threshold for final autocorrelogram

    Returns:
    torch.Tensor: Processed autocorrelograms of shape (N, 1, W, W)
    """
    N, C, W, _ = tiles.shape
    gauss = get_gaussian(oddenise(sigma*6), sigma)

    # Step 1: Compute autocorrelograms
    # Need to convert each tile to a full batch where each tile is a kernel
    result = torch.empty((N, C, W, W))
    for i in range(N):
        # Applying convolution for each tile with itself
        result[i] = F.conv2d(tiles[i].unsqueeze(0), tiles[i].unsqueeze(0), padding=W//2)

    # Normalize the autocorrelograms (optional but typical)
    result /= result.view(N, -1).max(1, keepdim=True)[0].view(N, 1, 1, 1)

    # Step 2: Locate the peaks
    # Assume peak is at the center for autocorrelation (common assumption, but you might need to adjust)
    peak_coords = (W//2, W//2)

    # Step 3: Mask out pixels within a radius 'r' from the peak
    for i in range(N):
        y, x = torch.meshgrid(torch.arange(W), torch.arange(W), indexing='ij')
        mask = ((x - peak_coords[1])**2 + (y - peak_coords[0])**2) <= r**2
        result[i, 0, mask] = 0
        mask = ((x - peak_coords[1])**2 + (y - peak_coords[0])**2) >= (W/2)**2
        result[i, 0, mask] = 0

    # Step 4: Apply Gaussian smoothing
    if sigma > 0:
        for i in range(N):
            result[i] = F.conv2d(result[i], gauss, padding=gauss.shape[-1]//2)

    # Step 5: Threshold the autocorrelograms
    result = torch.where(result >= eps, result, torch.zeros_like(result))

    return result

# given a connection field, and a value of sparsity, return a sparse mask
def get_sparsity_masks(cfs, delimiter, sparsity, keep_centre=False, thresh=0):

    cfs = cfs * delimiter

    shape = cfs.shape
    size = shape[-1]
    eye_mask = torch.eye(size**2, device=cfs.device).view(size**2, 1, size, size)

    if keep_centre:
        cfs = cfs * (1 - eye_mask) # mask away the central neuron
        
    cfs = cfs.view(shape[0], -1)

    if thresh:
        cfs = torch.relu(cfs - thresh)

    masks = torch.zeros(cfs.shape, device=cfs.device)

    for i, d in enumerate(delimiter):
        
        nonzero_connections = (d > 0).sum()

        if keep_centre:
            nonzero_connections -= 1
        
        num_samples = nonzero_connections * sparsity
        decimals = num_samples - int(num_samples)
        num_samples = int(num_samples)
        if random.random() < decimals:
                num_samples += 1

        if num_samples:
            indices = torch.multinomial(cfs[i], num_samples, replacement=False)
            masks[i, indices] = 1
        
    masks = masks.view(shape)

    if keep_centre:
        masks += eye_mask
    
    return masks


def find_norm_p(weights, masks=1, t=50, norm_flag=False, target_std=None, masses=None):

    p_trials = torch.linspace(0.01, 1, t)
    stds = []

    if target_std is None:
        _, target_std = get_masses_and_spreads(weights, norm_flag=norm_flag)
    
    sparse_weights = weights * masks
    
    if sparse_weights.sum([2,3]).min()==0:
        return torch.tensor([1.])

    for i in range(t):

        sw = sparse_weights**p_trials[i]
        sw /= sw.sum([2,3], keepdim=True) + 1e-11
        
        _, new_stds = get_masses_and_spreads(sw, norm_flag=norm_flag)
        
        diff = (new_stds - target_std).abs().cpu()
        stds.append(diff[None])

    stds = torch.cat(stds, dim=0)
    best_match = stds.min(dim=0)[1]
    p_vals = p_trials[best_match].cuda()[:, None, None, None]

    return p_vals


def find_val_loc(array, val):

    for idx, item in reversed(list(enumerate(array))):

        if item > val:

            diff = item - val

            if idx == len(array)-1:
                return idx 
                
            diff_next = val - array[idx+1]

            nudge = diff / (diff + diff_next)

            return idx + nudge         
    return 0


def get_wiring_length(lat_cfs, std_exc, trials=15):

    sparsity = min(1/(std_exc*2), 1)
    size = lat_cfs.shape[-1]
    max_r = size // 5
    radiuses = torch.linspace(1, max_r, trials)
    distance_tensor = generate_euclidean_space(size, size).to(device)
    mean_lengths = torch.zeros(trials)

    exc_r = round(std_exc*3)
    bse = generate_circles(size, size, exc_r).to(device)
    bse_cost = (bse * distance_tensor).mean()
    
    for idx, r in tqdm(enumerate(radiuses)):

        masks = generate_circles(size, size, r).to(device)
        masked_cfs = lat_cfs * masks

        sparsity_masks = get_sparsity_masks(masked_cfs, masks, sparsity)

        mean_lengths[idx] = (sparsity_masks * distance_tensor).mean() + bse_cost

    return mean_lengths 


def collect_wiring_pool(N_lrc, trialvar, r_0, phi_short=1):

    trials = N_lrc.shape[0]
    costs = np.zeros((2, trials, trials))
            
    for r in range(trials):
        for t in range(trials):

            pool_size = N_lrc[r]
            local_area = max(r_0, trialvar[t])**2
            local_pool_size = local_area * np.pi * 10

            phi_long_sp = local_area * 0.01 + 1
            phi_long_topo = local_area * 0.02 + 1
            phi_short = 1 #local_area * 0.005 + 1

            #k = np.sqrt(local_pool_size / pool_size)

            size_scaling = np.sqrt(local_area) + 5

            noise_penalty = 0.088*np.log(local_area / 400) + 0.55

            costs[0, r, t] = (pool_size + local_pool_size) / phi_long_sp * size_scaling / noise_penalty
            costs[1, r, t] = (pool_size / phi_long_topo + local_pool_size / phi_short) * size_scaling / noise_penalty

    costs = (np.round(np.log(costs)*10)).astype(int)

    return costs


# Function to count the pinwheels of a map
def count_pinwheels(orientation_map, peak, detect_window=5, count_window=5, n_bins=5, thresh=1):

    pinwheels = pinwheel_detection(orientation_map, peak, detect_window, torch.linspace(0, np.pi-np.pi/n_bins, n_bins))
    
    pinwheels = pinwheels < 2*pinwheels.mean()
    
    # use a sliding count_window to count how many distinct discontinuities are found
    count = 0
    pinwheels_copy = F.pad(pinwheels.clone(),(count_window//2,count_window//2,count_window//2,count_window//2))
    mask = torch.ones(count_window, count_window)
    mask[1:-1, 1:-1] -= torch.ones(count_window-2, count_window-2)
    skip = False
    final_size=pinwheels.shape[-1]

    for x in range(final_size):
        for y in range(final_size):
        
            curr_slice = pinwheels_copy[x:x+count_window, y:y+count_window]
        
            # if there is anything in the current count_window
            if curr_slice.sum():
        
                # and there is no activation on the borders of the count_window
                # eg: a pinwheel was fully within the count_window
                if (not skip) and (not (curr_slice*mask).sum()):
                
                    # then increase the counter and remove it from the discontinuity map
                    count +=1
                    pinwheels_copy[x:x+count_window, y:y+count_window] *= False
                    skip = True
        
            else:
                skip = False
                
    return count, pinwheels, pinwheels_copy



def gaussian_kernel(size, sigma=1.0):
    """Returns a 2D Gaussian kernel array."""
    if isinstance(size, tuple):
        size_x, size_y = size
    else:
        size_x = size_y = size  # Assume square if only one size is given
    
    ax = torch.linspace(-size_x // 2 + 1, size_x // 2, size_x)
    ay = torch.linspace(-size_y // 2 + 1, size_y // 2, size_y)
    xx, yy = torch.meshgrid(ax, ay, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / kernel.sum()

def von_mises_histogram(orientations, bins, gaussian_weights, k=1.0):
    """ Calculate the soft histogram for a given window. """
    orientation_diff = torch.cos(orientations.unsqueeze(-1) - bins)
    weights = torch.exp(k * orientation_diff)
    weighted_hist = weights * gaussian_weights.unsqueeze(-1)
    return weighted_hist.sum((0, 1))

def pinwheel_detection(orientation_map, peak, window_size, bins, k=20):

    sigma = window_size / 6
    
    pad_width = window_size // 2

    size = int(orientation_map.shape[-1] * 10/peak)
    orientation_map = F.interpolate(orientation_map[None, None], size=size)
    padded_map = torch.nn.functional.pad(orientation_map, (pad_width, pad_width, pad_width, pad_width), mode='reflect')

    gaussian_weights = gaussian_kernel((window_size, window_size), sigma)
    uniform_dist = torch.ones_like(bins) / len(bins)
    
    N = orientation_map.shape[-1]
    result_map = torch.zeros((N, N))

    for i in range(N):
        for j in range(N):
            window = padded_map[0, 0, i:i+window_size, j:j+window_size]
            if window.size(0) != window_size or window.size(1) != window_size:
                continue
            hist = von_mises_histogram(window, bins, gaussian_weights, k)
            hist = hist / hist.sum()  # Normalize histogram to sum to 1
            hist_log = torch.log(hist + 1e-10)  # Avoid log(0) by adding a small constant
            # Calculate KL divergence
            kl_div = torch.nn.functional.kl_div(hist_log, uniform_dist, reduction='sum')
            result_map[i, j] = kl_div

    return - result_map


def get_masses_and_spreads(cf, theta=0.95, norm_flag=False, masses=None):

    cf = cf / cf.sum([2,3], keepdim=True)

    sorted_vec = cf.view(cf.shape[0], -1).sort(dim=1, descending=True)
    locs = (sorted_vec[0].cumsum(dim=1) < theta) * (sorted_vec[0].cumsum(dim=1) > 0)

    if masses is None:
        masses = locs.sum(1).float()

    if norm_flag:
        norm_vec = sorted_vec[0] / (sorted_vec[0][:, 0, None] + 1e-11)
    else:
        norm_vec = sorted_vec[0]
    
    means = (norm_vec * locs).sum(1) / (masses + 1e-11)
    deviations = (norm_vec - means[:, None])**2
    spreads = (deviations * locs).sum(1) / (masses - 1)
    spreads = torch.sqrt(spreads)

    spreads[masses<2] = 0
    
    return masses, spreads

def angular_normalise(matrices, theta, num_bins=4, kappa=1):
    """
    Normalizes a batch of matrices using smooth angular weighting to avoid discretization artifacts.
    
    Args:
        matrices (torch.Tensor): Input tensor of shape (N², 1, N, N)
        offset (int): Center offset from image edges
        num_bins (int): Number of bins for density estimation (default: 360)
        kernel_width (float): Angular width of kernel in radians (default: 0.1 ~ 5.7°)
    
    Returns:
        torch.Tensor: Normalized matrices with same shape as input
    """

    device = theta.device
    B, C, N, _ = theta.shape

    slices = get_slices(theta, num_bins, kappa)
    
    # 6. Compute density estimate (soft histogram)
    density = torch.einsum('bxy,bxyk->bk', matrices.squeeze(1), slices.squeeze(1))  # [B, num_bins]

    inv_slices = slices * density[:,None,None,None]

    inv_slices = inv_slices.sum(-1)

    #plt.imshow(inv_slices[1200,0,:,:].cpu())
    #plt.show()
    #plt.imshow(slices[1200,0,:,:].sum(-1).cpu())
    #plt.show()
    #print(num_bins)

    matrices = matrices / inv_slices

    return matrices

    

def get_meshgrid(cutoff, offset=0):
    
    B, C, N, _ = cutoff.shape
    device = cutoff.device 
    
    # 1. Generate center coordinates for all matrices
    centers = torch.stack(torch.meshgrid(
        torch.arange(offset, N-offset, device=device),
        torch.arange(offset, N-offset, device=device),
        indexing='ij',
    )).permute(1, 2, 0).reshape(B, 2)

    # 2. Create coordinate grid for all matrices
    y, x = torch.meshgrid(torch.arange(N, device=device), 
                        torch.arange(N, device=device), indexing='ij')
    x = x.float().unsqueeze(0)  # Add batch dimension
    y = y.float().unsqueeze(0)

    # 3. Compute centered coordinates and angles
    cy = centers[:, 0].view(B, 1, 1, 1)
    cx = centers[:, 1].view(B, 1, 1, 1)
    x_centered = x - cx
    y_centered = y - cy

    return x_centered, y_centered

def get_angles(cutoff, offset=0):

    x_centered, y_centered = get_meshgrid(cutoff, offset)
    
    theta = torch.atan2(y_centered, x_centered) + math.pi  # [0, 2π)

    return theta

def get_slices(theta, num_bins, kappa):
    
    # 4. Create angular density estimation parameters
    bin_centers = torch.linspace(0, 2*math.pi, num_bins+1, device=theta.device)[:-1]  # Exclude last point

    # 5. Compute angular weights using von Mises distribution
    delta_theta = theta.unsqueeze(-1) - bin_centers + torch.rand(1, device=theta.device) * np.pi * 2
    # [B, N, N, num_bins]
    
    #slices = (torch.cos(delta_theta) > 2/3).float() # [B, N, N, num_bins]
    #slices_sin = torch.exp(torch.sin(delta_theta)) / np.e

    #sc = (torch.cos(delta_theta) > 0).float() # [B, N, N, num_bins]
    #ss = (torch.sin(delta_theta) > 0).float()

    slices = torch.exp(torch.cos(delta_theta) * kappa) / np.exp(kappa)

    #slices = slices > 1e-2

    #slices /= 2 * math.pi * torch.i0(torch.tensor(kappa, device=theta.device))  # Normalize

    #slices_cos /= slices_cos.sum(-1, keepdim=True)
    #slices_sin /= slices_sin.sum(-1, keepdim=True)

    #slices = torch.cat([slices_cos, slices_sin], dim=-1)

    slices /= slices.mean(-1, keepdim=True)

    #slices /= slices.mean([2,3], keepdim=True)

    return slices


def assign(probs, theta, target, kappa=1):

    b = probs.shape[0]
    source = probs.view(b, -1).clone().detach()  # Ensure we don't store gradients
    assignment = torch.zeros_like(source)  # Allocate memory once
    ranges = torch.arange(b, device=assignment.device)[:, None]

    n_s = int((target[0] > 0).sum())

    for s in range(n_s):

        masks = get_slices(theta, 1, kappa).view(b, -1) > 2/3
    
        #plt.imshow(masks[1000].view(90,90).cpu())
        #plt.show()

        max_indices = (source * masks).max(1, keepdim=True)[1]

        assignment[ranges, max_indices] = target[ranges, s]

        #plt.imshow(assignment[1000].view(90,90).cpu() + (probs[1000,0]>0).cpu()*1e-3)
        #plt.show()

        source[ranges, max_indices] = 0

        del masks

    return assignment

def fit_exponential(x, y, c=0):
    
    y_adj = y - c  # Adjust if there's an asymptote
    y_adj = np.where(y_adj <= 0, np.nan, y_adj)  # Avoid log of non-positive values

    # Apply log transformation
    log_y = np.log(y_adj)
    
    # Perform linear regression on transformed data
    A = np.vstack([x, np.ones_like(x)]).T
    m, log_a = np.linalg.lstsq(A, log_y, rcond=None)[0]  # Solves Ax = b
    
    a = np.exp(log_a)
    b = -m  # Since equation was -bx in the exponent

    return a, b, c

def batch_roll_2d(tensor, shifts):
    """
    Rolls a 4D tensor (B, 1, N, N) along dimensions 2 and 3 independently for each batch.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (B, 1, N, N).
        shifts (torch.Tensor): Shift values of shape (B, 2), where:
                               - shifts[:, 0] is for dim=2 (height)
                               - shifts[:, 1] is for dim=3 (width)
    
    Returns:
        torch.Tensor: Rolled tensor of the same shape.
    """
    B, _, N, M = tensor.shape  # B = batch size, N = height, M = width
    device = tensor.device

    # Create index matrices
    row_idx = torch.arange(N, device=device).repeat(B, 1)  # Shape: (B, N)
    col_idx = torch.arange(M, device=device).repeat(B, 1)  # Shape: (B, M)

    # Compute rolled indices for both dimensions
    rolled_row_idx = (row_idx - shifts[:, [1]]) % N  # Shift rows (dim=2)
    rolled_col_idx = (col_idx - shifts[:, [0]]) % M  # Shift columns (dim=3)

    # Expand dimensions for proper broadcasting
    rolled_row_idx = rolled_row_idx[:, None, :, None].expand(-1, 1, -1, M)  # Shape: (B, 1, N, M)
    rolled_col_idx = rolled_col_idx[:, None, None, :].expand(-1, 1, N, -1)  # Shape: (B, 1, N, M)

    # Apply gather to shift along both dimensions
    tensor = torch.gather(tensor, dim=2, index=rolled_row_idx)  # Shift along height (dim=2)
    tensor = torch.gather(tensor, dim=3, index=rolled_col_idx)  # Shift along width  (dim=3)

    return tensor


def exact_log_fit(x_points, y_points):
    """
    Finds the exact coefficients a and b for the logarithmic function y = a * ln(x) + b
    that passes through all given (x, y) points.
    
    Parameters:
        x_points (list or numpy array): The x-coordinates of the data points.
        y_points (list or numpy array): The y-coordinates of the data points.
    
    Returns:
        tuple: (a, b) coefficients of the logarithmic function
    """
    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")
    
    if any(x <= 0 for x in x_points):
        raise ValueError("All x values must be positive for the natural logarithm.")

    # Construct the matrix system Ax = B
    A = np.vstack([np.log(x_points), np.ones(len(x_points))]).T  # Design matrix
    B = np.array(y_points)  # Target values

    # Solve for [a, b] in the equation A * [a, b] = B
    a, b = np.linalg.solve(A.T @ A, A.T @ B)  # Exact solution using least squares

    return a, b


def histogram_mean_std(counts, hist_range):
    """
    Compute mean and std from histogram counts and range.

    Args:
        counts (torch.Tensor): 1D tensor of length n with bin counts.
        hist_range (tuple): (min, max) range of the histogram.

    Returns:
        (mean, std): torch.Tensors
    """
    n_bins = counts.numel()
    xmin, xmax = hist_range

    # Bin centers
    bin_edges = torch.linspace(xmin, xmax, n_bins + 1, device=counts.device, dtype=counts.dtype)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Normalize counts
    weights = counts / counts.sum()

    # First and second moment
    mean = (bin_centers * weights).sum()
    mean_sq = (bin_centers**2 * weights).sum()

    # Variance and std
    var = mean_sq - mean**2
    std = torch.sqrt(var)

    return mean, std


def reciprocal_connectivity(N, p):

    N = N**2
    
    # Sample only upper-triangle (excluding diagonal)
    upper = (torch.rand(N, N) < p).triu(diagonal=1)
    
    # Mirror to make symmetric
    A = upper | upper.t()
    
    # Optional: zero diagonal (no autapses)
    A.fill_diagonal_(1)
    
    return A.int()



def generate_sinusoidal_grating(size, wavelength, angle_deg, phase):
    """
    size: int, output image is size x size
    wavelength: float, in pixels
    angle_deg: float, orientation in degrees
    Returns: 2D tensor [1, size, size] (single-channel)
    """
    x = np.arange(size) - size/2
    y = np.arange(size) - size/2
    X, Y = np.meshgrid(x, y)
    
    # Convert angle to radians
    theta = np.deg2rad(angle_deg)
    phase = np.deg2rad(phase)
    
    # Rotate coordinates
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    
    # Sinusoidal grating
    grating = np.sin(2 * np.pi * X_rot / wavelength + phase)
    
    # Normalize to [0,1]
    grating = (grating - grating.min()) / (grating.max() - grating.min())
    
    return torch.tensor(grating, dtype=torch.float32).unsqueeze(0) * 0.8  # shape [1, size, size]




def radial_mean_angle_distance(theta):
    """
    Parameters
    ----------
    theta : ndarray, shape (N, N)
        Orientation map with values in [0, pi)

    Returns
    -------
    mean_dist_deg : ndarray, shape (N//2,)
        Mean orientation angle distance (degrees) vs radial distance
    """

    N = theta.shape[0]
    max_r = N // 2

    dist_sum = np.zeros(max_r)
    dist_count = np.zeros(max_r)

    for dx in range(-(max_r - 1), max_r):
        for dy in range(-(max_r - 1), max_r):
            if dx == 0 and dy == 0:
                continue

            r = int(round(np.sqrt(dx*dx + dy*dy)))
            if r <= 0 or r >= max_r:
                continue

            x0_min = max(0, -dx)
            x0_max = min(N, N - dx)
            y0_min = max(0, -dy)
            y0_max = min(N, N - dy)

            if x0_min >= x0_max or y0_min >= y0_max:
                continue

            a = theta[x0_min:x0_max, y0_min:y0_max]
            b = theta[x0_min+dx:x0_max+dx, y0_min+dy:y0_max+dy]

            # orientation distance modulo pi
            d = np.abs(a - b)
            d = np.minimum(d, np.pi - d)

            dist_sum[r] += d.sum()
            dist_count[r] += d.size

    mean_dist = np.zeros(max_r)
    valid = dist_count > 0
    mean_dist[valid] = dist_sum[valid] / dist_count[valid]

    # r = 0: self-distance is exactly zero
    #mean_dist[0] = mean_dist[1]

    # convert to degrees
    mean_dist_deg = mean_dist * (180.0 / np.pi)

    return mean_dist_deg

def gaussian_local_permutation(tensor: torch.Tensor, std: float, seed=None, distance_metric='euclidean'):
    """
    Applies a soft local random permutation via Gaussian-sampled swaps, tracks the permutation,
    and computes the average displacement distance of elements from original positions.
    
    For each position (i,j) in random order:
    - Sample displacement di ~ Normal(0, std), dj ~ Normal(0, std)
    - Target ti = clamp(round(i + di), 0, N-1), tj = clamp(round(j + dj), 0, N-1)
    - Swap with target (if different)
    
    This creates a probabilistic local shuffle where closer positions are more likely to be swapped,
    approximating a 2D Gaussian diffusion on the grid.
    
    Parameters:
        tensor: torch.Tensor of shape (N, N)
        std: float - standard deviation for the 2D Gaussian (in grid units); std=0: no shuffle, std~1: very local, std>>1: more global
        seed: optional int for reproducibility
        distance_metric: 'euclidean' (default) or 'manhattan'
    
    Returns:
        permuted_tensor: (N, N) - shuffled tensor
        perm_indices: (N*N,) long tensor - permuted.flatten()[k] == original.flatten()[perm_indices[k]]
        mean_displacement: float - average distance each element moved in grid units
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    if len(tensor.shape) != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("Input tensor must be square (N x N)")
    
    N = tensor.shape[0]
    device = tensor.device
    
    # Data: values being permuted
    data = tensor.flatten().clone()
    
    # Track original source index for each final position
    perm_indices = torch.arange(N * N, device=device, dtype=torch.long)
    
    # Random order of visits
    positions = [(i, j) for i in range(N) for j in range(N)]
    random.shuffle(positions)
    
    for i, j in positions:
        # Sample Gaussian displacements
        di = random.gauss(0, std)
        dj = random.gauss(0, std)
        
        # Round to nearest integer offset
        ti = max(0, min(N - 1, round(i + di)))
        tj = max(0, min(N - 1, round(j + dj)))
        
        src_idx = i * N + j
        tgt_idx = ti * N + tj
        
        if src_idx == tgt_idx:
            continue
        
        # Swap values (robust)
        src_val = data[src_idx].clone()
        tgt_val = data[tgt_idx].clone()
        data[src_idx] = tgt_val
        data[tgt_idx] = src_val
        
        # Swap origins
        src_origin = perm_indices[src_idx].clone()
        tgt_origin = perm_indices[tgt_idx].clone()
        perm_indices[src_idx] = tgt_origin
        perm_indices[tgt_idx] = src_origin
    
    permuted_tensor = data.reshape(N, N)
    
    # === Compute mean displacement ===
    # Original positions for each final slot
    orig_rows = perm_indices // N
    orig_cols = perm_indices % N
    
    # Final positions
    final_rows = torch.arange(N * N, device=device) // N
    final_cols = torch.arange(N * N, device=device) % N
    
    # Displacements
    d_rows = (final_rows - orig_rows).float()
    d_cols = (final_cols - orig_cols).float()
    
    if distance_metric == 'euclidean':
        distances = torch.sqrt(d_rows**2 + d_cols**2)
    elif distance_metric == 'manhattan':
        distances = d_rows.abs() + d_cols.abs()
    else:
        raise ValueError("distance_metric must be 'euclidean' or 'manhattan'")
    
    mean_displacement = distances.mean().item()
    
    return permuted_tensor, perm_indices, mean_displacement
