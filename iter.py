import numpy as np
import torch
from skimage.restoration import denoise_tv_chambolle as TV
import scipy.io as sio
import matplotlib.pyplot as plt
from network import Network
from utils import ProjOperator, FBPOperator, compare
import time

# Load data
data = sio.loadmat('/home/wujia/daima/DPMA/code/results/slice_100.mat')
reference = data['reference'].astype(np.float32)
reconstruction = data['recon'].astype(np.float32)

# Setup device
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# Define parameters
size = 512
angles = 64
lambda_param = 0.1
delta = 0.005

# Initialize operators
radon = ProjOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                    det_pixels=624, det_pixel_size=0.2, angles=angles, 
                    src_origin=950, det_origin=200)

radon_full = ProjOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                         det_pixels=624, det_pixel_size=0.2, angles=192, 
                         src_origin=950, det_origin=200)

fbp = FBPOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                  det_pixels=624, det_pixel_size=0.2, angles=angles, 
                  src_origin=950, det_origin=200, filter_type='Ram-Lak', 
                  frequency_scaling=0.7)

# Initialize variables
ata = fbp(radon(np.ones([size, size])))
v = np.zeros([size, size])
f = np.zeros([size, size])

# Record start time
start_time = time.time()

# Reconstruction process
for iter_num in range(300):
    if iter_num == 0:
        proj = radon(reference)
        proj_full = radon_full(reference)
        net_output = reconstruction.copy()
        compare(reference, net_output)
    else:
        # Update reconstruction
        residual = fbp(radon(reconstruction) - proj)
        reconstruction = reconstruction - 0.15 * (residual + lambda_param * 
                        (reconstruction - net_output - v - f)) / (ata + lambda_param)
        reconstruction[reconstruction < 0] = 0
        
        # TV denoising
        diff = reconstruction - net_output - f
        max_val = diff.max()
        min_val = diff.min()
        normalized = (diff - min_val) / (max_val - min_val)
        v = TV(normalized, weight=delta, max_num_iter=10) * (max_val - min_val) + min_val
        
        # Update auxiliary variable
        f = f + 1.0 * (v - reconstruction + net_output)
        
        if iter_num % 50 == 0:
            compare(reference, reconstruction)

# Calculate total time
total_time = time.time() - start_time
print(f"\nTotal time: {total_time:.2f} seconds")
print(f"Average time per iteration: {total_time/300:.2f} seconds")

# Calculate final metrics
mse, ssim, psnr = compare(reference, reconstruction)
print(f"MSE: {mse}, PSNR: {psnr}, SSIM: {ssim}")

# Visualize results
# Convert to HU units
reconstruction_HU = reconstruction * 2500 - 1024
reference_HU = reference * 2500 - 1024
difference = reference_HU - reconstruction_HU

# Display with window [-160, 240]
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

ax[0].imshow(np.clip(reference_HU, -160, 240), cmap='gray', vmin=-160, vmax=240)
ax[0].set_title('Reference')
ax[0].axis('off')

ax[1].imshow(np.clip(reconstruction_HU, -160, 240), cmap='gray', vmin=-160, vmax=240)
ax[1].set_title('DPKA')
ax[1].axis('off')

ax[2].imshow(difference, cmap='gray')
ax[2].set_title('Difference')
ax[2].axis('off')

plt.show()