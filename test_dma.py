import torch
from network import Network
from torch.utils.data import DataLoader
from utils import ProjOperator, FBPOperator, compare, DicomDataset, clear
import torch.nn.functional as F
from tqdm import tqdm
import os
import scipy.io as sio
import numpy as np

# Device setup
device = torch.device("cuda:0")
torch.cuda.set_device(0)

# Load models
Unet = Network(img_channel=1, width=16)
Unet.to(device)
Unet = torch.nn.DataParallel(Unet)
Unet.load_state_dict(torch.load('/home/wujia/daima/DPMA/results_60/my_model_0322.pth'))
Unet.eval()

Unet2 = Network(img_channel=1, width=16)
Unet2.to(device)
Unet2 = torch.nn.DataParallel(Unet2)
Unet2.load_state_dict(torch.load('/home/wujia/daima/DPMA/results_60/my_model_pic_0322.pth'))
Unet2.eval()

# Dataset and loader setup
val_dataset = DicomDataset('/home/wujia/daima/raw_data/test/L***/full_3mm')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

# Operators setup
size = 512
angles = 64
radon_pro = ProjOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                        det_pixels=624, det_pixel_size=0.2, angles=angles, 
                        src_origin=950, det_origin=200)

fbp_op = FBPOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                     det_pixels=624, det_pixel_size=0.2, angles=192, 
                     src_origin=950, det_origin=200, filter_type='Ram-Lak', 
                     frequency_scaling=0.7)

fbp_op_angel = FBPOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                          det_pixels=624, det_pixel_size=0.2, angles=angles,
                          src_origin=950, det_origin=200, filter_type='Ram-Lak',
                          frequency_scaling=0.7)

def val_fn(validlow):
    with torch.no_grad():
        proj_64 = radon_pro(validlow)
        proj_192 = F.interpolate(proj_64, size=(192, proj_64.shape[3]))
        unet_output = Unet(proj_192)
        ct_recon = fbp_op(unet_output)
        unet_output2 = Unet(ct_recon)
    return unet_output2, proj_64

def main():
    # Create results directory if it doesn't exist
    save_dir = '/home/wujia/daima/DPMA/code/results'
    os.makedirs(save_dir, exist_ok=True)
    
    # Metrics initialization
    running_val_loss = 0
    running_val_mse = 0
    running_val_psnr = 0
    l2_loss = torch.nn.MSELoss()
    
    print(f"Starting evaluation on {len(val_loader)} samples...")
    
    with torch.no_grad():
        val_loop = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, val_data in val_loop:
            val_data = val_data.to(device)
            
            # Forward pass
            val_unet_output, val_proj_64 = val_fn(val_data)
            val_unet_output = val_unet_output - fbp_op_angel(radon_pro(val_unet_output) - val_proj_64)
            val_final = Unet2(val_unet_output)
            
            # Compute metrics - using val_unet_output instead of val_final
            val_total_loss = l2_loss(val_unet_output, val_data)
            mse_recon, _, psnr_recon = compare(val_data.detach().cpu().numpy(),
                                             val_final.detach().cpu().numpy(),
                                             verbose=False)
            
            # Save the 100th slice
            if i == 99:  # 0-based indexing, so 99 is the 100th slice
                recon = clear(val_final)    
                reference = clear(val_data)
                
                # Remove batch dimension and any extra dimensions
                recon = np.squeeze(recon)
                reference = np.squeeze(reference)
                
                # Save as .mat file
                sio.savemat(os.path.join(save_dir, 'slice_100.mat'), 
                           {'recon': recon, 'reference': reference})
                print(f"\nSaved slice 100 to {save_dir}/slice_100.mat")
            
            # Update running metrics
            running_val_loss += val_total_loss.item()
            running_val_mse += mse_recon.item()
            running_val_psnr += psnr_recon.item()
            
            # Update progress bar
            val_loop.set_description('Testing Progress')
            val_loop.set_postfix(
                avg_loss=running_val_loss / (i + 1),
                avg_mse=running_val_mse / (i + 1),
                avg_psnr=running_val_psnr / (i + 1)
            )
    
    # Calculate final metrics
    num_samples = len(val_loader)
    final_loss = running_val_loss / num_samples
    final_mse = running_val_mse / num_samples
    final_psnr = running_val_psnr / num_samples
    
    print("\nFinal Results:")
    print(f"Average Loss: {final_loss:.4f}")
    print(f"Average MSE: {final_mse:.4f}")
    print(f"Average PSNR: {final_psnr:.4f}")

if __name__ == "__main__":
    main()