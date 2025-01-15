import os
import logging
from typing import Tuple, Dict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd

from network import Network
from utils import (
    ProjOperator, 
    FBPOperator,
    BackProjectionOperator,
    FilteredProjOperator,
    compare,
    DicomDataset,
    clear
)

def setup_logging(save_dir: str) -> None:
    """Setup logging configuration"""
    log_file = Path(save_dir) / 'training.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class Config:
    """Training configuration"""
    # Model parameters
    img_channels: int = 1
    model_width: int = 16
    
    # Data parameters
    size: int = 512
    angles: int = 64
    batch_size: int = 1
    num_workers: int = 4
    
    # Training parameters
    epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    
    # Paths
    train_data_path: str = '/home/wujia/daima/raw_data/full_3mm/L***/full_3mm'
    val_data_path: str = '/home/wujia/daima/raw_data/test/L***/full_3mm'
    save_dir: str = '/home/wujia/daima/DPMA/code/results_60'
    model1_path: str = '/home/wujia/daima/DPMA/results_60/my_model_0114.pth'
    model2_path: str = '/home/wujia/daima/DPMA/results_60/my_model_pic_0114.pth'
    metrics_file: str = 'my_model_pic_0114.csv'
    
    # Device
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

def setup_operators(cfg: Config) -> Tuple[ProjOperator, ProjOperator, FBPOperator, FBPOperator, BackProjectionOperator, FilteredProjOperator]:
    """Initialize operators"""
    common_params = {
        'N': cfg.size,
        'M': cfg.size,
        'pixel_size_x': 0.15,
        'pixel_size_y': 0.15,
        'det_pixels': 624,
        'det_pixel_size': 0.2,
        'src_origin': 950,
        'det_origin': 200
    }
    
    # Initialize all operators
    radon_pro = ProjOperator(**common_params, angles=cfg.angles)
    radon_pro_label = ProjOperator(**common_params, angles=192)
    
    fbp_op = FBPOperator(
        **common_params,
        angles=192,
        filter_type='Ram-Lak',
        frequency_scaling=0.7
    )
    
    fbp_op_angel = FBPOperator(
        **common_params,
        angles=cfg.angles,
        filter_type='Ram-Lak',
        frequency_scaling=0.7
    )
    
    bp_op_angel = BackProjectionOperator(**common_params, angles=cfg.angles)
    
    filtered_proj_op = FilteredProjOperator(
        **common_params,
        angles=cfg.angles,
        filter_type='Ram-Lak',
        frequency_scaling=0.7
    )
    
    # Make sure we return all 6 operators
    return (
        radon_pro,
        radon_pro_label,
        fbp_op,
        fbp_op_angel,
        bp_op_angel,
        filtered_proj_op
    )

@torch.no_grad()
def val_fn(data: torch.Tensor, model1: nn.Module, operators: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validation function for a single batch"""
    radon_pro, _, fbp_op, fbp_op_angel, _, _ = operators
    
    # Get 64-angle projection
    proj_64 = radon_pro(data)
    
    # Interpolate to 192 angles
    proj_192 = F.interpolate(proj_64, size=(192, proj_64.shape[3]))
    
    # First UNet output
    unet_output = model1(proj_192)
    
    # Use fbp_op_angel (64 angles) instead of fbp_op (192 angles)
    ct_recon = fbp_op(unet_output)
    ct_recon = model1(ct_recon)
    
    return ct_recon, proj_64

def train_epoch(
    model1: nn.Module,
    model2: nn.Module,
    loader: DataLoader,
    operators: Tuple,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch"""
    model1.eval()  # model1 is fixed
    model2.train()
    
    running_loss = running_mse = running_psnr = 0
    num_batches = len(loader)
    
    # Create progress bar with metrics
    pbar = tqdm(loader, desc="Training")
    
    for i, data in enumerate(pbar, 1):
        data = data.to(device)
        
        # Get intermediate results using val_fn
        input, proj_64 = val_fn(data, model1, operators)
        
        # Second UNet forward pass
        output = model2(input)
        
        # Calculate loss
        loss = F.mse_loss(output, data)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        mse, _, psnr = compare(
            data.cpu().numpy(),
            output.detach().cpu().numpy(),
            verbose=False
        )
        
        # Update running metrics
        running_loss += loss.item()
        running_mse += mse
        running_psnr += psnr
        
        # Update progress bar with current metrics
        current_loss = running_loss / i
        current_mse = running_mse / i
        current_psnr = running_psnr / i
        
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'MSE': f'{current_mse:.4f}',
            'PSNR': f'{current_psnr:.2f}'
        })
    
    # Calculate final averages
    return {
        'loss': running_loss / num_batches,
        'mse': running_mse / num_batches,
        'psnr': running_psnr / num_batches
    }

def validate(
    model1: nn.Module,
    model2: nn.Module,
    loader: DataLoader,
    operators: Tuple,
    device: torch.device
) -> Dict[str, float]:
    """Validate models"""
    model1.eval()
    model2.eval()
    
    running_loss = running_mse = running_psnr = 0
    num_batches = len(loader)
    
    # Create progress bar with metrics
    pbar = tqdm(loader, desc="Validating")
    
    with torch.no_grad():
        for i, data in enumerate(pbar, 1):
            data = data.to(device)
            
            # Get intermediate results
            input, proj_64 = val_fn(data, model1, operators)
            
            # Second UNet forward pass
            output = model2(input)
            
            # Calculate loss
            loss = F.mse_loss(output, data)
            
            # Calculate metrics
            mse, _, psnr = compare(
                data.cpu().numpy(),
                output.cpu().numpy(),
                verbose=False
            )
            
            # Update running metrics
            running_loss += loss.item()
            running_mse += mse
            running_psnr += psnr
            
            # Update progress bar with current metrics
            current_loss = running_loss / i
            current_mse = running_mse / i
            current_psnr = running_psnr / i
            
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'MSE': f'{current_mse:.4f}',
                'PSNR': f'{current_psnr:.2f}'
            })
    
    # Calculate final averages
    return {
        'loss': running_loss / num_batches,
        'mse': running_mse / num_batches,
        'psnr': running_psnr / num_batches
    }

def main():
    # Setup
    cfg = Config()
    setup_logging(cfg.save_dir)
    device = torch.device(cfg.device)
    torch.cuda.set_device(device)
    
    # Models
    model1 = Network(img_channel=cfg.img_channels, width=cfg.model_width)
    model1 = nn.DataParallel(model1).to(device)
    model1.load_state_dict(torch.load(cfg.model1_path))
    
    model2 = Network(img_channel=cfg.img_channels, width=cfg.model_width)
    model2 = nn.DataParallel(model2).to(device)
    
    # Data
    train_dataset = DicomDataset(cfg.train_data_path)
    val_dataset = DicomDataset(cfg.val_data_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    
    # Optimization
    optimizer = torch.optim.AdamW(
        model2.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-5
    )
    
    # Operators
    operators = setup_operators(cfg)
    
    # Metrics history
    metrics_history = {
        'train_loss': [], 'train_mse': [], 'train_psnr': [],
        'val_loss': [], 'val_mse': [], 'val_psnr': []
    }
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(cfg.epochs):
        logging.info(f"\nEpoch {epoch+1}/{cfg.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model1, model2, train_loader,
            operators, optimizer, device
        )
        scheduler.step()
        
        # Validate
        val_metrics = validate(
            model1, model2, val_loader,
            operators, device
        )
        
        # Log metrics
        logging.info(
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"MSE: {train_metrics['mse']:.4f}, "
            f"PSNR: {train_metrics['psnr']:.4f}"
        )
        logging.info(
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"MSE: {val_metrics['mse']:.4f}, "
            f"PSNR: {val_metrics['psnr']:.4f}"
        )
        
        # Update metrics history
        for k in metrics_history:
            metrics_history[k].append(
                train_metrics[k.split('_')[1]] if 'train' in k 
                else val_metrics[k.split('_')[1]]
            )
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = Path(cfg.save_dir) / cfg.model2_path
            torch.save(model2.state_dict(), save_path)
            logging.info(f"Saved best model to {save_path}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics_history)
        metrics_df.to_csv(Path(cfg.save_dir) / cfg.metrics_file)

if __name__ == '__main__':
    main()