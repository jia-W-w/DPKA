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

from network import Network
from utils import (
    ProjOperator, 
    FBPOperator, 
    compare, 
    DicomDataset, 
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
    save_dir: str = '/home/wujia/daima/DPMA/results_60/my_model_0322.pth'
    
    # Device
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Add pretrained model path
    pretrained_path: str = '/home/wujia/daima/DPMA/results_60/my_model_0114.pth'

def setup_operators(cfg: Config) -> Tuple[ProjOperator, ProjOperator, FBPOperator]:
    """Initialize projection and FBP operators"""
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
    
    radon_pro = ProjOperator(**common_params, angles=cfg.angles)
    radon_pro_label = ProjOperator(**common_params, angles=192)
    fbp_op = FBPOperator(
        **common_params, 
        angles=192, 
        filter_type='Ram-Lak',
        frequency_scaling=0.7
    )
    
    return radon_pro, radon_pro_label, fbp_op

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    operators: Tuple[ProjOperator, ProjOperator, FBPOperator],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    running_loss = running_mse = running_psnr = 0
    radon_pro, radon_pro_label, fbp_op = operators
    
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, data in loop:
        data = data.to(device)
        
        # Forward pass
        proj_64 = radon_pro(data)
        proj_192_label = radon_pro_label(data)
        proj_192 = F.interpolate(proj_64, size=(192, proj_64.shape[3]))
        
        unet_output = model(proj_192)
        loss1 = criterion(proj_192_label, unet_output)
        
        ct_recon = fbp_op(unet_output)
        loss2 = criterion(data, ct_recon)
        
        unet_output2 = model(ct_recon)
        loss3 = criterion(unet_output2, data)
        
        loss = 0.1 * loss1 + 0.1 * loss2 + loss3
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            mse, _, psnr = compare(
                data.cpu().numpy(),
                unet_output2.cpu().numpy(),
                verbose=False
            )
            running_loss += loss.item()
            running_mse += mse.item()
            running_psnr += psnr.item()
            
        # Update progress bar
        avg_loss = running_loss / (i + 1)
        avg_mse = running_mse / (i + 1)
        avg_psnr = running_psnr / (i + 1)
        loop.set_description("Training")
        loop.set_postfix(loss=avg_loss, mse=avg_mse, psnr=avg_psnr)
    
    return {
        'loss': avg_loss,
        'mse': avg_mse,
        'psnr': avg_psnr
    }

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    operators: Tuple[ProjOperator, ProjOperator, FBPOperator],
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    running_loss = running_mse = running_psnr = 0
    radon_pro, radon_pro_label, fbp_op = operators
    
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, data in loop:
        data = data.to(device)
        
        # Forward pass
        proj_64 = radon_pro(data)
        proj_192_label = radon_pro_label(data)
        proj_192 = F.interpolate(proj_64, size=(192, proj_64.shape[3]))
        
        unet_output = model(proj_192)
        loss1 = criterion(proj_192_label, unet_output)
        
        ct_recon = fbp_op(unet_output)
        loss2 = criterion(data, ct_recon)
        
        unet_output2 = model(ct_recon)
        loss3 = criterion(unet_output2, data)
        
        loss = 0.1 * loss1 + 0.1 * loss2 + loss3
        
        # Metrics
        mse, _, psnr = compare(
            data.cpu().numpy(),
            unet_output2.cpu().numpy(),
            verbose=False
        )
        running_loss += loss.item()
        running_mse += mse.item()
        running_psnr += psnr.item()
        
        # Update progress bar
        avg_loss = running_loss / (i + 1)
        avg_mse = running_mse / (i + 1)
        avg_psnr = running_psnr / (i + 1)
        loop.set_description("Validation")
        loop.set_postfix(loss=avg_loss, mse=avg_mse, psnr=avg_psnr)
    
    return {
        'loss': avg_loss,
        'mse': avg_mse,
        'psnr': avg_psnr
    }

def main():
    # Setup
    cfg = Config()
    setup_logging(cfg.save_dir)
    device = torch.device(cfg.device)
    torch.cuda.set_device(device)
    
    # Model
    model = Network(img_channel=cfg.img_channels, width=cfg.model_width)
    model = nn.DataParallel(model).to(device)
    
    # Model loading with error handling
    if cfg.pretrained_path:
        try:
            pretrained_state_dict = torch.load(cfg.pretrained_path)
            model.load_state_dict(pretrained_state_dict)
            logging.info(f"Loaded pretrained weights from {cfg.pretrained_path}")
        except Exception as e:
            logging.error(f"Failed to load pretrained weights: {e}")
            raise
    
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
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(cfg.epochs):
        logging.info(f"\nEpoch {epoch+1}/{cfg.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, operators,
            optimizer, criterion, device
        )
        scheduler.step()
        
        # Validate
        val_metrics = validate(
            model, val_loader, operators,
            criterion, device
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
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = Path(cfg.save_dir) / 'best_model.pth'
            torch.save(model.state_dict(), save_path)
            logging.info(f"Saved best model to {save_path}")

if __name__ == '__main__':
    main()