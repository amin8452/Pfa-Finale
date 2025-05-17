import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import time
import datetime
import trimesh

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from datasets.luneetee_dataset import LuneeteeDataset
from utils.metrics import compute_chamfer_distance, compute_normal_consistency, compute_fscore
from utils.visualization import visualize_reconstruction

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Hunyuan3D-2 for medical 3D reconstruction")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train(config, args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    train_dataset = LuneeteeDataset(
        root_dir=config["data"]["root_dir"],
        split=config["data"]["train_split"],
        image_size=config["data"].get("image_size", 512)
    )
    
    val_dataset = LuneeteeDataset(
        root_dir=config["data"]["root_dir"],
        split=config["data"]["val_split"],
        image_size=config["data"].get("image_size", 512)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"].get("num_workers", 4),
        collate_fn=LuneeteeDataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"].get("num_workers", 4),
        collate_fn=LuneeteeDataset.collate_fn
    )
    
    # Load model
    model_path = config["model"].get("pretrained_path", "tencent/Hunyuan3D-2")
    subfolder = config["model"].get("subfolder", "hunyuan3d-dit-v2-0")
    
    print(f"Loading model from {model_path}, subfolder: {subfolder}")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder=subfolder,
        variant="fp16" if config["train"].get("use_fp16", True) else None,
        device=device
    )
    
    # Set up optimizer
    trainable_params = []
    for name, param in pipeline.named_parameters():
        if config["train"].get("freeze_encoder", False) and "encoder" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_params.append(param)
    
    optimizer = optim.AdamW(
        trainable_params,
        lr=config["train"]["lr"],
        weight_decay=config["train"].get("weight_decay", 0.01)
    )
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["train"]["epochs"],
        eta_min=config["train"].get("min_lr", 1e-6)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        pipeline.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float("inf")
    for epoch in range(start_epoch, config["train"]["epochs"]):
        print(f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        # Training
        pipeline.train()
        train_loss = 0.0
        train_metrics = {"chamfer": 0.0, "normal_consistency": 0.0, "fscore": 0.0}
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            # Forward pass
            images = batch["image"].to(device)
            
            # Generate meshes
            outputs = pipeline(
                image=images,
                num_inference_steps=config["train"].get("num_inference_steps", 50),
                output_type="trimesh"
            )
            
            # Compute loss
            loss = 0.0
            batch_metrics = {"chamfer": 0.0, "normal_consistency": 0.0, "fscore": 0.0}
            
            for i, mesh in enumerate(outputs):
                # Convert ground truth mesh to trimesh format
                gt_vertices = batch["vertices"][i].cpu().numpy()
                gt_faces = batch["faces"][i].cpu().numpy()
                gt_mesh = trimesh.Trimesh(vertices=gt_vertices, faces=gt_faces)
                
                # Compute metrics
                chamfer = compute_chamfer_distance(mesh, gt_mesh)
                normal = compute_normal_consistency(mesh, gt_mesh)
                fscore = compute_fscore(mesh, gt_mesh, threshold=0.01)
                
                # Update metrics
                batch_metrics["chamfer"] += chamfer
                batch_metrics["normal_consistency"] += normal
                batch_metrics["fscore"] += fscore
                
                # Compute loss (weighted sum of metrics)
                sample_loss = (
                    config["train"].get("chamfer_weight", 1.0) * chamfer +
                    config["train"].get("normal_weight", 0.5) * (1 - normal) +
                    config["train"].get("fscore_weight", 1.0) * (1 - fscore)
                )
                loss += sample_loss
            
            # Average loss and metrics over batch
            loss /= len(outputs)
            for key in batch_metrics:
                batch_metrics[key] /= len(outputs)
                train_metrics[key] += batch_metrics[key]
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Average metrics over dataset
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation
        pipeline.eval()
        val_loss = 0.0
        val_metrics = {"chamfer": 0.0, "normal_consistency": 0.0, "fscore": 0.0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Forward pass
                images = batch["image"].to(device)
                
                # Generate meshes
                outputs = pipeline(
                    image=images,
                    num_inference_steps=config["train"].get("num_inference_steps", 50),
                    output_type="trimesh"
                )
                
                # Compute loss
                loss = 0.0
                batch_metrics = {"chamfer": 0.0, "normal_consistency": 0.0, "fscore": 0.0}
                
                for i, mesh in enumerate(outputs):
                    # Convert ground truth mesh to trimesh format
                    gt_vertices = batch["vertices"][i].cpu().numpy()
                    gt_faces = batch["faces"][i].cpu().numpy()
                    gt_mesh = trimesh.Trimesh(vertices=gt_vertices, faces=gt_faces)
                    
                    # Compute metrics
                    chamfer = compute_chamfer_distance(mesh, gt_mesh)
                    normal = compute_normal_consistency(mesh, gt_mesh)
                    fscore = compute_fscore(mesh, gt_mesh, threshold=0.01)
                    
                    # Update metrics
                    batch_metrics["chamfer"] += chamfer
                    batch_metrics["normal_consistency"] += normal
                    batch_metrics["fscore"] += fscore
                    
                    # Compute loss (weighted sum of metrics)
                    sample_loss = (
                        config["train"].get("chamfer_weight", 1.0) * chamfer +
                        config["train"].get("normal_weight", 0.5) * (1 - normal) +
                        config["train"].get("fscore_weight", 1.0) * (1 - fscore)
                    )
                    loss += sample_loss
                
                # Average loss and metrics over batch
                loss /= len(outputs)
                for key in batch_metrics:
                    batch_metrics[key] /= len(outputs)
                    val_metrics[key] += batch_metrics[key]
                
                val_loss += loss.item()
        
        # Average metrics over dataset
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Metrics: {train_metrics}")
        print(f"Val Metrics: {val_metrics}")
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": pipeline.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }
        
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch+1}.pt")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # Visualize some reconstructions
        if (epoch + 1) % config["train"].get("vis_interval", 5) == 0:
            vis_dir = output_dir / "visualizations" / f"epoch_{epoch+1}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (mesh, gt_mesh) in enumerate(zip(outputs[:5], batch["vertices"][:5], batch["faces"][:5])):
                gt_vertices = gt_mesh[0].cpu().numpy()
                gt_faces = gt_mesh[1].cpu().numpy()
                gt_mesh = trimesh.Trimesh(vertices=gt_vertices, faces=gt_faces)
                
                visualize_reconstruction(
                    mesh, 
                    gt_mesh, 
                    save_path=vis_dir / f"sample_{i}.png"
                )

def main():
    args = parse_args()
    config = load_config(args.cfg)
    
    if args.eval_only:
        # Implement evaluation-only mode
        pass
    else:
        train(config, args)

if __name__ == "__main__":
    main()
