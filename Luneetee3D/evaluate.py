import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import trimesh

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from datasets.luneetee_dataset import LuneeteeDataset
from utils.metrics import compute_all_metrics
from utils.visualization import visualize_reconstruction, visualize_mesh_with_texture

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Hunyuan3D-2 model")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--texture", action="store_true", help="Generate textured meshes")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate(config, args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    test_dataset = LuneeteeDataset(
        root_dir=config["data"]["root_dir"],
        split=config["data"]["test_split"],
        image_size=config["data"].get("image_size", 512)
    )
    
    # Load model
    model_path = config["model"].get("pretrained_path", "tencent/Hunyuan3D-2")
    subfolder = config["model"].get("subfolder", "hunyuan3d-dit-v2-0")
    
    print(f"Loading model from {model_path}, subfolder: {subfolder}")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder=subfolder,
        variant="fp16" if config["eval"].get("use_fp16", True) else None,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    pipeline.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Load texture pipeline if needed
    if args.texture:
        texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            model_path,
            subfolder=config["model"].get("texture_subfolder", "hunyuan3d-paint-v2-0"),
            variant="fp16" if config["eval"].get("use_fp16", True) else None,
            device=device
        )
    
    # Evaluation
    pipeline.eval()
    all_metrics = []
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_dataset, desc="Evaluating")):
            # Get sample data
            image = sample["image"].unsqueeze(0).to(device)
            patient_id = sample["patient_id"]
            
            # Generate mesh
            mesh = pipeline(
                image=image,
                num_inference_steps=config["eval"].get("num_inference_steps", 50),
                output_type="trimesh"
            )[0]
            
            # Convert ground truth mesh to trimesh format
            gt_vertices = sample["vertices"].cpu().numpy()
            gt_faces = sample["faces"].cpu().numpy()
            gt_mesh = trimesh.Trimesh(vertices=gt_vertices, faces=gt_faces)
            
            # Compute metrics
            metrics = compute_all_metrics(
                mesh, 
                gt_mesh, 
                num_samples=config["eval"].get("num_samples", 10000),
                threshold=config["eval"].get("fscore_threshold", 0.01)
            )
            
            # Add patient ID to metrics
            metrics["patient_id"] = patient_id
            all_metrics.append(metrics)
            
            # Save mesh
            mesh_dir = output_dir / "meshes"
            mesh_dir.mkdir(exist_ok=True)
            mesh.export(mesh_dir / f"{patient_id}.obj")
            
            # Generate and save textured mesh if requested
            if args.texture:
                textured_mesh = texture_pipeline(mesh, image=image)
                textured_mesh.export(mesh_dir / f"{patient_id}_textured.glb")
            
            # Generate visualizations if requested
            if args.visualize:
                vis_dir = output_dir / "visualizations"
                vis_dir.mkdir(exist_ok=True)
                
                # Visualize reconstruction
                visualize_reconstruction(
                    mesh, 
                    gt_mesh, 
                    save_path=vis_dir / f"{patient_id}_comparison.png"
                )
                
                # Visualize textured mesh if available
                if args.texture:
                    visualize_mesh_with_texture(
                        textured_mesh,
                        save_path=vis_dir / f"{patient_id}_textured.png"
                    )
    
    # Compute average metrics
    avg_metrics = {key: np.mean([m[key] for m in all_metrics if key != "patient_id"]) 
                  for key in all_metrics[0] if key != "patient_id"}
    
    # Print average metrics
    print("Average Metrics:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save metrics to file
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            "individual_metrics": all_metrics,
            "average_metrics": avg_metrics
        }, f, indent=2)
    
    return avg_metrics

def main():
    args = parse_args()
    config = load_config(args.cfg)
    evaluate(config, args)

if __name__ == "__main__":
    main()
