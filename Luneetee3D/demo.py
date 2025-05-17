import os
import argparse
import torch
from PIL import Image
import trimesh
from pathlib import Path

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.rembg import BackgroundRemover

def parse_args():
    parser = argparse.ArgumentParser(description="Demo for Luneetee 3D reconstruction")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="demo_output", help="Output directory")
    parser.add_argument("--texture", action="store_true", help="Generate textured mesh")
    parser.add_argument("--texture_model", type=str, default="tencent/Hunyuan3D-2", help="Path to texture model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load image
    image = Image.open(args.image).convert("RGBA")
    
    # Remove background if needed
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-dit-v2-0",
        variant="fp16",
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    pipeline.load_state_dict(checkpoint["model_state_dict"])
    
    # Generate mesh
    print("Generating 3D mesh...")
    mesh = pipeline(
        image=image,
        num_inference_steps=50,
        output_type="trimesh"
    )[0]
    
    # Save mesh
    mesh_path = output_dir / "mesh.obj"
    mesh.export(str(mesh_path))
    print(f"Mesh saved to {mesh_path}")
    
    # Generate textured mesh if requested
    if args.texture:
        print("Generating textured mesh...")
        texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            args.texture_model,
            subfolder="hunyuan3d-paint-v2-0",
            variant="fp16",
            device=device
        )
        
        textured_mesh = texture_pipeline(mesh, image=image)
        textured_mesh_path = output_dir / "textured_mesh.glb"
        textured_mesh.export(str(textured_mesh_path))
        print(f"Textured mesh saved to {textured_mesh_path}")
    
    print("Done!")

if __name__ == "__main__":
    main()
