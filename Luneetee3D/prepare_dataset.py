import os
import argparse
import shutil
from pathlib import Path
import random
import numpy as np
from PIL import Image
import trimesh
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for Luneetee 3D reconstruction")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw data")
    parser.add_argument("--output_dir", type=str, default="data/luneetee_3d", help="Output directory")
    parser.add_argument("--split_ratio", type=str, default="0.7,0.15,0.15", help="Train/val/test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def prepare_dataset(args):
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse split ratio
    split_ratio = [float(x) for x in args.split_ratio.split(",")]
    assert len(split_ratio) == 3, "Split ratio must have 3 values (train, val, test)"
    assert sum(split_ratio) == 1.0, "Split ratio must sum to 1.0"
    
    # Create output directories
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    for directory in [output_dir, train_dir, val_dir, test_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Get list of patient directories
    input_dir = Path(args.input_dir)
    patient_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    # Shuffle patient directories
    random.shuffle(patient_dirs)
    
    # Split patients into train, val, test
    n_patients = len(patient_dirs)
    n_train = int(n_patients * split_ratio[0])
    n_val = int(n_patients * split_ratio[1])
    
    train_patients = patient_dirs[:n_train]
    val_patients = patient_dirs[n_train:n_train+n_val]
    test_patients = patient_dirs[n_train+n_val:]
    
    print(f"Total patients: {n_patients}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")
    
    # Process each split
    process_split(train_patients, train_dir, "train")
    process_split(val_patients, val_dir, "val")
    process_split(test_patients, test_dir, "test")
    
    print(f"Dataset prepared successfully in {output_dir}")

def process_split(patient_dirs, output_dir, split_name):
    """Process a split of the dataset."""
    print(f"Processing {split_name} split...")
    
    for patient_dir in tqdm(patient_dirs, desc=f"Processing {split_name}"):
        # Create patient directory in output
        patient_id = patient_dir.name
        patient_output_dir = output_dir / patient_id
        patient_output_dir.mkdir(exist_ok=True)
        
        # Find image and mesh files
        image_files = list(patient_dir.glob("*.jpg")) + list(patient_dir.glob("*.png"))
        mesh_files = list(patient_dir.glob("*.obj")) + list(patient_dir.glob("*.ply")) + list(patient_dir.glob("*.stl"))
        
        if not image_files:
            print(f"Warning: No image files found for patient {patient_id}")
            continue
        
        if not mesh_files:
            print(f"Warning: No mesh files found for patient {patient_id}")
            continue
        
        # Copy image file (use the first one if multiple)
        image_file = image_files[0]
        image_output_path = patient_output_dir / "image_2d.jpg"
        
        # Convert image to RGBA and save as JPG
        image = Image.open(image_file)
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        image.save(image_output_path)
        
        # Copy mesh file (use the first one if multiple)
        mesh_file = mesh_files[0]
        mesh_output_path = patient_output_dir / mesh_file.name
        
        # Load and save mesh to ensure it's valid
        try:
            mesh = trimesh.load(mesh_file)
            mesh.export(mesh_output_path)
        except Exception as e:
            print(f"Error processing mesh for patient {patient_id}: {e}")
            continue

def main():
    args = parse_args()
    prepare_dataset(args)

if __name__ == "__main__":
    main()
