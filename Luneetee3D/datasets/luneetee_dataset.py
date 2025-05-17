import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import trimesh
import cv2
from pathlib import Path

class LuneeteeDataset(Dataset):
    """
    Dataset for Luneetee 3D medical reconstruction.
    
    This dataset loads 2D medical images and their corresponding 3D meshes
    for training the Hunyuan3D-2 model on medical reconstruction tasks.
    
    Args:
        root_dir (str): Root directory of the dataset
        split (str): Dataset split ('train', 'val', or 'test')
        transform (callable, optional): Optional transform to be applied on the images
        image_size (int): Size to resize images to
    """
    
    def __init__(self, root_dir, split='train', transform=None, image_size=512):
        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.image_size = image_size
        
        if not self.split_dir.exists():
            raise ValueError(f"Split directory {self.split_dir} does not exist")
        
        # Get all patient directories
        self.patient_dirs = [d for d in self.split_dir.iterdir() if d.is_dir()]
        
        # Validate that each patient directory has the required files
        self.valid_patients = []
        for patient_dir in self.patient_dirs:
            image_path = patient_dir / "image_2d.jpg"
            mesh_path = next(patient_dir.glob("*.obj"), None)
            if mesh_path is None:
                mesh_path = next(patient_dir.glob("*.ply"), None)
            if mesh_path is None:
                mesh_path = next(patient_dir.glob("*.stl"), None)
                
            if image_path.exists() and mesh_path is not None:
                self.valid_patients.append({
                    "patient_id": patient_dir.name,
                    "image_path": str(image_path),
                    "mesh_path": str(mesh_path)
                })
        
        print(f"Found {len(self.valid_patients)} valid patients in {split} split")
    
    def __len__(self):
        return len(self.valid_patients)
    
    def __getitem__(self, idx):
        patient_data = self.valid_patients[idx]
        
        # Load image
        image = Image.open(patient_data["image_path"]).convert("RGBA")
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        else:
            # Basic preprocessing: resize and convert to tensor
            image = np.array(image)
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)  # Convert to CxHxW format
        
        # Load mesh
        mesh = trimesh.load(patient_data["mesh_path"])
        
        # Convert mesh to a format suitable for training
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.int64)
        
        return {
            "patient_id": patient_data["patient_id"],
            "image": image,
            "vertices": vertices,
            "faces": faces,
            "mesh_path": patient_data["mesh_path"]
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-sized meshes
        """
        patient_ids = [item["patient_id"] for item in batch]
        images = torch.stack([item["image"] for item in batch])
        
        # We can't batch meshes directly due to variable sizes
        vertices = [item["vertices"] for item in batch]
        faces = [item["faces"] for item in batch]
        mesh_paths = [item["mesh_path"] for item in batch]
        
        return {
            "patient_id": patient_ids,
            "image": images,
            "vertices": vertices,
            "faces": faces,
            "mesh_path": mesh_paths
        }
