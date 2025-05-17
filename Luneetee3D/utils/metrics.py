import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

def compute_chamfer_distance(pred_mesh, gt_mesh, num_samples=10000):
    """
    Compute the Chamfer distance between two meshes.
    
    Args:
        pred_mesh (trimesh.Trimesh): Predicted mesh
        gt_mesh (trimesh.Trimesh): Ground truth mesh
        num_samples (int): Number of points to sample from each mesh
    
    Returns:
        float: Chamfer distance
    """
    # Sample points from meshes
    pred_points = pred_mesh.sample(num_samples)
    gt_points = gt_mesh.sample(num_samples)
    
    # Build KD-trees
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    
    # Compute distances
    dist_pred_to_gt, _ = pred_tree.query(gt_points)
    dist_gt_to_pred, _ = gt_tree.query(pred_points)
    
    # Compute Chamfer distance
    chamfer_dist = np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)
    
    return chamfer_dist

def compute_normal_consistency(pred_mesh, gt_mesh, num_samples=10000):
    """
    Compute the normal consistency between two meshes.
    
    Args:
        pred_mesh (trimesh.Trimesh): Predicted mesh
        gt_mesh (trimesh.Trimesh): Ground truth mesh
        num_samples (int): Number of points to sample from each mesh
    
    Returns:
        float: Normal consistency (0-1, higher is better)
    """
    # Sample points and normals from meshes
    pred_points, pred_face_idx = pred_mesh.sample(num_samples, return_index=True)
    gt_points, gt_face_idx = gt_mesh.sample(num_samples, return_index=True)
    
    # Get normals
    pred_normals = pred_mesh.face_normals[pred_face_idx]
    gt_normals = gt_mesh.face_normals[gt_face_idx]
    
    # Build KD-trees
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    
    # Find closest points
    _, idx_pred_to_gt = pred_tree.query(gt_points)
    _, idx_gt_to_pred = gt_tree.query(pred_points)
    
    # Compute normal consistency
    normal_consistency_pred_to_gt = np.abs(np.sum(pred_normals[idx_gt_to_pred] * gt_normals, axis=1))
    normal_consistency_gt_to_pred = np.abs(np.sum(gt_normals[idx_pred_to_gt] * pred_normals, axis=1))
    
    normal_consistency = (np.mean(normal_consistency_pred_to_gt) + np.mean(normal_consistency_gt_to_pred)) / 2
    
    return normal_consistency

def compute_fscore(pred_mesh, gt_mesh, threshold=0.01, num_samples=10000):
    """
    Compute the F-score between two meshes.
    
    Args:
        pred_mesh (trimesh.Trimesh): Predicted mesh
        gt_mesh (trimesh.Trimesh): Ground truth mesh
        threshold (float): Distance threshold for considering a point as correct
        num_samples (int): Number of points to sample from each mesh
    
    Returns:
        float: F-score (0-1, higher is better)
    """
    # Sample points from meshes
    pred_points = pred_mesh.sample(num_samples)
    gt_points = gt_mesh.sample(num_samples)
    
    # Build KD-trees
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    
    # Compute distances
    dist_pred_to_gt, _ = pred_tree.query(gt_points)
    dist_gt_to_pred, _ = gt_tree.query(pred_points)
    
    # Compute precision and recall
    precision = np.mean(dist_gt_to_pred < threshold)
    recall = np.mean(dist_pred_to_gt < threshold)
    
    # Compute F-score
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0
    
    return fscore

def compute_mesh_volume_difference(pred_mesh, gt_mesh):
    """
    Compute the volume difference between two meshes.
    
    Args:
        pred_mesh (trimesh.Trimesh): Predicted mesh
        gt_mesh (trimesh.Trimesh): Ground truth mesh
    
    Returns:
        float: Absolute volume difference
    """
    # Ensure meshes are watertight
    pred_mesh.fill_holes()
    gt_mesh.fill_holes()
    
    # Compute volumes
    pred_volume = pred_mesh.volume
    gt_volume = gt_mesh.volume
    
    # Compute volume difference
    volume_diff = abs(pred_volume - gt_volume)
    
    return volume_diff

def compute_surface_area_difference(pred_mesh, gt_mesh):
    """
    Compute the surface area difference between two meshes.
    
    Args:
        pred_mesh (trimesh.Trimesh): Predicted mesh
        gt_mesh (trimesh.Trimesh): Ground truth mesh
    
    Returns:
        float: Absolute surface area difference
    """
    # Compute surface areas
    pred_area = pred_mesh.area
    gt_area = gt_mesh.area
    
    # Compute area difference
    area_diff = abs(pred_area - gt_area)
    
    return area_diff

def compute_all_metrics(pred_mesh, gt_mesh, num_samples=10000, threshold=0.01):
    """
    Compute all metrics between two meshes.
    
    Args:
        pred_mesh (trimesh.Trimesh): Predicted mesh
        gt_mesh (trimesh.Trimesh): Ground truth mesh
        num_samples (int): Number of points to sample from each mesh
        threshold (float): Distance threshold for F-score
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        "chamfer_distance": compute_chamfer_distance(pred_mesh, gt_mesh, num_samples),
        "normal_consistency": compute_normal_consistency(pred_mesh, gt_mesh, num_samples),
        "fscore": compute_fscore(pred_mesh, gt_mesh, threshold, num_samples),
        "volume_difference": compute_mesh_volume_difference(pred_mesh, gt_mesh),
        "surface_area_difference": compute_surface_area_difference(pred_mesh, gt_mesh)
    }
    
    return metrics
