import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import cv2
from PIL import Image

def visualize_mesh(mesh, save_path=None, show=False):
    """
    Visualize a 3D mesh.
    
    Args:
        mesh (trimesh.Trimesh): Mesh to visualize
        save_path (str, optional): Path to save the visualization
        show (bool): Whether to show the visualization
    """
    # Create a new figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get mesh data
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Plot the mesh
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=faces, alpha=0.5, edgecolor='k', linewidth=0.2)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Set title
    ax.set_title("3D Mesh Visualization")
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close(fig)

def visualize_reconstruction(pred_mesh, gt_mesh, save_path=None, show=False):
    """
    Visualize a predicted mesh alongside the ground truth.
    
    Args:
        pred_mesh (trimesh.Trimesh): Predicted mesh
        gt_mesh (trimesh.Trimesh): Ground truth mesh
        save_path (str, optional): Path to save the visualization
        show (bool): Whether to show the visualization
    """
    # Create a new figure
    fig = plt.figure(figsize=(20, 10))
    
    # Plot predicted mesh
    ax1 = fig.add_subplot(121, projection='3d')
    pred_vertices = pred_mesh.vertices
    pred_faces = pred_mesh.faces
    ax1.plot_trisurf(pred_vertices[:, 0], pred_vertices[:, 1], pred_vertices[:, 2],
                    triangles=pred_faces, alpha=0.5, edgecolor='k', linewidth=0.2)
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.set_title("Predicted Mesh")
    
    # Plot ground truth mesh
    ax2 = fig.add_subplot(122, projection='3d')
    gt_vertices = gt_mesh.vertices
    gt_faces = gt_mesh.faces
    ax2.plot_trisurf(gt_vertices[:, 0], gt_vertices[:, 1], gt_vertices[:, 2],
                    triangles=gt_faces, alpha=0.5, edgecolor='k', linewidth=0.2)
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.set_title("Ground Truth Mesh")
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close(fig)

def visualize_mesh_with_texture(mesh, save_path=None, show=False):
    """
    Visualize a 3D mesh with texture.
    
    Args:
        mesh (trimesh.Trimesh): Mesh to visualize
        save_path (str, optional): Path to save the visualization
        show (bool): Whether to show the visualization
    """
    # Create a scene with the mesh
    scene = trimesh.Scene(mesh)
    
    # Render the scene
    img = scene.save_image(resolution=(1024, 1024), visible=True)
    
    # Convert to PIL Image
    img = Image.fromarray(img)
    
    # Save or show the image
    if save_path:
        img.save(save_path)
    
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

def visualize_mesh_comparison(meshes, titles, save_path=None, show=False):
    """
    Visualize multiple meshes side by side for comparison.
    
    Args:
        meshes (list): List of trimesh.Trimesh objects
        titles (list): List of titles for each mesh
        save_path (str, optional): Path to save the visualization
        show (bool): Whether to show the visualization
    """
    # Create a new figure
    fig = plt.figure(figsize=(5 * len(meshes), 10))
    
    # Plot each mesh
    for i, (mesh, title) in enumerate(zip(meshes, titles)):
        ax = fig.add_subplot(1, len(meshes), i + 1, projection='3d')
        vertices = mesh.vertices
        faces = mesh.faces
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                        triangles=faces, alpha=0.5, edgecolor='k', linewidth=0.2)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(title)
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close(fig)

def visualize_metrics_over_time(metrics_history, save_path=None, show=False):
    """
    Visualize metrics over training epochs.
    
    Args:
        metrics_history (dict): Dictionary of metrics over epochs
        save_path (str, optional): Path to save the visualization
        show (bool): Whether to show the visualization
    """
    # Create a new figure
    fig, axs = plt.subplots(len(metrics_history), 1, figsize=(10, 5 * len(metrics_history)))
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics_history.items()):
        ax = axs[i] if len(metrics_history) > 1 else axs
        ax.plot(values, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} over Training')
        ax.grid(True)
    
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close(fig)
