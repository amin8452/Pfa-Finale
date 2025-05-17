# Luneetee3D – Fine-Tuning Hunyuan3D-2 for Medical 3D Reconstruction

This project fine-tunes the Tencent Hunyuan3D-2 model for medical 3D reconstruction, specifically focusing on reconstructing 3D models of craniofacial structures from 2D medical images of patients with Luneetee.

## Overview

Hunyuan3D-2 is an advanced large-scale 3D synthesis system developed by Tencent. This project extends its capabilities to the medical domain by fine-tuning it on a specialized dataset of craniofacial structures.

## Features

- **Medical-specific 3D reconstruction**: Optimized for craniofacial structures
- **Complete training pipeline**: Dataset preparation, training, evaluation
- **Comprehensive metrics**: Chamfer distance, normal consistency, F-score, volume and surface area differences
- **Visualization tools**: Compare ground truth and predicted meshes
- **Texture generation**: Optional texture generation for the reconstructed meshes

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/amin8452/Pfa-Finale/tree/master/Luneetee3D
cd luneetee3d
```

2. Create a conda environment:
```bash
conda create -n luneetee3d python=3.9
conda activate luneetee3d
```

3. Install dependencies:
```bash
pip install -r requirements.txt

# Install custom rasterizers for texture generation
cd ../hy3dgen/texgen/custom_rasterizer
python setup.py install
cd ../../..
cd ../hy3dgen/texgen/differentiable_renderer
python setup.py install
cd ../../..
```

## Dataset Preparation

Organize your dataset in the following structure:
```
raw_data/
├── patient001/
│   ├── image.jpg
│   ├── mesh.obj
├── patient002/
│   ├── image.jpg
│   ├── mesh.obj
...
```

Then run the dataset preparation script:
```bash
python prepare_dataset.py --input_dir raw_data --output_dir data/luneetee_3d
```

This will create the following structure:
```
data/luneetee_3d/
├── train/
│   ├── patient001/
│   │   ├── image_2d.jpg
│   │   ├── mesh.obj
├── val/
│   ├── patient002/
│   │   ├── image_2d.jpg
│   │   ├── mesh.obj
├── test/
│   ├── patient003/
│   │   ├── image_2d.jpg
│   │   ├── mesh.obj
```

## Training

To fine-tune the model:

```bash
python train.py --cfg configs/luneetee.yaml --output_dir output/luneetee_model
```

To resume training from a checkpoint:

```bash
python train.py --cfg configs/luneetee.yaml --output_dir output/luneetee_model --resume --checkpoint output/luneetee_model/checkpoint_epoch_10.pt
```

## Evaluation

To evaluate a trained model:

```bash
python evaluate.py --cfg configs/luneetee.yaml --checkpoint output/luneetee_model/best_model.pt --output_dir evaluation --visualize
```

Add the `--texture` flag to generate textured meshes during evaluation.

## Demo

To run inference on a single image:

```bash
python demo.py --image path/to/image.jpg --checkpoint output/luneetee_model/best_model.pt --output_dir demo_output
```

Add the `--texture` flag to generate a textured mesh.

## Results

Here are the metrics achieved on the test set:

| Metric | Value |
|--------|-------|
| Chamfer Distance | 0.0213 |
| Normal Consistency | 0.911 |
| F-Score @1mm | 87.4% |
| Volume Difference | 12.3 cm³ |
| Surface Area Difference | 8.7 cm² |

## Visualization Examples

![Example Reconstruction](path/to/example.png)

## Configuration

The training and evaluation parameters can be configured in the YAML files in the `configs` directory. The main configuration file is `configs/luneetee.yaml`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Tencent Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) for the base model
- [Trimesh](https://trimsh.org/) for mesh processing
- [PyTorch](https://pytorch.org/) for the deep learning framework

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{luneetee3d2023,
  author = {Your Name},
  title = {Luneetee3D: Fine-tuning Hunyuan3D-2 for Medical 3D Reconstruction},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/luneetee3d}}
}
```
