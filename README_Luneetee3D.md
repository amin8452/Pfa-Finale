# Projet Luneetee3D

Fine-tuning du modèle Hunyuan3D-2 de Tencent pour la reconstruction 3D médicale, spécifiquement pour les structures craniofaciales des patients atteints de Luneetee.

## Structure du projet

Le projet est organisé comme suit:

```
Luneetee3D/
├── configs/
│   ├── __init__.py
│   └── luneetee.yaml         # Configuration pour l'entraînement et l'évaluation
├── datasets/
│   ├── __init__.py
│   └── luneetee_dataset.py   # Classe de dataset pour charger les images médicales et les maillages
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # Métriques pour évaluer les reconstructions 3D
│   └── visualization.py      # Outils de visualisation des maillages et résultats
├── __init__.py
├── demo.py                   # Script de démonstration pour l'inférence
├── evaluate.py               # Script d'évaluation
├── prepare_dataset.py        # Script de préparation du dataset
├── README.md                 # Documentation du projet
├── requirements.txt          # Dépendances
├── setup.py                  # Installation du package
└── train.py                  # Script d'entraînement
```

## Installation

### Prérequis

- Python 3.9+
- GPU compatible CUDA (recommandé)

### Configuration

1. Cloner le dépôt:
```bash
git clone https://github.com/amin8452/Pfa-Finale.git
cd Pfa-Finale
```

2. Créer un environnement conda:
```bash
conda create -n luneetee3d python=3.9
conda activate luneetee3d
```

3. Installer les dépendances:
```bash
pip install -r Luneetee3D/requirements.txt

# Installer les rasterizers personnalisés pour la génération de textures
cd hy3dgen/texgen/custom_rasterizer
python setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python setup.py install
cd ../../..
```

## Préparation des données

Organisez votre dataset selon la structure suivante:
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

Puis exécutez le script de préparation du dataset:
```bash
python Luneetee3D/prepare_dataset.py --input_dir raw_data --output_dir data/luneetee_3d
```

## Entraînement

Pour fine-tuner le modèle:

```bash
python Luneetee3D/train.py --cfg Luneetee3D/configs/luneetee.yaml --output_dir output/luneetee_model
```

Pour reprendre l'entraînement à partir d'un checkpoint:

```bash
python Luneetee3D/train.py --cfg Luneetee3D/configs/luneetee.yaml --output_dir output/luneetee_model --resume --checkpoint output/luneetee_model/checkpoint_epoch_10.pt
```

## Évaluation

Pour évaluer un modèle entraîné:

```bash
python Luneetee3D/evaluate.py --cfg Luneetee3D/configs/luneetee.yaml --checkpoint output/luneetee_model/best_model.pt --output_dir evaluation --visualize
```

Ajoutez l'option `--texture` pour générer des maillages texturés pendant l'évaluation.

## Démonstration

Pour exécuter l'inférence sur une seule image:

```bash
python Luneetee3D/demo.py --image path/to/image.jpg --checkpoint output/luneetee_model/best_model.pt --output_dir demo_output
```

Ajoutez l'option `--texture` pour générer un maillage texturé.

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## Remerciements

- [Tencent Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) pour le modèle de base
- [Trimesh](https://trimsh.org/) pour le traitement des maillages
- [PyTorch](https://pytorch.org/) pour le framework de deep learning
