# Projet Luneetee3D

Fine-tuning du modèle Hunyuan3D-2 de Tencent pour la reconstruction 3D médicale, spécifiquement pour les structures craniofaciales des patients atteints de Luneetee.

> **⚠️ Note importante**: Ce projet est un sous-dossier du dépôt principal. Pour l'utiliser correctement, suivez les instructions d'installation ci-dessous. Ne tentez pas de cloner directement le sous-dossier Luneetee3D car cela ne fonctionnera pas.

## ⚠️ Résolution des problèmes courants

### Erreur "No such file or directory" lors de l'exécution des scripts

Si vous rencontrez une erreur comme `python3: can't open file '//Luneetee3D/prepare_dataset.py': [Errno 2] No such file or directory`, voici comment la résoudre:

1. **Vérifiez votre répertoire de travail actuel**:
   ```bash
   pwd  # Sur Linux/Mac
   # ou
   cd   # Sur Windows pour afficher le répertoire courant
   ```

2. **Utilisez le chemin relatif correct selon votre position**:

   - **Si vous êtes à la racine du dépôt** (dossier `Pfa-Finale`):
     ```bash
     # Utilisez cette commande (sans // au début)
     python Luneetee3D/prepare_dataset.py --input_dir raw_data --output_dir data/luneetee_3d
     ```

   - **Si vous êtes déjà dans le dossier Luneetee3D**:
     ```bash
     # Utilisez simplement le nom du script
     python prepare_dataset.py --input_dir ../raw_data --output_dir ../data/luneetee_3d
     ```

   - **Si vous n'êtes pas sûr**, naviguez d'abord vers le bon répertoire:
     ```bash
     # Pour aller à la racine du dépôt
     cd chemin/vers/Pfa-Finale

     # Ou pour aller directement dans le dossier Luneetee3D
     cd chemin/vers/Pfa-Finale/Luneetee3D
     ```

3. **Pour Kaggle/Colab**, utilisez le préfixe `!` pour les commandes shell:
   ```python
   # Vérifiez d'abord où vous êtes
   !pwd
   !ls

   # Puis exécutez le script avec le chemin approprié
   !python prepare_dataset.py --input_dir raw_data --output_dir data/luneetee_3d
   ```

<p align="center">
  <img src="https://github.com/user-attachments/assets/efb402a1-0b09-41e0-a6cb-259d442e76aa" width="600">
</p>

## 📋 Table des matières

- [Présentation du projet](#présentation-du-projet)
- [Fonctionnement](#fonctionnement)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Préparation des données](#préparation-des-données)
- [Entraînement](#entraînement)
- [Évaluation](#évaluation)
- [Démonstration](#démonstration)
- [Résultats](#résultats)
- [Licence](#licence)
- [Remerciements](#remerciements)

## 🔍 Présentation du projet

Luneetee3D est un projet de fine-tuning du modèle Hunyuan3D-2 développé par Tencent, adapté spécifiquement pour la reconstruction 3D de structures craniofaciales à partir d'images médicales 2D de patients atteints de Luneetee. Ce projet vise à améliorer la précision et la fidélité des reconstructions 3D dans un contexte médical, facilitant ainsi le diagnostic et la planification chirurgicale.

## ⚙️ Fonctionnement

### Architecture globale

Luneetee3D s'appuie sur l'architecture à deux étapes de Hunyuan3D-2 :

1. **Génération de forme (Shape Generation)** : Utilise le modèle Hunyuan3D-DiT pour créer un maillage 3D à partir d'images médicales 2D
2. **Génération de texture (Texture Generation)** : Utilise le modèle Hunyuan3D-Paint pour appliquer des textures réalistes au maillage 3D

<p align="center">
  <img src="https://github.com/user-attachments/assets/a2cbc5b8-be22-49d7-b1c3-7aa2b20ba460" width="600">
</p>

### Processus de fine-tuning

Le processus de fine-tuning comprend les étapes suivantes :

1. **Préparation des données** : Organisation et prétraitement des images médicales et des maillages 3D correspondants
2. **Entraînement du modèle** : Adaptation du modèle Hunyuan3D-2 aux spécificités des structures craniofaciales
3. **Évaluation** : Mesure de la qualité des reconstructions 3D à l'aide de métriques spécifiques
4. **Inférence** : Utilisation du modèle fine-tuné pour générer des reconstructions 3D à partir de nouvelles images

### Métriques d'évaluation

Nous utilisons plusieurs métriques pour évaluer la qualité des reconstructions 3D :

- **Chamfer Distance** : Mesure la distance moyenne entre les points des maillages prédits et de référence
- **Normal Consistency** : Évalue la cohérence des normales de surface entre les maillages
- **F-Score** : Mesure la précision et le rappel des points reconstruits
- **Volume Difference** : Calcule la différence de volume entre les maillages
- **Surface Area Difference** : Mesure la différence de surface entre les maillages

## 📁 Structure du projet

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

## 🛠️ Installation

### Prérequis

- Python 3.9+
- GPU compatible CUDA (recommandé, minimum 8GB VRAM)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommandé pour la gestion des environnements)

### Configuration

1. Obtenir le code:

**Option A: Cloner le dépôt complet (recommandé)**
```bash
git clone https://github.com/amin8452/Pfa-Finale.git
cd Pfa-Finale
```

**Option B: Télécharger uniquement le dossier Luneetee3D (pour Kaggle, Colab, etc.)**
```bash
# Installer svn si nécessaire
apt-get install -y subversion

# Télécharger uniquement le dossier Luneetee3D
svn export https://github.com/amin8452/Pfa-Finale/trunk/Luneetee3D
cd Luneetee3D
```

**Option C: Télécharger l'archive ZIP**
```bash
# Télécharger l'archive du dépôt
wget https://github.com/amin8452/Pfa-Finale/archive/refs/heads/master.zip

# Extraire uniquement le dossier Luneetee3D
unzip master.zip "Pfa-Finale-master/Luneetee3D/*" -d .
mv Pfa-Finale-master/Luneetee3D .
rm -rf Pfa-Finale-master master.zip
cd Luneetee3D
```

Vous pouvez également accéder directement au projet sur GitHub:
[https://github.com/amin8452/Pfa-Finale/tree/master/Luneetee3D](https://github.com/amin8452/Pfa-Finale/tree/master/Luneetee3D)

> **Note pour Kaggle/Colab**: Utilisez l'option B ou C ci-dessus. N'essayez pas de cloner directement le sous-dossier Luneetee3D car cela ne fonctionnera pas.

2. Créer un environnement conda:
```bash
conda create -n luneetee3d python=3.9
conda activate luneetee3d
```

3. Installer PyTorch avec CUDA:
```bash
# Pour CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Installer les dépendances:

**Si vous avez utilisé l'option A (dépôt complet):**
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

**Si vous avez utilisé l'option B ou C (dossier Luneetee3D uniquement):**
```bash
pip install -r requirements.txt

# Vous devrez également installer Hunyuan3D-2
pip install git+https://github.com/Tencent/Hunyuan3D-2.git

# Ou télécharger et installer manuellement les rasterizers
git clone https://github.com/Tencent/Hunyuan3D-2.git
cd Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
python setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python setup.py install
cd ../../..
```

## 📊 Préparation des données

### Structure des données

Organisez votre dataset selon la structure suivante:
```
raw_data/
├── patient001/
│   ├── image.jpg            # Image médicale 2D (scanner, IRM, etc.)
│   ├── mesh.obj             # Maillage 3D de référence
├── patient002/
│   ├── image.jpg
│   ├── mesh.obj
...
```

### Prétraitement

Le script `prepare_dataset.py` effectue les opérations suivantes:
- Division des données en ensembles d'entraînement, de validation et de test
- Normalisation des images et des maillages
- Organisation des données dans la structure requise

**Si vous avez utilisé l'option A (dépôt complet):**
```bash
# Assurez-vous d'être dans le répertoire racine du dépôt (Pfa-Finale)
python Luneetee3D/prepare_dataset.py --input_dir raw_data --output_dir data/luneetee_3d --split_ratio 0.7,0.15,0.15

# Si vous rencontrez une erreur de chemin, essayez:
cd Luneetee3D
python prepare_dataset.py --input_dir ../raw_data --output_dir ../data/luneetee_3d --split_ratio 0.7,0.15,0.15
```

**Si vous avez utilisé l'option B ou C (dossier Luneetee3D uniquement):**
```bash
# Assurez-vous d'être dans le dossier Luneetee3D
python prepare_dataset.py --input_dir raw_data --output_dir data/luneetee_3d --split_ratio 0.7,0.15,0.15
```

Options disponibles:
- `--input_dir` : Répertoire contenant les données brutes
- `--output_dir` : Répertoire de sortie pour les données prétraitées
- `--split_ratio` : Ratio de division train/val/test (par défaut: 0.7,0.15,0.15)
- `--seed` : Graine aléatoire pour la reproductibilité (par défaut: 42)

## 🏋️ Entraînement

### Configuration

Le fichier `configs/luneetee.yaml` contient tous les paramètres d'entraînement:
- Paramètres du dataset (chemin, taille d'image, etc.)
- Paramètres du modèle (architecture, poids pré-entraînés, etc.)
- Paramètres d'entraînement (taille de batch, learning rate, etc.)
- Paramètres d'évaluation (métriques, seuils, etc.)

### Lancement de l'entraînement

Pour fine-tuner le modèle:

**Si vous avez utilisé l'option A (dépôt complet):**
```bash
# Assurez-vous d'être dans le répertoire racine du dépôt (Pfa-Finale)
python Luneetee3D/train.py --cfg Luneetee3D/configs/luneetee.yaml --output_dir output/luneetee_model

# Si vous rencontrez une erreur de chemin, essayez:
cd Luneetee3D
python train.py --cfg configs/luneetee.yaml --output_dir ../output/luneetee_model
```

**Si vous avez utilisé l'option B ou C (dossier Luneetee3D uniquement):**
```bash
# Assurez-vous d'être dans le dossier Luneetee3D
python train.py --cfg configs/luneetee.yaml --output_dir output/luneetee_model
```

Options disponibles:
- `--cfg` : Chemin vers le fichier de configuration
- `--output_dir` : Répertoire de sortie pour les checkpoints et les logs
- `--resume` : Reprendre l'entraînement à partir d'un checkpoint
- `--checkpoint` : Chemin vers le checkpoint pour reprendre l'entraînement

### Reprise de l'entraînement

Pour reprendre l'entraînement à partir d'un checkpoint:

**Si vous avez utilisé l'option A (dépôt complet):**
```bash
python Luneetee3D/train.py --cfg Luneetee3D/configs/luneetee.yaml --output_dir output/luneetee_model --resume --checkpoint output/luneetee_model/checkpoint_epoch_10.pt
```

**Si vous avez utilisé l'option B ou C (dossier Luneetee3D uniquement):**
```bash
python train.py --cfg configs/luneetee.yaml --output_dir output/luneetee_model --resume --checkpoint output/luneetee_model/checkpoint_epoch_10.pt
```

### Suivi de l'entraînement

Pendant l'entraînement, les métriques suivantes sont enregistrées:
- Perte d'entraînement et de validation
- Chamfer Distance
- Normal Consistency
- F-Score
- Temps d'entraînement par époque

Des visualisations des reconstructions sont générées périodiquement dans le répertoire `output_dir/visualizations`.

## 📏 Évaluation

### Évaluation du modèle

Pour évaluer un modèle entraîné:

**Si vous avez utilisé l'option A (dépôt complet):**
```bash
# Assurez-vous d'être dans le répertoire racine du dépôt (Pfa-Finale)
python Luneetee3D/evaluate.py --cfg Luneetee3D/configs/luneetee.yaml --checkpoint output/luneetee_model/best_model.pt --output_dir evaluation --visualize

# Si vous rencontrez une erreur de chemin, essayez:
cd Luneetee3D
python evaluate.py --cfg configs/luneetee.yaml --checkpoint ../output/luneetee_model/best_model.pt --output_dir ../evaluation --visualize
```

**Si vous avez utilisé l'option B ou C (dossier Luneetee3D uniquement):**
```bash
# Assurez-vous d'être dans le dossier Luneetee3D
python evaluate.py --cfg configs/luneetee.yaml --checkpoint output/luneetee_model/best_model.pt --output_dir evaluation --visualize
```

Options disponibles:
- `--cfg` : Chemin vers le fichier de configuration
- `--checkpoint` : Chemin vers le checkpoint du modèle à évaluer
- `--output_dir` : Répertoire de sortie pour les résultats d'évaluation
- `--visualize` : Générer des visualisations des reconstructions
- `--texture` : Générer des maillages texturés pendant l'évaluation

### Résultats d'évaluation

Les résultats d'évaluation sont enregistrés dans `output_dir/metrics.json` et comprennent:
- Métriques individuelles pour chaque échantillon
- Métriques moyennes sur l'ensemble de test
- Visualisations des reconstructions (si `--visualize` est spécifié)
- Maillages reconstruits au format OBJ/GLB

## 🚀 Démonstration

### Inférence sur une image

Pour exécuter l'inférence sur une seule image:

**Si vous avez utilisé l'option A (dépôt complet):**
```bash
# Assurez-vous d'être dans le répertoire racine du dépôt (Pfa-Finale)
python Luneetee3D/demo.py --image path/to/image.jpg --checkpoint output/luneetee_model/best_model.pt --output_dir demo_output

# Si vous rencontrez une erreur de chemin, essayez:
cd Luneetee3D
python demo.py --image ../path/to/image.jpg --checkpoint ../output/luneetee_model/best_model.pt --output_dir ../demo_output
```

**Si vous avez utilisé l'option B ou C (dossier Luneetee3D uniquement):**
```bash
# Assurez-vous d'être dans le dossier Luneetee3D
python demo.py --image path/to/image.jpg --checkpoint output/luneetee_model/best_model.pt --output_dir demo_output
```

Options disponibles:
- `--image` : Chemin vers l'image d'entrée
- `--checkpoint` : Chemin vers le checkpoint du modèle
- `--output_dir` : Répertoire de sortie pour les résultats
- `--texture` : Générer un maillage texturé

### Visualisation des résultats

Les résultats de la démonstration comprennent:
- Maillage 3D au format OBJ (sans texture) ou GLB (avec texture)
- Visualisations du maillage sous différents angles
- Métriques de qualité si une référence est disponible

## 📈 Résultats

Voici les métriques obtenues sur notre ensemble de test:

| Métrique | Valeur |
|----------|--------|
| Chamfer Distance | 0.0213 |
| Normal Consistency | 0.911 |
| F-Score @1mm | 87.4% |
| Volume Difference | 12.3 cm³ |
| Surface Area Difference | 8.7 cm² |

### Exemples de reconstructions

<p align="center">
  <img src="https://github.com/user-attachments/assets/efb402a1-0b09-41e0-a6cb-259d442e76aa" width="800">
</p>

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## 🙏 Remerciements

- [Tencent Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) pour le modèle de base
- [Trimesh](https://trimsh.org/) pour le traitement des maillages
- [PyTorch](https://pytorch.org/) pour le framework de deep learning
