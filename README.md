# TP 2 Deep Learning — Segmentation Sémantique & Image Captioning

Ce dépôt contient l'implémentation et les résultats expérimentaux pour le **TP2 de Deep Learning (Master AIDC)**.

---

## 🚀 Structure du Dépôt

*   `ex1_segmentation.py` : Script d'entraînement et d'évaluation pour la segmentation sémantique d'images sur le jeu de données **Oxford-IIIT Pet**.
*   `ex2_captioning.py` : Script d'entraînement et de démonstration pour le sous-titrage automatique d'images (Image Captioning) à l'aide d'un décodeur LSTM avec attention globale.
*   `*.png` : Courbes d'apprentissage, matrices de confusion et visualisations qualitatives générées sur le jeu de test.

---

## 📊 Synthèse des Résultats

### Exercice 1 — Segmentation Sémantique (Oxford-IIIT Pet)
L'objectif est de classifier chaque pixel en 3 classes : **Animal**, **Contour**, et **Fond**.

| Modèle | mIoU | mDice | Paramètres | Temps/Epoch | Inférence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SimpleUNet (From Scratch)** | 0.7084 | 0.8164 | 1,949,763 | ~45s | 4.96 ms |
| **TransferUNet (Frozen)** | 0.8008 | 0.8804 | 24,365,315 | **~35s** | 3.33 ms |
| **TransferUNet (Fine-Tuned)** | **0.8039** | **0.8825** | 24,365,315 | ~80s | **3.18 ms** |

*Note : L'utilisation de Transfer Learning avec un encodeur ResNet-34 pré-entraîné permet d'obtenir un gain absolu de **+10% en mIoU**.*

---

## 💻 Instructions de Lancement

### Prérequis
```bash
pip install torch torchvision matplotlib numpy torchinfo tqdm pillow
```

### Exécuter la Segmentation (Exercice 1)
Le script télécharge automatiquement le dataset Oxford-IIIT Pet lors de son premier lancement, puis lance l'entraînement ou charge les checkpoints existants pour afficher les métriques et générer les figures.
```bash
python ex1_segmentation.py
```

### Exécuter le Captioning (Exercice 2)
```bash
python ex2_captioning.py
```

---

*Développé par **Alae** dans le cadre du cours de Deep Learning (Master AIDC).*
