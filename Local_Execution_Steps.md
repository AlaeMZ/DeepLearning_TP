# Guide d'Exécution Locale et Rédaction du Rapport — TP Deep Learning

Ce document détaille les étapes exactes à suivre pour réaliser le TP localement (hors de Google Colab) et générer tous les éléments nécessaires à votre rapport final.

## 1. Préparation de l'environnement local

Puisque vous travaillez localement, assurez-vous d'avoir les prérequis suivants :
- **Python 3.8+**
- **Bibliothèques :** `torch`, `torchvision`, `matplotlib`, `seaborn`, `scikit-learn`, `nltk`, `opencv-python`, `torchinfo`, `pycocotools`.
- **Matériel :** Si vous avez un GPU local (NVIDIA), assurez-vous que CUDA est bien installé. Sinon, le code utilisera le CPU (ce qui sera beaucoup plus lent). Ajustez le `batch_size` (ex: 8, 16 ou 32) et le `num_workers` (ex: 4 ou 8) en fonction de la mémoire RAM et VRAM de votre machine.

---

## 2. Étapes de l'Exercice 1 : Segmentation (Oxford-IIIT Pet)

### Étape 2.1 : Téléchargement et Visualisation (Q1)
1. **Téléchargez** le dataset depuis [le site officiel](https://www.robots.ox.ac.uk/~vgg/data/pets/) et extrayez-le dans un dossier `./oxford-iiit-pet`.
2. **Exécutez le script Q1** pour générer l'image `visualisation_dataset.png`.
3. **Notez les pourcentages** de distribution des classes affichés dans la console.
4. **Dans le rapport :** Intégrez l'image, la distribution et ajoutez l'analyse sur le déséquilibre des classes et la stratégie de pondération proposée.

### Étape 2.2 : Augmentation et DataLoaders (Q2 & Q3)
1. **Implémentez** la classe `SyncAugmentation` pour la data augmentation géométrique synchronisée.
2. **Configurez les DataLoaders** (70/15/15). Ajustez le `batch_size` selon la mémoire de votre carte graphique locale.
3. **Dans le rapport :** Justifiez votre choix de `batch_size` en fonction de votre matériel local et expliquez l'importance du `NEAREST` pour la rotation du masque.

### Étape 2.3 : SimpleUNet et Entraînement (Q4, Q5 & Q6)
1. **Instanciez** le `SimpleUNet` et comptez les paramètres via `torchinfo`.
2. **Implémentez** la fonction `combined_loss`.
3. **Lancez l'entraînement** de 30 epochs. Le script sauvegardera `best_unet.pth` et générera `learning_curves_unet.png`.
4. **Dans le rapport :** Intégrez le nombre de paramètres, expliquez la fonction de perte combinée, insérez la courbe d'apprentissage et rédigez une analyse sur l'overfitting.

### Étape 2.4 : Évaluation et Matrice de confusion (Q7)
1. **Exécutez l'évaluation** sur le test set en chargeant `best_unet.pth`.
2. **Sauvegardez** la figure `confusion_matrix_unet.png` générée.
3. **Dans le rapport :** Intégrez la matrice et analysez les erreurs (ex: confusion entre contour et fond).

### Étape 2.5 : Transfer Learning (Q8, Q9 & Q10)
1. **Implémentez** la méthode `forward()` de `TransferUNet`.
2. **Entraînez** les deux versions : `frozen` (encodeur gelé) et `fine-tuned` (taux d'apprentissage différentiel).
3. **Mesurez** les temps par epoch et l'inférence sur votre machine.
4. **Dans le rapport :** Remplissez le tableau comparatif avec *vos* vrais résultats locaux (mIoU, mDice, temps) et ajoutez l'analyse comparative.

---

## 3. Étapes de l'Exercice 2 : Image Captioning (COCO)

### Étape 3.1 : Dataset et Vocabulaire (Q1 & Q2)
1. **Téléchargez** les annotations COCO (`captions_train2017.json`). *Note : Le dataset complet des images COCO est très lourd, assurez-vous d'avoir assez d'espace disque ou d'utiliser un sous-ensemble si votre machine est limitée.*
2. **Générez et sauvegardez** l'histogramme `caption_lengths.png`.
3. **Dans le rapport :** Insérez l'histogramme et justifiez le choix de `max_len=50`.

### Étape 3.2 : Modèle CNN-LSTM et Entraînement (Q3 & Q4)
1. **Vérifiez les dimensions** du modèle avec un forward pass factice.
2. **Lancez l'entraînement** avec Teacher Forcing pour 20 epochs et sauvegardez `perplexity_curve.png`.
3. **Dans le rapport :** Intégrez le tableau des dimensions, la courbe de perplexité et l'explication du Teacher Forcing.

### Étape 3.3 : Évaluation de la Génération (Q5)
1. **Implémentez** l'inférence `greedy_caption`.
2. **Calculez** le score BLEU-4 avec le set de test.
3. **Dans le rapport :** Comparez théoriquement Greedy Search vs Beam Search et indiquez votre score.

### Étape 3.4 : Attention Visuelle (Q6 & Q7)
1. **Modifiez** l'encodeur (`ImageEncoderSpatial`) et entraînez le décodeur avec attention.
2. **Exécutez le script de visualisation** de l'attention pour générer `attention.png`.
3. **Dans le rapport :** Intégrez l'image d'attention et commentez la pertinence sémantique des zones mises en évidence.

### Étape 3.5 : Bilan des Métriques (Q8)
1. **Compilez** les résultats finaux pour les métriques BLEU, METEOR, et CIDEr.
2. **Dans le rapport :** Remplissez le tableau bilan final et ajoutez la discussion sur les limites des métriques automatiques.

---

## 4. Rédaction et Structure du Rapport Final

Assemblez tous les éléments générés lors des étapes précédentes dans un document (ex: Word ou Markdown exporté en PDF) en respectant **exactement** cette structure :

1. **Page de garde**
2. **Table des matières**
3. **Introduction** (½ page)
4. **Exercice 1 — Segmentation**
   - Partie 1 : Dataset (Q1, Q2, Q3)
   - Partie 2 : SimpleUNet (Q4, Q5, Q6, Q7)
   - Partie 3 : Transfer Learning (Q8, Q9, Q10)
5. **Exercice 2 — Image Captioning**
   - Partie 1 : Dataset COCO (Q1, Q2)
   - Partie 2 : CNN-LSTM (Q3, Q4, Q5)
   - Partie 3 : Attention (Q6, Q7, Q8)
6. **Conclusion** (½ page)

### Rappels Cruciaux pour la notation :
- **Ne collez pas tout le code.** Ne mettez que les extraits essentiels (ex: fonctions clés, architecture réseau).
- **Chaque figure doit avoir :** un numéro, un titre, des axes étiquetés et une description.
- **Règle d'or :** Chaque résultat chiffré ou graphique **doit être analysé et commenté** comme indiqué dans le guide. Ne laissez jamais une image ou un chiffre sans explication textuelle.

---
*Ce guide suit précisément les consignes du fichier TP_Guide_Solution_Complet.md pour une exécution en environnement local.*
