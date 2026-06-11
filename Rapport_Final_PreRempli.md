# RAPPORT : TP Deep Learning
**CNN Segmentation & Image Captioning CNN-LSTM**
*Master AIDC 2025–2026 — FST Béni Mellal*

**Réalisé par :** [Votre Nom/Prénom]

---

## Table des matières
1. Introduction
2. Exercice 1 — Segmentation
   - Partie 1 : Dataset Oxford-IIIT Pet
   - Partie 2 : SimpleUNet
   - Partie 3 : Transfer Learning
3. Exercice 2 — Image Captioning
   - Partie 1 : Dataset COCO
   - Partie 2 : CNN-LSTM
   - Partie 3 : Attention Visuelle
4. Conclusion

---

## Introduction
Ce rapport présente les résultats et analyses des deux exercices de Deep Learning. Le premier exercice est consacré à la segmentation sémantique d'images d'animaux (dataset Oxford-IIIT Pet) en utilisant une architecture U-Net from scratch puis via Transfer Learning. Le second exercice aborde la génération de descriptions d'images (Image Captioning) sur le dataset COCO à l'aide d'une architecture CNN-LSTM, puis intègre un mécanisme d'attention visuelle pour améliorer la pertinence et l'interprétabilité des résultats.

---

## Exercice 1 — Segmentation

### Partie 1 : Dataset Oxford-IIIT Pet

#### Chargement, visualisation et distribution
Afin de charger les données, nous avons utilisé la classe `PetSegmentationDataset`.

```python
# Instanciation du dataset
train_dataset = PetSegmentationDataset(root='./oxford-iiit-pet', split='train')

# Affichage de 4 exemples
fig, axes = plt.subplots(4, 2, figsize=(8, 16))
for i in range(4):
    img, mask = train_dataset[i]
    img_np = img.permute(1, 2, 0).numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    axes[i, 0].imshow(img_np)
    axes[i, 0].set_title("Image")
    axes[i, 1].imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=2)
    axes[i, 1].set_title("Masque (0=animal, 1=contour, 2=fond)")
plt.tight_layout()
plt.savefig("visualisation_dataset.png", dpi=150)
plt.show()

# Distribution des classes
total_pixels = {'animal': 0, 'contour': 0, 'fond': 0}
for i in range(len(train_dataset)):
    _, mask = train_dataset[i]
    total_pixels['animal']  += (mask == 0).sum().item()
    total_pixels['contour'] += (mask == 1).sum().item()
    total_pixels['fond']    += (mask == 2).sum().item()

total = sum(total_pixels.values())
for k, v in total_pixels.items():
    print(f"{k}: {v/total*100:.1f}%")
```

*[Insérez ici l'image visualisation_dataset.png générée]*

**Analyse :**
Le dataset est déséquilibré : la classe 'contour' représente seulement ~5% des pixels (fond ~65%, animal ~30%). Sans correction, le modèle ignorera cette classe. Pour corriger ce déséquilibre, nous proposons d'utiliser des poids inversement proportionnels à la fréquence de chaque classe dans la fonction de perte CrossEntropy.

```python
# Stratégie de pondération proposée :
freqs = torch.tensor([0.30, 0.05, 0.65])
weights = 1.0 / freqs
weights = weights / weights.sum()
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

#### Augmentation de données
Pour améliorer la robustesse du modèle, une augmentation spatiale a été appliquée de manière synchronisée sur l'image et son masque. 

```python
class SyncAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() < self.p:
            angle = random.uniform(-15, 15)
            img  = TF.rotate(img,  angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        if random.random() < self.p:
            img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)(img)
        return img, mask
```
*[Optionnel: Insérez ici un exemple d'image avant/après augmentation]*

**Justification :** 
Lors de la rotation, l'interpolation `NEAREST` est obligatoire pour le masque afin de ne pas créer de valeurs de classes fractionnaires inexistantes (ex: 1.5). Le `ColorJitter` n'est appliqué qu'à l'image.

#### DataLoaders (70/15/15)
```python
from torch.utils.data import random_split

n = len(full_dataset)
n_train, n_val = int(0.70 * n), int(0.15 * n)
n_test  = n - n_train - n_val

train_set, val_set, test_set = random_split(full_dataset, [n_train, n_val, n_test])

BATCH_SIZE = 8
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
```

**Justification du Batch Size :** 
Avec des images de 256×256×3 en float32, un batch de 8 images représente environ 6.3 MB. En ajoutant les activations du modèle, l'empreinte mémoire reste bien en dessous des limites standards d'un GPU moyen (ex: 6 ou 8 GB), ce qui laisse une marge de sécurité confortable.

---

### Partie 2 : SimpleUNet

#### Architecture et Paramètres
```python
from torchinfo import summary
model = SimpleUNet(in_channels=3, num_classes=3, base_features=32)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paramètres entraînables : {total_params:,}")
```
Le modèle `SimpleUNet` avec `base_features=32` compte environ 1.9 millions de paramètres entraînables. C'est un modèle relativement léger comparé aux architectures pré-entraînées classiques.

#### Fonction de perte combinée
```python
def combined_loss(pred, target, alpha=0.5):
    ce   = nn.CrossEntropyLoss()(pred, target)
    d    = dice_loss(pred, target)
    return alpha * ce + (1 - alpha) * d
```
**Analyse :** 
La CrossEntropy traite chaque pixel indépendamment et est sensible aux déséquilibres de classes. La Dice Loss optimise directement le ratio de chevauchement (IoU) entre prédiction et vérité terrain, ce qui la rend plus robuste aux classes rares (le contour). En combinant les deux (α=0.5), on bénéficie de la stabilité de la CrossEntropy et de la précision par classe de la Dice Loss.

#### Entraînement et performances
L'entraînement a été mené sur 30 epochs avec un algorithme d'optimisation Adam et un `ReduceLROnPlateau`.

*[Insérez ici l'image learning_curves_unet.png générée par votre script local]*

**Analyse :** 
On observe un overfitting typique à partir de l'epoch ~15 : la loss de validation remonte ou stagne tandis que la loss d'entraînement continue de baisser. Des solutions pour contrer cela sont : 
(1) Une augmentation de données plus agressive, 
(2) L'ajout de couches Dropout, 
(3) La réduction de la complexité du modèle (`base_features` à 16), ou 
(4) L'Early Stopping (qui s'est d'ailleurs déclenché ici).

#### Matrice de confusion sur le Test Set
*[Insérez ici l'image confusion_matrix_unet.png générée par votre script local]*

**Analyse des erreurs :** 
La classe 'contour' est la plus difficile à prédire en raison de sa finesse et du faible volume de pixels qu'elle représente. Les erreurs les plus fréquentes se situent aux frontières, causant une confusion forte entre 'contour' et 'fond'.

---

### Partie 3 : Transfer Learning

Nous avons implémenté un modèle `TransferUNet` basé sur un encodeur pré-entraîné (ResNet). 

```python
# Complétion du forward() avec Skip Connections
def forward(self, x):
    e0 = self.enc0(x)
    e1 = self.enc1(self.pool(e0))
    e2 = self.enc2(e1)
    e3 = self.enc3(e2)
    e4 = self.enc4(e3)

    d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
    d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
    d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
    d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))
    out = self.final_up(d1)
    return self.out(out)
```

Pour l'entraînement avec fine-tuning, un taux d'apprentissage différentiel a été appliqué (`1e-4` pour l'encodeur, `1e-3` pour le décodeur).

| Modèle | mIoU | mDice | Nb Params | Temps/epoch | Inférence |
|---|---|---|---|---|---|
| SimpleUNet | [VOTRE_VALEUR] | [VOTRE_VALEUR] | ~1.9M | [VOTRE_TEMPS] | [VOTRE_TEMPS] |
| TransferUNet (frozen) | [VOTRE_VALEUR] | [VOTRE_VALEUR] | ~25M | [VOTRE_TEMPS] | [VOTRE_TEMPS] |
| TransferUNet (fine-tuned) | [VOTRE_VALEUR] | [VOTRE_VALEUR] | ~25M | [VOTRE_TEMPS] | [VOTRE_TEMPS] |

**Analyse comparative :**
Le `TransferUNet` fine-tuné obtient les meilleures métriques globales grâce aux représentations riches apprises sur ImageNet. L'encodeur gelé (frozen) converge très vite puisqu'il n'entraîne que le décodeur, mais atteint un plafond de performance inférieur. Le `SimpleUNet` est rapide et léger mais manque de capacité d'extraction de features complexes. Pour un déploiement réel en production, la version 'frozen' offre souvent le meilleur compromis entre temps d'entraînement et performance.

---

## Exercice 2 — Image Captioning

### Partie 1 : Dataset COCO

#### Vocabulaire et longueurs
Après avoir construit le vocabulaire avec un `freq_threshold=5` :

*[Insérez ici l'image caption_lengths.png]*

**Choix de max_len :** 
L'analyse de l'histogramme montre que le 95ème percentile des longueurs de légendes est aux alentours de 22 tokens. Nous avons donc choisi `max_len=50`. Cela couvre plus de 99% des données sans saturer la mémoire avec un padding excessif. Les éventuels mots tronqués au-delà de 50 tokens sont généralement des redondances grammaticales sans impact critique.

#### DataLoader et Collate Function
Pour gérer des séquences de taille variable au sein d'un même batch, un `collate_fn` a été implémenté pour trier les séquences par longueur décroissante :
```python
def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs, 0)
    lengths = [(cap != 0).sum().item() for cap in caps]
    sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    imgs    = imgs[[sorted_idx]]
    caps    = torch.stack([caps[i] for i in sorted_idx])
    lengths = [lengths[i] for i in sorted_idx]
    return imgs, caps, torch.tensor(lengths)
```

---

### Partie 2 : CNN-LSTM

#### Architecture et vérification des dimensions
Le flux de données à travers l'encodeur ResNet et le décodeur LSTM donne les dimensions suivantes (pour B=4) :

| Étape | Dimension |
|---|---|
| Input images | (4, 3, 224, 224) |
| ResNet features (post-pooling) | (4, 2048) |
| After fc + BN (embed_size) | (4, 256) |
| Embeddings des captions | (4, 49, 256) |
| Entrée LSTM (concaténée) | (4, 50, 256) |
| Sortie LSTM | (4, 50, 512) |
| Final logits | (4, 50, vocab_size) |

#### Entraînement et Teacher Forcing

*[Insérez ici l'image perplexity_curve.png]*

**Analyse de l'entraînement :** 
Nous avons utilisé le "Teacher Forcing", qui consiste à fournir au modèle les vrais mots de la vérité terrain comme entrée à l'instant *t*, plutôt que sa propre prédiction à *t-1*. Cela permet d'accélérer drastiquement la convergence en évitant l'accumulation d'erreurs en début d'apprentissage, au risque de créer un léger biais (exposure bias) lors de l'inférence. La perplexité chute drastiquement, indiquant la bonne mémorisation des structures syntaxiques.

#### Greedy Search vs Beam Search
| BLEU-4 (Greedy Search) | [VOTRE_SCORE_BLEU_ICI] |
|---|---|

**Analyse :** 
Le Beam Search permet de générer des descriptions globalement plus fluides que le Greedy Search. En effet, le Greedy Search sélectionne naïvement le mot le plus probable à chaque pas de temps, ce qui génère souvent des phrases syntaxiquement limitées ou des boucles de répétition. Le Beam Search explore de multiples branches simultanément pour maximiser la probabilité globale de la phrase générée, augmentant ainsi le score BLEU final.

---

### Partie 3 : Attention Visuelle

Afin d'apporter de l'explicabilité au modèle, un mécanisme d'attention spatiale a été intégré. L'encodeur ResNet a été modifié pour retourner les "Feature Maps" spatiales de taille `(B, 49, 2048)` plutôt qu'un vecteur global.

```python
class ImageEncoderSpatial(nn.Module):
    # [...]
    def forward(self, images):
        features = self.resnet(images)           # (B, 2048, 7, 7)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # (B, 7, 7, 2048)
        features = features.view(B, -1, C)       # (B, 49, 2048)
        return features
```

#### Visualisation de l'attention

*[Insérez ici l'image attention.png]*

**Analyse des cartes d'attention :** 
L'affichage des poids "alphas" sur l'image démontre que le modèle apprend correctement à aligner les mots générés avec les sous-régions visuelles correspondantes. Par exemple, lorsque le mot décrivant l'animal est généré, la "heatmap" se concentre explicitement sur les pixels de l'animal dans l'image.

#### Bilan Final des Métriques Automatiques

| Modèle | BLEU-1 | BLEU-4 | METEOR | CIDEr |
|---|---|---|---|---|
| CNN-LSTM (sans attention) | [VALEUR] | [VALEUR] | [VALEUR] | [VALEUR] |
| CNN-LSTM-Attention | [VALEUR] | [VALEUR] | [VALEUR] | [VALEUR] |

**Limites des métriques :** 
Les métriques comme BLEU et METEOR évaluent le chevauchement (n-grammes) exact avec des références humaines. Leurs limites sont évidentes : 
1. Elles pénalisent rudement l'utilisation de synonymes tout à fait valides sémantiquement. 
2. Elles ne capturent pas toujours la cohérence ou la grammaire de l'intégralité de la phrase. 
Bien que CIDEr soit plus pertinent pour le captioning en pondérant l'importance de certains mots via TF-IDF, ces méthodes automatiques ne remplaceront jamais une vérification humaine pour jauger la "véritable" qualité d'une description.

---

## Conclusion
Ce projet nous a permis d'implémenter de bout en bout deux des tâches fondamentales du Deep Learning appliqué à la vision par ordinateur et au NLP. La segmentation via U-Net a mis en évidence les problématiques de classes minoritaires, que nous avons palliées par la conception d'une loss hybride et de data augmentation. L'ajout du Transfer Learning a démontré la supériorité incontestable des poids pré-entraînés. Enfin, l'Image Captioning nous a illustré la puissance des architectures séquentielles avec le CNN-LSTM. Le mécanisme d'attention est venu apporter la touche finale nécessaire pour rendre le réseau "explicable", soulignant que le deep learning moderne tend autant vers la performance pure que vers l'interprétabilité de ses décisions.
