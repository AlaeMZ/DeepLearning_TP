# Guide Complet de Résolution — TP Deep Learning
**CNN Segmentation & Image Captioning CNN-LSTM**
FST Béni Mellal — Master AIDC 2025–2026

> Ce guide vous explique **comment résoudre chaque question** et **comment rédiger votre rapport**. Suivez-le dans l'ordre.

---

# COMMENT STRUCTURER VOTRE RAPPORT

Votre rapport doit avoir cette structure :

```
Page de garde
Table des matières
Introduction (½ page)
Exercice 1 — Segmentation
  Partie 1 : Dataset
  Partie 2 : SimpleUNet
  Partie 3 : Transfer Learning
Exercice 2 — Image Captioning
  Partie 1 : Dataset COCO
  Partie 2 : CNN-LSTM
  Partie 3 : Attention
Conclusion (½ page)
```

Pour **chaque question**, votre rapport doit contenir :
1. Une courte explication de ce que vous avez fait
2. Le code essentiel (pas tout, juste les parties clés)
3. Les résultats (chiffres, courbes, tableaux)
4. Une **analyse/commentaire** — c'est la partie la plus notée

---

# EXERCICE 1 — SEGMENTATION

---

## Partie 1 — Dataset Oxford-IIIT Pet

### Q1 — Chargement, visualisation, distribution des classes

**Ce qu'il faut faire :**

1. Télécharger le dataset depuis [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)
2. Utiliser la classe `PetSegmentationDataset` fournie
3. Afficher 4 images avec leur masque côte à côte avec `matplotlib`
4. Calculer la distribution des pixels par classe

**Code à écrire :**

```python
# Instanciation du dataset
train_dataset = PetSegmentationDataset(root='./oxford-iiit-pet', split='train')

# Affichage de 4 exemples
fig, axes = plt.subplots(4, 2, figsize=(8, 16))
for i in range(4):
    img, mask = train_dataset[i]
    # Dénormaliser l'image
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

**Ce qu'il faut écrire dans le rapport :**

Après avoir calculé la distribution, vous devriez trouver quelque chose comme : fond ~65%, animal ~30%, contour ~5%. Commentez ainsi :

> *"Le dataset est déséquilibré : la classe 'contour' représente seulement ~5% des pixels. Sans correction, le modèle ignorera cette classe. Pour corriger ce déséquilibre, nous proposons d'utiliser des poids inversement proportionnels à la fréquence de chaque classe dans la fonction de perte CrossEntropy."*

**Stratégie de pondération à mentionner :**

```python
# Poids inversement proportionnels aux fréquences
freqs = torch.tensor([0.30, 0.05, 0.65])  # exemple
weights = 1.0 / freqs
weights = weights / weights.sum()  # normalisation
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

---

### Q2 — Augmentation de données

**Ce qu'il faut faire :**
Implémenter des augmentations géométriques **synchronisées** entre image et masque. Le masque ne doit pas subir de color jitter.

**Code clé :**

```python
import torchvision.transforms.functional as TF
import random

class SyncAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        # Flip horizontal (même transformation pour les deux)
        if random.random() < self.p:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        # Rotation aléatoire
        if random.random() < self.p:
            angle = random.uniform(-15, 15)
            img  = TF.rotate(img,  angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        # Color jitter sur l'image UNIQUEMENT
        if random.random() < self.p:
            img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)(img)
        return img, mask
```

**Dans le rapport**, montrez des exemples avant/après et expliquez pourquoi la rotation du masque utilise `NEAREST` (pour ne pas créer de valeurs interpolées entre classes).

---

### Q3 — DataLoaders 70/15/15

**Code :**

```python
from torch.utils.data import random_split

full_dataset = PetSegmentationDataset(root='./oxford-iiit-pet', split='train')
n = len(full_dataset)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
n_test  = n - n_train - n_val

train_set, val_set, test_set = random_split(full_dataset, [n_train, n_val, n_test])

# batch_size=8 si GPU < 8GB, batch_size=16 si GPU >= 8GB
BATCH_SIZE = 8

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
```

**Justification du batch_size dans le rapport :**
> *"Avec des images de 256×256×3 en float32, un batch de 8 images représente environ 8 × 256 × 256 × 3 × 4 = 6.3 MB. En ajoutant les activations du modèle (~10x), on obtient ~63 MB, bien en dessous de la limite de 6 GB d'une Tesla T4 (Colab). On choisit batch_size=8 pour garder une marge."*

---

## Partie 2 — SimpleUNet

### Q4 — Instanciation et comptage des paramètres

```python
from torchinfo import summary

model = SimpleUNet(in_channels=3, num_classes=3, base_features=32)

# Nombre de paramètres
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paramètres entraînables : {total_params:,}")

# Architecture complète
summary(model, input_size=(1, 3, 256, 256))
```

**Dans le rapport**, notez le nombre de paramètres (environ 1.9M pour base_features=32) et commentez la taille par rapport aux modèles pré-entraînés.

---

### Q5 — Fonction de perte combinée

**Code :**

```python
def combined_loss(pred, target, alpha=0.5):
    ce   = nn.CrossEntropyLoss()(pred, target)
    d    = dice_loss(pred, target)
    return alpha * ce + (1 - alpha) * d
```

**Explication à rédiger dans le rapport :**
> *"La CrossEntropy traite chaque pixel indépendamment et est sensible aux déséquilibres de classes. La Dice Loss optimise directement le ratio de chevauchement (IoU) entre prédiction et vérité terrain, ce qui la rend plus robuste aux classes rares. En combinant les deux avec α=0.5, on bénéficie de la stabilité de la CrossEntropy en début d'entraînement et de la précision par classe de la Dice Loss."*

---

### Q6 — Entraînement 30 epochs

**Code d'entraînement complet :**

```python
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

best_val_loss   = float('inf')
patience_count  = 0
EARLY_STOP      = 5
train_losses, val_losses = [], []

for epoch in range(30):
    # --- TRAIN ---
    model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss  = combined_loss(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # --- VALIDATION ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            val_loss += combined_loss(preds, masks).item()
    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        patience_count = 0
        torch.save(model.state_dict(), 'best_unet.pth')
    else:
        patience_count += 1
        if patience_count >= EARLY_STOP:
            print(f"Early stopping à l'epoch {epoch+1}")
            break

    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# Courbes
plt.figure(figsize=(8,4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Courbes d\'apprentissage — SimpleUNet')
plt.legend()
plt.savefig('learning_curves_unet.png', dpi=150)
plt.show()
```

**Analyse à rédiger (très important) :**
> *"On observe un overfitting à partir de l'epoch ~15 : la val_loss remonte tandis que la train_loss continue de baisser. Solutions proposées : (1) augmentation de données plus agressive, (2) ajout de Dropout dans ConvBlock, (3) réduction de base_features de 32 à 16, (4) early stopping (déjà implémenté)."*

---

### Q7 — Évaluation sur le test set

**Code des métriques :**

```python
def compute_iou_dice(pred_mask, true_mask, num_classes=3):
    iou_per_class  = []
    dice_per_class = []
    for cls in range(num_classes):
        tp = ((pred_mask == cls) & (true_mask == cls)).sum().float()
        fp = ((pred_mask == cls) & (true_mask != cls)).sum().float()
        fn = ((pred_mask != cls) & (true_mask == cls)).sum().float()
        iou  = tp / (tp + fp + fn + 1e-6)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
        iou_per_class.append(iou.item())
        dice_per_class.append(dice.item())
    mIoU  = sum(iou_per_class)  / num_classes
    mDice = sum(dice_per_class) / num_classes
    return mIoU, mDice, iou_per_class, dice_per_class

# Matrice de confusion
from sklearn.metrics import confusion_matrix
import seaborn as sns

all_preds, all_true = [], []
model.load_state_dict(torch.load('best_unet.pth'))
model.eval()
with torch.no_grad():
    for imgs, masks in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu()
        all_preds.append(preds.flatten())
        all_true.append(masks.flatten())

all_preds = torch.cat(all_preds).numpy()
all_true  = torch.cat(all_true).numpy()

cm = confusion_matrix(all_true, all_preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['animal','contour','fond'],
            yticklabels=['animal','contour','fond'])
plt.title('Matrice de confusion — SimpleUNet')
plt.savefig('confusion_matrix_unet.png', dpi=150)
plt.show()
```

**Commentaire attendu dans le rapport :**
> *"La classe 'contour' est la plus difficile à prédire (IoU le plus bas) car elle est très fine et représente peu de pixels. Les erreurs les plus fréquentes sont la confusion entre 'contour' et 'fond' (pixels de bordure classifiés comme fond)."*

---

## Partie 3 — Transfer Learning

### Q8 — Compléter le forward() de TransferUNet

**Solution complète :**

```python
def forward(self, x):
    # Encodage
    e0 = self.enc0(x)           # (B, 64,  H/2,  W/2)
    e1 = self.enc1(self.pool(e0))  # (B, 256, H/4,  W/4)
    e2 = self.enc2(e1)          # (B, 512, H/8,  W/8)
    e3 = self.enc3(e2)          # (B, 1024,H/16, W/16)
    e4 = self.enc4(e3)          # (B, 2048,H/32, W/32)

    # Décodage avec skip connections
    d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))  # (B, 512, H/16, W/16)
    d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))  # (B, 256, H/8,  W/8)
    d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))  # (B, 64,  H/4,  W/4)
    d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))  # (B, 32,  H/2,  W/2)
    out = self.final_up(d1)                                # (B, 32,  H,    W)
    return self.out(out)                                   # (B, num_classes, H, W)
```

**Attention aux dimensions** — dans le rapport, faites un tableau des dimensions à chaque étape (les profs adorent ça).

---

### Q9 — Frozen vs Fine-tuning

**Code du fine-tuning différentiel :**

```python
model_frozen = TransferUNet(freeze_encoder=True).to(device)
model_ft     = TransferUNet(freeze_encoder=False).to(device)

# Optimizer pour le fine-tuning (lr différentiel)
encoder_params = (list(model_ft.enc0.parameters()) +
                  list(model_ft.enc1.parameters()) +
                  list(model_ft.enc2.parameters()) +
                  list(model_ft.enc3.parameters()) +
                  list(model_ft.enc4.parameters()))
decoder_params = (list(model_ft.up4.parameters())  +
                  list(model_ft.dec4.parameters()) +
                  list(model_ft.up3.parameters())  +
                  list(model_ft.dec3.parameters()) +
                  list(model_ft.up2.parameters())  +
                  list(model_ft.dec2.parameters()) +
                  list(model_ft.up1.parameters())  +
                  list(model_ft.dec1.parameters()) +
                  list(model_ft.final_up.parameters()) +
                  list(model_ft.out.parameters()))

optimizer_ft = torch.optim.Adam([
    {'params': encoder_params, 'lr': 1e-4, 'weight_decay': 1e-4},
    {'params': decoder_params, 'lr': 1e-3, 'weight_decay': 1e-4},
])
```

---

### Q10 — Tableau comparatif

**Dans le rapport, remplissez ce tableau avec vos vrais résultats :**

| Modèle | mIoU | mDice | Nb Params | Temps/epoch | Inférence |
|---|---|---|---|---|---|
| SimpleUNet | ~0.42 | ~0.55 | ~1.9M | ~45s | ~12ms |
| TransferUNet (frozen) | ~0.58 | ~0.70 | ~25M | ~35s | ~18ms |
| TransferUNet (fine-tuned) | ~0.65 | ~0.77 | ~25M | ~80s | ~18ms |

*(Les valeurs ci-dessus sont indicatives, remplacez par vos résultats réels)*

**Analyse à rédiger :**
> *"Le TransferUNet fine-tuné obtient les meilleures métriques grâce aux représentations riches apprises sur ImageNet. L'encodeur gelé converge plus vite car il entraîne moins de paramètres, mais atteint un plateau plus bas. Le SimpleUNet, malgré moins de paramètres, souffre d'un manque de capacité d'extraction de features. Pour un déploiement en production, le TransferUNet frozen offre le meilleur compromis temps/performance."*

---

# EXERCICE 2 — IMAGE CAPTIONING

---

## Partie 1 — Dataset COCO

### Q1 — Construction du vocabulaire

```python
# Construire le vocabulaire
vocab = Vocabulary(freq_threshold=5)

# Collecter toutes les captions
coco_train = COCO('annotations/captions_train2017.json')
all_captions = [[ann['caption'] for ann in coco_train.imgToAnns[img_id]]
                 for img_id in list(coco_train.imgs.keys())[:5000]]
vocab.build(all_captions)

# Histogramme des longueurs
lengths = [len(Vocabulary.tokenize(ann['caption']))
           for ann in coco_train.anns.values()]
plt.hist(lengths, bins=30, edgecolor='black')
plt.axvline(x=50, color='red', linestyle='--', label='max_len=50')
plt.xlabel('Longueur de la caption (tokens)')
plt.ylabel('Fréquence')
plt.title('Distribution des longueurs de captions')
plt.legend()
plt.savefig('caption_lengths.png', dpi=150)
plt.show()
print(f"95e percentile : {np.percentile(lengths, 95):.0f} tokens")
```

**Justification du max_len dans le rapport :**
> *"Le 95e percentile des longueurs est ~22 tokens. On choisit max_len=50 pour couvrir 99%+ des captions sans gaspiller trop de mémoire. Les tokens au-delà sont tronqués, ce qui est acceptable car les captions longues ont généralement des informations redondantes en fin de phrase."*

---

### Q2 — DataLoader avec collate_fn

```python
def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs, 0)
    # Longueurs réelles (compter les tokens avant padding)
    lengths = [(cap != 0).sum().item() for cap in caps]
    # Trier par longueur décroissante (requis par pack_padded_sequence)
    sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    imgs    = imgs[[sorted_idx]]
    caps    = torch.stack([caps[i] for i in sorted_idx])
    lengths = [lengths[i] for i in sorted_idx]
    return imgs, caps, torch.tensor(lengths)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          collate_fn=collate_fn, num_workers=2)
```

---

## Partie 2 — CNN-LSTM

### Q3 — Vérification des dimensions

**Dans le rapport, montrez ce tableau de vérification :**

```python
# Forward pass de vérification
encoder = ImageEncoder(embed_size=256).to(device)
decoder = CaptionDecoder(embed_size=256, hidden_size=512,
                          vocab_size=len(vocab), num_layers=2).to(device)

# Batch fictif
dummy_imgs = torch.randn(4, 3, 224, 224).to(device)
dummy_caps = torch.randint(0, len(vocab), (4, 50)).to(device)
dummy_lens = torch.tensor([50, 45, 40, 35])

features = encoder(dummy_imgs)
print(f"Encoder output : {features.shape}")   # [4, 256]

outputs = decoder(features, dummy_caps, dummy_lens)
print(f"Decoder output : {outputs.shape}")    # [4, 49, vocab_size]
```

| Étape | Dimension |
|---|---|
| Input images | (B=4, 3, 224, 224) |
| ResNet features | (B, 2048, 7, 7) |
| After AdaptiveAvgPool | (B, 2048, 1, 1) → (B, 2048) |
| After fc + BN | (B, 256) = embed_size |
| Embeddings (captions) | (B, 49, 256) |
| LSTM input (img+embed) | (B, 50, 256) |
| LSTM output | (B, 50, 512) |
| Final logits | (B, 50, vocab_size) |

---

### Q4 — Entraînement avec Teacher Forcing

```python
criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore le padding
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=3e-4)

perplexities = []

for epoch in range(20):
    encoder.train(); decoder.train()
    epoch_loss = 0
    for imgs, caps, lengths in train_loader:
        imgs, caps = imgs.to(device), caps.to(device)
        features = encoder(imgs)
        outputs  = decoder(features, caps, lengths)
        targets  = caps[:, 1:]
        T = min(outputs.size(1), targets.size(1))
        loss = criterion(outputs[:,:T].reshape(-1, outputs.size(-1)),
                         targets[:,:T].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss   = epoch_loss / len(train_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    perplexities.append(perplexity)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Perplexité: {perplexity:.2f}")

# Courbe de perplexité
plt.plot(perplexities)
plt.xlabel('Epoch'); plt.ylabel('Perplexité')
plt.title('Courbe de perplexité — CNN-LSTM')
plt.savefig('perplexity_curve.png', dpi=150)
```

**Explication du Teacher Forcing dans le rapport :**
> *"Le Teacher Forcing consiste à utiliser la vraie caption comme entrée du LSTM à chaque pas de temps, au lieu de réutiliser la prédiction précédente. Cela accélère la convergence mais crée un écart entre entraînement et inférence (exposure bias). En pratique, une perplexité de 10-20 sur le train set est un bon indicateur de convergence."*

---

### Q5 — Greedy vs Beam Search + BLEU-4

**Génération greedy :**

```python
def greedy_caption(encoder, decoder, image, vocab, max_len=50, device='cpu'):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        feature    = encoder(image.unsqueeze(0).to(device))
        inputs     = feature.unsqueeze(1)
        hidden     = None
        result     = []
        for _ in range(max_len):
            out, hidden = decoder.lstm(inputs, hidden)
            word_id = decoder.fc(out.squeeze(1)).argmax(dim=1).item()
            if word_id == vocab.stoi['<EOS>']:
                break
            result.append(vocab.itos.get(word_id, '<UNK>'))
            inputs = decoder.embed(torch.tensor([[word_id]]).to(device))
    return ' '.join(result)
```

**Calcul du BLEU-4 :**

```python
from nltk.translate.bleu_score import corpus_bleu

references, hypotheses = [], []
for imgs, caps, _ in test_loader:
    for i in range(len(imgs)):
        pred = greedy_caption(encoder, decoder, imgs[i], vocab, device=device)
        ref  = [vocab.itos.get(t.item(), '') for t in caps[i]
                if t.item() not in [0, 1, 2]]  # exclure PAD, SOS, EOS
        references.append([ref])
        hypotheses.append(pred.split())

bleu4 = corpus_bleu(references, hypotheses,
                     weights=(0.25, 0.25, 0.25, 0.25))
print(f"BLEU-4 (greedy) : {bleu4:.4f}")
```

**Commentaire dans le rapport :**
> *"Le Beam Search produit des captions plus fluides et grammaticalement correctes que le Greedy. Le Greedy tend à générer des phrases courtes et répétitives car il choisit toujours le mot le plus probable. Avec beam_size=3, le BLEU-4 augmente en général de 1-2 points."*

---

## Partie 3 — Attention Visuelle

### Q6 — ImageEncoder spatial + entraînement avec attention

**Modifier l'encoder pour retourner la feature map :**

```python
class ImageEncoderSpatial(nn.Module):
    def __init__(self, fine_tune=False):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)   # (B, 2048, 7, 7)
        for param in self.resnet[:6].parameters():
            param.requires_grad = fine_tune

    def forward(self, images):
        features = self.resnet(images)           # (B, 2048, 7, 7)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)  # (B, 7, 7, 2048)
        features = features.view(B, -1, C)        # (B, 49, 2048)
        return features
```

**Boucle d'entraînement avec attention :**

```python
encoder_att = ImageEncoderSpatial().to(device)
decoder_att = AttentionDecoder(embed_size=256, hidden_size=512,
                                vocab_size=len(vocab)).to(device)

for epoch in range(15):
    encoder_att.train(); decoder_att.train()
    for imgs, caps, lengths in train_loader:
        imgs, caps = imgs.to(device), caps.to(device)
        enc_out = encoder_att(imgs)                     # (B, 49, 2048)
        preds, alphas = decoder_att(enc_out, caps, lengths)
        targets = caps[:, 1:]
        T = min(preds.size(1), targets.size(1))
        loss = criterion(preds[:,:T].reshape(-1, preds.size(-1)),
                         targets[:,:T].reshape(-1))
        # Régularisation doubly stochastic (optionnelle mais recommandée)
        loss += 1.0 * ((1 - alphas.sum(dim=1)) ** 2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```

---

### Q7 — Visualisation des cartes d'attention

```python
import cv2

def visualize_attention(image, caption_words, alphas, filename='attention.png'):
    """
    image        : tensor (3, 224, 224) normalisé
    caption_words: liste de mots générés
    alphas       : tensor (num_words, 49)
    """
    # Dénormalisation
    img = image.permute(1,2,0).numpy()
    img = img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
    img = np.clip(img, 0, 1)

    n_words = min(len(caption_words), 12)
    fig, axes = plt.subplots(2, (n_words+1)//2 + 1, figsize=(20, 6))
    axes = axes.flatten()

    axes[0].imshow(img)
    axes[0].set_title("Image originale", fontsize=8)
    axes[0].axis('off')

    for i, (word, alpha) in enumerate(zip(caption_words[:n_words],
                                           alphas[:n_words])):
        # Reshape alpha (49,) → (7,7) puis interpoler à (224,224)
        att_map = alpha.reshape(7, 7).numpy()
        att_map = cv2.resize(att_map, (224, 224))
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)

        axes[i+1].imshow(img)
        axes[i+1].imshow(att_map, alpha=0.5, cmap='jet')
        axes[i+1].set_title(word, fontsize=10, fontweight='bold')
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
```

**Commentaire attendu dans le rapport :**
> *"On observe que l'attention se concentre sur les régions sémantiquement pertinentes : pour le mot 'dog', l'attention couvre la zone de l'animal ; pour 'running', elle se concentre sur les pattes. Cela valide que le mécanisme apprend effectivement à aligner les mots avec les régions visuelles correspondantes."*

---

### Q8 — Tableau bilan final

| Modèle | BLEU-1 | BLEU-4 | METEOR | CIDEr |
|---|---|---|---|---|
| CNN-LSTM (sans attention) | ~0.58 | ~0.18 | ~0.21 | ~0.52 |
| CNN-LSTM-Attention | ~0.65 | ~0.23 | ~0.24 | ~0.68 |

*(Valeurs indicatives — remplacez par vos résultats réels)*

**Discussion des limites des métriques automatiques :**
> *"Les métriques BLEU et METEOR mesurent la correspondance n-gramme avec des références humaines, mais présentent plusieurs limites : (1) elles pénalisent les synonymes valides ('chien' vs 'animal'), (2) elles ne capturent pas la cohérence globale de la description, (3) elles sont sensibles à l'ordre des mots. CIDEr est plus adapté au captioning car il pondère les mots rares. Cependant, aucune métrique automatique ne remplace une évaluation humaine pour juger la qualité et la fluidité des descriptions générées."*

---

# CHECKLIST DU RAPPORT FINAL

Avant de rendre, vérifiez que votre rapport contient :

**Exercice 1 :**
- [ ] 4 images avec masques côte à côte (Q1)
- [ ] Histogramme de distribution des classes (Q1)
- [ ] Exemples avant/après augmentation (Q2)
- [ ] Architecture complète du modèle (Q4)
- [ ] Courbes train/val loss (Q6)
- [ ] Matrice de confusion (Q7)
- [ ] Tableau comparatif des 3 modèles avec valeurs réelles (Q10)

**Exercice 2 :**
- [ ] Histogramme des longueurs de captions (Q1)
- [ ] Tableau des dimensions à chaque étape (Q3)
- [ ] Courbe de perplexité (Q4)
- [ ] Comparaison greedy vs beam search (Q5)
- [ ] Visualisation des cartes d'attention (Q7)
- [ ] Tableau bilan BLEU/METEOR/CIDEr (Q8)

**Pour chaque figure :** titre, légendes des axes, numéro de figure, caption descriptive.

**Règle d'or :** Chaque résultat doit être **commenté**. Un chiffre sans analyse = 0 point.

---

*Guide rédigé pour le TP Deep Learning — Master AIDC, FST Béni Mellal 2025–2026*
