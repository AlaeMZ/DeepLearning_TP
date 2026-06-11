import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import resnet34, ResNet34_Weights
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchinfo import summary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du device : {device}")

# ==========================================
# CONFIGURATION GLOBALE
# ==========================================
EPOCHS      = 30
BATCH_SIZE  = 8
EARLY_STOP  = 5
NUM_WORKERS = 0   # Mettez 2 ou 4 si vous etes sur Linux/Mac

# ==========================================
# PARTIE 1 : DATASET ET AUGMENTATION
# ==========================================

class SyncAugmentation:
    """Augmentation geometrique synchronisee image+masque."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() < self.p:
            angle = random.uniform(-15, 15)
            img  = TF.rotate(img, angle)
            # NEAREST obligatoire : evite les valeurs de classes fractionnaires
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        if random.random() < self.p:
            img = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)(img)
        return img, mask


class PetSegmentationDataset(Dataset):
    def __init__(self, root, apply_aug=False):
        os.makedirs(root, exist_ok=True)
        self.dataset = OxfordIIITPet(
            root=root, split='trainval', target_types='segmentation', download=True
        )
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256),
                              interpolation=transforms.InterpolationMode.NEAREST),
        ])
        self.sync_aug = SyncAugmentation(p=0.5) if apply_aug else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        img  = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        # Mapping torchvision -> guide:
        # torchvision: 1=animal, 2=fond, 3=contour
        # guide:       0=animal, 1=contour, 2=fond
        mask_mapped = torch.zeros_like(mask)
        mask_mapped[mask == 1] = 0
        mask_mapped[mask == 3] = 1
        mask_mapped[mask == 2] = 2

        if self.sync_aug:
            mask_mapped = mask_mapped.unsqueeze(0)
            img, mask_mapped = self.sync_aug(img, mask_mapped)
            mask_mapped = mask_mapped.squeeze(0)

        return img, mask_mapped


# ==========================================
# PARTIE 2 : MODELES
# ==========================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)


class SimpleUNet(nn.Module):
    """U-Net leger construit from scratch."""
    def __init__(self, in_channels=3, num_classes=3, base_features=32):
        super().__init__()
        f = base_features
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f,   f*2)
        self.enc3 = ConvBlock(f*2, f*4)
        self.enc4 = ConvBlock(f*4, f*8)
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(f*8 + f*4, f*4)
        self.dec2 = ConvBlock(f*4 + f*2, f*2)
        self.dec1 = ConvBlock(f*2 + f,   f)
        self.out  = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.out(d1)


class TransferUNet(nn.Module):
    """
    U-Net avec encodeur ResNet-34 pre-entraine.
    freeze_encoder=True  -> encodeur gele (frozen)
    freeze_encoder=False -> fine-tuning complet
    """
    def __init__(self, num_classes=3, freeze_encoder=True):
        super().__init__()
        # Essai de chargement des poids pre-entraines (necessite internet)
        try:
            resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            print("  Poids ImageNet charges avec succes.")
        except Exception as e:
            print(f"  AVERTISSEMENT: telechargement des poids impossible ({type(e).__name__}).")
            print("  Utilisation d'un encodeur aleatoire (TransferUNet sans pre-entrainement).")
            resnet = resnet34(weights=None)

        # Encodeur : couches ResNet-34
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 ch, /2
        self.pool = resnet.maxpool                                           # /4
        self.enc1 = resnet.layer1   # 64 ch,  /4
        self.enc2 = resnet.layer2   # 128 ch, /8
        self.enc3 = resnet.layer3   # 256 ch, /16
        self.enc4 = resnet.layer4   # 512 ch, /32

        if freeze_encoder:
            for p in list(self.enc0.parameters()) + \
                     list(self.enc1.parameters()) + \
                     list(self.enc2.parameters()) + \
                     list(self.enc3.parameters()) + \
                     list(self.enc4.parameters()):
                p.requires_grad = False

        # Decodeur
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(256 + 256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 32)

        self.final_up = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        # Encodage
        e0 = self.enc0(x)           # (B, 64,  H/2,  W/2)
        e1 = self.enc1(self.pool(e0))  # (B, 64,  H/4,  W/4)
        e2 = self.enc2(e1)          # (B, 128, H/8,  W/8)
        e3 = self.enc3(e2)          # (B, 256, H/16, W/16)
        e4 = self.enc4(e3)          # (B, 512, H/32, W/32)

        # Decodage avec skip connections
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))
        out = self.final_up(d1)
        return self.out(out)


# ==========================================
# PARTIE 3 : LOSS ET METRIQUES
# ==========================================

def dice_loss(pred, target, smooth=1e-6):
    pred = F.softmax(pred, dim=1)
    target_oh = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    intersection = (pred * target_oh).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_loss(pred, target, alpha=0.5):
    ce = nn.CrossEntropyLoss()(pred, target)
    d  = dice_loss(pred, target)
    return alpha * ce + (1 - alpha) * d


def compute_iou_dice(pred_mask, true_mask, num_classes=3):
    iou_per_class, dice_per_class = [], []
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


# ==========================================
# PARTIE 4 : FONCTIONS D'ENTRAINEMENT / EVALUATION
# ==========================================

def train_model(model, train_loader, val_loader, model_name, epochs=EPOCHS):
    """Entraine un modele et sauvegarde les courbes + le meilleur checkpoint."""
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    # Pour le fine-tuning, on utilise un lr differentiel
    if hasattr(model, 'enc0') and model_name == 'TransferUNet_FineTuned':
        enc_params = (list(model.enc0.parameters()) +
                      list(model.enc1.parameters()) +
                      list(model.enc2.parameters()) +
                      list(model.enc3.parameters()) +
                      list(model.enc4.parameters()))
        dec_params = (list(model.up4.parameters())  +
                      list(model.dec4.parameters()) +
                      list(model.up3.parameters())  +
                      list(model.dec3.parameters()) +
                      list(model.up2.parameters())  +
                      list(model.dec2.parameters()) +
                      list(model.up1.parameters())  +
                      list(model.dec1.parameters()) +
                      list(model.final_up.parameters()) +
                      list(model.out.parameters()))
        optimizer = torch.optim.Adam([
            {'params': enc_params, 'lr': 1e-4, 'weight_decay': 1e-4},
            {'params': dec_params, 'lr': 1e-3, 'weight_decay': 1e-4},
        ])

    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_val_loss = float('inf')
    patience_cnt  = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        t0 = time.time()

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
                val_loss += combined_loss(model(imgs), masks).item()
        val_loss /= len(val_loader)

        epoch_time = time.time() - t0
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), f'best_{model_name}.pth')
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOP:
                print(f"  Early stopping a l'epoch {epoch+1}")
                break

        print(f"  Epoch {epoch+1:02d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {epoch_time:.1f}s/epoch")

    # Courbes
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Courbes d'apprentissage — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'learning_curves_{model_name}.png', dpi=150)
    plt.close()
    print(f"  --> Courbes sauvegardees : learning_curves_{model_name}.png")
    return best_val_loss


def evaluate_model(model, model_name, test_loader):
    """Calcule mIoU, mDice et genere la matrice de confusion."""
    model.load_state_dict(torch.load(f'best_{model_name}.pth', map_location=device))
    model.eval()

    all_preds, all_true = [], []
    infer_times = []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            t0   = time.time()
            preds = model(imgs).argmax(dim=1).cpu()
            infer_times.append((time.time() - t0) / imgs.size(0) * 1000)  # ms/image
            all_preds.append(preds.flatten())
            all_true.append(masks.flatten())

    all_preds = torch.cat(all_preds).numpy()
    all_true  = torch.cat(all_true).numpy()

    mIoU, mDice, iou_cls, dice_cls = compute_iou_dice(
        torch.tensor(all_preds), torch.tensor(all_true)
    )
    avg_infer = np.mean(infer_times)

    print(f"  mIoU  = {mIoU:.4f}")
    print(f"  mDice = {mDice:.4f}")
    print(f"  IoU par classe : animal={iou_cls[0]:.3f}, contour={iou_cls[1]:.3f}, fond={iou_cls[2]:.3f}")
    print(f"  Dice par classe : animal={dice_cls[0]:.3f}, contour={dice_cls[1]:.3f}, fond={dice_cls[2]:.3f}")
    print(f"  Inference moyenne : {avg_infer:.2f} ms/image")

    # Matrice de confusion
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['animal', 'contour', 'fond'],
                yticklabels=['animal', 'contour', 'fond'])
    plt.title(f'Matrice de confusion — {model_name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=150)
    plt.close()
    print(f"  --> Matrice sauvegardee : confusion_matrix_{model_name}.png")

    return mIoU, mDice, avg_infer, iou_cls, dice_cls


# ==========================================
# EXECUTION PRINCIPALE
# ==========================================
if __name__ == "__main__":

    # ---- Q1 : Chargement et visualisation ----
    print("\n=== CHARGEMENT DU DATASET ===")
    full_aug    = PetSegmentationDataset(root='./oxford-iiit-pet', apply_aug=True)
    full_noaug  = PetSegmentationDataset(root='./oxford-iiit-pet', apply_aug=False)

    # Visualisation 4 images
    print("Generation de visualisation_dataset.png...")
    fig, axes = plt.subplots(4, 2, figsize=(8, 16))
    for i in range(4):
        img, mask = full_noaug[i]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=2)
        axes[i, 1].set_title("Masque (0=animal, 1=contour, 2=fond)")
    plt.tight_layout()
    plt.savefig("visualisation_dataset.png", dpi=150)
    plt.close()

    # Distribution des classes
    print("Calcul de la distribution (sur 500 images)...")
    total_pixels = {'animal': 0, 'contour': 0, 'fond': 0}
    for i in range(min(500, len(full_noaug))):
        _, mask = full_noaug[i]
        total_pixels['animal']  += (mask == 0).sum().item()
        total_pixels['contour'] += (mask == 1).sum().item()
        total_pixels['fond']    += (mask == 2).sum().item()
    total = sum(total_pixels.values())
    for k, v in total_pixels.items():
        print(f"  {k}: {v/total*100:.1f}%")

    # ---- Q3 : DataLoaders 70/15/15 ----
    print("\n=== CREATION DES DATALOADERS ===")
    n       = len(full_noaug)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val
    print(f"  Total: {n} | Train: {n_train} | Val: {n_val} | Test: {n_test}")

    g = torch.Generator().manual_seed(42)
    train_idx, val_idx, test_idx = random_split(range(n), [n_train, n_val, n_test], generator=g)

    from torch.utils.data import Subset
    train_set = Subset(full_aug,   list(train_idx))
    val_set   = Subset(full_noaug, list(val_idx))
    test_set  = Subset(full_noaug, list(test_idx))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    results = {}

    def run_model(model, model_name, total_params):
        """Entraine le modele si pas de checkpoint, sinon charge et evalue directement."""
        checkpoint_path = f'best_{model_name}.pth'
        if os.path.exists(checkpoint_path):
            print(f"  [SKIP] Checkpoint trouve : {checkpoint_path} — Passage direct a l'evaluation.")
            t_ep = 0.0  # temps inconnu
        else:
            t0 = time.time()
            train_model(model, train_loader, val_loader, model_name, epochs=EPOCHS)
            t_ep = (time.time() - t0) / EPOCHS

        print(f"  Evaluation {model_name} sur le test set...")
        mIoU, mDice, infer, iou_cls, dice_cls = evaluate_model(model, model_name, test_loader)
        return {'mIoU': mIoU, 'mDice': mDice, 'params': total_params,
                'epoch_time': t_ep, 'infer_ms': infer, 'iou_cls': iou_cls, 'dice_cls': dice_cls}

    # ========================================
    # Q4-Q7 : SIMPLEUNET
    # ========================================
    print("\n=== MODELE 1 : SimpleUNet ===")
    model_unet   = SimpleUNet(in_channels=3, num_classes=3, base_features=32).to(device)
    total_params = sum(p.numel() for p in model_unet.parameters() if p.requires_grad)
    print(f"  Parametres entrainables : {total_params:,}")
    results['SimpleUNet'] = run_model(model_unet, 'SimpleUNet', total_params)

    # ========================================
    # Q8-Q9 : TRANSFERUNET FROZEN
    # ========================================
    print("\n=== MODELE 2 : TransferUNet (Frozen) ===")
    model_frozen   = TransferUNet(num_classes=3, freeze_encoder=True).to(device)
    total_params_f = sum(p.numel() for p in model_frozen.parameters())
    trainable_f    = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    print(f"  Total params: {total_params_f:,} | Entrainables: {trainable_f:,}")
    results['TransferUNet_Frozen'] = run_model(model_frozen, 'TransferUNet_Frozen', total_params_f)

    # ========================================
    # Q9 : TRANSFERUNET FINE-TUNED
    # ========================================
    print("\n=== MODELE 3 : TransferUNet (Fine-Tuned) ===")
    model_ft        = TransferUNet(num_classes=3, freeze_encoder=False).to(device)
    total_params_ft = sum(p.numel() for p in model_ft.parameters())
    trainable_ft    = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print(f"  Total params: {total_params_ft:,} | Entrainables: {trainable_ft:,}")
    results['TransferUNet_FineTuned'] = run_model(model_ft, 'TransferUNet_FineTuned', total_params_ft)

    # ========================================
    # EXTRA VISUALIZATIONS FOR GAPS
    # ========================================
    print("\n=== GENERATION DES EXTRA VISUALISATIONS ===")
    
    # 1. Augmentation Comparison
    print("Generation de augmentation_comparison.png...")
    class ForceAugmentation:
        def __call__(self, img, mask):
            img, mask = TF.hflip(img), TF.hflip(mask)
            img = TF.rotate(img, 15)
            mask = TF.rotate(mask, 15, interpolation=TF.InterpolationMode.NEAREST)
            img = transforms.ColorJitter(brightness=0.4, contrast=0.4)(img)
            return img, mask
            
    force_aug = ForceAugmentation()
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    indices = [0, 10, 20]
    for i, idx in enumerate(indices):
        img_noaug, mask_noaug = full_noaug[idx]
        img_np = img_noaug.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(mask_noaug.numpy(), cmap='tab10', vmin=0, vmax=2)
        axes[i, 1].set_title("Masque Original")
        axes[i, 1].axis('off')

        # Augmentation forcée
        raw_img, raw_mask = full_noaug.dataset[idx]
        img_t = full_noaug.img_transform(raw_img)
        mask_t = full_noaug.mask_transform(raw_mask)
        mask_t = torch.as_tensor(np.array(mask_t), dtype=torch.long)
        mask_mapped = torch.zeros_like(mask_t)
        mask_mapped[mask_t == 1] = 0
        mask_mapped[mask_t == 3] = 1
        mask_mapped[mask_t == 2] = 2

        img_aug, mask_aug = force_aug(img_t, mask_mapped.unsqueeze(0))
        mask_aug = mask_aug.squeeze(0)

        img_aug_np = img_aug.permute(1, 2, 0).numpy()
        img_aug_np = img_aug_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_aug_np = np.clip(img_aug_np, 0, 1)

        axes[i, 2].imshow(img_aug_np)
        axes[i, 2].set_title("Augmente")
        axes[i, 2].axis('off')
        axes[i, 3].imshow(mask_aug.numpy(), cmap='tab10', vmin=0, vmax=2)
        axes[i, 3].set_title("Masque Augmente")
        axes[i, 3].axis('off')
    plt.tight_layout()
    plt.savefig("augmentation_comparison.png", dpi=150)
    plt.close()

    # 2. 6 Predictions Comparison
    print("Generation de predictions_comparison.png...")
    models = {
        'SimpleUNet': model_unet,
        'TransferUNet_Frozen': model_frozen,
        'TransferUNet_FineTuned': model_ft
    }
    fig, axes = plt.subplots(6, 5, figsize=(15, 18))
    pred_indices = [5, 12, 18, 25, 30, 42]
    for i, idx in enumerate(pred_indices):
        img, mask = test_set[idx]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        axes[i, 0].imshow(img_np)
        if i == 0: axes[i, 0].set_title("Image Originale")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=2)
        if i == 0: axes[i, 1].set_title("Vérité Terrain (GT)")
        axes[i, 1].axis('off')

        img_batch = img.unsqueeze(0).to(device)
        for col_idx, (m_name, model) in enumerate(models.items()):
            model.eval()
            with torch.no_grad():
                pred = model(img_batch).argmax(dim=1).squeeze(0).cpu().numpy()
            axes[i, 2 + col_idx].imshow(pred, cmap='tab10', vmin=0, vmax=2)
            if i == 0: axes[i, 2 + col_idx].set_title(m_name)
            axes[i, 2 + col_idx].axis('off')
    plt.tight_layout()
    plt.savefig("predictions_comparison.png", dpi=150)
    plt.close()

    # ========================================
    # Q10 : TABLEAU COMPARATIF FINAL
    # ========================================
    print("\n" + "="*70)
    print("=== TABLEAU COMPARATIF FINAL (Q10) ===")
    print("="*70)
    print(f"{'Modele':<28} {'mIoU':>6} {'mDice':>7} {'Params':>10} {'Infer':>8}")
    print("-"*70)
    for name, r in results.items():
        print(f"{name:<28} {r['mIoU']:>6.4f} {r['mDice']:>7.4f} "
              f"{r['params']:>10,} {r['infer_ms']:>7.2f}ms")
    print("="*70)

    # Print detailed tables for copy-pasting
    print("\n=== DETAILED METRICS BY CLASS FOR LATEX ===")
    print("Model | Class | IoU | Dice")
    print("-" * 40)
    for name, r in results.items():
        classes = ['animal', 'contour', 'fond']
        for c_idx, cls_name in enumerate(classes):
            print(f"{name} | {cls_name} | {r['iou_cls'][c_idx]:.4f} | {r['dice_cls'][c_idx]:.4f}")
        print("-" * 40)

    print("\nTous les fichiers ont ete generes dans le dossier courant.")
    print("Exercice 1 TERMINE !")
