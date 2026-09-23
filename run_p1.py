import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.show = lambda *a, **k: None
import sys
IN_COLAB = False

# ---- cell 5 ----
import os, glob, math, copy, time, random
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE, '| torch', torch.__version__)

# ---- cell 6 ----
@dataclass
class Config:
    # paths
    data_root: str = '/content' if IN_COLAB else os.environ.get('HV_DATA_ROOT', 'HV-AI-2025')   # relative to the repo root
    test_dir: str = ''                 # '' -> auto-discover (see find_test_dir)
    out_dir: str = '/content/outputs' if IN_COLAB else os.environ.get('HV_OUT_DIR', 'outputs')
    # data
    img_size: int = 224
    cache_short_side: int = 256        # images are decoded once and cached at this short side
    val_frac: float = 0.2
    repeats: int = 3                   # fresh augmentations of each image per epoch ("more samples")
    vflip_p: float = 0.2               # vertical-flip augmentation (robustness to unknown test orientation)
    rot_deg: float = 20.0
    # training
    epochs: int = 40
    warmup_epochs: int = 2
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    dropout: float = 0.2
    label_smoothing: float = 0.1
    loss_mode: str = 'weighted'        # 'weighted' (class-weighted CE) | 'logit_adjust' (plain CE + post-hoc τ·log prior)
    weight_scheme: str = 'inv_freq'    # 'inv_freq' | 'effective' (Cui et al., beta=0.999)
    mix_prob: float = 0.5              # probability a batch is mixed (MixUp or CutMix, 50/50)
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    ema_decay: float = 0.995
    amp: bool = True
    num_workers: int = 8
    seed: int = 42
    retrain_full: bool = True          # final Phase 1 model trained on 100% of labeled data

CFG = Config()
CFG.epochs = int(os.environ.get('HV_EPOCHS', CFG.epochs))
CFG.repeats = int(os.environ.get('HV_REPEATS', CFG.repeats))
CFG.retrain_full = os.environ.get('HV_RETRAIN_FULL', '1') == '1'
CFG.test_dir = os.environ.get('HV_TEST_DIR', CFG.test_dir)
PREDICT_ONLY = os.environ.get('HV_PREDICT_ONLY', '0') == '1'   # load saved checkpoint, skip training
SKIP_VAL = os.environ.get('HV_SKIP_VAL', '0') == '1'           # skip the 80/20 val run (parallel full retrain)
TRAIN_ONLY = os.environ.get('HV_TRAIN_ONLY', '0') == '1'       # stop after saving the checkpoint
print(asdict(CFG))
os.makedirs(CFG.out_dir, exist_ok=True)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(CFG.seed)

# ---- cell 8 ----
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def load_image(path, vflip=False, short_side=CFG.cache_short_side):
    """Open an image as canonical grayscale, upright, with the short side capped at `short_side`."""
    img = Image.open(path)
    img = img.convert('L')                      # standardize color -> grayscale (labeled set is 100% grayscale)
    if vflip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    w, h = img.size
    s = short_side / min(w, h)
    if s < 1:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    return img

def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))

def load_folder(folder, names, vflip=False):
    return [load_image(os.path.join(folder, n), vflip=vflip) for n in names]

# ---- cell 9 ----
LABELED_DIR = os.path.join(CFG.data_root, 'labeled_data')
labels_df = pd.read_csv(os.path.join(LABELED_DIR, 'labeled_data.csv'))   # columns: img_name, label

CLASSES = sorted(labels_df['label'].unique())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
labels_df['y'] = labels_df['label'].map(CLASS_TO_IDX)

t0 = time.time()
labeled_images = load_folder(os.path.join(LABELED_DIR, 'images'), labels_df['img_name'])
labeled_y = labels_df['y'].to_numpy()
print(f'loaded {len(labeled_images)} labeled images in {time.time()-t0:.1f}s')

class_counts = np.bincount(labeled_y, minlength=NUM_CLASSES)
print(pd.DataFrame({'class': CLASSES, 'count': class_counts}).set_index('class').T)

# ---- cell 10 ----
fig, ax = plt.subplots(figsize=(9, 3))
ax.bar(CLASSES, class_counts, color='#4a7ab5')
ax.set_title('Labeled class distribution'); ax.tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.show()

# ---- cell 12 ----
idx_all = np.arange(len(labeled_images))
idx_tr, idx_val = train_test_split(idx_all, test_size=CFG.val_frac, stratify=labeled_y, random_state=CFG.seed)
print('train', len(idx_tr), '| val', len(idx_val))
print('val per class:', np.bincount(labeled_y[idx_val], minlength=NUM_CLASSES))

# ---- cell 14 ----
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def to_tensor_3ch():
    return [T.Grayscale(num_output_channels=3), T.ToTensor(), T.Normalize(MEAN, STD)]

def build_transforms(cfg):
    size = cfg.img_size
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.35, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=cfg.vflip_p),
        T.RandomApply([T.RandomRotation(cfg.rot_deg, fill=0)], p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
        T.RandomAutocontrast(p=0.2),
        *to_tensor_3ch(),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    weak = T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        *to_tensor_3ch(),
    ])
    strong = T.Compose([
        T.RandomResizedCrop(size, scale=(0.35, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=10),
        *to_tensor_3ch(),
        T.RandomErasing(p=0.5, scale=(0.02, 0.25)),
    ])
    center = [T.Resize(size), T.CenterCrop(size)]
    full = [T.Resize((size, size))]
    eval_views = {
        'center':       T.Compose(center + to_tensor_3ch()),
        'center_hflip': T.Compose(center + [T.RandomHorizontalFlip(p=1.0)] + to_tensor_3ch()),
        'full':         T.Compose(full + to_tensor_3ch()),
        'full_hflip':   T.Compose(full + [T.RandomHorizontalFlip(p=1.0)] + to_tensor_3ch()),
    }
    return {'train': train, 'weak': weak, 'strong': strong, 'eval': eval_views['center'], 'eval_views': eval_views}

TFMS = build_transforms(CFG)

# ---- cell 15 ----
class ImageDataset(Dataset):
    """In-memory PIL images; `repeats` makes one epoch show each image several times with fresh augmentations.
    Unlabeled samples get label -1 so labeled/unlabeled batches share one collate path."""
    def __init__(self, images, labels=None, transform=None, repeats=1):
        self.images, self.labels, self.transform, self.repeats = images, labels, transform, repeats

    def __len__(self):
        return len(self.images) * self.repeats

    def __getitem__(self, i):
        i %= len(self.images)
        x = self.transform(self.images[i])
        y = -1 if self.labels is None else int(self.labels[i])
        return x, y

def subset(seq, idx):
    return [seq[i] for i in idx]

def make_loader(images, labels, transform, cfg, train, repeats=1):
    ds = ImageDataset(images, labels, transform, repeats=repeats)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, drop_last=train,
                      num_workers=cfg.num_workers, pin_memory=DEVICE.type == 'cuda',
                      persistent_workers=cfg.num_workers > 0)

# ---- cell 16 ----
def denorm(x):
    return (x * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).clamp(0, 1)

def show_augmentations(img, tfm, n=8, title=''):
    fig, axes = plt.subplots(1, n + 1, figsize=(2 * (n + 1), 2.2))
    axes[0].imshow(img, cmap='gray'); axes[0].set_title('original')
    for ax in axes[1:]:
        ax.imshow(denorm(tfm(img)).permute(1, 2, 0))
    for ax in axes: ax.axis('off')
    fig.suptitle(title); plt.tight_layout(); plt.show()

for k in [0, 300, 600]:
    show_augmentations(labeled_images[k], TFMS['train'], title=f'train aug — {labels_df.label[k]}')

# ---- cell 18 ----
def build_model(num_classes=NUM_CLASSES, dropout=CFG.dropout, pretrained=True):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model

def param_groups(model, cfg):
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [{'params': backbone, 'lr': cfg.lr_backbone}, {'params': head, 'lr': cfg.lr_head}]

def class_weights(counts, scheme='inv_freq', beta=0.999):
    counts = torch.as_tensor(counts, dtype=torch.float)
    if scheme == 'inv_freq':
        w = 1.0 / counts
    elif scheme == 'effective':                      # Cui et al. 2019, class-balanced loss
        w = (1 - beta) / (1 - beta ** counts)
    else:
        w = torch.ones_like(counts)
    return w / w.mean()

def build_criterion(cfg, counts):
    weight = class_weights(counts, cfg.weight_scheme) if cfg.loss_mode == 'weighted' else None
    # logit_adjust -> plain (unweighted) CE; the prior correction happens at inference only
    return nn.CrossEntropyLoss(weight=None if weight is None else weight.to(DEVICE),
                               label_smoothing=cfg.label_smoothing)

# ---- cell 19 ----
def mix_batch(x, y, cfg):
    """MixUp or CutMix (50/50) with probability cfg.mix_prob. Returns x, y_a, y_b, lam;
    loss = lam*CE(y_a) + (1-lam)*CE(y_b), exact because CE is linear in the target."""
    if random.random() >= cfg.mix_prob:
        return x, y, y, 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    if random.random() < 0.5:
        lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
        x = lam * x + (1 - lam) * x[perm]
    else:
        lam = float(np.random.beta(cfg.cutmix_alpha, cfg.cutmix_alpha))
        H, W = x.shape[2:]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)       # correct lam for the clipped box
    return x, y, y[perm], lam

class ModelEMA:
    """Exponential moving average of weights (BN buffers included); used for evaluation and pseudo-labeling."""
    def __init__(self, model, decay):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))   # warm up the average early on
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

def warmup_cosine(optimizer, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

# ---- cell 21 ----
@torch.no_grad()
def predict_logits(model, images, transform, cfg):
    model.eval()
    loader = make_loader(images, None, transform, cfg, train=False)
    out = []
    for x, _ in loader:
        with torch.autocast(DEVICE.type, enabled=cfg.amp and DEVICE.type == 'cuda'):
            out.append(model(x.to(DEVICE, non_blocking=True)).float().cpu())
    return torch.cat(out)

def predict_tta(model, images, cfg, views=('center', 'center_hflip', 'full', 'full_hflip')):
    """Average softmax over TTA views; returns log-probs so logit adjustment still applies."""
    probs = torch.stack([predict_logits(model, images, TFMS['eval_views'][v], cfg).softmax(1) for v in views]).mean(0)
    return probs.clamp_min(1e-8).log()

def adjust_logits(logits, prior, tau):
    return logits - tau * torch.log(torch.as_tensor(prior, dtype=torch.float)) if tau else logits

def compute_metrics(logits, y):
    pred = logits.argmax(1).numpy(); y = np.asarray(y)
    per_class = np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(NUM_CLASSES)])
    return {'acc': float((pred == y).mean()), 'macro_acc': float(np.nanmean(per_class)), 'per_class': per_class}

def tune_tau(logits, y, prior, grid=np.linspace(0, 2, 21)):
    scores = [(compute_metrics(adjust_logits(logits, prior, t), y)['macro_acc'], t) for t in grid]
    best = max(scores)
    return best[1], best[0]

# ---- cell 22 ----
def train_one_epoch(model, ema, loader, criterion, optimizer, scheduler, scaler, cfg):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        x, ya, yb, lam = mix_batch(x, y, cfg)
        with torch.autocast(DEVICE.type, enabled=cfg.amp and DEVICE.type == 'cuda'):
            logits = model(x)
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update(); scheduler.step()
        ema.update(model)
        total += loss.item() * x.size(0); n += x.size(0)
    return total / n

def fit(train_imgs, train_y, cfg, val_imgs=None, val_y=None, tag='run'):
    """Train ResNet-18 on (train_imgs, train_y). If val data is given, track EMA macro-acc each epoch
    and keep the best EMA weights; otherwise keep the final EMA weights."""
    seed_everything(cfg.seed)
    model = build_model().to(DEVICE)
    criterion = build_criterion(cfg, np.bincount(train_y, minlength=NUM_CLASSES))
    optimizer = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg.weight_decay)
    loader = make_loader(train_imgs, train_y, TFMS['train'], cfg, train=True, repeats=cfg.repeats)
    steps = cfg.epochs * len(loader)
    scheduler = warmup_cosine(optimizer, cfg.warmup_epochs * len(loader), steps)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=cfg.amp and DEVICE.type == 'cuda')
    ema = ModelEMA(model, cfg.ema_decay)

    history, best = [], (-1.0, None, -1)
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, ema, loader, criterion, optimizer, scheduler, scaler, cfg)
        row = {'epoch': epoch, 'loss': loss}
        if val_imgs is not None:
            m = compute_metrics(predict_logits(ema.ema, val_imgs, TFMS['eval'], cfg), val_y)
            row.update(val_acc=m['acc'], val_macro=m['macro_acc'])
            if m['macro_acc'] > best[0]:
                best = (m['macro_acc'], copy.deepcopy(ema.ema.state_dict()), epoch)
        history.append(row)
        print(f"[{tag}] ep {epoch:3d}/{cfg.epochs} loss {loss:.4f}"
              + (f" | val acc {row['val_acc']:.4f} macro {row['val_macro']:.4f}" if val_imgs is not None else '')
              + f" | {time.time()-t0:.1f}s")

    final = ema.ema
    if best[1] is not None:
        final.load_state_dict(best[1])
        print(f'[{tag}] best EMA macro-acc {best[0]:.4f} @ epoch {best[2]}')
    return final, pd.DataFrame(history)

# ---- cell 23 ----
def plot_history(hist, title=''):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].plot(hist.epoch, hist.loss); ax[0].set_title('train loss')
    if 'val_macro' in hist:
        ax[1].plot(hist.epoch, hist.val_acc, label='acc'); ax[1].plot(hist.epoch, hist.val_macro, label='macro acc')
        ax[1].legend(); ax[1].set_title('val (EMA)')
    fig.suptitle(title); plt.tight_layout(); plt.show()

def report(logits, y, title=''):
    m = compute_metrics(logits, y)
    print(f"{title} acc {m['acc']:.4f} | macro acc {m['macro_acc']:.4f}")
    print(pd.Series(m['per_class'], index=CLASSES).round(3).to_string())
    cm = confusion_matrix(y, logits.argmax(1).numpy(), labels=range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=60); ax.set_yticks(range(NUM_CLASSES), CLASSES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8, color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('pred'); ax.set_ylabel('true'); ax.set_title(title); plt.tight_layout(); plt.show()
    return m

# ---- cell 25 ----
tr_imgs, tr_y = subset(labeled_images, idx_tr), labeled_y[idx_tr]
val_imgs, val_y = subset(labeled_images, idx_val), labeled_y[idx_val]

if not (PREDICT_ONLY or SKIP_VAL):
    model_val, hist = fit(tr_imgs, tr_y, CFG, val_imgs, val_y, tag='p1-val')
    plot_history(hist, 'Phase 1 (80% train)')

# ---- cell 26 ----
prior = np.bincount(tr_y, minlength=NUM_CLASSES) / len(tr_y)
TAU = 0.0
if not (PREDICT_ONLY or SKIP_VAL):
  val_logits = predict_tta(model_val, val_imgs, CFG)
  if CFG.loss_mode == 'logit_adjust':
    TAU, macro = tune_tau(val_logits, val_y, prior)
    print(f'tuned tau = {TAU:.2f} (val macro acc {macro:.4f})')
  report(adjust_logits(val_logits, prior, TAU), val_y, 'Phase 1 val (TTA)')

# ---- cell 28 ----
if PREDICT_ONLY:
    ckpt = torch.load(os.path.join(CFG.out_dir, 'phase1_resnet18.pt'), map_location=DEVICE, weights_only=False)
    model_p1 = build_model(pretrained=False).to(DEVICE); model_p1.load_state_dict(ckpt['state_dict']); TAU = ckpt['tau']
    print('loaded', os.path.join(CFG.out_dir, 'phase1_resnet18.pt'))
else:
    if CFG.retrain_full:
        model_p1, hist_full = fit(labeled_images, labeled_y, CFG, tag='p1-full')
    else:
        model_p1 = model_val
    torch.save({'state_dict': model_p1.state_dict(), 'classes': CLASSES, 'tau': TAU, 'cfg': asdict(CFG)},
               os.path.join(CFG.out_dir, 'phase1_resnet18.pt'))
    print('saved', os.path.join(CFG.out_dir, 'phase1_resnet18.pt'))
if TRAIN_ONLY:
    sys.exit(0)

# ---- cell 30 ----
def find_test_dir(cfg):
    if cfg.test_dir:
        return cfg.test_dir
    roots = [cfg.data_root, '.', '/content']
    cands = [d for r in roots for d in glob.glob(os.path.join(r, '**', '*test*'), recursive=True) if os.path.isdir(d)]
    cands += [os.path.join(d, 'images') for d in cands]
    cands = [d for d in dict.fromkeys(cands) if os.path.isdir(d) and list_images(d)]
    if not cands:
        raise FileNotFoundError('test images not found; set CFG.test_dir')
    return max(cands, key=lambda d: len(list_images(d)))

@torch.no_grad()
def detect_vflip(model, folder, names, cfg, n=300):
    """True if the (sampled) images look upside down to the model, i.e. flipped copies get higher confidence."""
    sample = random.Random(cfg.seed).sample(list(names), min(n, len(names)))
    conf = {}
    for flip in (False, True):
        imgs = load_folder(folder, sample, vflip=flip)
        conf[flip] = predict_logits(model, imgs, TFMS['eval'], cfg).softmax(1).max(1).values.mean().item()
    print(f'{folder}: mean confidence as-is {conf[False]:.3f} | flipped {conf[True]:.3f}')
    return conf[True] > conf[False]

def write_predictions(names, logits, path):
    df = pd.DataFrame({'path': names, 'predicted_label': [CLASSES[i] for i in logits.argmax(1).tolist()]})
    df.to_csv(path, index=False)
    print(f'wrote {len(df)} rows -> {path}')
    print(df.predicted_label.value_counts().to_string())
    return df

# ---- cell 31 ----
TEST_DIR = find_test_dir(CFG)
test_names = list_images(TEST_DIR)
test_vflip = detect_vflip(model_p1, TEST_DIR, test_names, CFG)
test_images = load_folder(TEST_DIR, test_names, vflip=test_vflip)
print(f'{len(test_images)} test images from {TEST_DIR} (vflip={test_vflip})')

test_logits_p1 = adjust_logits(predict_tta(model_p1, test_images, CFG), prior, TAU)
torch.save(test_logits_p1, os.path.join(CFG.out_dir, 'phase1_test_logits.pt'))
p1_df = write_predictions(test_names, test_logits_p1, os.path.join(CFG.out_dir, 'phase1_predictions.csv'))
print(p1_df.head())
