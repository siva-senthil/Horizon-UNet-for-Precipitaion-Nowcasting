# ==============================================================================
# Imports & Seeds
# ==============================================================================
import math
import time
import os
import random
from pathlib import Path

import numpy as np
import h5py
from tqdm.auto import tqdm
from skimage.measure import label as skimage_label, regionprops  # aliased to avoid shadowing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import GradScaler, autocast
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba2

# Enable TF32 on Ampere+ GPUs for speed (no accuracy loss for this task)
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Prevent CUDA OOM fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
SUPPORTS_BF16 = torch.cuda.is_available() and _cc[0] >= 8
# Use bfloat16 on Ampere+, float16 elsewhere (P100 etc.)
AMP_DTYPE = torch.bfloat16 if SUPPORTS_BF16 else torch.float16

print(f'Device: {DEVICE}  |  AMP dtype: {AMP_DTYPE}')


# ==============================================================================
# Hardware Compatibility Patch for Mamba on Pre-Ampere GPUs
# ==============================================================================
# The official Triton kernels for Mamba's memory-efficient path require
# compute capability >= 8.0 (Ampere or newer). On older GPUs such as the
# Tesla P100 (cc 6.0), these kernels fail silently. The two patches below
# replace the CUDA-optimised internals with numerically equivalent pure-PyTorch
# implementations so training works on any GPU generation.

def _fast_parallel_scan(x, dt, A, B, C, chunk_size,
                         D=None, z=None, dt_bias=None, initial_states=None,
                         seq_idx=None, cu_seqlens=None, dt_softplus=True,
                         dt_limit=(0.0, float('inf')), return_final_states=False,
                         return_varlen_states=False):
    """
    Pure-PyTorch parallel scan replacing mamba_chunk_scan_combined.

    Implements the selective SSM recurrence in log-space for numerical
    stability, then projects to output via Einstein summation.
    """
    B_sz, L, nheads, headdim = x.shape
    ngroups = B.shape[2]
    g = nheads // ngroups

    if dt_bias is not None:
        dt = dt + dt_bias.view(1, 1, nheads)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = dt.clamp(1e-4, 1.0).float()

    # Expand group B and C to match number of heads
    B_exp = B.repeat_interleave(g, dim=2).float()
    C_exp = C.repeat_interleave(g, dim=2).float()

    A_real = -torch.exp(A.float().clamp(-20, 0))
    alpha = torch.exp(dt * A_real.view(1, 1, nheads))

    v = torch.einsum('blhp,blhn->blhpn', x.float(), B_exp)

    # Log-space cumulative product for numerical stability
    log_alpha = torch.log(alpha.clamp(min=1e-8))
    cum_fac = torch.exp(torch.cumsum(log_alpha, dim=1))

    h0_contrib = (cum_fac[:, :, :, None, None] * initial_states.float().unsqueeze(1)
                  if initial_states is not None else 0.0)

    v_norm = v / cum_fac[:, :, :, None, None].clamp(min=1e-8)
    h_all = cum_fac[:, :, :, None, None] * torch.cumsum(v_norm, dim=1) + h0_contrib
    h_all = h_all.clamp(-100, 100)

    y = torch.einsum('blhpn,blhn->blhp', h_all, C_exp)
    if D is not None:
        y = y + x.float() * D.view(1, 1, nheads, 1)

    final_state = h_all[:, -1] if return_final_states else None
    return (y, final_state) if return_final_states else y


def pure_pytorch_causal_conv1d(x, weight, bias=None, seq_idx=None,
                                initial_states=None, return_final_states=False,
                                activation=None):
    """
    Pure-PyTorch causal 1-D convolution replacing the CUDA-optimised kernel.
    Pads left and trims right to maintain causality.
    """
    pad = weight.shape[-1] - 1
    out = F.conv1d(x, weight.unsqueeze(1), bias=bias, padding=pad, groups=x.shape[1])
    out = out[..., :-pad] if pad > 0 else out
    if activation in ["silu", "swish"]:
        out = F.silu(out)
    return (out, None) if return_final_states else out


# Inject patches once at import time
import causal_conv1d.causal_conv1d_interface as _cc1d
import mamba_ssm.modules.mamba2 as _mamba2_mod
import mamba_ssm.ops.triton.ssd_combined as _ssd_mod
import torch._dynamo
torch._dynamo.config.suppress_errors = True

if not getattr(_cc1d, '_patched', False):
    _cc1d.causal_conv1d_fn = pure_pytorch_causal_conv1d
    _mamba2_mod.causal_conv1d_fn = pure_pytorch_causal_conv1d
    _cc1d._patched = True

if not getattr(_ssd_mod, '_patched', False):
    _ssd_mod.mamba_chunk_scan_combined = _fast_parallel_scan
    _mamba2_mod.mamba_chunk_scan_combined = _fast_parallel_scan
    _ssd_mod._patched = True


# ==============================================================================
# Configuration
# ==============================================================================
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    WORK_DIR = Path('./outputs')
    # LOCAL_DIR: directory containing paired *_vil.h5 and *_ir069.h5 event files
    # Update this to point to your SEVIR data directory.
    LOCAL_DIR = Path('./data/sevir')
    CHECKPOINT_DIR = WORK_DIR / 'checkpoints'

    # ── Data ───────────────────────────────────────────────────────────────────
    NUM_EVENTS = 1000
    # Frame window: use frames 13–25 as past input (13 frames = 65 minutes)
    PAST_START = 13
    PAST_END = 26
    IMG_SIZE = 128                  # spatial resolution after downsampling
    TARGET_FRAMES = [5, 11]         # future frame indices → t+30 min, t+60 min
    T_OUT = len(TARGET_FRAMES)      # 2 output horizons

    # ── Cross-Validation ───────────────────────────────────────────────────────
    CV_FOLDS = 4                    # rolling-origin folds
    CV_VAL_FRAC = 0.15
    CV_TEST_FRAC = 0.15

    # ── Loss weights ───────────────────────────────────────────────────────────
    HORIZON_WEIGHTS = [0.4, 0.6]    # up-weight t+60 to combat temporal degradation
    LAMBDA_EXTREME = 4.0            # weight for extreme-intensity pixels (93rd pct.)
    EXTREME_WARMUP = 8              # epochs to linearly ramp LAMBDA_EXTREME from 1→4
    EXTREME_THRESH = 0.93           # normalised VIL threshold for extreme mask
    MOD_THR = 0.54                  # moderate-rain threshold used in area loss
    LAMBDA_GRAD = 2.0               # spatial gradient preservation weight

    # ── Model ──────────────────────────────────────────────────────────────────
    LATENT_DIM = 64
    MAMBA_D_STATE = 16
    MAMBA_D_CONV = 4
    MAMBA_EXPAND = 2
    MAMBA_HEADDIM = 32
    USE_CHECKPOINT = True           # gradient checkpointing (saves VRAM during training)
    BOTTLENECK_LAYERS = 2
    DROPOUT_P = 0.1

    # ── Training ───────────────────────────────────────────────────────────────
    BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    NUM_WORKERS = 2
    PREFETCH_FACTOR = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_ACCUM = 2                  # effective batch size = BATCH_SIZE * GRAD_ACCUM
    MAX_GRAD_NORM = 1.0
    USE_AMP = True
    EPOCHS = 40
    SAVE_EVERY = 5                  # save periodic checkpoints every N epochs
    PATIENCE = 8                    # early stopping patience (monitored: val loss)
    MIN_DELTA = 1e-4
    LR_WARMUP_EPOCHS = 7            # linear warmup before cosine annealing
    CV_EPOCHS = 10                  # shorter training loop for CV folds
    CV_LR_WARMUP = 2

    # ── Evaluation ─────────────────────────────────────────────────────────────
    FSS_SCALES = [4, 8, 16, 32]
    FULL_EVAL_EVERY = 1


config = Config()
config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Derived constants used throughout the script
T_PAST = config.PAST_END - config.PAST_START   # 13 input frames
T_OUT = config.T_OUT                           # 2 output horizons

print(f'Past frames : {T_PAST}  |  Target frames: {config.TARGET_FRAMES}  (t+30, t+60 min)')
print(f'Epochs      : {config.EPOCHS}  |  Effective batch: {config.BATCH_SIZE * config.GRAD_ACCUM}')


# ==============================================================================
# Dataset
# ==============================================================================
class SEVIRDualHorizonDataset(Dataset):
    """
    Loads paired VIL + IR069 event H5 files from a local directory.

    Each event is stored as two files:
        <event_id>_vil.h5    — datasets: 'past'   (13, H, W)
                                          'future' (12, H, W)
        <event_id>_ir069.h5  — same layout

    Returns:
        x  (T_PAST, 2, IMG_SIZE, IMG_SIZE)  — normalised [VIL, IR069] stack
        y  (T_OUT,  1, IMG_SIZE, IMG_SIZE)  — normalised target VIL frames
    """

    def __init__(self, root_dir, augment: bool = False):
        self.root = Path(root_dir)
        self.vil_files = sorted(self.root.glob('*_vil.h5'))
        self._tgt = (config.IMG_SIZE, config.IMG_SIZE)
        self._frames = config.TARGET_FRAMES
        self.augment = augment
        if not self.vil_files:
            print(f'WARNING: No *_vil.h5 files found in {root_dir}')

    def __len__(self):
        return len(self.vil_files)

    def __getitem__(self, idx):
        vp = self.vil_files[idx]
        eid = vp.stem.rsplit('_vil', 1)[0]

        with h5py.File(vp, 'r') as v, \
             h5py.File(self.root / f'{eid}_ir069.h5', 'r') as ir:
            pv_np = v['past'][:]
            fv_np = v['future'][self._frames]
            pi_np = ir['past'][:]

        # Normalise: VIL via log1p-scale, IR via linear scale
        pv_np = (np.log1p(pv_np.astype(np.float32)) - 3.5) / 1.5
        fv_np = (np.log1p(fv_np.astype(np.float32)) - 3.5) / 1.5
        pi_np = pi_np.astype(np.float32) / 350.0

        def _resize(arr):
            # Bilinear interpolation to target spatial resolution
            t = torch.from_numpy(arr).unsqueeze(1)
            if t.shape[-2:] != self._tgt:
                t = F.interpolate(t, size=self._tgt, mode='bilinear', align_corners=False)
            return t

        pv = _resize(pv_np)
        fv = _resize(fv_np)
        pi = _resize(pi_np)

        # Random horizontal and vertical flipping for augmentation
        # Intensity is NOT augmented to preserve physical validity.
        if self.augment:
            if random.random() > 0.5:
                pv = torch.flip(pv, [-1])
                fv = torch.flip(fv, [-1])
                pi = torch.flip(pi, [-1])
            if random.random() > 0.5:
                pv = torch.flip(pv, [-2])
                fv = torch.flip(fv, [-2])
                pi = torch.flip(pi, [-2])

        # Stack VIL and IR along the channel dimension
        return torch.cat([pv, pi], dim=1), fv


def make_rolling_origin_folds(dataset, k: int):
    """
    Temporal rolling-origin cross-validation splits.

    Preserves chronological order to prevent data leakage. Each fold
    uses strictly earlier events for training and later events for
    validation and test.
    """
    n = len(dataset)
    block_size = n // (k + 2)
    all_idx = list(range(n))
    splits = []
    for fold in range(k):
        train_end = fold * block_size + block_size
        val_start = train_end
        val_end = val_start + block_size
        test_start = val_end
        test_end = test_start + block_size if fold < k - 1 else n
        splits.append((
            all_idx[:train_end],
            all_idx[val_start:val_end],
            all_idx[test_start:test_end],
        ))
    return splits


# Build datasets and splits
full_ds = SEVIRDualHorizonDataset(config.LOCAL_DIR, augment=False)
full_ds_aug = SEVIRDualHorizonDataset(config.LOCAL_DIR, augment=True)
cv_splits = make_rolling_origin_folds(full_ds, k=config.CV_FOLDS)

# 70/15/15 hold-out split for main training
_n = len(full_ds)
_tr_end = int(_n * 0.70)
_va_end = int(_n * 0.85)
train_idx = list(range(0, _tr_end))
val_idx = list(range(_tr_end, _va_end))
test_idx = list(range(_va_end, _n))

print(f'Dataset: {_n} events  |  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}')

train_ds = Subset(full_ds_aug, train_idx)
val_ds = Subset(full_ds, val_idx)
test_ds = Subset(full_ds, test_idx)


def _seed_worker(wid):
    """Ensure each DataLoader worker has a different but deterministic seed."""
    np.random.seed(SEED + wid)
    random.seed(SEED + wid)


def make_loader(dataset, batch_size: int, shuffle: bool,
                drop_last: bool = False, seed_offset: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(SEED + seed_offset)
    kw = dict(
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.NUM_WORKERS > 0,
        timeout=120,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    if config.NUM_WORKERS > 0:
        kw['prefetch_factor'] = config.PREFETCH_FACTOR
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last, **kw)


train_loader = make_loader(train_ds, config.BATCH_SIZE, shuffle=True, drop_last=True, seed_offset=0)
val_loader = make_loader(val_ds, config.VAL_BATCH_SIZE, shuffle=False, drop_last=False, seed_offset=1)
test_loader = make_loader(test_ds, config.VAL_BATCH_SIZE, shuffle=False, drop_last=False, seed_offset=2)

# Threshold names and horizon labels used in metrics dictionaries
CSI_THRESHOLDS = {'CSI_light': -0.45, 'CSI_mod': 0.54, 'CSI_heavy': 0.93}
HORIZONS = ['t+30', 't+60']

xs, ys = next(iter(train_loader))
print(f'Batch — x: {xs.shape}  y: {ys.shape}')


# ==============================================================================
# Model Architecture (HorizonUNet / Causal-Mamba U-Net)
# ==============================================================================

class SafeMamba2(Mamba2):
    """
    Mamba2 with the memory-efficient Triton path disabled.

    The memory-efficient path requires compute capability >= 8.0.
    Disabling it ensures correctness on older architectures (P100, V100).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mem_eff_path = False


class Mamba2DBlock(nn.Module):
    """
    Isotropic 2-D scanning block.

    Two SafeMamba2 SSMs scan the input independently — one along rows
    (horizontal), one along columns (vertical). A learnable sigmoid gate
    fuses the two output feature maps, allowing the network to adapt to
    directional asymmetry in local storm structure.

    This addresses the 1-D directional bias of standard SSMs when applied
    to meteorological 2-D spatial fields.
    """

    def __init__(self, dim: int, cfg):
        super().__init__()
        kw = dict(d_model=dim, d_state=cfg.MAMBA_D_STATE, d_conv=cfg.MAMBA_D_CONV,
                  expand=cfg.MAMBA_EXPAND, headdim=cfg.MAMBA_HEADDIM)
        self.mh = SafeMamba2(**kw)    # horizontal scan
        self.mw = SafeMamba2(**kw)    # vertical scan
        # Gate initialised to 0 → sigmoid(0)=0.5, balanced fusion at start
        self.gate = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xl = x.permute(0, 2, 3, 1)   # (B, H, W, C)

        # Horizontal scan: treat each column position as a batch, rows as sequence
        yh = self.mh(xl.permute(0, 2, 1, 3).reshape(B * W, H, C))
        yh = yh.reshape(B, W, H, C).permute(0, 3, 2, 1)   # → (B, C, H, W)

        # Vertical scan: treat each row position as a batch, columns as sequence
        yw = self.mw(xl.reshape(B * H, W, C))
        yw = yw.reshape(B, H, W, C).permute(0, 3, 1, 2)   # → (B, C, H, W)

        g = torch.sigmoid(self.gate)
        return g * yh + (1.0 - g) * yw


class Mamba2DLayer(nn.Module):
    """
    Residual wrapper: GroupNorm → Mamba2DBlock → Dropout2d, with skip.
    Gradient checkpointing is applied during training to reduce peak VRAM.
    """

    def __init__(self, dim: int, cfg):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.block = Mamba2DBlock(dim, cfg)
        self.drop = nn.Dropout2d(p=cfg.DROPOUT_P)
        self.use_ckpt = cfg.USE_CHECKPOINT

    def _fwd(self, x):
        return self.drop(self.block(self.norm(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gradient checkpointing: recompute activations on backward pass
        # to trade computation for reduced VRAM usage.
        if self.training and self.use_ckpt:
            return checkpoint(self._fwd, x, use_reentrant=False) + x
        return self._fwd(x) + x


class EncBlock(nn.Module):
    """Encoder stage: strided Conv2d (halves spatial) → GroupNorm → GELU → Mamba2DLayer."""

    def __init__(self, ic: int, oc: int, cfg):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic, oc, 3, stride=2, padding=1),
            nn.GroupNorm(8, oc),
            nn.GELU(),
        )
        self.mamba = Mamba2DLayer(oc, cfg)

    def forward(self, x):
        return self.mamba(self.conv(x))


class DecBlock(nn.Module):
    """Decoder stage: ConvTranspose2d (doubles spatial) → GroupNorm → GELU → Dropout → Mamba2DLayer."""

    def __init__(self, ic: int, oc: int, cfg):
        super().__init__()
        self.up = nn.ConvTranspose2d(ic, oc, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(8, oc)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(p=cfg.DROPOUT_P)
        self.mamba = Mamba2DLayer(oc, cfg)

    def forward(self, x):
        return self.mamba(self.drop(self.act(self.norm(self.up(x)))))


def _sinusoidal_pos(T: int, C: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding for T positions and C channels."""
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, C, 2, dtype=torch.float32) * -(math.log(10000.0) / C))
    enc = torch.zeros(T, C)
    enc[:, 0::2] = torch.sin(pos * div)
    enc[:, 1::2] = torch.cos(pos * div[:C // 2])
    return enc.unsqueeze(-1).unsqueeze(-1)   # (T, C, 1, 1)


class HorizonUNet(nn.Module):
    """
    Causal-Mamba U-Net (HorizonUNet).

    Input:  (B, T_PAST, 2, H, W)   — 13 past frames of [VIL, IR069], normalised
    Output: (B, T_OUT,  1, H, W)   — predicted VIL at t+30 and t+60

    Architecture:
    - 4-level encoder (strided Conv + Mamba2DLayer at each level)
    - 2-layer Mamba bottleneck
    - 4-level decoder (ConvTranspose + Mamba2DLayer + skip connections)
    - Multi-horizon prediction head (Conv → tanh-scaled output)

    Positional embeddings:
        Sinusoidal embeddings (fixed) + learned residual are injected into
        the flattened temporal-channel input to give the SSMs temporal context.

    Output activation:
        tanh scaled to [out_shift - out_scale, out_shift + out_scale]
        which spans the normalised VIL distribution.
    """

    def __init__(self, cfg):
        super().__init__()
        C = cfg.LATENT_DIM
        Cin = T_PAST * 2   # 13 frames × 2 modalities = 26 input channels

        # Fixed sinusoidal embeddings + learnable residual correction
        self.register_buffer('frame_emb_sin', _sinusoidal_pos(T_PAST, 2) * 0.1)
        self.frame_emb_res = nn.Parameter(torch.zeros(T_PAST, 2, 1, 1))

        # Feature stem: projects 26-channel input to latent dimension
        self.stem = nn.Sequential(
            nn.Conv2d(Cin, C, 3, padding=1),
            nn.GroupNorm(8, C),
            nn.GELU(),
        )

        # Encoder: C → 2C → 4C → 8C → 8C (spatial 128→64→32→16→8)
        self.enc1 = EncBlock(C,     2 * C, cfg)
        self.enc2 = EncBlock(2 * C, 4 * C, cfg)
        self.enc3 = EncBlock(4 * C, 8 * C, cfg)
        self.enc4 = EncBlock(8 * C, 8 * C, cfg)

        # Bottleneck: 2 sequential Mamba2DLayers at the smallest spatial scale
        self.bn = nn.Sequential(*[Mamba2DLayer(8 * C, cfg)
                                   for _ in range(cfg.BOTTLENECK_LAYERS)])

        # Skip connection projection at enc4 level
        self.s4_proj = nn.Sequential(
            nn.Conv2d(8 * C, 8 * C, 3, padding=1),
            nn.GroupNorm(8, 8 * C),
            nn.GELU(),
        )

        # Decoder: skip connections double the input channels at each level
        self.dec1 = DecBlock(16 * C, 4 * C, cfg)   # bn + s4_proj → 4C
        self.dec2 = DecBlock(12 * C, 2 * C, cfg)   # d1 + s3    → 2C
        self.dec3 = DecBlock(6 * C,  C,     cfg)   # d2 + s2    → C
        self.dec4 = DecBlock(3 * C,  C,     cfg)   # d3 + s1    → C

        # Prediction head: maps to T_OUT channels then constrains to valid range
        self.head = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),
            nn.GroupNorm(8, C),
            nn.GELU(),
            nn.Conv2d(C, T_OUT, 1),
        )

        # Output scale/shift: tanh(x) * 1.92 - 0.42 spans ~[-2.34, 1.50]
        # which covers the normalised VIL distribution.
        self.register_buffer('out_scale', torch.tensor(1.92))
        self.register_buffer('out_shift', torch.tensor(-0.42))

    def forward(self, x_past: torch.Tensor) -> torch.Tensor:
        B, T, C2, H, W = x_past.shape

        # Add positional embeddings and flatten temporal+channel axes
        frame_emb = self.frame_emb_sin + self.frame_emb_res
        x = (x_past + frame_emb.unsqueeze(0)).reshape(B, T * C2, H, W)

        # Encoder
        h = self.stem(x)
        s1 = self.enc1(h)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        bn = self.bn(s4)
        s4p = self.s4_proj(s4)

        # Decoder with skip connections
        d1 = self.dec1(torch.cat([bn,  s4p], dim=1))
        d2 = self.dec2(torch.cat([d1,  s3],  dim=1))
        d3 = self.dec3(torch.cat([d2,  s2],  dim=1))
        d4 = self.dec4(torch.cat([d3,  s1],  dim=1))

        raw = self.head(d4)
        out = self.out_scale * torch.tanh(raw) + self.out_shift
        return out.unsqueeze(2)   # (B, T_OUT, 1, H, W)


# Instantiate and verify model
model = HorizonUNet(config).to(DEVICE)
total_p = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {total_p:,}')

with torch.no_grad():
    _o = model(torch.randn(2, T_PAST, 2, 128, 128, device=DEVICE))
    print(f'Forward check: {_o.shape}  nan={torch.isnan(_o).any()}'
          f'  range=[{_o.min():.3f}, {_o.max():.3f}]')


# ==============================================================================
# Physics-Informed Composite Loss
# ==============================================================================
class ForecastLoss(nn.Module):
    """
    Multi-component physics-informed loss for extreme precipitation nowcasting.

    Components:
    1. Charbonnier (robust L1 base)
    2. Spatial gradient preservation via Sobel operators
    3. Extreme-intensity up-weighting (pixels > 93rd percentile)
    4. Heavy-rain under-prediction penalty (asymmetric, favours safety)
    5. Background sparsity (penalises false alarms in clear-sky regions)
    6. Area consistency (predicted vs. actual rain coverage)
    7. Mean bias correction

    Horizon weighting: t+60 weighted 0.6 vs t+30 at 0.4 to prioritise
    longer-range skill (the harder and more operationally valuable task).

    The extreme-intensity weight is linearly warmed up over 8 epochs to
    prevent early training instability.
    """

    def __init__(self, cfg):
        super().__init__()
        self.weights = cfg.HORIZON_WEIGHTS
        self.lx = cfg.LAMBDA_EXTREME
        self.et = cfg.EXTREME_THRESH
        self.lg = cfg.LAMBDA_GRAD
        self.extreme_warmup = cfg.EXTREME_WARMUP
        self.current_epoch = 1
        self.mod_thr = cfg.MOD_THR

        # Sobel kernels for horizontal and vertical gradient computation
        sx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                          dtype=torch.float32).view(1, 1, 3, 3) / 8.
        sy = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                          dtype=torch.float32).view(1, 1, 3, 3) / 8.
        self.register_buffer('sx', sx)
        self.register_buffer('sy', sy)

    def _grad(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient magnitude."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        sx = self.sx.to(device=x.device, dtype=x.dtype)
        sy = self.sy.to(device=x.device, dtype=x.dtype)
        return torch.sqrt(
            F.conv2d(x, sx, padding=1) ** 2 +
            F.conv2d(x, sy, padding=1) ** 2 + 1e-8
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        # Linearly ramp extreme weight from 1.0 to self.lx over warmup epochs
        lx_eff = 1.0 + (self.lx - 1.0) * min(1.0, self.current_epoch / max(1, self.extreme_warmup))
        total = torch.zeros((), device=pred.device, dtype=pred.dtype)
        losses = {}

        for hi, w in enumerate(self.weights):
            p = pred[:, hi]
            t = target[:, hi]
            diff = p - t
            n_pix = float(p.numel())

            # 1. Charbonnier base loss (robust alternative to L1)
            L_char = torch.sqrt(diff ** 2 + 1e-6).mean()

            # 2. Gradient loss preserves storm cell boundaries
            L_grad = torch.sqrt((self._grad(p) - self._grad(t)) ** 2 + 1e-6).mean()

            # 3. Extreme-intensity up-weighting
            mask_ext = (t > self.et).float()
            L_ext = (mask_ext * diff.float() ** 2).sum() / n_pix

            # 4. Heavy-rain asymmetric under-prediction penalty
            # Raised to weight 3.0 (vs. 1.5 in earlier ablations) to improve PODheavy
            mask_heavy = (t > 0.95).float()
            n_heavy = mask_heavy.sum().clamp(min=1)
            L_heavy = (mask_heavy * F.relu(t - p)).sum() / n_heavy

            # 5. Background sparsity — suppresses false alarms in clear-sky areas
            bg_mask = (t < -0.5).float()
            L_sparse = (bg_mask * p.float().clamp(min=-0.5) ** 2).sum() / n_pix

            # 6a. Area consistency — moderate threshold
            p_area = torch.sigmoid(10.0 * (p - self.mod_thr)).mean(dim=[-1, -2])
            t_area = (t > self.mod_thr).float().mean(dim=[-1, -2])
            L_area = (F.relu(p_area - t_area) ** 2).mean()
            L_area_under = (F.relu(t_area - p_area) ** 2).mean()

            # 6b. Area consistency — heavy threshold
            p_area_h = torch.sigmoid(10.0 * (p - 0.95)).mean(dim=[-1, -2])
            t_area_h = (t > 0.95).float().mean(dim=[-1, -2])
            L_area_heavy_under = (F.relu(t_area_h - p_area_h) ** 2).mean()

            # 7. Mean bias correction
            L_mean_bias = F.mse_loss(
                p.float().mean(dim=[-1, -2, -3]),
                t.float().mean(dim=[-1, -2, -3]).detach()
            ).mean()

            L_h = (L_char
                   + self.lg    * L_grad
                   + lx_eff     * L_ext
                   + 3.0        * L_sparse
                   + 5.0        * L_area
                   + 0.5        * L_area_under
                   + 10.0       * L_mean_bias
                   + 3.0        * L_heavy
                   + 2.0        * L_area_heavy_under)

            losses[f'h{hi}'] = L_h.item()
            total = total + w * L_h

        losses['total'] = total
        return losses


# ==============================================================================
# Metrics & Evaluation
# ==============================================================================
def compute_mae(pred, target):
    return F.l1_loss(pred, target).item()

def compute_rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)).item()

def compute_bias(pred, target):
    return (pred - target).mean().item()


def _binary_counts(pred, target, thr):
    """Return (TP, FP, FN) for a given threshold."""
    p = pred > thr
    t = target > thr
    return (p & t).sum().item(), (p & ~t).sum().item(), (~p & t).sum().item()


def _fss_components(pred, target, thr, scale):
    """Return (MSE_neighbourhood, reference_sum) for FSS computation."""
    p_bin = (pred > thr).float()
    t_bin = (target > thr).float()
    k = 2 * scale + 1
    fp = F.avg_pool2d(p_bin, kernel_size=k, stride=1, padding=scale)
    ft = F.avg_pool2d(t_bin, kernel_size=k, stride=1, padding=scale)
    return ((fp - ft) ** 2).sum().item(), (fp ** 2 + ft ** 2).sum().item()


def compute_sal(pred, target, threshold: float = 0.5):
    """
    Structure-Amplitude-Location (SAL) score.

    Returns three scalars:
        S  — structure: normalised difference in object volumes
        A  — amplitude: normalised domain-mean intensity difference
        L  — location: normalised centroid displacement (lower is better)

    Uses skimage_label (aliased at import) to avoid shadowing the 'label'
    built-in in list comprehensions elsewhere in the script.
    """
    p_np = pred.cpu().numpy().squeeze()
    t_np = target.cpu().numpy().squeeze()

    # Convert normalised values back to physical units for SAL
    pred_phys = np.clip(np.expm1((p_np * 1.5) + 3.5), 0, None)
    target_phys = np.clip(np.expm1((t_np * 1.5) + 3.5), 0, None)
    thr_phys = np.expm1((threshold * 1.5) + 3.5)

    # Amplitude component
    sm = pred_phys.mean() + target_phys.mean()
    A = 2 * (pred_phys.mean() - target_phys.mean()) / (sm + 1e-8)

    def objs(f, th):
        regs = regionprops(skimage_label(f > th, connectivity=1), intensity_image=f)
        if not regs:
            return 0., 0.
        vol = sum(r.area for r in regs)
        return vol, sum(r.area * r.mean_intensity for r in regs) / (vol + 1e-8)

    # Structure component
    Vp, _ = objs(pred_phys, thr_phys)
    Vo, _ = objs(target_phys, thr_phys)
    S = 2 * (Vp - Vo) / (Vp + Vo + 1e-8) if Vp + Vo else 0.

    def cen(f, th):
        y, x = np.where(f > th)
        return np.array([x.mean(), y.mean()]) if len(x) else np.zeros(2)

    # Location component: centroid displacement normalised by domain diagonal
    H, W = pred_phys.shape
    L = np.linalg.norm(
        cen(pred_phys, thr_phys) - cen(target_phys, thr_phys)
    ) / np.sqrt(H ** 2 + W ** 2)
    return S, A, L


def compute_sal_for_model(mdl, loader, sal_thr: float, desc: str = 'SAL') -> dict:
    """Compute mean SAL across all batches in loader. Returns {horizon: {S,A,L}}."""
    mdl.eval()
    acc = {tag: {'S': 0., 'A': 0., 'L': 0.} for tag in HORIZONS}
    n = 0
    with torch.no_grad():
        for xp, yf in tqdm(loader, desc=desc, leave=False):
            xp = xp.to(DEVICE)
            yf = yf.to(DEVICE)
            pred = mdl(xp)
            for b in range(xp.shape[0]):
                for hi, tag in enumerate(HORIZONS):
                    S, A, L = compute_sal(pred[b, hi, 0], yf[b, hi, 0], threshold=sal_thr)
                    acc[tag]['S'] += S
                    acc[tag]['A'] += A
                    acc[tag]['L'] += L
            n += 1
    return {tag: {k: v / n for k, v in acc[tag].items()} for tag in HORIZONS}


def compute_crps_mc(mdl, xp, yf, n_members: int = 10) -> dict:
    """
    Continuous Ranked Probability Score via Monte Carlo Dropout ensemble.

    Temporarily sets the model to train() mode to activate Dropout, then
    generates n_members stochastic forward passes as the ensemble.
    """
    mdl.train()
    try:
        with torch.no_grad():
            members = torch.stack([mdl(xp).cpu() for _ in range(n_members)])
    finally:
        mdl.eval()

    yf_cpu = yf.cpu()
    results = {}
    for hi, tag in enumerate(HORIZONS):
        ens = members[:, :, hi]
        tgt = yf_cpu[:, hi]
        acc = F.l1_loss(ens, tgt.unsqueeze(0).expand_as(ens))
        spread = (ens.unsqueeze(0) - ens.unsqueeze(1)).abs().mean()
        results[f'crps_{tag}'] = (acc - 0.5 * spread).item()
    return results


def evaluate_split(mdl, loader, loss_fn, desc: str = 'eval') -> dict:
    """
    Evaluate model on a DataLoader.

    Computes MAE, RMSE, Bias, CSI/POD/FAR at three thresholds (light/moderate/heavy),
    and FSS at multiple spatial scales for both t+30 and t+60 horizons.
    """
    mdl.eval()
    lin = {'loss': 0., 'n': 0}
    for tag in HORIZONS:
        lin[f'mae_{tag}'] = lin[f'rmse_{tag}'] = lin[f'bias_{tag}'] = 0.
    counts = {tag: {nm: [0., 0., 0.] for nm in CSI_THRESHOLDS} for tag in HORIZONS}
    fss_c = {tag: {nm: {sc: [0., 0.] for sc in config.FSS_SCALES}
                   for nm in CSI_THRESHOLDS} for tag in HORIZONS}

    with torch.no_grad():
        for xp, yf in tqdm(loader, desc=desc, leave=False):
            xp = xp.to(DEVICE, non_blocking=True)
            yf = yf.to(DEVICE, non_blocking=True)
            pred = mdl(xp)
            lin['loss'] += loss_fn(pred, yf)['total'].item()
            lin['n'] += 1
            for hi, tag in enumerate(HORIZONS):
                p = pred[:, hi].float()
                t = yf[:, hi].float()
                lin[f'mae_{tag}'] += compute_mae(p, t)
                lin[f'rmse_{tag}'] += compute_rmse(p, t)
                lin[f'bias_{tag}'] += compute_bias(p, t)
                for nm, thr in CSI_THRESHOLDS.items():
                    tp, fp, fn = _binary_counts(p, t, thr)
                    counts[tag][nm][0] += tp
                    counts[tag][nm][1] += fp
                    counts[tag][nm][2] += fn
                    for sc in config.FSS_SCALES:
                        mn, mr = _fss_components(p, t, thr, sc)
                        fss_c[tag][nm][sc][0] += mn
                        fss_c[tag][nm][sc][1] += mr

    n = lin['n']
    out = {'loss': lin['loss'] / n}
    for tag in HORIZONS:
        out[f'mae_{tag}'] = lin[f'mae_{tag}'] / n
        out[f'rmse_{tag}'] = lin[f'rmse_{tag}'] / n
        out[f'bias_{tag}'] = lin[f'bias_{tag}'] / n
        for nm in CSI_THRESHOLDS:
            tp, fp, fn = counts[tag][nm]
            out[f'{nm}_csi_{tag}'] = tp / (tp + fp + fn + 1e-9)
            out[f'{nm}_pod_{tag}'] = tp / (tp + fn + 1e-9)
            out[f'{nm}_far_{tag}'] = fp / (tp + fp + 1e-9)
            for sc in config.FSS_SCALES:
                mn = fss_c[tag][nm][sc][0]
                mr = fss_c[tag][nm][sc][1]
                out[f'fss_s{sc}_{nm}_{tag}'] = 1.0 - mn / (mr + 1e-9) if mr > 1e-9 else 1.0
    return out


def _inject_crps_sal(metrics_dict: dict, crps_dict: dict, sal_dict: dict):
    """Merge CRPS and SAL results into an existing metrics dictionary in-place."""
    for tag in HORIZONS:
        metrics_dict[f'crps_{tag}'] = crps_dict.get(f'crps_{tag}', float('nan'))
        metrics_dict[f'sal_S_{tag}'] = sal_dict[tag]['S']
        metrics_dict[f'sal_A_{tag}'] = sal_dict[tag]['A']
        metrics_dict[f'sal_L_{tag}'] = sal_dict[tag]['L']


SAL_THR = list(CSI_THRESHOLDS.values())[1]   # use moderate threshold for SAL


# ==============================================================================
# Early Stopping & Trainer
# ==============================================================================
class EarlyStopping:
    """Stops training when validation metric has not improved for `patience` epochs."""

    def __init__(self, patience: int, min_delta: float = 0.):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0

    def __call__(self, val: float) -> bool:
        if val < self.best - self.min_delta:
            self.best = val
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """
    Training loop with:
    - AdamW optimiser
    - Linear warmup + cosine annealing LR schedule
    - Automatic Mixed Precision (AMP) with GradScaler
    - Gradient accumulation (effective batch = BATCH_SIZE * GRAD_ACCUM)
    - Gradient clipping (max norm 1.0)
    - Early stopping monitored on validation loss
    - Periodic and best-checkpoint saving
    """

    def __init__(self, mdl, cfg):
        self.model = mdl
        self.cfg = cfg
        self.loss_fn = ForecastLoss(cfg)
        self.es = EarlyStopping(cfg.PATIENCE, cfg.MIN_DELTA)
        self.scaler = GradScaler('cuda', enabled=cfg.USE_AMP, growth_interval=500)
        self.best_val = float('inf')

        # History for later plotting
        self.history = {'train_loss': [], 'val_loss': []}
        for tag in HORIZONS:
            for m in ['mae', 'rmse', 'bias']:
                self.history[f'{m}_{tag}'] = []
            for nm in CSI_THRESHOLDS:
                self.history[f'{nm}_csi_{tag}'] = []
                self.history[f'{nm}_pod_{tag}'] = []
            for sc in cfg.FSS_SCALES:
                self.history[f'fss_s{sc}_CSI_mod_{tag}'] = []

        self.opt = optim.AdamW(mdl.parameters(),
                               lr=cfg.LEARNING_RATE,
                               weight_decay=cfg.WEIGHT_DECAY)

        def lr_lam(ep):
            # Linear warmup for the first LR_WARMUP_EPOCHS epochs, then cosine decay
            if cfg.LR_WARMUP_EPOCHS > 0 and ep < cfg.LR_WARMUP_EPOCHS:
                return (ep + 1) / cfg.LR_WARMUP_EPOCHS
            p = (ep - cfg.LR_WARMUP_EPOCHS) / max(1, cfg.EPOCHS - cfg.LR_WARMUP_EPOCHS)
            return max(0.5 * (1 + math.cos(math.pi * p)), 0.02)  # floor at 2% of initial LR

        self.sched = optim.lr_scheduler.LambdaLR(self.opt, lr_lam)
        self.sched.step()

    def train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        acc_loss = 0.
        self.opt.zero_grad()
        self.loss_fn.current_epoch = epoch
        pbar = tqdm(enumerate(loader), total=len(loader),
                    desc=f'Ep {epoch:3d} [train]', leave=False)
        for step, (xp, yf) in pbar:
            xp = xp.to(DEVICE, non_blocking=True)
            yf = yf.to(DEVICE, non_blocking=True)
            with autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=self.cfg.USE_AMP):
                pred = self.model(xp)
                losses = self.loss_fn(pred, yf)
                # Divide by GRAD_ACCUM so gradients are averaged over accumulation steps
                loss = losses['total'] / self.cfg.GRAD_ACCUM
            self.scaler.scale(loss).backward()
            if (step + 1) % self.cfg.GRAD_ACCUM == 0 or (step + 1) == len(loader):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.MAX_GRAD_NORM)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()
            acc_loss += losses['total'].item()
            pbar.set_postfix(loss=f'{losses["total"].item():.4f}')
        return acc_loss / len(loader)

    def train(self, tr_loader, va_loader):
        print('=' * 80)
        print('  HorizonUNet Training  —  t+30 min  +  t+60 min  Nowcasting')
        print('=' * 80)
        for epoch in range(1, self.cfg.EPOCHS + 1):
            t0 = time.time()
            tm = self.train_epoch(tr_loader, epoch)
            vm = evaluate_split(self.model, va_loader, self.loss_fn, desc=f'Ep {epoch} [val]')
            self.sched.step()
            elapsed = time.time() - t0

            # Record history
            self.history['train_loss'].append(tm)
            self.history['val_loss'].append(vm['loss'])
            for tag in HORIZONS:
                for m in ['mae', 'rmse', 'bias']:
                    self.history[f'{m}_{tag}'].append(vm[f'{m}_{tag}'])
                for nm in CSI_THRESHOLDS:
                    self.history[f'{nm}_csi_{tag}'].append(vm[f'{nm}_csi_{tag}'])
                    self.history[f'{nm}_pod_{tag}'].append(vm[f'{nm}_pod_{tag}'])
                for sc in self.cfg.FSS_SCALES:
                    self.history[f'fss_s{sc}_CSI_mod_{tag}'].append(vm[f'fss_s{sc}_CSI_mod_{tag}'])

            for tag in HORIZONS:
                ratio = vm[f'rmse_{tag}'] / (vm[f'mae_{tag}'] + 1e-8)
                fss_str = '  '.join(
                    f'FSS@{sc}px={vm[f"fss_s{sc}_CSI_mod_{tag}"]:.3f}'
                    for sc in self.cfg.FSS_SCALES
                )
                print(f'Ep{epoch:3d} [{elapsed:.0f}s] {tag} | '
                      f'tr={tm:.4f} val={vm["loss"]:.4f} | '
                      f'MAE={vm[f"mae_{tag}"]:.4f} RMSE={vm[f"rmse_{tag}"]:.4f} '
                      f'R/M={ratio:.2f} Bias={vm[f"bias_{tag}"]:+.4f} | '
                      f'CSI_M={vm[f"CSI_mod_csi_{tag}"]:.3f} '
                      f'POD_M={vm[f"CSI_mod_pod_{tag}"]:.3f} | '
                      f'{fss_str} | lr={self.opt.param_groups[0]["lr"]:.1e}')

            # Save best checkpoint
            if vm['loss'] < self.best_val:
                self.best_val = vm['loss']
                torch.save(
                    {'epoch': epoch, 'model_state': self.model.state_dict(),
                     'val_loss': self.best_val, 'val_metrics': vm},
                    config.CHECKPOINT_DIR / 'best_model.pt'
                )
                print(f'          Best model saved (val={self.best_val:.4f})')

            # Periodic snapshots
            if epoch % config.SAVE_EVERY == 0:
                torch.save(
                    {'epoch': epoch, 'model_state': self.model.state_dict()},
                    config.CHECKPOINT_DIR / f'ckpt_ep{epoch:03d}.pt'
                )

            if self.es(vm['loss']):
                print(f'Early stopping at epoch {epoch}')
                break
            torch.cuda.empty_cache()

        print(f'Training complete.  Best val loss: {self.best_val:.4f}')


# ==============================================================================
# Train
# ==============================================================================
trainer = Trainer(model, config)
trainer.train(train_loader, val_loader)


# ==============================================================================
# Evaluation — Main Model (HorizonUNet / Causal-Mamba)
# ==============================================================================
# Load the best checkpoint saved during training
best_ckpt = config.CHECKPOINT_DIR / 'best_model.pt'
if best_ckpt.exists():
    ck = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck['model_state'])
    print(f'Loaded best model (epoch {ck["epoch"]}, val_loss={ck["val_loss"]:.4f})')

loss_fn = ForecastLoss(config)
loss_fn.current_epoch = config.EPOCHS

print('\nEvaluating Causal-Mamba on TEST set...')
test_metrics = evaluate_split(model, test_loader, loss_fn, desc='Mamba Test')

# CRPS via Monte Carlo Dropout (10-member ensemble)
print('Computing CRPS (MC dropout, 10 members)...')
crps_acc = {f'crps_{tag}': 0. for tag in HORIZONS}
nb_crps = 0
model.eval()
with torch.no_grad():
    for xp, yf in tqdm(test_loader, desc='CRPS', leave=False):
        xp = xp.to(DEVICE)
        yf = yf.to(DEVICE)
        cr = compute_crps_mc(model, xp, yf, n_members=10)
        for tag in HORIZONS:
            crps_acc[f'crps_{tag}'] += cr[f'crps_{tag}']
        nb_crps += 1
for k in crps_acc:
    crps_acc[k] /= nb_crps

# SAL decomposition
print('Computing SAL...')
mamba_sal = compute_sal_for_model(model, test_loader, SAL_THR, desc='Mamba SAL')

# Merge CRPS and SAL into the test metrics dictionary
_inject_crps_sal(test_metrics, crps_acc, mamba_sal)

# Print summary
print(f'\nCausal-Mamba CRPS:  ' +
      '  '.join(f'{tag}={crps_acc[f"crps_{tag}"]:.4f}' for tag in HORIZONS))
for tag in HORIZONS:
    d = mamba_sal[tag]
    print(f'Causal-Mamba SAL [{tag}]:  S={d["S"]:.3f}  A={d["A"]:.3f}  L={d["L"]:.3f}')

print('\n-- Test Set Summary --')
for tag in HORIZONS:
    print(f'\n{tag}:')
    print(f'  MAE={test_metrics[f"mae_{tag}"]:.4f}  '
          f'RMSE={test_metrics[f"rmse_{tag}"]:.4f}  '
          f'Bias={test_metrics[f"bias_{tag}"]:+.4f}  '
          f'CRPS={test_metrics[f"crps_{tag}"]:.4f}')
    print(f'  CSI_heavy={test_metrics["CSI_heavy_csi_" + tag]:.4f}  '
          f'POD_heavy={test_metrics["CSI_heavy_pod_" + tag]:.4f}  '
          f'FAR_heavy={test_metrics["CSI_heavy_far_" + tag]:.4f}')
    print(f'  FSS@32px={test_metrics[f"fss_s32_CSI_mod_{tag}"]:.4f}')
    print(f'  SAL-S={test_metrics[f"sal_S_{tag}"]:.4f}  '
          f'SAL-A={test_metrics[f"sal_A_{tag}"]:.4f}  '
          f'SAL-L={test_metrics[f"sal_L_{tag}"]:.4f}')

print('\nEvaluation complete.')
