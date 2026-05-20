import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ── Shared constants ───────────────────────────────────────────────────────────
T_PAST = 13   # past frames fed to the model
T_OUT  = 2    # output horizons: index-0 → t+30 min, index-1 → t+60 min


# ── Parallel-scan fallback (pure-PyTorch, P100-compatible) ────────────────────
def _fast_parallel_scan(x, dt, A, B, C, chunk_size,
                         D=None, z=None, dt_bias=None, initial_states=None,
                         seq_idx=None, cu_seqlens=None, dt_softplus=True,
                         dt_limit=(0.0, float("inf")), return_final_states=False,
                         return_varlen_states=False):
    B_sz, L, nheads, headdim = x.shape
    ngroups = B.shape[2]
    g = nheads // ngroups
    if dt_bias is not None:
        dt = dt + dt_bias.view(1, 1, nheads)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = dt.clamp(1e-4, 1.0).float()
    B_exp = B.repeat_interleave(g, dim=2).float()
    C_exp = C.repeat_interleave(g, dim=2).float()
    A_real = -torch.exp(A.float().clamp(-20, 0))
    alpha = torch.exp(dt * A_real.view(1, 1, nheads))
    v = torch.einsum("blhp,blhn->blhpn", x.float(), B_exp)
    log_alpha = torch.log(alpha.clamp(min=1e-8))
    cum_fac = torch.exp(torch.cumsum(log_alpha, dim=1))
    h0 = (cum_fac[:, :, :, None, None] * initial_states.float().unsqueeze(1)
          if initial_states is not None else 0.0)
    v_norm = v / cum_fac[:, :, :, None, None].clamp(min=1e-8)
    h_all = cum_fac[:, :, :, None, None] * torch.cumsum(v_norm, dim=1) + h0
    h_all = h_all.clamp(-100, 100)
    y = torch.einsum("blhpn,blhn->blhp", h_all, C_exp)
    if D is not None:
        y = y + x.float() * D.view(1, 1, nheads, 1)
    final_state = h_all[:, -1] if return_final_states else None
    return (y, final_state) if return_final_states else y


def apply_mamba_patches() -> bool:
    """
    Inject the pure-PyTorch causal-conv1d and parallel-scan implementations
    into mamba_ssm's internals.  Must be called BEFORE constructing HorizonUNet.
    Returns True on success, False if mamba_ssm / causal_conv1d are not installed.
    """
    try:
        import causal_conv1d.causal_conv1d_interface as _cc1d
        import mamba_ssm.modules.mamba2 as _mamba2_mod
        import mamba_ssm.ops.triton.ssd_combined as _ssd_mod
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

        def _pure_causal_conv1d(x, weight, bias=None, seq_idx=None,
                                 initial_states=None, return_final_states=False,
                                 activation=None):
            pad = weight.shape[-1] - 1
            out = F.conv1d(x, weight.unsqueeze(1), bias=bias,
                           padding=pad, groups=x.shape[1])
            if pad > 0:
                out = out[..., :-pad]
            if activation in ("silu", "swish"):
                out = F.silu(out)
            return (out, None) if return_final_states else out

        if not getattr(_cc1d, "_patched", False):
            _cc1d.causal_conv1d_fn   = _pure_causal_conv1d
            _mamba2_mod.causal_conv1d_fn = _pure_causal_conv1d
            _cc1d._patched = True

        if not getattr(_ssd_mod, "_patched", False):
            _ssd_mod.mamba_chunk_scan_combined   = _fast_parallel_scan
            _mamba2_mod.mamba_chunk_scan_combined = _fast_parallel_scan
            _ssd_mod._patched = True

        return True

    except Exception:           # ImportError or any Triton/CUDA issue
        return False


# ── Model Config (must match training) ────────────────────────────────────────
class ModelConfig:
    LATENT_DIM        = 64
    MAMBA_D_STATE     = 16
    MAMBA_D_CONV      = 4
    MAMBA_EXPAND      = 2
    MAMBA_HEADDIM     = 32
    USE_CHECKPOINT    = False   # disable during inference (no grad needed)
    BOTTLENECK_LAYERS = 2
    DROPOUT_P         = 0.1


# ── Mamba building blocks ──────────────────────────────────────────────────────
def _build_safe_mamba2(d_model, d_state, d_conv, expand, headdim):
    from mamba_ssm import Mamba2

    class _SafeMamba2(Mamba2):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.use_mem_eff_path = False

    return _SafeMamba2(d_model=d_model, d_state=d_state,
                       d_conv=d_conv, expand=expand, headdim=headdim)


class Mamba2DBlock(nn.Module):
    """Scan rows and columns independently; fuse with a learned sigmoid gate."""

    def __init__(self, dim, cfg):
        super().__init__()
        kw = dict(d_model=dim, d_state=cfg.MAMBA_D_STATE, d_conv=cfg.MAMBA_D_CONV,
                  expand=cfg.MAMBA_EXPAND, headdim=cfg.MAMBA_HEADDIM)
        self.mh   = _build_safe_mamba2(**kw)
        self.mw   = _build_safe_mamba2(**kw)
        self.gate = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        xl = x.permute(0, 2, 3, 1)                                  # B H W C
        yh = self.mh(xl.permute(0, 2, 1, 3).reshape(B * W, H, C))  # BW H C
        yh = yh.reshape(B, W, H, C).permute(0, 3, 2, 1)            # B C H W
        yw = self.mw(xl.reshape(B * H, W, C))                       # BH W C
        yw = yw.reshape(B, H, W, C).permute(0, 3, 1, 2)            # B C H W
        g  = torch.sigmoid(self.gate)
        return g * yh + (1.0 - g) * yw


class Mamba2DLayer(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.norm     = nn.GroupNorm(8, dim)
        self.block    = Mamba2DBlock(dim, cfg)
        self.drop     = nn.Dropout2d(p=cfg.DROPOUT_P)
        self.use_ckpt = cfg.USE_CHECKPOINT

    def _fwd(self, x):
        return self.drop(self.block(self.norm(x)))

    def forward(self, x):
        if self.training and self.use_ckpt:
            return checkpoint(self._fwd, x, use_reentrant=False) + x
        return self._fwd(x) + x


# ── Encoder / Decoder blocks ───────────────────────────────────────────────────
class EncBlock(nn.Module):
    def __init__(self, ic, oc, cfg):
        super().__init__()
        self.conv  = nn.Sequential(nn.Conv2d(ic, oc, 3, stride=2, padding=1),
                                   nn.GroupNorm(8, oc), nn.GELU())
        self.mamba = Mamba2DLayer(oc, cfg)

    def forward(self, x):
        return self.mamba(self.conv(x))


class DecBlock(nn.Module):
    def __init__(self, ic, oc, cfg):
        super().__init__()
        self.up   = nn.ConvTranspose2d(ic, oc, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(8, oc)
        self.act  = nn.GELU()
        self.drop = nn.Dropout2d(p=cfg.DROPOUT_P)
        self.mamba = Mamba2DLayer(oc, cfg)

    def forward(self, x):
        return self.mamba(self.drop(self.act(self.norm(self.up(x)))))


# ── Positional encoding helper ─────────────────────────────────────────────────
def _sinusoidal_pos(T, C):
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, C, 2, dtype=torch.float32)
                    * -(math.log(10000.0) / C))
    enc = torch.zeros(T, C)
    enc[:, 0::2] = torch.sin(pos * div)
    enc[:, 1::2] = torch.cos(pos * div[: C // 2])
    return enc.unsqueeze(-1).unsqueeze(-1)    # T C 1 1


# ── Main model ─────────────────────────────────────────────────────────────────
class HorizonUNet(nn.Module):
    """
    Causal-Mamba U-Net.
    Input:  (B, T_PAST, 2, H, W)  – channel-0=VIL, channel-1=IR069
    Output: (B, T_OUT,  1, H, W)  – predicted VIL (normalised)
    """

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        C   = cfg.LATENT_DIM
        Cin = T_PAST * 2

        self.register_buffer("frame_emb_sin", _sinusoidal_pos(T_PAST, 2) * 0.1)
        self.frame_emb_res = nn.Parameter(torch.zeros(T_PAST, 2, 1, 1))

        self.stem = nn.Sequential(nn.Conv2d(Cin, C, 3, padding=1),
                                   nn.GroupNorm(8, C), nn.GELU())

        self.enc1 = EncBlock(C,     2 * C, cfg)
        self.enc2 = EncBlock(2 * C, 4 * C, cfg)
        self.enc3 = EncBlock(4 * C, 8 * C, cfg)
        self.enc4 = EncBlock(8 * C, 8 * C, cfg)

        self.bn      = nn.Sequential(*[Mamba2DLayer(8 * C, cfg)
                                        for _ in range(cfg.BOTTLENECK_LAYERS)])
        self.s4_proj = nn.Sequential(nn.Conv2d(8 * C, 8 * C, 3, padding=1),
                                      nn.GroupNorm(8, 8 * C), nn.GELU())

        self.dec1 = DecBlock(16 * C, 4 * C, cfg)
        self.dec2 = DecBlock(12 * C, 2 * C, cfg)
        self.dec3 = DecBlock( 6 * C,     C, cfg)
        self.dec4 = DecBlock( 3 * C,     C, cfg)

        self.head = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.GroupNorm(8, C), nn.GELU(),
            nn.Conv2d(C, T_OUT, 1))

        self.register_buffer("out_scale", torch.tensor(1.92))
        self.register_buffer("out_shift", torch.tensor(-0.42))

    def forward(self, x_past):
        B, T, C2, H, W = x_past.shape
        frame_emb = self.frame_emb_sin + self.frame_emb_res
        x  = (x_past + frame_emb.unsqueeze(0)).reshape(B, T * C2, H, W)
        h  = self.stem(x)
        s1 = self.enc1(h)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        bn  = self.bn(s4)
        s4p = self.s4_proj(s4)
        d1 = self.dec1(torch.cat([bn,  s4p], 1))
        d2 = self.dec2(torch.cat([d1,  s3 ], 1))
        d3 = self.dec3(torch.cat([d2,  s2 ], 1))
        d4 = self.dec4(torch.cat([d3,  s1 ], 1))
        raw = self.head(d4)
        return (self.out_scale * torch.tanh(raw) + self.out_shift).unsqueeze(2)


# ── Checkpoint loader ──────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: str = "cpu") -> HorizonUNet:
    """
    Load HorizonUNet from a saved checkpoint.

    Checkpoint format (from trainer):
        { 'model_state': state_dict, 'epoch': int, 'val_loss': float, ... }
    Or plain state_dict.
    """
    patches_ok = apply_mamba_patches()
    if not patches_ok:
        raise ImportError(
            "mamba_ssm / causal_conv1d not found. "
            "Install them in the same environment used for training."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)    # handle both formats

    cfg   = ModelConfig()
    cfg.USE_CHECKPOINT = False               # no grad-checkpointing at inference
    model = HorizonUNet(cfg)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
