"""
Toy re-implementation of AEO (Adaptive Entropy-aware Optimization, ICLR 2025)
for Multimodal Open-Set Test-Time Adaptation, on the Iris dataset.

Setup
-----
* Two "modalities": sepal features (length, width) and petal features (length, width).
* Open set: the model is trained on K = 2 known classes; the held-out class
  appears only at test time as "unknown".
* Distribution shift: the target domain applies a per-modality sensor
  recalibration (scale + offset) and Gaussian noise.
* Test-time adaptation updates only BatchNorm affine parameters (Tent-style),
  without labels and without source data.

Methods compared: Source-only, Tent (entropy minimisation), AEO (UAE + AMP).
Metrics: accuracy on known target samples, AUROC and FPR@95%TPR for
known-vs-unknown detection using the fused prediction's normalised entropy.
"""
import argparse
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

MODALITIES = {"sepal": [0, 1], "petal": [2, 3]}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_domains(unknown_class, seed, shift):
    X, y = load_iris(return_X_y=True)
    X = X.astype(np.float32)
    known_classes = [c for c in range(3) if c != unknown_class]

    known_mask = y != unknown_class
    Xk, yk = X[known_mask], y[known_mask]
    yk = np.array([known_classes.index(c) for c in yk])  # relabel to 0..K-1

    # Source = half of the known data; target = other half + every unknown sample.
    Xs, Xt_known, ys, yt_known = train_test_split(
        Xk, yk, test_size=0.5, stratify=yk, random_state=seed)
    Xt_unk = X[~known_mask]
    Xt = np.concatenate([Xt_known, Xt_unk])
    yt = np.concatenate([yt_known, -np.ones(len(Xt_unk), dtype=int)])  # -1 = unknown

    # Domain shift on the target: each "sensor" is miscalibrated differently.
    rng = np.random.default_rng(seed)
    Xt = Xt.copy()
    Xt[:, MODALITIES["sepal"]] = Xt[:, MODALITIES["sepal"]] * (1 + 0.15 * shift) + 0.4 * shift
    Xt[:, MODALITIES["petal"]] = Xt[:, MODALITIES["petal"]] * (1 - 0.10 * shift) - 0.3 * shift
    Xt += rng.normal(0, 0.25 * shift, Xt.shape).astype(np.float32)

    # Standardise with *source* statistics only (target stats are unknown a priori).
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    Xs, Xt = (Xs - mu) / sd, (Xt - mu) / sd
    return (torch.tensor(Xs), torch.tensor(ys)), (torch.tensor(Xt), torch.tensor(yt)), len(known_classes)


# ----------------------------------------------------------------------------
# Multimodal model: one encoder + unimodal head per modality, plus a fusion head
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, d_in, d_hid=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.BatchNorm1d(d_hid), nn.ReLU(),
            nn.Linear(d_hid, d_hid), nn.BatchNorm1d(d_hid), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class MultimodalNet(nn.Module):
    def __init__(self, num_classes, d_hid=16):
        super().__init__()
        self.encoders = nn.ModuleDict({m: Encoder(len(idx), d_hid) for m, idx in MODALITIES.items()})
        self.uni_heads = nn.ModuleDict({m: nn.Linear(d_hid, num_classes) for m in MODALITIES})
        self.fusion_head = nn.Linear(d_hid * len(MODALITIES), num_classes)

    def forward(self, x):
        feats = {m: self.encoders[m](x[:, idx]) for m, idx in MODALITIES.items()}
        uni_logits = {m: self.uni_heads[m](f) for m, f in feats.items()}
        fused_logits = self.fusion_head(torch.cat([feats[m] for m in MODALITIES], dim=1))
        return fused_logits, uni_logits


def train_source(model, Xs, ys, epochs=300, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        fused, uni = model(Xs)
        loss = F.cross_entropy(fused, ys) + sum(F.cross_entropy(l, ys) for l in uni.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
def norm_entropy(logits):
    """Per-sample entropy normalised to [0, 1] by log K."""
    p = logits.softmax(1)
    return -(p * logits.log_softmax(1)).sum(1) / math.log(logits.shape[1])


def tent_loss(fused, uni):
    return norm_entropy(fused).mean()


def aeo_loss(fused, uni, alpha, beta, lam_amp=1.0, lam_div=1.0):
    h_fused = norm_entropy(fused)
    # Adaptive weight: < 0 for confident (likely known) samples -> minimise entropy,
    #                  > 0 for uncertain (likely unknown) samples -> maximise entropy.
    w = torch.tanh(beta * (h_fused - alpha)).detach()

    # (a) UAE: unknown-aware adaptive entropy optimisation on the fused prediction.
    l_uae = (-w * h_fused).mean()

    # (b) AMP: adaptive entropy on each unimodal prediction, plus adaptive
    #     cross-modal discrepancy (agree on knowns, disagree on unknowns).
    h_uni = torch.stack([norm_entropy(l) for l in uni.values()]).mean(0)
    probs = [l.softmax(1) for l in uni.values()]
    disc = torch.stack([(probs[i] - probs[j]).abs().sum(1) / 2  # total-variation distance in [0, 1]
                        for i in range(len(probs)) for j in range(i + 1, len(probs))]).mean(0)
    l_amp = (-w * h_uni).mean() + (-w * disc).mean()

    # Diversity regulariser: keep the batch-level class marginal spread out
    # so that known samples do not collapse onto a single class.
    p_bar = fused.softmax(1).mean(0)
    l_div = 1 + (p_bar * p_bar.clamp_min(1e-8).log()).sum() / math.log(fused.shape[1])

    return l_uae + lam_amp * l_amp + lam_div * l_div


# ----------------------------------------------------------------------------
# Test-time adaptation
# ----------------------------------------------------------------------------
def configure_for_tta(model):
    """Tent-style: train mode for BN (batch stats), only BN affine params trainable."""
    model.train()
    params = []
    for m in model.modules():
        for p in m.parameters(recurse=False):
            p.requires_grad_(isinstance(m, nn.BatchNorm1d))
            if isinstance(m, nn.BatchNorm1d):
                params.append(p)
        if isinstance(m, nn.BatchNorm1d):
            m.track_running_stats = False
            m.running_mean, m.running_var = None, None
    return params


def adapt(model, Xt, loss_fn, steps, batch_size, lr, seed):
    params = configure_for_tta(model)
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        for idx in torch.randperm(len(Xt), generator=g).split(batch_size):
            if len(idx) < 2:
                continue
            fused, uni = model(Xt[idx])
            loss = loss_fn(fused, uni)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, Xt, yt, use_batch_stats, score_fn="msp"):
    model.train(use_batch_stats)  # adapted models have no running stats -> batch stats
    fused, _ = model(Xt)
    # Unknown-ness score (higher = more likely unknown). The paper's detector uses
    # S(x) = MSP and flags S(x) < eta as unknown, so we negate MSP here.
    if score_fn == "msp":
        score = -fused.softmax(1).max(1).values.numpy()
    else:
        score = norm_entropy(fused).numpy()
    is_unk = (yt == -1).numpy()
    known = ~is_unk

    acc = (fused.argmax(1).numpy()[known] == yt.numpy()[known]).mean()
    auroc = roc_auc_score(is_unk, score)
    thr = np.percentile(score[known], 95)  # accept 95% of known samples
    fpr95 = (score[is_unk] <= thr).mean()  # fraction of unknowns wrongly accepted
    return acc, auroc, fpr95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unknown-class", type=int, default=0,
                    help="Iris class held out as unknown (0=setosa, 1=versicolor, 2=virginica)")
    ap.add_argument("--shift", type=float, default=1.0, help="Domain-shift strength (0 = none)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.4, help="Entropy threshold for known/unknown")
    ap.add_argument("--beta", type=float, default=3.0, help="Sharpness of the tanh weight")
    ap.add_argument("--steps", type=int, default=10, help="Passes over the target stream")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--score", choices=["msp", "entropy"], default="msp",
                    help="Unknown-detection score S(x): max softmax prob (paper default) or entropy")
    args = ap.parse_args()

    names = ["Source-only", "Tent", "UAE only", "AEO"]
    results = {n: [] for n in names}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        (Xs, ys), (Xt, yt), K = make_domains(args.unknown_class, seed, args.shift)
        source = train_source(MultimodalNet(K), Xs, ys)

        results["Source-only"].append(evaluate(source, Xt, yt, use_batch_stats=False, score_fn=args.score))
        for name, fn in [("Tent", tent_loss),
                         ("UAE only", lambda f, u: aeo_loss(f, u, args.alpha, args.beta, lam_amp=0.0)),
                         ("AEO", lambda f, u: aeo_loss(f, u, args.alpha, args.beta))]:
            model = adapt(copy.deepcopy(source), Xt, fn, args.steps, args.batch_size, args.lr, seed)
            results[name].append(evaluate(model, Xt, yt, use_batch_stats=True, score_fn=args.score))

    cls = load_iris().target_names[args.unknown_class]
    print(f"Unknown class: {cls} | shift={args.shift} | score={args.score} | {args.seeds} seeds (mean ± std)\n")
    print(f"{'Method':<12} {'Known Acc ↑':>14} {'AUROC ↑':>14} {'FPR95 ↓':>14}")
    for n in names:
        r = np.array(results[n]) * 100
        m, s = r.mean(0), r.std(0)
        print(f"{n:<12} " + " ".join(f"{m[i]:>7.1f} ± {s[i]:<4.1f}" for i in range(3)))


if __name__ == "__main__":
    main()
