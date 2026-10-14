#!/usr/bin/env python
"""
P2-B training (review_followup3, 2026-10-14).  Spec: P2/LEARNED_EXTRAPOLATOR_PROTOCOL.md section 4 (locked; checked).
CPU only (torch threads <= 8, run under nice -n 19).  Training uses ONLY training-drive rows of
P2/_extrap_data/samples.npz; validation drives are used only for the hyper-parameter grid criterion and are reported.

  python p2_extrap_train.py grid  <H> <lr>             (primary input, seed 0)  -> _extrap_models/grid_H<H>_lr<lr>.pt
  python p2_extrap_train.py final <variant> <seed>     (variant prefix | prefix_ego; uses the selected H, lr)
  python p2_extrap_train.py select                     -> P2/extrap_hparam_selection.csv, _extrap_models/selected.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p2_extrap_common as X  # noqa: E402
from p2_extrap_data import check_lock, OUT, DD  # noqa: E402

MD = f"{OUT}/_extrap_models"
EPOCHS, BATCH, WD = 15, 2048, 1e-4
GRID = [(256, 1e-3), (256, 3e-4), (512, 1e-3), (512, 3e-4)]
torch.set_num_threads(8)


def build_rows(S, idx, variant):
    """all j = 0..8 for the plan frames idx -> features, targets, mask, j."""
    dec = S["dec"][idx]; ego = S["ego"][idx] if variant == "prefix_ego" else None
    Fs, Ts, Ms, Js = [], [], [], []
    for j in range(X.N_ACT - 1):
        jj = np.full(len(idx), j)
        Fs.append(X.prefix_features(dec, jj, ego)); t, m = X.targets(dec, jj)
        Ts.append(t); Ms.append(m); Js.append(jj)
    return np.concatenate(Fs), np.concatenate(Ts), np.concatenate(Ms), np.concatenate(Js)


class MLP(torch.nn.Module):
    def __init__(self, d_in, H, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu)); self.register_buffer("sd", torch.as_tensor(sd))
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, H), torch.nn.GELU(), torch.nn.Linear(H, H), torch.nn.GELU(),
                                       torch.nn.Linear(H, H), torch.nn.GELU(), torch.nn.Linear(H, X.OUT_SLOTS * 3))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).view(-1, X.OUT_SLOTS, 3)


def masked_loss(pred, tgt, m):
    e = ((pred - tgt) ** 2).sum(-1) * m
    return e.sum() / m.sum()


@torch.no_grad()
def val_pos_err(model, Fv, Tv, Mv):
    tot = n = 0.0
    for s in range(0, len(Fv), 65536):
        p = model(torch.from_numpy(Fv[s:s + 65536]))
        d = torch.linalg.norm(p[..., :2] - torch.from_numpy(Tv[s:s + 65536, :, :2]), dim=-1) * X.POS_SCALE
        m = torch.from_numpy(Mv[s:s + 65536]).float()
        tot += float((d * m).sum()); n += float(m.sum())
    return tot / n


def train(name, variant, H, lr, seed):
    check_lock()
    os.makedirs(MD, exist_ok=True)
    S = dict(np.load(f"{DD}/samples.npz", allow_pickle=False))
    tr = np.where(S["split"] == 0)[0]; va = np.where(S["split"] == 1)[0]
    Ft, Tt, Mt, _ = build_rows(S, tr, variant)
    Fv, Tv, Mv, _ = build_rows(S, va, variant)
    mu = Ft.mean(0); sd = np.maximum(Ft.std(0), 1e-3)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = MLP(Ft.shape[1], H, mu.astype(np.float32), sd.astype(np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    steps = EPOCHS * math.ceil(len(Ft) / BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps)))
    Ftt, Ttt, Mtt = torch.from_numpy(Ft), torch.from_numpy(Tt), torch.from_numpy(Mt).float()
    log = []; t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.from_numpy(rng.permutation(len(Ft)))
        tl = 0.0; nb = 0
        model.train()
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            loss = masked_loss(model(Ftt[b]), Ttt[b], Mtt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += float(loss); nb += 1
        model.eval()
        ve = val_pos_err(model, Fv, Tv, Mv)
        log.append({"run": name, "variant": variant, "H": H, "lr": lr, "seed": seed, "epoch": ep + 1, "train_loss": tl / nb,
                    "val_mean_pos_err_m": ve, "elapsed_s": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
    torch.save({"state_dict": model.state_dict(), "d_in": Ft.shape[1], "H": H, "lr": lr, "seed": seed, "variant": variant,
                "epochs": EPOCHS, "n_train_rows": len(Ft), "n_val_rows": len(Fv)}, f"{MD}/{name}.pt")
    pd.DataFrame(log).to_csv(f"{MD}/{name}_log.csv", index=False)


def load(name):
    ck = torch.load(f"{MD}/{name}.pt", map_location="cpu")
    m = MLP(ck["d_in"], ck["H"], np.zeros(ck["d_in"], np.float32), np.ones(ck["d_in"], np.float32))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def select():
    rows = []
    for H, lr in GRID:
        lg = pd.read_csv(f"{MD}/grid_H{H}_lr{lr:g}_log.csv")
        rows.append({"H": H, "lr": lr, "final_val_mean_pos_err_m": float(lg.val_mean_pos_err_m.iloc[-1]),
                     "final_train_loss": float(lg.train_loss.iloc[-1])})
    D = pd.DataFrame(rows).sort_values("final_val_mean_pos_err_m")
    D["selected"] = False; D.iloc[0, D.columns.get_loc("selected")] = True
    D.to_csv(f"{OUT}/extrap_hparam_selection.csv", index=False)
    best = D.iloc[0]
    json.dump({"H": int(best.H), "lr": float(best.lr), "criterion": "final-epoch validation-drive mean position error (m)"},
              open(f"{MD}/selected.json", "w"), indent=1)
    print(D)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "grid":
        H, lr = int(sys.argv[2]), float(sys.argv[3])
        train(f"grid_H{H}_lr{lr:g}", "prefix", H, lr, 0)
    elif mode == "final":
        sel = json.load(open(f"{MD}/selected.json"))
        v, sd = sys.argv[2], int(sys.argv[3])
        train(f"final_{v}_s{sd}", v, sel["H"], sel["lr"], sd)
    elif mode == "select":
        select()
