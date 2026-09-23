"""A3 sequence detector on the frozen rating model's per-move outputs.

Plan: analysis/anomaly-extra-arms-plan.md section 5 (A3).
BiGRU over engine-free per-move features from rating_cheap.npz, per-ply
instance logits, gated attention (MIL) pooling over suspect-side plies.
Validation-only grid H in {64,128} x dropout in {0.1,0.3} (seed 0); the best
val-AUC config is the only one scored on withheld conditions; --seed 1 then
retrains that chosen config (read from the seed-0 tuning log).

  python extra_arm_a3.py --seed 0 [--split v2|grouped] [--smoke]
  python extra_arm_a3.py --seed 1                     # needs A3 seed 0 DONE
"""
import argparse
import json
import os
import time

import numpy as np

import extra_arms_common as C

FEATURES = ["r_hat_suspect", "r_hat_other", "dev_signed", "d_t", "alpha_n", "run_mean", "run_std",
            "first_diff", "second_diff", "is_capture", "is_check", "material_balance",
            "clock_remaining", "clock_delta", "clock_delta_missing", "is_suspect_move", "ply_pos"]
BINARY = {"is_capture", "is_check", "clock_delta_missing", "is_suspect_move", "ply_pos"}
GRID = [(64, 0.1), (64, 0.3), (128, 0.1), (128, 0.3)]
MAXLEN = 100


def build_ply_matrix(data):
    pp = data.pp
    nply_full = np.diff(data.gs)
    band_per_ply = np.repeat(np.asarray(_full_band(data), np.float32), nply_full)
    nply_per_ply = np.repeat(nply_full.astype(np.float32), nply_full)
    cd = pp["clock_delta"].astype(np.float32)
    cols = {
        "r_hat_suspect": pp["r_hat_suspect"], "r_hat_other": pp["r_hat_other"],
        "dev_signed": pp["r_hat_suspect"] - band_per_ply, "d_t": pp["d_t"],
        "alpha_n": pp["alpha"] * nply_per_ply, "run_mean": pp["run_mean"], "run_std": pp["run_std"],
        "first_diff": pp["first_diff"], "second_diff": pp["second_diff"],
        "is_capture": pp["is_capture"], "is_check": pp["is_check"],
        "material_balance": pp["material_balance"], "clock_remaining": pp["clock_remaining"],
        "clock_delta": cd, "clock_delta_missing": (~np.isfinite(cd)).astype(np.float32),
        "is_suspect_move": pp["is_suspect_move"], "ply_pos": pp["ply_idx"] / 100.0,
    }
    return np.stack([np.asarray(cols[f], np.float32) for f in FEATURES], axis=1)


def _full_band(data):
    # data.band may be a smoke subset; per-ply broadcast needs every game
    z = np.load(os.path.join(C.FEAT_DIR, "rating_cheap.npz"), allow_pickle=True)
    return z["band"].astype(np.float32)


def standardize(M, train_ply_mask):
    mu = np.zeros(M.shape[1], np.float32); sd = np.ones(M.shape[1], np.float32)
    for j, f in enumerate(FEATURES):
        if f in BINARY:
            continue
        col = M[train_ply_mask, j]
        col = col[np.isfinite(col)]
        mu[j] = col.mean(); sd[j] = max(col.std(), 1e-6)
    M = (M - mu) / sd
    M[~np.isfinite(M)] = 0.0
    return M, mu, sd


def make_model(nin, hidden, dropout):
    import torch
    import torch.nn as nn

    class MILGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(nin, hidden), nn.GELU(), nn.Dropout(dropout))
            self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True,
                              dropout=dropout)
            self.drop = nn.Dropout(dropout)
            self.inst = nn.Linear(2 * hidden, 1)
            self.att_v = nn.Linear(2 * hidden, hidden)
            self.att_u = nn.Linear(2 * hidden, hidden)
            self.att_w = nn.Linear(hidden, 1)
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, x, lengths, pool_mask):
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            h = self.inp(x)
            packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
            out = self.drop(out)
            s = self.inst(out).squeeze(-1)
            e = self.att_w(torch.tanh(self.att_v(out)) * torch.sigmoid(self.att_u(out))).squeeze(-1)
            e = e.masked_fill(~pool_mask, -1e4)
            a = torch.softmax(e, dim=1)
            return (a * s).sum(1) + self.bias, s, a

    return MILGRU()


class Batcher:
    """Gathers padded (B, 100, F) batches from the flat ply matrix on GPU."""

    def __init__(self, M, data, device):
        import torch
        self.torch = torch
        self.M = torch.from_numpy(M).to(device)
        self.start = torch.from_numpy(data.start.astype(np.int64)).to(device)
        self.nply = torch.from_numpy(np.minimum(data.nply, MAXLEN).astype(np.int64)).to(device)
        self.y = torch.from_numpy(data.y.astype(np.float32)).to(device)
        self.ar = torch.arange(MAXLEN, device=device)
        self.sus_col = FEATURES.index("is_suspect_move")
        self.susraw = torch.from_numpy(data.pp["is_suspect_move"] > 0.5).to(device)

    def get(self, gidx):
        t = self.torch
        g = t.as_tensor(gidx, device=self.M.device)
        st = self.start[g]; n = self.nply[g]
        valid = self.ar[None, :] < n[:, None]
        idx = (st[:, None] + self.ar[None, :]).clamp(max=self.M.shape[0] - 1)
        x = self.M[idx] * valid[..., None]
        pool = self.susraw[idx] & valid
        pool = pool | (~pool.any(1, keepdim=True) & valid)  # games with no suspect ply: pool all
        return x, n, pool, self.y[g], idx, valid


def predict(model, bat, gidx, bs=2048):
    import torch
    model.eval()
    gs, ps, pa, pi = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(gidx), bs):
            x, n, pool, _, idx, valid = bat.get(gidx[i:i + bs])
            logit, s, a = model(x, n, pool)
            gs.append(torch.sigmoid(logit).float().cpu().numpy())
            pi.append(idx[valid].cpu().numpy()); ps.append(s[valid].float().cpu().numpy()); pa.append(a[valid].float().cpu().numpy())
    return np.concatenate(gs), np.concatenate(pi), np.concatenate(ps), np.concatenate(pa)


def train_one(M, data, split, hidden, dropout, seed, device, max_epochs, log_rows, tag):
    import torch
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    bat = Batcher(M, data, device)
    tr = np.where(split == "train")[0]; va = np.where(split == "val")[0]
    model = make_model(M.shape[1], hidden, dropout).to(device)
    npos = data.y[tr].sum(); nneg = len(tr) - npos
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([nneg / max(npos, 1)], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best, best_state, bad, patience = -1.0, None, 0, 5
    for ep in range(max_epochs):
        model.train(); t0 = time.time(); tot = 0.0
        perm = rng.permutation(tr)
        for i in range(0, len(perm), 512):
            x, n, pool, y, _, _ = bat.get(perm[i:i + 512])
            logit, _, _ = model(x, n, pool)
            loss = lossf(logit, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss) * len(y)
        pv, _, _, _ = predict(model, bat, va)
        vauc = float(roc_auc_score(data.y[va], pv))
        row = {"tag": tag, "hidden": hidden, "dropout": dropout, "seed": seed, "epoch": ep,
               "train_loss": tot / len(tr), "val_auc": vauc, "sec": round(time.time() - t0, 1)}
        log_rows.append(row); C.log(json.dumps(row))
        if vauc > best:
            best, bad = vauc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, bat, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="v2", choices=["v2", "grouped"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    C.require_gate(a.smoke)
    arm = "A3" + ("g" if a.split == "grouped" else "") + f"_seed{a.seed}"

    def body(d):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = C.Data(smoke=a.smoke)
        split = data.split(a.split)
        max_epochs = 2 if a.smoke else 40
        M = build_ply_matrix(data)
        train_plies = np.zeros(M.shape[0], bool)
        for i in np.where(split == "train")[0]:
            train_plies[data.start[i]:data.start[i] + data.nply[i]] = True
        M, mu, sd = standardize(M, train_plies)
        tuning = []
        if a.seed == 0:
            grid = GRID[:2] if a.smoke else GRID
            best_cfg, best_auc, best_fit = None, -1, None
            for hidden, dropout in grid:
                fit = train_one(M, data, split, hidden, dropout, 0, device, max_epochs, tuning,
                                f"grid_h{hidden}_p{dropout}")
                if fit[2] > best_auc:
                    best_cfg, best_auc, best_fit = (hidden, dropout), fit[2], fit
                else:
                    del fit
                torch.cuda.empty_cache()
            model, bat, vauc = best_fit  # seed 0: the chosen grid model itself
        else:
            src = os.path.join(C.SMOKE_ROOT if a.smoke else C.OUT_ROOT,
                               "A3" + ("g" if a.split == "grouped" else "") + "_seed0", "tuning_log.json")
            with open(src) as f:
                best_cfg = tuple(json.load(f)["chosen"])
            model, bat, vauc = train_one(M, data, split, best_cfg[0], best_cfg[1], a.seed, device,
                                         max_epochs, tuning, f"seed{a.seed}_chosen")
        C.log(f"chosen config hidden={best_cfg[0]} dropout={best_cfg[1]}")
        C.write_json(os.path.join(d, "tuning_log.json"),
                     {"grid": GRID, "chosen": list(best_cfg), "selection": "max val AUC, seed 0",
                      "rows": tuning})
        allg = np.arange(data.n)
        gscore, pidx, ps, pa = predict(model, bat, allg)
        ply_s = np.full(M.shape[0], np.nan, np.float32); ply_s[pidx] = ps
        ply_a = np.full(M.shape[0], np.nan, np.float32); ply_a[pidx] = pa
        torch.save(model.state_dict(), os.path.join(d, "model.pt"))
        C.atomic_savez(os.path.join(d, "scores.npz"), gid=data.gid, split=split.astype(str), y=data.y,
                       score=gscore, ply_s=ply_s, ply_a=ply_a, feat_mu=mu, feat_sd=sd)
        n_boot = 50 if a.smoke else C.N_BOOT
        a0 = None
        if a.split == "v2":
            a0s, a0g = C.load_scores("A0", a.smoke)
            if np.array_equal(a0g, data.gid):
                a0 = a0s
        eng = C.load_engine(data)
        res = C.base_results(arm, a.smoke, a.split, {
            "model": "BiGRU 2-layer + gated-attention MIL pooling", "features": FEATURES,
            "chosen_config": {"hidden": best_cfg[0], "dropout": best_cfg[1]}, "seed": a.seed,
            "best_val_auc": vauc,
            "deviations": ["plan text says 'linear 16 to H'; the input width is 17 because the "
                           "pre-registered clock_delta missing indicator is a separate column"],
            "game_level": C.game_level_report(data, split, gscore, n_boot, a0_score=a0),
            "localization_instance_logit": C.localization_report(data, split, ply_s, engine=eng),
            "localization_attention": C.localization_report(data, split, ply_a),
        })
        C.write_json(os.path.join(d, "results.json"), res)

    C.run_arm(arm, a.smoke, body)


if __name__ == "__main__":
    main()
