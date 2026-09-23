"""A4 end-to-end CNN-BiLSTM detector initialized from the frozen rating model.

Plan: analysis/anomaly-extra-arms-plan.md section 5 (A4).
Frozen CNN (per-board encoder), fine-tuned BiLSTM + causal Bahdanau attention
(lr 1e-4), new gated-attention MIL head with per-ply instance logits (lr 1e-3).

Stages (both resumable):
  --stage cache : read corpus pickles (read-only), run the frozen CNN trunk once,
                  write float16 per-ply embeddings + standardized clocks, one
                  shard per cell under extra_arms/A4_cache/. Not a result; may
                  run before the A0 gate.
  --stage train : needs A0_GATE_PASS; trains, scores, writes results.json.

FLAG (plan): a board-level model can learn to recognize engine-style moves.
Reported in its own row with the withheld-band, low-rate, hard-negative and
engine-proximity checks.
"""
import argparse
import glob
import json
import os
import pickle
import time

import numpy as np

import extra_arms_common as C

EMB_DIM = 256
MAXLEN = 100


def cache_dir(smoke):
    d = os.path.join(C.SMOKE_ROOT if smoke else C.OUT_ROOT, "A4_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _read_game(fp):
    import torch  # noqa: F401  (pickles hold torch tensors)
    import extract_detector_features_v2 as X
    with open(fp, "rb") as f:
        raw = pickle.load(f)
    pos = np.stack([p.numpy() for p in raw["Positions"][:MAXLEN]]).astype(np.uint8)
    n = pos.shape[0]
    clk_z, _ = X._clocks_z_from_strings(raw.get("Clocks", []) or [], n)
    return pos, clk_z.numpy().astype(np.float32)


def load_frozen(device):
    import extract_detector_features_v2 as X
    return X._load_model(C.CHECKPOINT, device)


def cnn_trunk(model, boards):
    import torch.nn.functional as F
    x = F.leaky_relu(model.bn1(model.conv1(boards)))
    x = model.pool(x)
    x = F.leaky_relu(model.bn2(model.conv2(x)))
    x = model.pool(x)
    x = F.leaky_relu(model.bn3(model.conv3(x)))
    x = model.pool(x)
    x = F.leaky_relu(model.bn4(model.conv4(x)))
    return x.reshape(x.shape[0], -1)  # dropout1 is identity in eval mode


def stage_cache(smoke, workers):
    import torch
    from multiprocessing import Pool
    data = C.Data(smoke=smoke)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen(device)
    cd = cache_dir(smoke)
    cells = np.array([g.split("/")[0] for g in data.gid])
    t0 = time.time(); done = 0
    with Pool(workers) as pool:
        for cell in sorted(set(cells.tolist())):
            sp = os.path.join(cd, f"{cell}.npz")
            idx = np.where(cells == cell)[0]
            if os.path.exists(sp):
                z = np.load(sp)
                if np.array_equal(z["gid"].astype(str), data.gid[idx]):
                    continue
            files = [os.path.join(C.CORPUS_DIR, g) for g in data.gid[idx]]
            embs, clks = [], []
            for j, (pos, clk) in enumerate(pool.imap(_read_game, files, chunksize=8)):
                if pos.shape[0] != data.nply[idx[j]]:
                    raise RuntimeError(f"ply mismatch {data.gid[idx[j]]}: {pos.shape[0]} vs {data.nply[idx[j]]}")
                embs.append(pos); clks.append(clk)
            boards = np.concatenate(embs)
            out = np.empty((boards.shape[0], EMB_DIM), np.float16)
            with torch.no_grad():
                for i in range(0, boards.shape[0], 16384):
                    b = torch.from_numpy(boards[i:i + 16384]).to(device).float()
                    out[i:i + 16384] = cnn_trunk(model, b).half().cpu().numpy()
            tmp = sp + ".tmp.npz"
            np.savez(tmp, gid=data.gid[idx], emb=out, clk=np.concatenate(clks), nply=data.nply[idx])
            os.replace(tmp, sp)
            done += len(idx)
            C.log(f"cache {cell}: {len(idx)} games, {boards.shape[0]} plies, {done/(time.time()-t0):.1f} g/s")
    with open(os.path.join(cd, "CACHE_DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S %z\n"))


def load_cache(data, smoke):
    cd = cache_dir(smoke)
    cells = np.array([g.split("/")[0] for g in data.gid])
    offs = np.zeros(data.n + 1, np.int64)
    offs[1:] = np.cumsum(np.minimum(data.nply, MAXLEN))
    E = np.empty((offs[-1], EMB_DIM), np.float16)
    K = np.empty(offs[-1], np.float32)
    for cell in sorted(set(cells.tolist())):
        idx = np.where(cells == cell)[0]
        z = np.load(os.path.join(cd, f"{cell}.npz"))
        assert np.array_equal(z["gid"].astype(str), data.gid[idx]), cell
        pos = 0
        emb = z["emb"]; clk = z["clk"]
        for i in idx:  # games of a cell are contiguous in data order but copy per game to be safe
            n = int(data.nply[i])
            E[offs[i]:offs[i] + n] = emb[pos:pos + n]; K[offs[i]:offs[i] + n] = clk[pos:pos + n]
            pos += n
    return E, K, offs


def make_model(device):
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    base = load_frozen(torch.device("cpu"))

    class A4(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = base.lstm
            self.attention = base.attention
            self.drop = nn.Dropout(0.5)
            H = 128
            self.inst = nn.Linear(H, 1)
            self.att_v = nn.Linear(H, 64)
            self.att_u = nn.Linear(H, 64)
            self.att_w = nn.Linear(64, 1)
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, emb, clk, lengths, pool_mask):
            x = torch.cat([self.drop(emb), clk.unsqueeze(2)], dim=2)
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=emb.shape[1])
            ctx, _ = self.attention.forward_causal(out, lengths)
            h = out + ctx
            s = self.inst(h).squeeze(-1)
            e = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
            e = e.masked_fill(~pool_mask, -1e4)
            a = torch.softmax(e, dim=1)
            return (a * s).sum(1) + self.bias, s, a

    m = A4().to(device)
    backbone = list(m.lstm.parameters()) + list(m.attention.parameters())
    head = [p for n, p in m.named_parameters() if not (n.startswith("lstm.") or n.startswith("attention."))]
    return m, backbone, head


class Batcher:
    def __init__(self, E, K, offs, data):
        import torch
        self.t = torch
        self.E = torch.from_numpy(E); self.K = torch.from_numpy(K)
        self.offs = offs; self.data = data
        self.sus = data.pp["is_suspect_move"]
        self.ar = np.arange(MAXLEN)

    def get(self, gidx, device):
        t = self.t
        n = np.minimum(self.data.nply[gidx], MAXLEN)
        valid = self.ar[None, :] < n[:, None]
        rows = (self.offs[gidx][:, None] + self.ar[None, :])
        rows = np.where(valid, rows, 0)
        ridx = t.from_numpy(rows.reshape(-1))
        emb = self.E.index_select(0, ridx).view(len(gidx), MAXLEN, EMB_DIM)
        clk = self.K.index_select(0, ridx).view(len(gidx), MAXLEN)
        vt = t.from_numpy(valid)
        emb = emb * vt[..., None]; clk = clk * vt
        prow = self.data.start[gidx][:, None] + self.ar[None, :]
        sus = np.where(valid, self.sus[np.minimum(prow, len(self.sus) - 1)] > 0.5, False)
        pool = sus | (~sus.any(1, keepdims=True) & valid)
        return (emb.to(device, non_blocking=True).float(), clk.to(device, non_blocking=True),
                t.from_numpy(n.astype(np.int64)).to(device), t.from_numpy(pool).to(device),
                t.from_numpy(self.data.y[gidx].astype(np.float32)).to(device), prow, valid)


def predict(model, bat, gidx, device, bs=128):
    import torch
    model.eval()
    gs, pi, ps, pa = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(gidx), bs):
            emb, clk, n, pool, _, prow, valid = bat.get(gidx[i:i + bs], device)
            logit, s, a = model(emb, clk, n, pool)
            gs.append(torch.sigmoid(logit).cpu().numpy())
            pi.append(prow[valid]); ps.append(s.cpu().numpy()[valid]); pa.append(a.cpu().numpy()[valid])
    return np.concatenate(gs), np.concatenate(pi), np.concatenate(ps), np.concatenate(pa)


def stage_train(smoke):
    C.require_gate(smoke)
    cd = cache_dir(smoke)
    while not os.path.exists(os.path.join(cd, "CACHE_DONE")):
        C.log("waiting for A4 cache"); time.sleep(300)

    def body(d):
        import torch
        from sklearn.metrics import roc_auc_score
        seed = 0
        torch.manual_seed(seed); np.random.seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = C.Data(smoke=smoke)
        split = data.split("v2")
        E, K, offs = load_cache(data, smoke)
        bat = Batcher(E, K, offs, data)
        model, backbone, head = make_model(device)
        opt = torch.optim.AdamW([{"params": backbone, "lr": 1e-4}, {"params": head, "lr": 1e-3}],
                                weight_decay=1e-4)
        tr = np.where(split == "train")[0]; va = np.where(split == "val")[0]
        npos = data.y[tr].sum(); nneg = len(tr) - npos
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([nneg / max(npos, 1)], device=device))
        rng = np.random.default_rng(seed)
        rows, best, best_state, bad = [], -1.0, None, 0
        max_epochs = 1 if smoke else 15
        for ep in range(max_epochs):
            model.train(); t0 = time.time(); tot = 0.0
            perm = rng.permutation(tr)
            # batch 256 as pre-registered, as 4 micro-batches of 64: the causal Bahdanau
            # attention materializes (B, 100, 100, 64) and B=256 does not fit 24 GB.
            for i in range(0, len(perm), 256):
                chunk = perm[i:i + 256]
                opt.zero_grad()
                for j in range(0, len(chunk), 64):
                    emb, clk, n, pool, y, _, _ = bat.get(chunk[j:j + 64], device)
                    logit, _, _ = model(emb, clk, n, pool)
                    loss = lossf(logit, y) * (len(y) / len(chunk))
                    loss.backward()
                    tot += float(loss) * len(chunk)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            pv, _, _, _ = predict(model, bat, va, device)
            vauc = float(roc_auc_score(data.y[va], pv))
            row = {"epoch": ep, "train_loss": tot / len(tr), "val_auc": vauc, "sec": round(time.time() - t0, 1)}
            rows.append(row); C.log(json.dumps(row))
            C.write_json(os.path.join(d, "train_log.json"), {"rows": rows})
            if vauc > best:
                best, bad = vauc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(d, "model_best.pt"))
            else:
                bad += 1
                if bad >= 3:
                    break
        model.load_state_dict(best_state)
        gscore, pidx, ps, pa = predict(model, bat, np.arange(data.n), device)
        total_plies = len(data.pp["is_suspect_move"])
        ply_s = np.full(total_plies, np.nan, np.float32); ply_s[pidx] = ps
        ply_a = np.full(total_plies, np.nan, np.float32); ply_a[pidx] = pa
        C.atomic_savez(os.path.join(d, "scores.npz"), gid=data.gid, split=split.astype(str), y=data.y,
                       score=gscore, ply_s=ply_s, ply_a=ply_a)
        a0s, a0g = C.load_scores("A0", smoke)
        a0 = a0s if np.array_equal(a0g, data.gid) else None
        n_boot = 50 if smoke else C.N_BOOT
        res = C.base_results("A4", smoke, "v2", {
            "model": "frozen rating CNN -> fine-tuned BiLSTM + causal Bahdanau attention -> MIL head",
            "flag": ("board-level model may recognize engine-style moves directly; on this corpus that "
                     "can approach the leaked engine-agreement signal. Reported separately; read with "
                     "withheld_band, low-rate, strength_check and engine_proximity."),
            "seed": seed, "best_val_auc": best, "train_log": rows,
            "game_level": C.game_level_report(data, split, gscore, n_boot, a0_score=a0),
            "localization_instance_logit": C.localization_report(data, split, ply_s, engine=C.load_engine(data)),
            "localization_attention": C.localization_report(data, split, ply_a),
        })
        C.write_json(os.path.join(d, "results.json"), res)

    C.run_arm("A4", smoke, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cache", "train"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.stage == "cache":
        stage_cache(a.smoke, a.workers)
    else:
        stage_train(a.smoke)


if __name__ == "__main__":
    main()
