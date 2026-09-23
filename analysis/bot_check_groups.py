#!/usr/bin/env python3
"""Bot-account check, stage 1 (HPC, read-only on data): build three game groups
from the frozen model's per-game test errors and read player names + ply counts
from the archived corpus pickles.

Per-game error = mean(white_err, black_err). Filter: >= 40 plies.
Groups (200 games each, in rank order after the filter):
  worst  = highest error, lowest = lowest error, random = seeded permutation.
"""
import csv, json, pickle, sys, tarfile
from multiprocessing import Pool
from pathlib import Path
import numpy as np

CSV = "analysis/heldout_test_eval/attn_tuned__best.csv"
ARCH = Path("data/corpus_archive_transfer")
OUT = Path("scratch/bot-check")
N, MINPLY, NCAND, SEED = 200, 40, 700, 20260917

def read_month(args):
    month, wanted = args
    wanted = set(wanted); got = {}
    with tarfile.open(ARCH / f"{month}.tar.gz", "r:gz") as tf:
        for m in tf:
            base = m.name.rsplit("/", 1)[-1]
            if base in wanted:
                d = pickle.load(tf.extractfile(m))
                got[f"{month}_{base[:-4]}"] = {"White": d.get("White"), "Black": d.get("Black"),
                                               "nplies": len(d.get("Moves", [])), "Result": d.get("Result")}
                if len(got) == len(wanted):
                    break
    return got

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(CSV)))
    ids = np.array([r["game_id"] for r in rows])
    err = np.array([(float(r["white_err"]) + float(r["black_err"])) / 2 for r in rows])
    order = np.argsort(err, kind="stable")
    rng = np.random.default_rng(SEED)
    cand = {"worst": list(ids[order[::-1][:NCAND]]), "lowest": list(ids[order[:NCAND]]),
            "random": list(ids[rng.permutation(len(ids))[:NCAND]])}
    by_month = {}
    for g in set(x for v in cand.values() for x in v):
        month, _, rest = g.partition("_game_")
        by_month.setdefault(month, []).append(f"game_{rest}.pkl")
    info = {}
    with Pool(8) as p:
        for got in p.imap_unordered(read_month, sorted(by_month.items())):
            info.update(got)
    errmap = dict(zip(ids, err)); tc = {r["game_id"]: r["time_control"] for r in rows}
    out = {"seed": SEED, "min_plies": MINPLY, "n_test_games": len(rows), "groups": {}}
    for name, lst in cand.items():
        kept, rejected = [], 0
        for g in lst:
            gi = info.get(g)
            if gi is None:
                print("missing", g, file=sys.stderr); continue
            if gi["nplies"] < MINPLY:
                rejected += 1; continue
            kept.append({"game_id": g, "err": float(errmap[g]), "time_control": tc[g], **gi})
            if len(kept) == N:
                break
        assert len(kept) == N, (name, len(kept))
        out["groups"][name] = {"games": kept, "rejected_short": rejected}
    json.dump(out, open(OUT / "groups.json", "w"), indent=1)
    for k, v in out["groups"].items():
        e = [x["err"] for x in v["games"]]
        print(k, "rejected", v["rejected_short"], "err range", min(e), max(e))

if __name__ == "__main__":
    main()
