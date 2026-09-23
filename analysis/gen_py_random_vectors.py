"""Emit CPython random.Random reference vectors in a dependency-free flat format.

Format: one record per line, "<name> <value> <value> ...". Regenerate with:
    python gen_vectors.py analysis/scanner-rs/tests/py_random_vectors.txt
"""
import random, sys

lines = []
def rec(name, vals):
    lines.append(name + " " + " ".join(str(v) for v in vals))

# 1. raw 32-bit stream
r = random.Random(42)
rec("seed42_getrandbits32", [r.getrandbits(32) for _ in range(24)])

# 2. mixed widths across seed shapes (0, small, >32-bit, 64-bit)
WIDTHS = [1,2,3,7,8,17,31,32,33,53,64,64,32,5]
rec("widths", WIDTHS)
for seed in (0, 1, 42, 12345, 2**31, 2**32, 2**64 - 1):
    r = random.Random(seed)
    rec(f"mixed_{seed}", [r.getrandbits(k) for k in WIDTHS])

# 3. the exact call the reservoir makes, across the interesting range
r = random.Random(42)
rec("randint_reservoir_42", [r.randint(0, e - 1) for e in range(30001, 33001)])
r = random.Random(42)
rec("randint_small_42", [r.randint(0, e - 1) for e in range(1, 501)])

# 4. end-to-end Algorithm R, mirroring preprocess_lichess.py:147-154
def replay(verdicts, max_games, seed):
    rng = random.Random(seed)
    reservoir, eligible = [], 0
    for ordinal, b in enumerate(verdicts):
        if b != 1:
            continue
        eligible += 1
        if len(reservoir) < max_games:
            reservoir.append(ordinal)
        else:
            j = rng.randint(0, eligible - 1)
            if j < max_games:
                reservoir[j] = ordinal
    return reservoir

bits = [1 if (i * 2654435761) % 1000 < 70 else 0 for i in range(400000)]
rec("algoR_stream_len", [len(bits)])
rec("algoR_stream_eligible", [sum(bits)])
for mg, sd in ((1000, 42), (1000, 7), (30000, 42), (137, 99)):
    rec(f"algoR_{mg}_{sd}", replay(bits, mg, sd))

open(sys.argv[1], "w").write("\n".join(lines) + "\n")
print("wrote", sys.argv[1], "records:", len(lines))
