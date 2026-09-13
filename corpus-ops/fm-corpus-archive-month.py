import sys, os, tarfile, pickle, random

if len(sys.argv) != 2:
    sys.exit(f"usage: {sys.argv[0]} <month>  (e.g. 2020-01)")
month = sys.argv[1]
D_ROOT = "/mnt/d/firstmate/projects/cs_thesis_2/local-run/data/processed_games"
C_ROOT = "/mnt/c/Users/bacsa/Downloads/fm-corpus-processed"
ARCHIVE_DIR = "/mnt/d/firstmate/projects/cs_thesis_2/local-run/data/processed_games_archive"

os.makedirs(ARCHIVE_DIR, exist_ok=True)
dst = os.path.join(ARCHIVE_DIR, f"{month}.tar.gz")
tmp_dst = dst + ".tmp"

sources = []
for root in (D_ROOT, C_ROOT):
    d = os.path.join(root, month)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".pkl") or f == ".sampling_summary.json":
                sources.append(os.path.join(d, f))

pkl_sources = [s for s in sources if s.endswith(".pkl")]
if len(pkl_sources) < 30000:
    print(f"{month}: ABORT - only {len(pkl_sources)} source pkls found, expected >=30000", flush=True)
    sys.exit(1)

# 2026-08-24: guard against a same-basename collision between D_ROOT and
# C_ROOT (both are legitimate sources for a split-storage month). tarfile
# happily writes two members with an identical name with no error, and the
# count-based verification below can't detect it (both copies still count),
# so extraction later would silently shadow one file's data. Caught in code
# review; verified empirically that no actual month has ever collided (the
# 5 known split months have disjoint index ranges), but the code shouldn't
# rely on that holding forever without checking.
basenames = [os.path.basename(s) for s in sources]
dupes = {n for n in basenames if basenames.count(n) > 1}
if dupes:
    print(f"{month}: ABORT - {len(dupes)} filename collision(s) between source dirs: {sorted(dupes)[:5]}", flush=True)
    sys.exit(1)

with tarfile.open(tmp_dst, "w:gz", compresslevel=6) as tf:
    for s in sources:
        arcname = os.path.basename(s)
        tf.add(s, arcname=arcname)

# verify: member count and a content spot-check straight from the archive
with tarfile.open(tmp_dst, "r:gz") as tf:
    names = tf.getnames()
    pkl_names = [n for n in names if n.endswith(".pkl")]
    if len(pkl_names) != len(pkl_sources):
        print(f"{month}: ABORT - archive has {len(pkl_names)} pkl members, expected {len(pkl_sources)}", flush=True)
        os.remove(tmp_dst)
        sys.exit(1)
    random.seed(7)
    sample = random.sample(pkl_names, min(20, len(pkl_names)))
    for name in sample:
        member = tf.extractfile(name)
        obj = pickle.load(member)
        assert set(obj.keys()) == {'Black','BlackElo','Clocks','Moves','Positions','Result','Time','White','WhiteElo'}, f"{month}: bad schema in {name}"

os.replace(tmp_dst, dst)

# only now delete the originals
for s in sources:
    os.remove(s)
for root in (D_ROOT, C_ROOT):
    d = os.path.join(root, month)
    if os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d)

archived_size = os.path.getsize(dst)
print(f"{month}: OK archived={len(pkl_names)} pkls, {archived_size/1e6:.1f}MB, source dirs cleaned", flush=True)
