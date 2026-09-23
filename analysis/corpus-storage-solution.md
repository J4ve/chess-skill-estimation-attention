# Full-corpus training data storage: recommendation

Date: 2026-08-24. Scope: `CLAUDE_TASK_2.md`.

Subject: how `ChessGamesDataset` (`prototype/src/chess_rating_net.py:54`) should
read ~2.46M games for the full-corpus run, given ~318GB of NVMe scratch and a
corpus that is ~500GB as loose `.pkl` files.

## Recommendation

**One SQLite blob store: `corpus.sqlite`, keyed by the original `.pkl` basename,
one zlib-compressed pickle per row.** Built by
`prototype/src/build_corpus_store.py`, read through a new `GameBlobStore` class
threaded into `ChessGamesDataset`. Both are in this branch and both run.

- **Size: ~12GB** for the whole corpus, against 318GB free. Measured, not
  assumed (see §2).
- **Random access: one primary-key lookup.** No global-index-across-82-archives
  logic, because there are no archives to index across.
- **Dependencies: none.** `sqlite3` and `zlib` are both Python standard library.
  Nothing new enters the `ratingnet2` conda env.
- **The frozen split survives.** `split_manifest.json`'s sha256 is computed over
  the sorted `.pkl` basename list, and the store enumerates that identical list.
  Verified.
- **`--data_dir` is unchanged.** A directory containing `corpus.sqlite` uses the
  store; a directory of loose `.pkl` files behaves exactly as it does today. The
  170k flat directory keeps working with no flag change.

## The reframing that decides this

The brief states the problem as "~500GB does not fit in ~318GB." That is true of
the *loose* representation, but it is not the constraint that actually binds, and
treating it as the binding constraint makes the problem look much harder than it
is.

**The corpus already fits on the NVMe disk today.** It is sitting there as ~82
`.tar.gz` files at ~107MB each, roughly 8.8GB total. Nothing about the full
corpus is too big for that disk. The real problem is narrower:

> The only random-access representation the trainer knows how to read is loose
> `.pkl` files, and *that* representation is 500GB.

Once it is put that way, the question stops being "how do we fit 500GB into
318GB" and becomes "what random-access format keeps the compression the corpus
already has." Every per-file-compressed format answers that, so the capacity
constraint stops binding and the decision turns on read latency, multi-worker
safety, conversion risk and simplicity instead.

The premise this rests on is that per-file compression is nearly as good as the
solid stream, which is not obvious and which the brief explicitly worried about.
So it was measured.

---

# 1. What was measured, and how

No HPC access, and `torch`/`numpy`/`h5py`/`lmdb` are not installed on this
machine, so nothing here is a torch-level benchmark. What was built instead is a
**byte-level model of the payload**, which is what compression and read cost
actually respond to.

A game pickle is dominated by `Positions`: a list of `torch.float32` tensors of
shape `(12, 8, 8)` from `format_data.board_to_array` (`format_data.py:17-30`).
That is 3072 bytes each, of which at most 32 entries are `1.0` and the rest are
`0.0` - about 4% density. torch pickles such a tensor as a small header plus the
raw little-endian buffer, so a `bytes` object with the same layout has the same
entropy. Games were generated with realistic move counts (mean 66, capped at
100), the real nine-key schema, and pieces coming off the board as the game
progresses.

The model checks out against the real corpus on both numbers that can be
compared:

| | real corpus | this model |
|---|---|---|
| mean uncompressed pickle | ~203 KB/game | 197.8 KB/game |
| solid `tar.gz` ratio | ~57x (6.1GB → 107MB) | 47.7x |

The model is slightly *pessimistic* on compression, which is the safe direction
for a sizing argument.

Probe scripts are in this session's scratchpad (`bench_formats.py`,
`bench_reads.py`, `bench_handle.py`, `bench_mp.py`, `test_store.py`). The
shipped equivalents, which run against real data, are
`build_corpus_store.py verify` and `analysis/scripts/check_corpus_store.py`.

# 2. Per-file compression costs almost nothing

This is the load-bearing measurement. 2000 modelled games, ~405MB loose:

| format | size | ratio | → 2.46M games | fits 318GB? |
|---|---|---|---|---|
| loose `.pkl` (status quo) | 405.1 MB | 1.0x | **498 GB** | **no** |
| solid `tar.gz` (today, no random access) | 8.5 MB | 47.7x | 10.4 GB | yes |
| **ZIP deflate (per-member)** | 8.7 MB | 46.3x | 10.8 GB | yes |
| packed `.bin` + index (zlib/record) | 8.6 MB | 47.1x | 10.6 GB | yes |
| **SQLite blobs (zlib/blob)** | 10.0 MB | 40.7x | **12.2 GB** | **yes** |
| ZIP stored (uncompressed) | 405.3 MB | 1.0x | 498 GB | no |

**Per-file compression gives up 2.9% of the solid-stream ratio.** The worry that
independent compression would forfeit cross-file redundancy is not borne out,
and the reason is structural: the compressibility comes from long runs of zero
bytes *inside* each position tensor, not from similarity *between* games. DEFLATE
finds those runs in a 200KB window without needing to see another game.

So the break-even matters less than it looks: 2.46M games must average under
**126 KB on disk** to fit in 318GB. The store averages about **5 KB**. That is
**~26x of headroom**, not a tight fit. Even if the real corpus compressed three
times worse than modelled, it would still fit with room to spare.

SQLite is the largest of the compressed options, by about 15% over a raw packed
file, which is page and B-tree overhead. At 12GB against 318GB free, paying 1.6GB
for a real database is not a meaningful cost.

# 3. Why SQLite rather than the alternatives

Once nothing is disqualified on size, the criteria that remain are: true random
access, safety under `num_workers=8`, dependency cost, conversion risk, and how
much new code has to be correct.

**ZIP (per-month or consolidated)** - the brief's leading candidate, and a
perfectly reasonable answer. It loses on two specifics. Consolidated into one
archive, the central directory holds 2.46M entries, and `zipfile.ZipFile` parses
the whole thing into Python objects on open; at roughly 200+ bytes of object
overhead per entry that is several hundred MB **per worker process**, times 8
workers. Kept per-month, each worker still eventually touches all 82 archives
under a global shuffle, so it arrives at the same place by a longer route, plus
the index-mapping logic the brief asks about. SQLite's index lives on disk in a
B-tree and is paged in as needed, so per-worker memory is bounded by the page
cache rather than by corpus size.

**Packed binary + sidecar index** - the fastest option measured (0.072 ms/game)
and the smallest. It loses on operational properties, not performance: resuming a
half-finished conversion means reasoning about byte offsets, a truncated write
leaves a file that looks valid, and inspecting it requires bespoke tooling. It
is the right answer if reads ever become the bottleneck, and the store could be
migrated to it later without touching the Dataset, since both sit behind the
same `load(name)` interface.

**LMDB** - not installed in `ratingnet2`, so it adds a dependency. It also stores
values uncompressed by default, so either the values get compressed by hand
(at which point it is a key-value store with the same shape as the SQLite one, for
an added dependency) or the map is ~500GB and the problem comes back. Its
memory-mapped design is a genuine advantage at larger-than-RAM scale, which this
corpus is not.

**HDF5** - not installed either, and `h5py` plus `fork` is a known hazard that
needs care in exactly the multi-worker setting this has to survive. Chunked
compression is designed for slab reads out of large arrays, not for 2.46M
independently-addressed variable-length blobs.

**WebDataset-style shards** - disqualified on methodology rather than
engineering. It replaces global shuffling with a shuffle buffer, which changes
the training procedure. This thesis has a pre-registered methodology, a tuned
`lr=3e-4`, and a frozen split guarded by a sha256. Changing how examples are
ordered is not a storage decision, and it is not one to make silently while
solving a disk-space problem.

*Honest limitation:* ZIP, packed-binary and SQLite were measured here. LMDB,
HDF5 and WebDataset were assessed by reasoning, because the libraries are not
installed on this machine and the brief rules out installing things. The six
subagents assigned to evaluate them independently did not return - the account
hit its monthly spend limit mid-run - so those three lines are one analysis, not
two.

# 4. The measured ZIP overhead is probably an artifact, and it is worth re-testing

The brief reports a real measurement from 2026-08-24: loose 3.791 ms/game, ZIP
5.396 ms/game, a +1.605 ms/game penalty, and asks that it be taken seriously
rather than assumed away. Taking it seriously turns up a likely explanation.

The brief describes the ZIP access pattern as `zipfile.ZipFile(path).read(member)`
- a **fresh `ZipFile` per read**. Constructing a `ZipFile` parses the entire
central directory. Measured on this machine, on a 2000-member archive:

| pattern | ms/game |
|---|---|
| one handle reused (the per-worker pattern) | 0.077 |
| `ZipFile(path)` constructed per read | 4.175 |

**A fresh handle per read costs 4.1 ms/game at 2000 members, 54x the reused
handle.** The reported +1.605 ms/game is the right order of magnitude for
directory-parse overhead at that archive size, and it is not an intrinsic
property of ZIP.

The consequence matters more than the diagnosis: that cost **scales with member
count**. At a real month's ~30k members the same pattern would extrapolate to
tens of ms per game. A benchmark on a 2000-game archive would have understated
the penalty by more than an order of magnitude, and the format would have looked
merely "42% slower" while being unusable in production.

**This is a hypothesis about someone else's benchmark, not a finding.** It is
falsifiable in about five minutes: re-run the same 2000-game test with the
`ZipFile` opened once outside the loop. If the overhead collapses toward zero,
the artifact explanation holds. If it stays near +1.6 ms, it is real
decompression cost and the store's own timing should be checked with the same
scrutiny. `analysis/scripts/check_corpus_store.py` prints the store's number in
the same methodology so the two are directly comparable.

For reference, the cost decomposition on this machine is `zlib.decompress`
0.054 ms + `pickle.loads` 0.025 ms per game, i.e. decompression is about 69% of
the CPU cost of a compressed read and is well under a tenth of a millisecond.

# 5. Whether prefetch hides the cost, with the arithmetic

The brief asks whether `--num_workers 8` prefetch hides compression overhead
behind GPU compute. It depends entirely on which epoch time is the real one, and
the repo gives two that do not agree.

`CLAUDE_TASK_2.md` says **85 s/epoch** at 170k games on NVMe.
`analysis/stage2-sweep-results.md` says **~3.8 min/epoch** in the seed rerun and
~5.6 min/epoch during the sweep. Those are 85s versus 228s for nominally the same
data at the same scale.

Per-game budget = epoch time / 170,138 games. With 8 workers each worker has 8x
that to deliver one game:

| epoch time | aggregate budget | per-worker budget | loose read (3.791 ms) uses | ZIP read (5.396 ms) uses |
|---|---|---|---|---|
| 85 s | 0.50 ms/game | 4.00 ms/game | **95%** - saturated | **135%** - becomes the bottleneck |
| 228 s | 1.34 ms/game | 10.72 ms/game | 35% - lots of slack | 50% - absorbed easily |

**At 85 s/epoch the pipeline is already nearly IO-bound and the brief's worry is
justified**: adding 1.6 ms/game would push per-worker cost past the budget and
epochs would lengthen by roughly a third. **At 228 s/epoch there is enough slack
that it disappears.** This is the single most important thing to measure before
the full run, and it is cheap to measure.

Two things push the real answer toward "hidden", though:

1. **The +1.6 ms may not exist** once the handle is reused (§4). The store's
   measured decompression cost is ~0.05 ms/game.
2. **Page cache reverses the comparison at full-corpus scale.** The 2000-game
   benchmark had *both* formats fully cached, so it measured CPU with IO removed
   - the one regime where compression can only lose. At full corpus the loose
   set is ~500GB and cannot be cached, so every read is a real 203KB NVMe read,
   while the store is ~12GB and sits entirely in page cache on any node with
   modest RAM. Compressed-in-cache should beat uncompressed-on-disk.

That second point is an **assumption to verify, not a fact**: it needs the node
to have enough free RAM to hold ~12GB of page cache alongside the training
process. `free -g` settles it. If the node is memory-starved the argument weakens,
though the store still wins on every other axis.

# 6. What this requires that is not obviously true

Stated explicitly, per the brief's item 3.

1. **The archives must be rebuilt into the new format** - decompress then
   recompress. The loose `.pkl` files are gone (`fm-corpus-archive-month.py`
   deletes them after verifying), so this is real work, not a rename.
2. **Peak transient disk is the output only, ~12GB.** This is better than the
   brief assumes. Archives are streamed with `tarfile.open(path, "r|gz")`, which
   is forward-only and holds one member in memory at a time, so a month is never
   extracted to disk. There is no ~6.1GB-per-month transient cost.
3. **Conversion CPU is roughly 40 minutes single-threaded** for 2.46M games
   (measured: 0.883 ms/game to zlib-compress at level 6), plus streaming ~500GB
   through gunzip. `--jobs 8` brings compression to a few minutes; end to end,
   expect the gunzip stream and disk writes to dominate. Locally the converter
   sustained ~2,500 games/s with `--jobs 4`, which would be ~16 minutes for the
   full corpus, but that is a Mac SSD reading small archives and should not be
   quoted as an HPC estimate.
4. **All 82 archives must be present** before the store is used for a run, or
   the split will be computed over a partial corpus. The brief says 60+ of 82 are
   staged. The converter is resumable, so it can be run repeatedly as archives
   land, but `split_manifest.json` must not be created until the corpus is
   complete - once written it is enforced by sha256 and adding games afterwards
   makes the trainer abort at `chess_rating_net.py:567`.
5. **`immutable=1` means what it says.** The store is opened read-only with
   locking disabled. Do not point a training run at a store that is still being
   built.
6. **Game ids must be unique across months.** If two archives contain the same
   `.pkl` basename the converter aborts rather than overwriting, because a silent
   overwrite would drop a game and change the corpus. This has not been checked
   against the real archives; it will surface immediately if it happens.

# 7. How to verify it, without trusting this branch

`analysis/scripts/check_corpus_store.py` exists for this. It re-derives
everything rather than reading any claim from this document:

```
python src/build_corpus_store.py build   --archives ~/<workdir>/data/corpus_archive_transfer --store /tmp/ratingnet_store --jobs 8
python src/build_corpus_store.py verify  --archives ~/<workdir>/data/corpus_archive_transfer --store /tmp/ratingnet_store
python analysis/scripts/check_corpus_store.py --store /tmp/ratingnet_store --loose /tmp/ratingnet_data_flat
```

`verify` streams every archive again and compares every game byte-for-byte with
what the store returns. `check_corpus_store.py` adds: name-list identity against
`os.listdir`, split-hash invariance, tensor-by-tensor round-trip equality, 8
forked workers reading after the parent has already opened a connection, survival
of a spawn-style pickle round-trip, and a loose-vs-store timing comparison in the
same methodology as the 2026-08-24 ZIP test.

The cheapest meaningful real-data check is to build a store from the **170k flat
directory's own months**, point `--data_dir` at it, and confirm a single training
epoch reproduces the same `split_manifest.json` sha256
(`4be0f9e8de8372dcc302f164d271b871841e82b8843c1cefbd4d25b119faacc6`) and a
comparable epoch time. That exercises the whole path against data whose expected
behaviour is already known.

# 8. What was tested, and what was not

**Tested, on a 2000-game synthetic corpus built to the real schema and byte
layout, packed into three `.tar.gz` month archives by the same `tarfile` calls
the real archiver uses:**

- build → 0 failures, 47.4x compression, ~2,500 games/s at `--jobs 4`
- `verify` → 0 byte-identical mismatches, 0 missing, 0 schema mismatches, 2000/2000
- split manifest sha256 identical between loose directory and store
- round-trip equality over all 2000 games
- 8 forked workers x 300 reads after the parent opened a connection → 0 corrupt reads
- spawn-style pickle round-trip → store still reads
- resume: deleting an archive's rows and re-running rebuilds only that archive
- duplicate game id across archives → aborts with a clear message, writes nothing
- read cost: store 0.072 ms/game vs loose 0.066 ms/game, +0.005 ms/game

**Not tested, and it matters:**

- Anything against real data. The synthetic corpus models the payload's byte
  layout and entropy, not torch's exact pickle format. Real pickles may compress
  somewhat differently - though with 26x of headroom the conclusion is not
  sensitive to that.
- Anything on HPC hardware, or on spinning-disk `/home`.
- Anything inside a real `DataLoader` or a real training loop. The fork test uses
  `multiprocessing.Pool`, which is the same mechanism, but it is not the same
  code path.
- Whether the real archives contain duplicate game ids across months.
- Whether the node has enough free RAM for the page-cache argument in §5.
- The `85 s` versus `228 s` epoch-time discrepancy is unresolved, and §5 shows the
  answer to the brief's prefetch question depends on which is right.
