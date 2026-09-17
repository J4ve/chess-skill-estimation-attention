"""Write the model-free test fixtures for the LightGBM and CNN-BiLSTM parity tests (local).

tests/fixtures/method_parity_synthetic_caught.npz: the synthetic_caught sample's
float16 CNN embeddings and standardized clocks (the A4 head's inputs), made with
the app's own code and RATINGNET_CHECKPOINT; expected scores for both methods
come from the HPC code path (hpc_scores.json), not from the app.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import api  # noqa: E402
import cnn_bilstm_detector  # noqa: E402

api.MODEL, _, _, _ = api._load_model()
pgn = open(os.path.join(ROOT, "src/static/samples/synthetic_caught.pgn")).read()
positions, clocks, *_ = api._pgn_to_tensor_inputs(pgn)
emb = cnn_bilstm_detector.embed_positions(api.MODEL, positions)

hpc = json.load(open(os.path.join(HERE, "hpc_scores.json")))
index = json.load(open(os.path.join(HERE, "index.json")))
gid = {e["side"]: e["gid"] for e in index if e["source"] == "synthetic_caught.pgn"}
np.savez_compressed(
    os.path.join(ROOT, "tests/fixtures/method_parity_synthetic_caught.npz"),
    emb=emb.numpy().astype(np.float16),
    clocks_z=clocks.reshape(-1).numpy().astype(np.float32),
    expected_lgbm=np.array([hpc["hpc"][gid[s]]["lgbm_a0g"] for s in ("white", "black")]),
    expected_cnn_bilstm=np.array([hpc["hpc"][gid[s]]["cnn_bilstm_a4"] for s in ("white", "black")]),
)
print(emb.shape)
