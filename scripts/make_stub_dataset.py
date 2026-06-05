"""Generate stub dataset for pipeline smoke tests (before receiving real dataset.npz)."""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.io import save_dataset, load_dataset

OUT_PATH = 'data/processed/dataset_v0.npz'

if os.path.exists(OUT_PATH):
    print(f'Already exists: {OUT_PATH}, skipping.')
    sys.exit(0)

N, C, H, W = 200, 3, 32, 32
rng = np.random.default_rng(42)
X = rng.random((N, C, H, W)).astype(np.float32)
y = rng.integers(0, 6, size=(N,)).astype(np.int64)
meta = {'H': H, 'W': W, 'N': N, 'created_by': 'make_stub_dataset.py', 'note': 'stub'}

save_dataset(X, y, meta, OUT_PATH)

sha = hashlib.sha256(open(OUT_PATH, 'rb').read()).hexdigest()
print(f'Stub saved: X={X.shape}, y={y.shape}')
print(f'sha256: {sha}')

dataset_md = f'## dataset_v0.npz\n\nN={N}, shape=(N,3,{H},{W}), labels 0-5 (stub)\n\nsha256: {sha}\n'
with open('DATASET.md', 'w') as f:
    f.write(dataset_md)
print('DATASET.md written.')

# Verify round-trip
X2, y2, meta2 = load_dataset(OUT_PATH)
assert X2.shape == (N, C, H, W), f'shape mismatch: {X2.shape}'
assert y2.shape == (N,), f'y shape mismatch: {y2.shape}'
print('validate_schema: PASS')
