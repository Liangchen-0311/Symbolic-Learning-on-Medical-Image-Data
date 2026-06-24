#!/usr/bin/env python3
"""Formula length vs discriminability: is max_sequence_length binding?"""
import json, numpy as np
from pathlib import Path

v = json.load(open(Path('outputs/brain_dir3/validated_formulas.json')))
L = np.array([len(f['body'].split()) for f in v])
Q = np.array([f.get('quality_anova_f', 0) for f in v])
print(f"n={len(v)}  body length: min={L.min()} max={L.max()} mean={L.mean():.1f} median={np.median(L):.0f}")

print("\nlength histogram (tokens: count):")
for t in range(int(L.min()), int(L.max()) + 1):
    c = int((L == t).sum())
    if c:
        print(f"  {t:2d}: {c:3d} " + "#" * (c // 4))

order = np.argsort(Q)[::-1]
print()
for k in [20, 50, 100]:
    top = order[:k]
    print(f"top-{k:3d} by quality: mean_len={L[top].mean():.1f}  max_len={L[top].max()}  "
          f"frac>=13={np.mean(L[top] >= 13):.2f}  frac>=14={np.mean(L[top] >= 14):.2f}")
print(f"ALL          : mean_len={L.mean():.1f}  "
      f"frac>=13={np.mean(L >= 13):.2f}  frac>=14={np.mean(L >= 14):.2f}  frac>=15={np.mean(L >= 15):.2f}")
print(f"\ncorr(length, quality) = {np.corrcoef(L, Q)[0, 1]:.3f}")
print("mean quality by length bucket:")
for lo, hi in [(1, 7), (8, 10), (11, 12), (13, 14), (15, 99)]:
    m = (L >= lo) & (L <= hi)
    if m.sum():
        print(f"  len {lo:2d}-{hi:2d}: n={int(m.sum()):3d}  mean_q={Q[m].mean():7.1f}  max_q={Q[m].max():7.1f}")
