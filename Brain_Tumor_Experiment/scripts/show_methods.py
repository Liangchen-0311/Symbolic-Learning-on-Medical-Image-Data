#!/usr/bin/env python3
import json, sys
from pathlib import Path
run = sys.argv[1] if len(sys.argv) > 1 else 'outputs/brain_dir3_fine'
d = json.load(open(Path(run) / 'classifier_results.json'))
print(f"run: {run}")
print(f"BEST: {d['method']}  (selected by balanced accuracy)")
am = d.get('all_methods', {}); pc = d.get('all_per_class', {})
cv = d.get('cv_accs', {}); cvb = d.get('cv_bal_accs', {})
for m, r in am.items():
    per = {k: round(v, 3) for k, v in pc.get(m, {}).items()}
    mark = '  <-- BEST' if m == d['method'] else ''
    print(f"  {m:18s} test_acc={r['acc']:.4f} bal={r.get('bal_acc', 0):.4f} "
          f"AUC={r['auc']:.4f} cv_acc={cv.get(m, 0):.3f}{mark}")
    print(f"      per-class: {per}")
