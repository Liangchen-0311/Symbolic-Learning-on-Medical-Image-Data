#!/usr/bin/env python3
"""One-shot: derive train_thirddata.py from train_brain.py (ThirdData/BUSI specifics)."""
s = open('train_brain.py').read()

s = s.replace("Brain Tumor MRI Dataset'", "ThirdData'")  # DATA_DIR constant
s = s.replace(
    "BRAIN_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']",
    "BRAIN_NAMES = ['benign', 'malignant', 'normal']",
)

old = '''    def __init__(self, split='train', transform=None):
        self.transform = transform
        self.samples = []
        disk = 'Testing' if split == 'test' else 'Training'
        rng = np.random.RandomState(SEED)
        for cls_idx, cls in enumerate(BRAIN_NAMES):
            cdir = os.path.join(DATA_DIR, disk, cls)
            if not os.path.isdir(cdir):
                continue
            files = sorted(f for f in os.listdir(cdir) if f.lower().endswith(_EXTS))
            if split in ('train', 'val'):
                perm = rng.permutation(len(files))
                n_val = int(round(len(files) * VAL_FRACTION))
                val_idx = set(perm[:n_val].tolist())
                keep = ([i for i in range(len(files)) if i in val_idx] if split == 'val'
                        else [i for i in range(len(files)) if i not in val_idx])
                files = [files[i] for i in keep]
            for f in files:
                self.samples.append((os.path.join(cdir, f), cls_idx))'''

new = '''    def __init__(self, split='train', transform=None):
        self.transform = transform
        self.samples = []
        # BUSI/ThirdData already provides train/val/test on disk -> read directly.
        for cls_idx, cls in enumerate(BRAIN_NAMES):
            cdir = os.path.join(DATA_DIR, split, cls)
            if not os.path.isdir(cdir):
                continue
            for f in sorted(os.listdir(cdir)):
                if f.startswith('._') or 'mask' in f.lower():
                    continue
                if f.lower().endswith(_EXTS):
                    self.samples.append((os.path.join(cdir, f), cls_idx))'''

assert old in s, "dataset block not found"
s = s.replace(old, new)
open('train_thirddata.py', 'w').write(s)
print("train_thirddata.py written")
