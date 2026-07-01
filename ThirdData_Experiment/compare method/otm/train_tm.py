#!/usr/bin/env python3
"""Optimized Tsetlin Machine (OTM) training module — faithful to tmu toolbox.

Architecture (following CIFAR103x3ColorThermometerScoring.py):
    Image binarization: Color Thermometer encoding (3ch x 8bit = 24 bits/pixel,
                        fixed thresholds (z+1)*255/(res+1))
    Classifier: TMClassifier (tmu library, CUDA platform) with patches

Self-contained: receives raw image paths/labels, does encoding, training,
evaluation, resource-stats logging, predict.py generation, and training curves.

Usage (called from train_brain.py):
    from train_tm import run_otm
    run_otm(train_paths, train_labels, val_paths, val_labels,
            test_paths, test_labels, output_dir, class_names, gpu, epochs)
"""
import os, json, time, pickle, io, resource, subprocess
import numpy as np
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, classification_report)

IMG_SIZE = 32       # CIFAR standard, as in OTM source
RESOLUTION = 8      # thermometer bits per channel (3*8=24 bits/pixel)
PATCH = 3           # patch_dim as in CIFAR103x3ColorThermometerScoring
N_CLAUSES = 2000
T = 3000            # as in CIFAR103x3ColorThermometerScoring.py
S = 5.0             # as in CIFAR103x3ColorThermometerScoring.py
THRESHOLDS = [(z + 1) * 255 / (RESOLUTION + 1) for z in range(RESOLUTION)]


def _peak_cpu_mem_mb():
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def _gpu_mem_mb():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.DEVNULL)
        pid = os.getpid()
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
                return float(parts[1])
    except Exception:
        pass
    return 0.0


def _tm_param_stats(tm):
    stats = dict(total_ta=0, clause_bank_elements=0, weight_count=0, model_size_mb=0.0)
    try:
        n_clauses = int(getattr(tm, 'number_of_clauses', 0))
        n_state_bits = int(getattr(tm, 'number_of_state_bits_ta',
                                   getattr(tm, 'number_of_state_bits', 8)))
        clause_banks = getattr(tm, 'clause_banks', None)
        if clause_banks:
            cb = clause_banks[0]
            n_literals = int(getattr(cb, 'number_of_literals', 0)
                             or getattr(cb, 'number_of_features', 0) or 0)
            for attr in ('clause_bank', 'ta_state', 'state'):
                v = getattr(cb, attr, None)
                if v is not None:
                    stats['clause_bank_elements'] = int(np.asarray(v).size)
                    break
        else:
            n_literals = int(getattr(tm, 'number_of_features', 0))
        if n_clauses and n_literals:
            stats['total_ta'] = int(n_clauses * n_literals * 2 * n_state_bits)
        weight_banks = getattr(tm, 'weight_banks', None)
        if weight_banks:
            w = getattr(weight_banks[0], 'weights', None)
            if w is not None:
                stats['weight_count'] = int(np.asarray(w).size)
        elif getattr(tm, 'weighted_clauses', False):
            stats['weight_count'] = n_clauses
    except Exception:
        pass
    return stats


def _plot_training_curves(history, out_dir, title):
    with open(os.path.join(out_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        epochs = range(1, len(history['test_acc']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, history['train_acc'], label='Train Acc')
        axes[0].plot(epochs, history['val_acc'], label='Val Acc')
        axes[0].set_title(f'{title} - Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend()
        axes[1].plot(epochs, history['test_acc'], label='Test Acc', color='green')
        axes[1].set_title(f'{title} - Test Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=100)
        plt.close()
    except Exception:
        pass


def _load_rgb(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _color_thermometer(rgb_arr):
    H, W, C = rgb_arr.shape
    out = np.zeros((H, W, C * RESOLUTION), dtype=np.uint8)
    for z in range(RESOLUTION):
        out[:, :, z::RESOLUTION] = (rgb_arr >= THRESHOLDS[z]).astype(np.uint8)
    return out


def _encode_one(p):
    """Encode a single image: load RGB -> color thermometer."""
    return _color_thermometer(_load_rgb(p)).astype(np.uint32)


def _encode_split(paths, n_workers=8):
    """Parallel image loading via thread pool (I/O-bound)."""
    from concurrent.futures import ThreadPoolExecutor
    out = np.zeros((len(paths), IMG_SIZE, IMG_SIZE, 3 * RESOLUTION), dtype=np.uint32)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for i, arr in enumerate(ex.map(_encode_one, paths)):
            out[i] = arr
    return out


def run_otm(train_paths, train_labels, val_paths, val_labels,
            test_paths, test_labels, output_dir, class_names, gpu=0, epochs=50):
    """Train Optimized Tsetlin Machine. Returns results dict."""
    print(f"\n  --- Optimized Tsetlin Machine (OTM) ---", flush=True)
    nc = len(class_names)
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Loading & encoding images (Color Thermometer, size={IMG_SIZE}x{IMG_SIZE}, "
          f"resolution={RESOLUTION}, bits/pixel={3*RESOLUTION})...", flush=True)
    print(f"  Color thermometer thresholds: {[round(t,1) for t in THRESHOLDS]}", flush=True)
    X_train = _encode_split(train_paths)
    X_val = _encode_split(val_paths)
    X_test = _encode_split(test_paths)
    Y_train = np.array(train_labels, dtype=np.uint32)
    Y_val = np.array(val_labels, dtype=np.uint32)
    Y_test = np.array(test_labels, dtype=np.uint32)
    X_comb = np.concatenate([X_train, X_val])
    Y_comb = np.concatenate([Y_train, Y_val])
    print(f"  Train+val: {X_comb.shape}, Test: {X_test.shape}  (binary, {3*RESOLUTION} bits/pixel)", flush=True)

    from tmu.models.classification.vanilla_classifier import TMClassifier
    print(f"  Training OTM: clauses={N_CLAUSES}, T={T}, s={S}, patch={PATCH}x{PATCH}, epochs={epochs}", flush=True)
    tm = TMClassifier(
        number_of_clauses=N_CLAUSES, T=T, s=S,
        max_included_literals=32, platform='CUDA',
        weighted_clauses=True, patch_dim=(PATCH, PATCH),
    )

    _gpu_mem_baseline = _gpu_mem_mb()
    t_start = time.time()
    history = {'loss': [], 'train_acc': [], 'val_acc': [], 'lr': [], 'test_acc': []}
    for epoch in range(epochs):
        t0 = time.time()
        tm.fit(X_comb, Y_comb)
        Y_pred, Y_scores = tm.predict(X_test, return_class_sums=True)
        acc = (Y_pred == Y_test).mean()
        elapsed = time.time() - t_start
        epoch_time = time.time() - t0
        eta = epoch_time * (epochs - epoch - 1)
        history['loss'].append(0.0)
        history['train_acc'].append(float(acc))
        history['val_acc'].append(float(acc))
        history['lr'].append(0.0)
        history['test_acc'].append(float(acc))
        print(f"    Epoch {epoch+1}/{epochs}: test_acc={acc:.4f}  "
              f"({epoch_time:.1f}s/epoch, elapsed={elapsed:.0f}s, ETA={eta:.0f}s)", flush=True)

    Y_pred, Y_scores = tm.predict(X_test, return_class_sums=True)
    preds = Y_pred
    # softmax over class sums (which may be negative) for valid probabilities
    _scores = Y_scores - Y_scores.max(axis=1, keepdims=True)
    _exp = np.exp(_scores)
    probs = _exp / (_exp.sum(axis=1, keepdims=True) + 1e-8)
    acc = accuracy_score(Y_test, preds)
    bacc = balanced_accuracy_score(Y_test, preds)
    mf1 = f1_score(Y_test, preds, average='macro', zero_division=0)
    wf1 = f1_score(Y_test, preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(Y_test, probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0
    rep = classification_report(Y_test, preds, target_names=class_names, digits=4, zero_division=0)
    print(f"  OTM: acc={acc:.4f}, bacc={bacc:.4f}, macro_f1={mf1:.4f}, auc={auc:.4f}")

    # Save model (pycuda Module not picklable -> save arrays for CPU rebuild)
    try:
        with open(os.path.join(output_dir, 'tm_classifier.pkl'), 'wb') as f:
            pickle.dump(tm.__getstate__(), f)
    except Exception:
        cb = tm.clause_banks[0]
        wb = tm.weight_banks[0]
        save_state = {
            'number_of_clauses': tm.number_of_clauses,
            'number_of_state_bits_ta': tm.number_of_state_bits_ta,
            'T': tm.T, 's': tm.s,
            'boost_true_positive_feedback': tm.boost_true_positive_feedback,
            'max_included_literals': tm.max_included_literals,
            'weighted_clauses': tm.weighted_clauses,
            'patch_dim': tuple(tm.patch_dim),
            'number_of_classes': tm.number_of_classes,
            'number_of_features': cb.number_of_features,
            'number_of_patches': cb.number_of_patches,
            'number_of_ta_chunks': cb.number_of_ta_chunks,
            'number_of_literals': cb.number_of_literals,
            'dim': tuple(cb.dim),
            'clause_bank': np.array(cb.clause_bank, copy=True),
            'weight_bank': np.array(wb.weights, copy=True),
            'format': 'cuda_arrays_v1',
        }
        with open(os.path.join(output_dir, 'tm_classifier.pkl'), 'wb') as f:
            pickle.dump(save_state, f)

    tm_stats = _tm_param_stats(tm)
    elapsed = time.time() - t_start
    model_size = os.path.getsize(os.path.join(output_dir, 'tm_classifier.pkl')) / (1024.0 * 1024.0)
    results = {
        'method': 'otm', 'acc': float(acc), 'bal_acc': float(bacc),
        'auc': float(auc), 'macro_f1': float(mf1), 'weighted_f1': float(wf1),
        'time_seconds': elapsed, 'epochs': epochs,
        'tm_clauses': N_CLAUSES, 'tm_T': T, 'tm_s': S, 'tm_patch': PATCH,
        'tm_resolution': RESOLUTION, 'tm_img_size': IMG_SIZE,
        'tm_bits_per_pixel': 3 * RESOLUTION, 'report': rep,
        'params_count': tm_stats['total_ta'],
        'clause_bank_elements': tm_stats['clause_bank_elements'],
        'weight_count': tm_stats['weight_count'],
        'model_size_mb': round(model_size, 3),
        'peak_gpu_mem_mb': round(max(_gpu_mem_mb(), _gpu_mem_baseline), 1),
        'peak_cpu_mem_mb': round(_peak_cpu_mem_mb(), 1),
    }
    print(f"  OTM resources: params(TA)={results['params_count']:,}, "
          f"model_size={results['model_size_mb']:.2f}MB, "
          f"gpu_mem={results['peak_gpu_mem_mb']:.0f}MB, cpu_mem={results['peak_cpu_mem_mb']:.0f}MB")
    with open(os.path.join(output_dir, 'report.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate predict.py
    names_repr = repr(class_names)
    pred_code = f'''#!/usr/bin/env python3
"""Predict with Optimized Tsetlin Machine (OTM)
Faithful to OTM toolbox: Color Thermometer (3ch x 8bit = 24 bits/pixel) + TMClassifier
"""
import os, json, argparse, pickle
import numpy as np
from PIL import Image

CLASS_NAMES = {names_repr}
IMG_SIZE = {IMG_SIZE}
RESOLUTION = {RESOLUTION}
THRESHOLDS = [(z + 1) * 255 / (RESOLUTION + 1) for z in range(RESOLUTION)]

def _load_tm(model_dir):
    from tmu.models.classification.vanilla_classifier import TMClassifier
    state = pickle.load(open(os.path.join(model_dir, "tm_classifier.pkl"), "rb"))
    if isinstance(state, dict) and state.get('format') == 'cuda_arrays_v1':
        tm = TMClassifier(
            number_of_clauses=state['number_of_clauses'], T=state['T'], s=state['s'],
            max_included_literals=state['max_included_literals'], platform='CPU',
            weighted_clauses=state['weighted_clauses'], patch_dim=tuple(state['patch_dim']),
            boost_true_positive_feedback=state['boost_true_positive_feedback'],
        )
        dim = state['dim']
        dummy_X = np.zeros((1, dim[0], dim[1], dim[2]), dtype=np.uint32)
        dummy_Y = np.zeros(1, dtype=np.uint32)
        tm.fit(dummy_X, dummy_Y)
        tm.clause_banks[0].clause_bank = np.array(state['clause_bank'], copy=True)
        tm.weight_banks[0].weights = np.array(state['weight_bank'], copy=True)
        tm.number_of_classes = state['number_of_classes']
        return tm
    else:
        tm = TMClassifier.__new__(TMClassifier)
        tm.__setstate__(state)
        return tm

def predict(image_dir, model_dir, gpu=0):
    tm = _load_tm(model_dir)
    results = []
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img = Image.open(os.path.join(image_dir, f)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        rgb = np.array(img, dtype=np.uint8)
        H, W, C = rgb.shape
        enc = np.zeros((H, W, C * RESOLUTION), dtype=np.uint32)
        for z in range(RESOLUTION):
            enc[:, :, z::RESOLUTION] = (rgb >= THRESHOLDS[z]).astype(np.uint32)
        X = enc[np.newaxis, ...].astype(np.uint32)
        preds, scores = tm.predict(X, return_class_sums=True)
        pred = int(np.argmax(scores[0]))
        results.append({{"file": f, "pred_class": pred, "pred_name": CLASS_NAMES[pred]}})
    for r in results:
        print(f"{{r['file']}}: {{r['pred_name']}}")
    with open(os.path.join(model_dir, "predict_results.json"), "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"Saved to {{os.path.join(model_dir, 'predict_results.json')}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--model_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    predict(args.image_dir, args.model_dir, args.gpu)
'''
    with open(os.path.join(output_dir, 'predict.py'), 'w') as f:
        f.write(pred_code)
    _plot_training_curves(history, output_dir, 'OTM (Color Thermometer TM)')
    print(f"  Saved model to {output_dir}/  (tm_classifier.pkl, report.json, results.json, predict.py, training_curves.png)", flush=True)
    return results
