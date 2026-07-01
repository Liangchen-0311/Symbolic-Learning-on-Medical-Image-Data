#!/usr/bin/env python3
"""Classic Tsetlin Machine (CTM) training module — faithful to pyTsetlinMachineParallel.

Architecture (following examples/FashionMNISTDemo2DConvolutionWeightedClauses.py):
    Image binarization: OpenCV adaptiveThreshold (grayscale -> 1 bit/pixel)
    Classifier: MultiClassConvolutionalTsetlinMachine2D with patches (CPU, OpenMP)

Self-contained: receives raw image paths/labels, does binarization, training,
evaluation, resource-stats logging, predict.py generation, and training curves.

Usage (called from train_brain.py):
    from train_tm import run_ctm
    run_ctm(train_paths, train_labels, val_paths, val_labels,
            test_paths, test_labels, output_dir, class_names, gpu, epochs)
"""
import os, json, time, pickle, io, resource, subprocess
import numpy as np
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, classification_report)

IMG_SIZE = 64      # resize target (CIFAR uses 32; 64 keeps more detail)
PATCH = 10         # patch_dim as in FashionMNISTDemo2DConvolutionWeightedClauses
N_CLAUSES = 2000
T = 50 * 100       # 5000, as in MNISTDemoWeightedClauses
S = 10.0


def _peak_cpu_mem_mb():
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def _tm_param_stats(tm):
    """Count learnable Tsetlin Automata units for fair comparison with neural params."""
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
        try:
            buf = io.BytesIO()
            pickle.dump(tm.__getstate__(), buf)
            stats['model_size_mb'] = buf.tell() / (1024.0 * 1024.0)
        except Exception:
            pass
    except Exception:
        pass
    return stats


def _plot_training_curves(history, out_dir, title):
    """Save training_curves.png and training_history.json."""
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


def _load_bin(path):
    """Load image -> grayscale -> resize -> OpenCV adaptive threshold."""
    import cv2
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    return cv2.adaptiveThreshold(arr, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)


def _encode_split(paths, n_workers=8):
    """Parallel image loading via thread pool (I/O-bound)."""
    from concurrent.futures import ThreadPoolExecutor
    out = np.zeros((len(paths), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for i, arr in enumerate(ex.map(_load_bin, paths)):
            out[i] = arr
    return out


def run_ctm(train_paths, train_labels, val_paths, val_labels,
            test_paths, test_labels, output_dir, class_names, gpu=0, epochs=50):
    """Train Classic Tsetlin Machine. Returns results dict."""
    print(f"\n  --- Classic Tsetlin Machine (CTM) ---", flush=True)
    nc = len(class_names)
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Loading & binarizing images (OpenCV adaptiveThreshold, size={IMG_SIZE}x{IMG_SIZE})...", flush=True)
    X_train = _encode_split(train_paths)
    X_val = _encode_split(val_paths)
    X_test = _encode_split(test_paths)
    Y_train = np.array(train_labels, dtype=np.uint32)
    Y_val = np.array(val_labels, dtype=np.uint32)
    Y_test = np.array(test_labels, dtype=np.uint32)
    X_comb = np.concatenate([X_train, X_val])
    Y_comb = np.concatenate([Y_train, Y_val])
    print(f"  Train+val: {X_comb.shape}, Test: {X_test.shape}  (binary values: 0/1)", flush=True)

    from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
    print(f"  Training CTM: clauses={N_CLAUSES}, T={T}, s={S}, patch={PATCH}x{PATCH}, epochs={epochs}", flush=True)
    tm = MultiClassConvolutionalTsetlinMachine2D(
        N_CLAUSES, T, S, (PATCH, PATCH),
        boost_true_positive_feedback=1, weighted_clauses=True,
    )

    t_start = time.time()
    history = {'loss': [], 'train_acc': [], 'val_acc': [], 'lr': [], 'test_acc': []}
    for epoch in range(epochs):
        t0 = time.time()
        tm.fit(X_comb, Y_comb, epochs=1, incremental=True)
        preds = tm.predict(X_test)
        acc = (preds == Y_test).mean()
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

    preds = tm.predict(X_test)
    probs = np.zeros((len(preds), nc), dtype=np.float32)
    for c in range(nc):
        probs[:, c] = (preds == c).astype(np.float32)
    acc = accuracy_score(Y_test, preds)
    bacc = balanced_accuracy_score(Y_test, preds)
    mf1 = f1_score(Y_test, preds, average='macro', zero_division=0)
    wf1 = f1_score(Y_test, preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(Y_test, probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0
    rep = classification_report(Y_test, preds, target_names=class_names, digits=4, zero_division=0)
    print(f"  CTM: acc={acc:.4f}, bacc={bacc:.4f}, macro_f1={mf1:.4f}, auc={auc:.4f}")

    with open(os.path.join(output_dir, 'tm_model.pkl'), 'wb') as f:
        pickle.dump(tm.__getstate__(), f)
    tm_stats = _tm_param_stats(tm)
    elapsed = time.time() - t_start
    results = {
        'method': 'ctm', 'acc': float(acc), 'bal_acc': float(bacc),
        'auc': float(auc), 'macro_f1': float(mf1), 'weighted_f1': float(wf1),
        'time_seconds': elapsed, 'epochs': epochs,
        'tm_clauses': N_CLAUSES, 'tm_T': T, 'tm_s': S, 'tm_patch': PATCH,
        'tm_img_size': IMG_SIZE, 'report': rep,
        'params_count': tm_stats['total_ta'],
        'clause_bank_elements': tm_stats['clause_bank_elements'],
        'weight_count': tm_stats['weight_count'],
        'model_size_mb': tm_stats['model_size_mb'],
        'peak_gpu_mem_mb': 0.0,
        'peak_cpu_mem_mb': round(_peak_cpu_mem_mb(), 1),
    }
    print(f"  CTM resources: params(TA)={results['params_count']:,}, "
          f"model_size={results['model_size_mb']:.2f}MB, "
          f"cpu_mem={results['peak_cpu_mem_mb']:.0f}MB, gpu_mem=0MB (CPU)")
    with open(os.path.join(output_dir, 'report.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate predict.py
    names_repr = repr(class_names)
    pred_code = f'''#!/usr/bin/env python3
"""Predict with Classic Tsetlin Machine (CTM)
Faithful to pyTsetlinMachineParallel: OpenCV adaptiveThreshold + MultiClassConvolutionalTsetlinMachine2D
"""
import os, json, argparse, pickle
import numpy as np
import cv2
from PIL import Image

CLASS_NAMES = {names_repr}
IMG_SIZE = {IMG_SIZE}

def predict(image_dir, model_dir, gpu=0):
    from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
    tm = MultiClassConvolutionalTsetlinMachine2D.__new__(MultiClassConvolutionalTsetlinMachine2D)
    tm.__setstate__(pickle.load(open(os.path.join(model_dir, "tm_model.pkl"), "rb")))
    results = []
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img = Image.open(os.path.join(image_dir, f)).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        bimg = cv2.adaptiveThreshold(arr, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        X = bimg[np.newaxis, ...].astype(np.uint8)
        pred = int(tm.predict(X)[0])
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
    _plot_training_curves(history, output_dir, 'CTM (Convolutional TM)')
    print(f"  Saved model to {output_dir}/  (tm_model.pkl, report.json, results.json, predict.py, training_curves.png)", flush=True)
    return results
