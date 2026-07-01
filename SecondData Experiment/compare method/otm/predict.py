#!/usr/bin/env python3
"""Predict with Optimized Tsetlin Machine (OTM)
Faithful to OTM toolbox: Color Thermometer (3ch x 8bit = 24 bits/pixel) + TMClassifier
"""
import os, json, argparse, pickle
import numpy as np
from PIL import Image

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
IMG_SIZE = 32
RESOLUTION = 8
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
        results.append({"file": f, "pred_class": pred, "pred_name": CLASS_NAMES[pred]})
    for r in results:
        print(f"{r['file']}: {r['pred_name']}")
    with open(os.path.join(model_dir, "predict_results.json"), "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"Saved to {os.path.join(model_dir, 'predict_results.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--model_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    predict(args.image_dir, args.model_dir, args.gpu)
