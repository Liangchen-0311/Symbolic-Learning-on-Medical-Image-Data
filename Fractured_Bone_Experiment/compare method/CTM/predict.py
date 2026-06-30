#!/usr/bin/env python3
"""Predict with Classic Tsetlin Machine (CTM)
Faithful to pyTsetlinMachineParallel: OpenCV adaptiveThreshold + MultiClassConvolutionalTsetlinMachine2D

Usage:
    python predict.py --image_dir /path/to/images --model_dir /path/to/model
"""
import os, json, argparse, pickle
import numpy as np
import cv2
from PIL import Image

FRACTURE_NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Linear',
                  'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
                  'Transverse Displaced', 'Transverse']
IMG_SIZE = 64


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
        results.append({"file": f, "pred_class": pred, "pred_name": FRACTURE_NAMES[pred]})
    for r in results:
        print(f"{r['file']}: {r['pred_name']}")
    out_path = os.path.join(model_dir, "predict_results.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--model_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    predict(args.image_dir, args.model_dir, args.gpu)
