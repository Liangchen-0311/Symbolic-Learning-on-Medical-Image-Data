#!/usr/bin/env python3
"""Predict with DINOv2 + Linear Probe"""
import os, json, argparse, pickle
import torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image

FRACTURE_NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Linear', 'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral', 'Transverse Displaced', 'Transverse']

def predict(image_dir, model_dir, gpu=0):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
    backbone.eval().to(device)
    scaler = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), "rb"))
    clf = pickle.load(open(os.path.join(model_dir, "classifier.pkl"), "rb"))
    transform = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    results = []
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img = Image.open(os.path.join(image_dir, f)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = backbone(tensor).cpu().numpy()
        feat_s = scaler.transform(feat)
        pred = clf.predict(feat_s)[0]
        prob = clf.predict_proba(feat_s)[0]
        results.append({"file": f, "pred_class": int(pred), "pred_name": FRACTURE_NAMES[pred],
                         "confidence": float(prob[pred])})
    for r in results:
        print(f"{r['file']}: {r['pred_name']} ({r['confidence']:.4f})")
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
