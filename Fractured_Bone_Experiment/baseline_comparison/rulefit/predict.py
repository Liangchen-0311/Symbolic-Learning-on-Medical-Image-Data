#!/usr/bin/env python3
"""Predict with MobileNetV2 + RuleFit"""
import os, json, argparse, pickle
import torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image

FRACTURE_NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Linear', 'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral', 'Transverse Displaced', 'Transverse']

def predict(image_dir, model_dir, gpu=0):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    backbone = models.mobilenet_v2(weights=None)
    backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, 10)
    backbone.load_state_dict(torch.load(os.path.join(model_dir, "backbone_weights.pth"), map_location="cpu"))
    backbone = backbone.to(device).eval()
    scaler = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), "rb"))
    selector = pickle.load(open(os.path.join(model_dir, "selector.pkl"), "rb"))
    rule_model = pickle.load(open(os.path.join(model_dir, "rule_model.pkl"), "rb"))
    feat_ext = nn.Sequential(*list(backbone.features.children()))
    pool = nn.AdaptiveAvgPool2d(1)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
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
            feat = pool(feat_ext(tensor)).flatten(1).cpu().numpy()
        feat_s = scaler.transform(feat)
        feat_sel = selector.transform(feat_s)
        pred = rule_model.predict(feat_sel)[0]
        results.append({"file": f, "pred_class": int(pred), "pred_name": FRACTURE_NAMES[pred]})
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
