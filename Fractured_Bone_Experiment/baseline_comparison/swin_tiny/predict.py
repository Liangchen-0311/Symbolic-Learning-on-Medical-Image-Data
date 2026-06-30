#!/usr/bin/env python3
"""Predict with Swin-Tiny (vit_b_16)"""
import os, json, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
from PIL import Image

FRACTURE_NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Linear', 'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral', 'Transverse Displaced', 'Transverse']

def create_model(num_classes=10):
    import timm
    return timm.create_model('vit_b_16', pretrained=False, num_classes=num_classes)

def predict(image_dir, weights_path, gpu=0):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = create_model()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model = model.to(device).eval()
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
            out = model(tensor)
            if isinstance(out, tuple):
                out = out[0]
            prob = F.softmax(out, dim=1)
            pred = out.argmax(1).item()
        results.append({"file": f, "pred_class": pred, "pred_name": FRACTURE_NAMES[pred],
                         "confidence": prob[0, pred].item()})
    for r in results:
        print(f"{r['file']}: {r['pred_name']} ({r['confidence']:.4f})")
    out_path = os.path.join(os.path.dirname(weights_path), "predict_results.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_weights.pth"))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    predict(args.image_dir, args.weights, args.gpu)
