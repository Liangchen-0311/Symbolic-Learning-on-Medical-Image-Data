#!/usr/bin/env python3
"""Predict with CBM (original design: independent concept heads)"""
import os, json, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

FRACTURE_NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Linear', 'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral', 'Transverse Displaced', 'Transverse']
FRACTURE_CONCEPTS = ['cortical_break', 'fracture_line_horizontal', 'fracture_line_oblique_45', 'fracture_line_oblique_135', 'fracture_line_vertical', 'displacement', 'bone_fragment', 'soft_tissue_swelling']

class CBMModel(nn.Module):
    def __init__(self, num_concepts=8, num_classes=10, expand_dim=64):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        backbone = models.resnet34(weights=None)
        self.backbone = backbone
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.concept_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, expand_dim), nn.ReLU(), nn.Linear(expand_dim, 1))
            for _ in range(num_concepts)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, expand_dim), nn.ReLU(), nn.Linear(expand_dim, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        concept_logits = torch.cat([head(features) for head in self.concept_heads], dim=1)
        concept_probs = torch.sigmoid(concept_logits)
        logits = self.classifier(concept_probs)
        return logits, concept_probs, concept_logits

def predict(image_dir, weights_path, gpu=0):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = CBMModel()
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
            logits, concepts, _ = model(tensor)
            prob = F.softmax(logits, dim=1)
            pred = logits.argmax(1).item()
            cv = {FRACTURE_CONCEPTS[i]: float(concepts[0, i].item()) for i in range(len(FRACTURE_CONCEPTS))}
        results.append({"file": f, "pred_class": pred, "pred_name": FRACTURE_NAMES[pred],
                         "confidence": prob[0, pred].item(), "concepts": cv})
    for r in results:
        print(f"{r['file']}: {r['pred_name']} ({r['confidence']:.4f})")
        tc = sorted(r['concepts'].items(), key=lambda x: -x[1])[:3]
        print(f"  Top concepts: {tc}")
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
