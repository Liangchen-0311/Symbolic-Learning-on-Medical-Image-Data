#!/usr/bin/env python3
"""Predict with CRL (original design: BinarizeLayer + UnionLayer + LRLayer)"""
import os, json, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

FRACTURE_NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Linear', 'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral', 'Transverse Displaced', 'Transverse']
FRACTURE_CONCEPTS = ['cortical_break', 'fracture_line_horizontal', 'fracture_line_oblique_45', 'fracture_line_oblique_135', 'fracture_line_vertical', 'displacement', 'bone_fragment', 'soft_tissue_swelling']

class _GradGraft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y): return X
    @staticmethod
    def backward(ctx, grad_output): return None, grad_output.clone()

class _Binarizer(torch.autograd.Function):
    @staticmethod
    def forward(_, concepts): return (concepts.detach() > 0.0).float()
    @staticmethod
    def backward(_, grad_output): return grad_output.clone()

class _BinarizeLayer(nn.Module):
    def __init__(self, n_concepts, use_not=True):
        super().__init__()
        self.n_concepts = n_concepts
        self.use_not = use_not
        self.output_dim = 2 * n_concepts if use_not else n_concepts
    def forward(self, x):
        x = _Binarizer.apply(x)
        if self.use_not: x = torch.cat((x, 1 - x), dim=1)
        return x

class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        y = -1.0 / (-1.0 + torch.sum(torch.log(X + 1e-10), dim=1))
        ctx.save_for_backward(X, y)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        X, y = ctx.saved_tensors
        return grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + 1e-10))

class _ConjunctionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.W = nn.Parameter(0.5 * torch.rand(self.input_dim, self.output_dim))
    def forward(self, x):
        res_tilde = self._continuous_forward(x)
        res_bar = self._binarized_forward(x)
        return _GradGraft.apply(res_bar, res_tilde)
    def _continuous_forward(self, x):
        if self.use_not: x = torch.cat((x, 1 - x), dim=1)
        return _Product.apply(1 - (1 - x).unsqueeze(-1) * self.W)
    @torch.no_grad()
    def _binarized_forward(self, x):
        if self.use_not: x = torch.cat((x, 1 - x), dim=1)
        Wb = _Binarizer.apply(self.W - 0.5)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb, dim=1)

class _DisjunctionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.W = nn.Parameter(0.5 * torch.rand(self.input_dim, self.output_dim))
    def forward(self, x):
        res_tilde = self._continuous_forward(x)
        res_bar = self._binarized_forward(x)
        return _GradGraft.apply(res_bar, res_tilde)
    def _continuous_forward(self, x):
        if self.use_not: x = torch.cat((x, 1 - x), dim=1)
        return 1 - _Product.apply(1 - x.unsqueeze(-1) * self.W)
    @torch.no_grad()
    def _binarized_forward(self, x):
        if self.use_not: x = torch.cat((x, 1 - x), dim=1)
        Wb = _Binarizer.apply(self.W - 0.5)
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb, dim=1)

class _UnionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim * 2
        self.con_layer = _ConjunctionLayer(input_dim, output_dim, use_not=use_not)
        self.dis_layer = _DisjunctionLayer(input_dim, output_dim, use_not=use_not)
    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)

class _LRLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc1(x)

class CRLModel(nn.Module):
    def __init__(self, num_concepts=8, num_classes=10, l1=256, l2=256, use_not=True, use_skip=True, temperature=1.0):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.use_not = use_not
        self.use_skip = use_skip
        backbone = models.resnet34(weights=None)
        self.backbone = backbone
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.concept_predictor = nn.Linear(in_features, num_concepts)
        self.t = nn.Parameter(torch.log(torch.tensor([temperature])))
        self.layer_list = nn.ModuleList()
        dim_list = [num_concepts, l1, l2, num_classes]
        prev_layer_dim = None
        for idx, dim in enumerate(dim_list):
            if idx == 0:
                layer = _BinarizeLayer(dim, use_not)
            elif idx == len(dim_list) - 1:
                layer = _LRLayer(prev_layer_dim, dim)
            else:
                layer_use_not = True if idx != 1 else False
                layer = _UnionLayer(prev_layer_dim, dim, use_not=layer_use_not)
            prev_layer_dim = layer.output_dim
            if use_skip and idx >= 3:
                skip_dim = self.layer_list[-2].output_dim
                prev_layer_dim += skip_dim
            self.layer_list.append(layer)
        self._skip_indices = {}
        for idx in range(3, len(dim_list)):
            self._skip_indices[idx] = idx - 2
    def forward(self, x):
        features = self.backbone(x)
        concept_logits = self.concept_predictor(features)
        concept_probs = torch.sigmoid(concept_logits)
        h = concept_logits
        skip_cache = {}
        for idx, layer in enumerate(self.layer_list):
            if idx in self._skip_indices:
                skip_idx = self._skip_indices[idx]
                h = torch.cat((h, skip_cache[skip_idx]), dim=1)
            h = layer(h)
            if idx in self._skip_indices.values():
                skip_cache[idx] = h
        logits = h / torch.exp(self.t)
        return logits, concept_probs, concept_logits

def predict(image_dir, weights_path, gpu=0):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = CRLModel()
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
