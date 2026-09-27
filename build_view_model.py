"""
Определитель ракурса ТС (спереди / сзади / сбоку) для поправки ранжирования.
DINOv2-small (без дообучения) на кропе ТС, признак = [CLS, среднее патчей], логистическая регрессия
по ручной разметке команды (team_labels/*.csv, 1 500 кадров train). Голова вшивается в ONNX:
на выходе вероятности классов. Проверка: точность на размеченных кадрах отложенных ТС.

    python build_view_model.py --data <dataset> --out weights
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import Dinov2Model

from reid.data import make_val_split, read_csv
from reid.loader import CropDataset, eval_tf

CLASSES = ["front", "rear", "side"]


class ViewNet(torch.nn.Module):
    def __init__(self, backbone, W, b):
        super().__init__()
        self.backbone = backbone
        self.fc = torch.nn.Linear(W.shape[1], W.shape[0])
        self.fc.weight.data = torch.tensor(W, dtype=torch.float32)
        self.fc.bias.data = torch.tensor(b, dtype=torch.float32)

    def forward(self, image):
        h = self.backbone(pixel_values=image).last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
        f = torch.nn.functional.normalize(f, dim=1)
        return torch.softmax(self.fc(f), 1)


def features(model, ids, crops, dev):
    dl = torch.utils.data.DataLoader(CropDataset(ids, crops, eval_tf(224)), batch_size=64)
    out = []
    with torch.no_grad():
        for x in dl:
            h = model(pixel_values=x.to(dev)).last_hidden_state.float()
            out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu())
    return torch.nn.functional.normalize(torch.cat(out), dim=1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--out", default="weights")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train = read_csv(f"{args.data}/train.csv")
    _, gt = make_val_split(train)
    lab = pd.concat([pd.read_csv(f) for f in glob.glob("team_labels/*.csv") or glob.glob("data_labels/*.csv")]).drop_duplicates("image_id")
    lab = lab[lab.view.isin(CLASSES)].reset_index(drop=True)
    bb = Dinov2Model.from_pretrained("facebook/dinov2-small").to(dev).eval()
    F = features(bb, list(lab.image_id), args.crops, dev)

    held = lab.image_id.isin(gt.image_id).values          # кадры отложенных ТС — только для проверки
    clf = LogisticRegression(max_iter=3000, C=3).fit(F[~held], lab.view[~held])
    acc = (clf.predict(F[held]) == lab.view[held]).mean()
    print(f"точность на отложенных ТС: {acc:.3f} ({held.sum()} кадров)")

    clf = LogisticRegression(max_iter=3000, C=3).fit(F, lab.view)   # финальная — на всей разметке
    order = [list(clf.classes_).index(c) for c in CLASSES]
    net = ViewNet(bb.float().cpu(), clf.coef_[order], clf.intercept_[order]).eval()
    os.makedirs(args.out, exist_ok=True)
    torch.onnx.export(net, torch.zeros(1, 3, 224, 224), f"{args.out}/view.onnx", input_names=["image"],
                      output_names=["prob"], dynamic_axes={"image": {0: "b"}, "prob": {0: "b"}}, opset_version=17)
    meta_path = f"{args.out}/meta.json"
    meta = json.load(open(meta_path, encoding="utf-8"))
    meta["view"] = {"onnx": "view.onnx", "size": 224, "classes": CLASSES, "beta": 0.08}
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=1)
    print("сохранено:", f"{args.out}/view.onnx", os.path.getsize(f"{args.out}/view.onnx") // 2**20, "МБ")


if __name__ == "__main__":
    main()
