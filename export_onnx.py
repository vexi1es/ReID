"""
Экспорт в ONNX для инференса без torch (Docker, onnxruntime):

  weights/reid.onnx  (N,3,224,224) -> L2-эмбеддинг ТС (N,1536)   — обученная модель
  weights/bg.onnx    (N,3,224,392) -> L2-признак фона кадра (N,768) — DINOv2-small, для камеры

    python export_onnx.py --ckpt runs/final/last.pt
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import Dinov2Model

from reid.camfeat import H as BG_H, W as BG_W
from reid.model import ReIDModel


class ReID(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.nn.functional.normalize(self.m(x)[1], dim=1)


class Background(torch.nn.Module):
    def __init__(self, name="facebook/dinov2-small"):
        super().__init__()
        self.m = Dinov2Model.from_pretrained(name)

    def forward(self, x):
        h = self.m(pixel_values=x).last_hidden_state
        return torch.nn.functional.normalize(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1), dim=1)


def export(net, shape, path):
    net = net.eval()
    x = torch.randn(*shape)
    torch.onnx.export(net, x, path, input_names=["image"], output_names=["embedding"],
                      dynamic_axes={"image": {0: "n"}, "embedding": {0: "n"}},
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    ref = net(x).detach().numpy()
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    print(f"{path}: {os.path.getsize(path) / 1e6:.0f} МБ, max |torch - onnx| = {np.abs(ref - got).max():.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="facebook/dinov2-base")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--name", default="reid.onnx", help="имя файла модели ТС")
    ap.add_argument("--weight", type=float, default=1.0, help="вес модели в ансамбле")
    ap.add_argument("--skip-bg", action="store_true", help="bg.onnx уже экспортирован")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    log = os.path.join(os.path.dirname(args.ckpt), "log.json")
    if os.path.exists(log):                      # разрешение и бэкбон — как при обучении
        a = json.load(open(log))["args"]
        args.backbone, args.size = a.get("backbone", args.backbone), a.get("size", args.size)
    m = ReIDModel(args.backbone, grad_ckpt=False)
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    with torch.no_grad():
        export(ReID(m), (2, 3, args.size, args.size), f"{args.out}/{args.name}")
        if not args.skip_bg:
            export(Background(), (2, 3, BG_H, BG_W), f"{args.out}/bg.onnx")
    # meta.json: список моделей ансамбля; запись с тем же именем файла заменяется
    path = f"{args.out}/meta.json"
    meta = json.load(open(path)) if os.path.exists(path) else {}
    models = [x for x in meta.get("models", []) if x["onnx"] != args.name]
    models.append({"onnx": args.name, "size": args.size, "weight": args.weight,
                   "backbone": args.backbone, "ckpt": args.ckpt})
    json.dump({"models": models}, open(path, "w"), indent=1)
    print(f"{path}: " + ", ".join(f"{x['onnx']} {x['size']}px x{x['weight']}" for x in models))


if __name__ == "__main__":
    main()
