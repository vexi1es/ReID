"""
Модель для карты внимания в сервисе: тот же дообученный ViT (runs/final), что и reid.onnx,
но с двумя выходами — эмбеддингом и «вкладами патчей».

Эмбеддинг = L2(BN(concat(CLS, mean(patch)))). BN в режиме инференса — поэлементное a*x + b, поэтому
половина эмбеддинга «среднее по патчам» линейна по патчам:
    emb_mean = (a / ||y||) * (1/N) * sum_i p_i + const.
Вклад патча i запроса в косинус с кандидатом c: <(a / (N·||y||)) * p_i, c_mean>. Выход «patches» —
это (a / (N·||y||)) * p_i; сервис умножает его на половину эмбеддинга кандидата и получает точное
разложение этой части сходства по участкам кадра (метод разложения сходства, Stylianou et al., WACV 2019).

    python export_explain.py --ckpt runs/final/last.pt --out weights
"""
import argparse
import os

import numpy as np
import onnxruntime as ort
import torch

from reid.model import ReIDModel


class Explain(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.backbone, self.bn = m.backbone, m.bn
        d = m.backbone.config.hidden_size
        a = m.bn.weight / torch.sqrt(m.bn.running_var + m.bn.eps)
        self.register_buffer("a_mean", a[d:].detach().clone())
        self.d = d

    def forward(self, image):
        h = self.backbone(pixel_values=image).last_hidden_state
        p = h[:, 1:]
        y = self.bn(torch.cat([h[:, 0], p.mean(1)], 1))
        n = y.norm(dim=1, keepdim=True)
        emb = y / n
        patches = p * self.a_mean / (p.shape[1] * n[:, :, None])
        return emb, patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/final/last.pt")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()
    m = ReIDModel("facebook/dinov2-base", grad_ckpt=False)
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    net = Explain(m).eval()
    x = torch.randn(2, 3, args.size, args.size)
    path = f"{args.out}/reid_explain.onnx"
    with torch.no_grad():
        torch.onnx.export(net, x, path, input_names=["image"], output_names=["embedding", "patches"],
                          dynamic_axes={"image": {0: "n"}, "embedding": {0: "n"}, "patches": {0: "n"}},
                          opset_version=17, dynamo=False)
        ref = net(x)[0].numpy()
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    emb, patches = s.run(None, {"image": x.numpy()})
    base = ort.InferenceSession(f"{args.out}/reid.onnx", providers=["CPUExecutionProvider"])
    same = base.run(None, {"image": x.numpy()})[0]
    print(f"{path}: {os.path.getsize(path) / 1e6:.0f} МБ; |torch-onnx| {np.abs(ref - emb).max():.1e}; "
          f"cos с reid.onnx {(emb * same).sum(1).min():.6f}; patches {patches.shape}")


if __name__ == "__main__":
    main()
