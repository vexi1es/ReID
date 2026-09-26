"""
Дообучение re-id модели. Валидация — отложенные ТС train по протоколу evaluate.py.

    python train.py --data <папка dataset> --crops crops --out runs/v1
    python train.py ... --full        # финальная модель: учимся на всём train
"""
import argparse
import json
import math
import os
import time

import numpy as np
import psutil
import torch

from reid.data import make_val_split, read_csv
from reid.loader import CropDataset, PKSampler, extract, train_tf
from reid.model import ArcFace, ReIDModel, batch_hard_triplet
from validate import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--backbone", default="facebook/dinov2-base")
    ap.add_argument("--freeze", type=int, default=6)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--P", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5, help="lr бэкбона")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16"])
    args = ap.parse_args()

    # ~90% ресурсов: фоновый приоритет, ноутбук остаётся отзывчивым
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if psutil.WINDOWS else 10)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train = read_csv(f"{args.data}/train.csv")
    rest, gt = make_val_split(train, seed=0)
    tr = train if args.full else rest
    classes = {v: i for i, v in enumerate(sorted(tr.vehicle_id.unique()))}
    labels = tr.vehicle_id.map(classes).values
    print(f"train: {len(tr)} кадров, {len(classes)} ТС; val: {len(gt)} кадров")

    ds = CropDataset(tr.image_id, args.crops, train_tf(args.size), labels)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.P * args.K, sampler=PKSampler(labels, args.P, args.K, args.seed),
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0)

    model = ReIDModel(args.backbone, args.freeze, grad_ckpt=bool(args.grad_ckpt)).to(dev)
    head = ArcFace(model.dim, len(classes)).to(dev)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
        {"params": model.bn.parameters(), "lr": args.head_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.05)
    base = [g["lr"] for g in opt.param_groups]
    steps = args.epochs * len(dl)
    warm = len(dl)
    # ConvNeXt DINOv3 переполняет fp16 (inf-градиенты -> GradScaler молча пропускает все шаги):
    # для него bf16 — диапазон как у fp32, скейлер не нужен
    amp_dtype = (torch.bfloat16 if model.is_timm and dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)   # T4/P100 (Kaggle) без bf16 -> fp16 + GradScaler
    if args.amp != "auto":
        amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=dev == "cuda" and amp_dtype == torch.float16)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best, log, step, start = -1.0, [], 0, 1
    state_path = f"{args.out}/state.pt"
    if os.path.exists(state_path):          # продолжение после сбоя с последней эпохи
        st = torch.load(state_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        best, log, step, start = st["best"], st["log"], st["step"], st["epoch"] + 1
        print(f"resume: эпоха {start}, best {best:.4f}", flush=True)

    for ep in range(start, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            k = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * k
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=dev == "cuda"):
                feat, bn = model(x)
            loss = ce(head(bn.float(), y), y) + batch_hard_triplet(feat.float(), y)
            if not torch.isfinite(loss):
                print("NaN/inf loss — выход с кодом 3", flush=True)
                raise SystemExit(3)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            step += 1
        rec = {"epoch": ep, "loss": tot / len(dl), "sec": round(time.time() - t0)}

        if not args.full and (ep % args.eval_every == 0 or ep == args.epochs):
            emb = extract(model, gt.image_id, args.crops, args.size, dev, workers=args.workers)
            rm, (t, cm) = score(gt, emb)
            rec.update({"mAP@10": rm["mAP@10"], "R1": rm["Rank-1"], "R5": rm["Rank-5"],
                        "F1": cm["F1"], "thr": float(t), "PR-AUC": cm["PR-AUC"]})
            if rm["mAP@10"] > best:
                best = rm["mAP@10"]
                torch.save(model.state_dict(), f"{args.out}/best.pt")
        log.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        json.dump({"args": vars(args), "log": log}, open(f"{args.out}/log.json", "w"), indent=1)
        torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "best": best, "log": log, "step": step, "epoch": ep},
                   state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)   # атомарно: сбой не портит чекпойнт

    torch.save(model.state_dict(), f"{args.out}/last.pt")
    if os.path.exists(state_path):
        os.remove(state_path)                          # прогон завершён — состояние больше не нужно
    print(f"best mAP@10 {best:.4f}")


if __name__ == "__main__":
    main()
