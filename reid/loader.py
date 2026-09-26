"""Датасет по кешу кропов, PK-сэмплер и аугментации."""
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Sampler

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def train_tf(size):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.Pad(10, padding_mode="edge"),
        T.RandomCrop(size),
        # только яркость/контраст: цвет кузова — ядро идентичности, hue не трогаем
        T.ColorJitter(brightness=0.25, contrast=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.5, value="random"),
    ])


def eval_tf(size):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class CropDataset(Dataset):
    def __init__(self, image_ids, crops_dir, tf, labels=None):
        self.ids, self.dir, self.tf = list(image_ids), crops_dir, tf
        self.labels = None if labels is None else np.asarray(labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x = self.tf(Image.open(f"{self.dir}/{self.ids[i]}.jpg").convert("RGB"))
        return x if self.labels is None else (x, int(self.labels[i]))


class PKSampler(Sampler):
    """P идентичностей × K кадров в батче (для batch-hard triplet)."""

    def __init__(self, labels, P=12, K=4, seed=0):
        self.by_id = {}
        for i, y in enumerate(labels):
            self.by_id.setdefault(int(y), []).append(i)
        self.P, self.K, self.rng = P, K, random.Random(seed)

    def __len__(self):
        return len(self.by_id) // self.P * self.P * self.K

    def __iter__(self):
        ids = list(self.by_id)
        self.rng.shuffle(ids)
        for b in range(len(ids) // self.P):
            for y in ids[b * self.P:(b + 1) * self.P]:
                idx = self.by_id[y]
                pick = self.rng.sample(idx, self.K) if len(idx) >= self.K else self.rng.choices(idx, k=self.K)
                yield from pick


@torch.no_grad()
def extract(model, image_ids, crops_dir, size, device, batch=64, workers=4):
    """L2-эмбеддинги с flip-TTA, в порядке image_ids."""
    dl = torch.utils.data.DataLoader(CropDataset(image_ids, crops_dir, eval_tf(size)),
                                     batch_size=batch, num_workers=workers)
    model.eval()
    out = []
    for x in dl:
        x = x.to(device, non_blocking=True)
        # ConvNeXt DINOv3: на инференсе только fp32 — bf16 у финальной модели искажает
        # эмбеддинг (косинус с fp32 0.959), fp16 рискует переполнением; ViT — fp16 (== fp32)
        timm_model = getattr(model, "is_timm", False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda" and not timm_model):
            e = model.embed(x) + model.embed(torch.flip(x, dims=[3]))
        out.append(torch.nn.functional.normalize(e.float(), dim=1).cpu())
    return torch.cat(out).numpy()
