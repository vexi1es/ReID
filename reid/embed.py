"""Эмбеддинги ТС: DINOv2 (замороженный) + flip-TTA."""
import numpy as np
import torch
import torchvision.transforms as T
from transformers import Dinov2Model

from .data import crop

SIZE = 224
_tf = T.Compose([
    T.Resize((SIZE, SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class Embedder:
    def __init__(self, name="facebook/dinov2-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Dinov2Model.from_pretrained(name).to(self.device).eval()

    @torch.no_grad()
    def _forward(self, x):
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            h = self.model(pixel_values=x).last_hidden_state.float()
        cls = torch.nn.functional.normalize(h[:, 0], dim=1)
        patch = torch.nn.functional.normalize(h[:, 1:].mean(1), dim=1)
        return torch.cat([cls, patch], 1)

    def __call__(self, df, images_dir, batch=32):
        out = []
        for i in range(0, len(df), batch):
            x = torch.stack([_tf(crop(images_dir, r)) for r in df.iloc[i:i + batch].itertuples()])
            x = x.to(self.device)
            e = self._forward(x) + self._forward(torch.flip(x, dims=[3]))
            out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
            print(f"\r  embed {min(i + batch, len(df))}/{len(df)}", end="", flush=True)
        print()
        return np.concatenate(out).astype(np.float32)
