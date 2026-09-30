from __future__ import annotations

from pathlib import Path
import os
import numpy as np
from PIL import Image


FULL_VISUAL_MODEL = "open_clip:ViT-B-32/laion2b_s34b_b79k"


class LightweightVisualEmbedder:
    """Handcrafted visual descriptor used only as the lightweight fallback."""

    name = "lightweight-visual-v1"
    supports_text = False

    def encode_pil(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize((96, 96))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        features: list[float] = []
        for channel in range(3):
            hist, _ = np.histogram(array[:, :, channel], bins=8, range=(0, 1), density=True)
            features.extend(hist.tolist())
        hsv = np.asarray(resized.convert("HSV"), dtype=np.float32) / 255.0
        for channel in range(3):
            hist, _ = np.histogram(hsv[:, :, channel], bins=8, range=(0, 1), density=True)
            features.extend(hist.tolist())
        luminance = np.asarray(resized.convert("L").resize((8, 8)), dtype=np.float32).reshape(-1) / 255.0
        features.extend(luminance.tolist())
        gray = np.asarray(resized.convert("L"), dtype=np.float32) / 255.0
        gradient_y, gradient_x = np.gradient(gray)
        magnitude = np.hypot(gradient_x, gradient_y)
        angle = (np.arctan2(gradient_y, gradient_x) + np.pi) % (2 * np.pi)
        gradient_hist, _ = np.histogram(angle, bins=12, range=(0, 2 * np.pi), weights=magnitude)
        features.extend(gradient_hist.tolist())
        vector = np.asarray(features, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-9)

    def encode_path(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return self.encode_pil(image)

    def encode_paths(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        del batch_size
        return np.vstack([self.encode_path(path) for path in paths]).astype(np.float32)


class OpenCLIPEmbedder:
    """OpenCLIP image/text encoder used by FINDX_MODE=full."""

    name = FULL_VISUAL_MODEL
    supports_text = True

    def __init__(self):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "FINDX_MODE=full requires open_clip_torch. Install backend/requirements-full.txt. "
                "FindX will not silently downgrade FULL mode to handcrafted visual features."
            ) from exc
        self.open_clip = open_clip
        self.torch = torch
        self.model_name = os.getenv("FINDX_CLIP_MODEL", "ViT-B-32")
        self.pretrained = os.getenv("FINDX_CLIP_PRETRAINED", "laion2b_s34b_b79k")
        self.cache_dir = os.getenv("OPENCLIP_CACHE_DIR") or None
        requested_device = os.getenv("FINDX_DEVICE", "auto")
        if requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested_device
        kwargs = {"pretrained": self.pretrained}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name, **kwargs)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model = self.model.to(self.device).eval()
        self.name = f"open_clip:{self.model_name}/{self.pretrained}"

    def encode_pil(self, image: Image.Image) -> np.ndarray:
        with self.torch.no_grad():
            tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
            vector = self.model.encode_image(tensor)
            vector = vector / vector.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return vector[0].detach().cpu().numpy().astype(np.float32)

    def encode_path(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return self.encode_pil(image)

    def encode_paths(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        rows: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                tensors = []
                for path in batch_paths:
                    with Image.open(path) as image:
                        tensors.append(self.preprocess(image.convert("RGB")))
                batch = self.torch.stack(tensors).to(self.device)
                vectors = self.model.encode_image(batch)
                vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                rows.append(vectors.detach().cpu().numpy().astype(np.float32))
        return np.vstack(rows)

    def encode_text(self, text: str) -> np.ndarray:
        with self.torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            vector = self.model.encode_text(tokens)
            vector = vector / vector.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return vector[0].detach().cpu().numpy().astype(np.float32)


def make_visual(mode: str = "lightweight"):
    if mode == "full":
        return OpenCLIPEmbedder()
    if mode == "lightweight":
        return LightweightVisualEmbedder()
    raise ValueError("FINDX_MODE must be 'lightweight' or 'full'")
