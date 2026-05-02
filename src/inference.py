"""Inference utilities including sliding-window and Dask parallel execution."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

from .train import build_model, load_config
from .utils import IMAGENET_MEAN, IMAGENET_STD, gaussian_weight_map, select_device

try:
    import dask
    from dask import delayed

    HAS_DASK = True
except Exception:
    dask = None
    delayed = None
    HAS_DASK = False

try:
    from dask_cuda import LocalCUDACluster  # type: ignore
    from distributed import Client  # type: ignore

    HAS_DASK_CUDA = True
except Exception:
    LocalCUDACluster = None
    Client = None
    HAS_DASK_CUDA = False


def preprocess_pil(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image into a normalized tensor."""

    tensor = TF.to_tensor(image.convert("RGB"))
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


@torch.no_grad()
def infer_probability_tile(
    model: torch.nn.Module,
    tile_a: torch.Tensor,
    tile_b: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run inference for one tile and return a CPU numpy probability map."""

    tile_a = tile_a.unsqueeze(0).to(device)
    tile_b = tile_b.unsqueeze(0).to(device)
    output = model(tile_a, tile_b)
    probability = output["probability"] if isinstance(output, dict) else output[2]
    return probability.squeeze().detach().cpu().numpy()


def generate_tile_coords(image_size: Tuple[int, int], tile_size: int, overlap: int) -> List[Tuple[int, int]]:
    """Generate sliding-window tile coordinates."""

    height, width = image_size
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size for sliding-window inference.")
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    if ys[-1] != height - tile_size:
        ys.append(height - tile_size)
    if xs[-1] != width - tile_size:
        xs.append(width - tile_size)
    return [(y, x) for y in ys for x in xs]


def sliding_window_inference(
    model: torch.nn.Module,
    image_a: Image.Image,
    image_b: Image.Image,
    device: torch.device,
    tile_size: int = 256,
    overlap: int = 128,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run weighted sliding-window inference on a full-resolution image pair.

    Args:
        model: Trained segmentation model.
        image_a: Time-1 image.
        image_b: Time-2 image.
        device: Torch device.
        tile_size: Sliding window size.
        overlap: Tile overlap in pixels.
        threshold: Probability threshold for binary prediction.

    Returns:
        Tuple of ``(probability_map, binary_mask)``.
    """

    tensor_a = preprocess_pil(image_a)
    tensor_b = preprocess_pil(image_b)
    _, height, width = tensor_a.shape
    coords = generate_tile_coords((height, width), tile_size=tile_size, overlap=overlap)
    weight = gaussian_weight_map(tile_size).numpy()
    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for top, left in coords:
        tile_a = tensor_a[:, top : top + tile_size, left : left + tile_size]
        tile_b = tensor_b[:, top : top + tile_size, left : left + tile_size]
        probability = infer_probability_tile(model, tile_a, tile_b, device)
        prob_sum[top : top + tile_size, left : left + tile_size] += probability * weight
        weight_sum[top : top + tile_size, left : left + tile_size] += weight

    probability_map = prob_sum / np.clip(weight_sum, 1e-6, None)
    binary_mask = (probability_map >= threshold).astype(np.uint8)
    return probability_map, binary_mask


def dask_sliding_window_inference(
    model: torch.nn.Module,
    image_a: Image.Image,
    image_b: Image.Image,
    device: torch.device,
    tile_size: int = 256,
    overlap: int = 128,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run delayed sliding-window inference with Dask when available."""

    if not HAS_DASK:
        return sliding_window_inference(model, image_a, image_b, device, tile_size, overlap, threshold)

    tensor_a = preprocess_pil(image_a)
    tensor_b = preprocess_pil(image_b)
    _, height, width = tensor_a.shape
    coords = generate_tile_coords((height, width), tile_size=tile_size, overlap=overlap)
    weight = gaussian_weight_map(tile_size).numpy()
    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    tasks = []
    for top, left in coords:
        tile_a = tensor_a[:, top : top + tile_size, left : left + tile_size]
        tile_b = tensor_b[:, top : top + tile_size, left : left + tile_size]
        tasks.append((top, left, delayed(infer_probability_tile)(model, tile_a, tile_b, device)))

    client = None
    cluster = None
    try:
        if HAS_DASK_CUDA:
            cluster = LocalCUDACluster()
            client = Client(cluster)
            results = dask.compute(*[task for _, _, task in tasks])
        else:
            results = dask.compute(*[task for _, _, task in tasks], scheduler="threads")
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()

    for (top, left, _), probability in zip(tasks, results):
        prob_sum[top : top + tile_size, left : left + tile_size] += probability * weight
        weight_sum[top : top + tile_size, left : left + tile_size] += weight

    probability_map = prob_sum / np.clip(weight_sum, 1e-6, None)
    binary_mask = (probability_map >= threshold).astype(np.uint8)
    return probability_map, binary_mask


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run inference on a new image pair.")
    parser.add_argument("--config", type=Path, default=Path("configs/full.yaml"), help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint path.")
    parser.add_argument("--img_a", type=Path, required=True, help="Time-1 image path.")
    parser.add_argument("--img_b", type=Path, required=True, help="Time-2 image path.")
    parser.add_argument("--output", type=Path, default=Path("results/inference_mask.png"), help="Output mask path.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config = load_config(args.config)
    device = select_device(config.get("device"))
    model = build_model(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    image_a = Image.open(args.img_a).convert("RGB")
    image_b = Image.open(args.img_b).convert("RGB")
    probability_map, binary_mask = dask_sliding_window_inference(
        model,
        image_a,
        image_b,
        device,
        tile_size=int(config["inference"]["tile_size"]),
        overlap=int(config["inference"]["overlap"]),
        threshold=float(config["inference"]["threshold"]),
    )
    output = Image.fromarray((binary_mask * 255).astype(np.uint8))
    output.save(args.output)
    print(f"Saved binary prediction to {args.output}")


if __name__ == "__main__":
    main()
