"""Dataset and dataloader utilities for LEVIR-CD with optional RAPIDS loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .utils import IMAGENET_MEAN, IMAGENET_STD

try:
    from cucim import CuImage  # type: ignore

    HAS_CUCIM = True
except Exception:
    CuImage = None
    HAS_CUCIM = False


SPLIT_NAMES = ("train", "val", "test")
FOLDER_A_CANDIDATES = ("A", "a", "img1", "T1", "t1")
FOLDER_B_CANDIDATES = ("B", "b", "img2", "T2", "t2")
FOLDER_LABEL_CANDIDATES = ("label", "labels", "gt", "GT", "mask")


@dataclass
class DatasetLayout:
    """Resolved LEVIR-CD folder layout."""

    root: Path
    folder_a: str
    folder_b: str
    folder_label: str


def find_data_root(search_root: Path) -> DatasetLayout:
    """Auto-detect the LEVIR-CD data root and folder names.

    Args:
        search_root: Top-level search path.

    Returns:
        Resolved dataset layout.
    """

    if not search_root.exists():
        raise FileNotFoundError(
            f"Dataset search root not found: {search_root}. Download LEVIR-CD first or update data.root in the config."
        )
    for directory in search_root.rglob("*"):
        if not directory.is_dir():
            continue
        children = {child.name for child in directory.iterdir() if child.is_dir()}
        if not {"train", "val", "test"}.issubset(children):
            continue
        folder_a = next((name for name in FOLDER_A_CANDIDATES if (directory / "train" / name).is_dir()), None)
        folder_b = next((name for name in FOLDER_B_CANDIDATES if (directory / "train" / name).is_dir()), None)
        folder_label = next(
            (name for name in FOLDER_LABEL_CANDIDATES if (directory / "train" / name).is_dir()),
            None,
        )
        if folder_a and folder_b and folder_label:
            return DatasetLayout(root=directory, folder_a=folder_a, folder_b=folder_b, folder_label=folder_label)
    raise FileNotFoundError(
        f"Could not auto-detect LEVIR-CD under {search_root}. Expected train/val/test with A, B, and label folders."
    )


class LEVIRChangeDataset(Dataset):
    """LEVIR-CD dataset with synchronized paired transforms.

    Args:
        layout: Resolved dataset layout.
        split: Dataset split name.
        crop_size: Output crop size.
        is_train: Whether to use random cropping and flips.
        use_cucim: Try GPU-native cuCIM loading when available.
    """

    def __init__(
        self,
        layout: DatasetLayout,
        split: str,
        crop_size: int,
        is_train: bool,
        use_cucim: bool = False,
    ) -> None:
        super().__init__()
        if split not in SPLIT_NAMES:
            raise ValueError(f"Invalid split '{split}'. Expected one of {SPLIT_NAMES}.")
        self.layout = layout
        self.split = split
        self.crop_size = crop_size
        self.is_train = is_train
        self.use_cucim = use_cucim and HAS_CUCIM
        split_root = layout.root / split
        self.files_a = sorted((split_root / layout.folder_a).glob("*.png"))
        self.files_b = sorted((split_root / layout.folder_b).glob("*.png"))
        self.files_l = sorted((split_root / layout.folder_label).glob("*.png"))
        if not self.files_a:
            raise FileNotFoundError(f"No images found in {(split_root / layout.folder_a)}. Check the dataset path.")
        if not (len(self.files_a) == len(self.files_b) == len(self.files_l)):
            raise RuntimeError(
                f"File count mismatch in split '{split}': A={len(self.files_a)}, B={len(self.files_b)}, mask={len(self.files_l)}."
            )

    def __len__(self) -> int:
        return len(self.files_a)

    def _load_pil(self, path: Path, mode: str) -> Image.Image:
        return Image.open(path).convert(mode)

    def _load_image(self, path: Path, mode: str) -> Image.Image:
        if self.use_cucim and mode == "RGB":
            try:
                cu_image = CuImage(str(path))
                array = cu_image.read_region(location=(0, 0), size=cu_image.shape[:2][::-1], level=0)
                return Image.fromarray(np.asarray(array)[..., :3]).convert(mode)
            except Exception:
                return self._load_pil(path, mode)
        return self._load_pil(path, mode)

    def _apply_spatial(self, image_a: Image.Image, image_b: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image, Image.Image]:
        crop = self.crop_size
        if self.is_train:
            if image_a.height < crop or image_a.width < crop:
                raise ValueError(
                    f"Crop size {crop} is larger than image size {(image_a.width, image_a.height)} in split '{self.split}'."
                )
            top = torch.randint(0, image_a.height - crop + 1, (1,)).item()
            left = torch.randint(0, image_a.width - crop + 1, (1,)).item()
            image_a = TF.crop(image_a, top, left, crop, crop)
            image_b = TF.crop(image_b, top, left, crop, crop)
            mask = TF.crop(mask, top, left, crop, crop)
            if torch.rand(1).item() > 0.5:
                image_a = TF.hflip(image_a)
                image_b = TF.hflip(image_b)
                mask = TF.hflip(mask)
            if torch.rand(1).item() > 0.5:
                image_a = TF.vflip(image_a)
                image_b = TF.vflip(image_b)
                mask = TF.vflip(mask)
        else:
            image_a = TF.center_crop(image_a, crop)
            image_b = TF.center_crop(image_b, crop)
            mask = TF.center_crop(mask, crop)
        return image_a, image_b, mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_a = self._load_image(self.files_a[index], "RGB")
        image_b = self._load_image(self.files_b[index], "RGB")
        mask = self._load_image(self.files_l[index], "L")
        image_a, image_b, mask = self._apply_spatial(image_a, image_b, mask)

        tensor_a = TF.to_tensor(image_a)
        tensor_b = TF.to_tensor(image_b)
        tensor_a = TF.normalize(tensor_a, IMAGENET_MEAN, IMAGENET_STD)
        tensor_b = TF.normalize(tensor_b, IMAGENET_MEAN, IMAGENET_STD)
        mask_np = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        tensor_m = torch.from_numpy(mask_np).unsqueeze(0)

        return {
            "image_a": tensor_a,
            "image_b": tensor_b,
            "mask": tensor_m,
            "path_a": str(self.files_a[index]),
            "path_b": str(self.files_b[index]),
            "path_mask": str(self.files_l[index]),
        }


def create_dataloaders(config: Dict) -> Tuple[LEVIRChangeDataset, LEVIRChangeDataset, LEVIRChangeDataset, DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders from a config dictionary."""

    data_cfg = config["data"]
    train_cfg = config["training"]
    layout = find_data_root(Path(data_cfg["root"]))
    train_ds = LEVIRChangeDataset(
        layout=layout,
        split="train",
        crop_size=int(data_cfg["crop_size"]),
        is_train=True,
        use_cucim=bool(data_cfg.get("use_cucim", False)),
    )
    val_ds = LEVIRChangeDataset(
        layout=layout,
        split="val",
        crop_size=int(data_cfg["eval_crop_size"]),
        is_train=False,
        use_cucim=bool(data_cfg.get("use_cucim", False)),
    )
    test_ds = LEVIRChangeDataset(
        layout=layout,
        split="test",
        crop_size=int(data_cfg["eval_crop_size"]),
        is_train=False,
        use_cucim=bool(data_cfg.get("use_cucim", False)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def compute_positive_class_weight(dataset: LEVIRChangeDataset) -> float:
    """Estimate positive class weight from the training set.

    Args:
        dataset: Training dataset.

    Returns:
        ``total_pixels / changed_pixels``.
    """

    changed_pixels = 0.0
    total_pixels = 0.0
    for index in range(len(dataset)):
        sample = dataset[index]
        mask = sample["mask"]
        changed_pixels += float(mask.sum())
        total_pixels += float(mask.numel())
    if changed_pixels <= 0:
        raise RuntimeError("Training set contains zero positive pixels. Recheck the label preprocessing pipeline.")
    return total_pixels / changed_pixels

