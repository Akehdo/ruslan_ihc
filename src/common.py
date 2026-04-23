import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_tiny,
    resnet18,
    resnet50,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = SRC_DIR / "data"
HER2_DATASET_DIR = DATA_DIR / "HER2_dataset"
PATCH_DATASET_DIR = HER2_DATASET_DIR / "her_2_patch" / "Patch-based-dataset"
WSI_DATASET_DIR = HER2_DATASET_DIR / "her_2_wsi"

TRAIN_DIR = PATCH_DATASET_DIR / "train_data_patch"
TEST_DIR = PATCH_DATASET_DIR / "test_data_patch"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = CHECKPOINTS_DIR

NUM_CLASSES = 4
IMAGE_SIZE = 224
SUPPORTED_MODELS = ("resnet18", "resnet50", "convnext", "convnext_tiny")


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def configure_torch_performance(device: torch.device, deterministic: bool = True) -> None:
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_project_dirs() -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_dataset_dirs() -> None:
    required_dirs = {
        "patch dataset": PATCH_DATASET_DIR,
        "train split": TRAIN_DIR,
        "test split": TEST_DIR,
    }

    missing = [f"{name}: {path}" for name, path in required_dirs.items() if not path.exists()]
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(
            "HER2 patch dataset was not found in the expected project structure:\n"
            f"{missing_text}"
        )


def get_transforms(use_augmentation: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    return train_transform, eval_transform


def get_model(model_name: str) -> nn.Module:
    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name in {"convnext", "convnext_tiny"}:
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def build_run_name(model_name: str, seed: int, with_augmentation: bool = True) -> str:
    aug_tag = "with_aug" if with_augmentation else "no_aug"
    return f"{model_name}_{aug_tag}_seed_{seed}"
