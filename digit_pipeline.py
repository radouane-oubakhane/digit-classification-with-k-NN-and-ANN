from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


@dataclass
class ExperimentConfig:
    seed: int = 42
    data_root: str = "data"
    train_per_class: int = 700
    val_per_class: int = 100
    test_per_class: int = 200
    tchebichef_order: int = 3
    knn_k_values: Tuple[int, ...] = (1, 3, 5, 7, 11)
    knn_chunk_size: int = 512
    batch_size: int = 256
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dims: Tuple[int, int] = (256, 128)
    dropout: float = 0.2
    patience: int = 5
    allow_fake_fallback: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fake_dataset_to_arrays(size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    fake_dataset = datasets.FakeData(
        size=size,
        image_size=(1, 28, 28),
        num_classes=10,
        random_offset=0,
        transform=None,
        target_transform=None,
    )
    images = np.empty((size, 28, 28), dtype=np.uint8)
    labels = np.empty((size,), dtype=np.int64)

    for idx in range(size):
        image, label = fake_dataset[idx]
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        images[idx] = arr
        labels[idx] = int(label)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(size)
    return images[perm], labels[perm]


def _load_mnist_arrays(data_root: str, allow_fake_fallback: bool, seed: int) -> Dict[str, np.ndarray]:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        train_ds = datasets.MNIST(root=str(root), train=True, download=True)
        test_ds = datasets.MNIST(root=str(root), train=False, download=True)
        return {
            "source": "MNIST (torchvision)",
            "train_images": train_ds.data.numpy().astype(np.uint8),
            "train_labels": train_ds.targets.numpy().astype(np.int64),
            "test_images": test_ds.data.numpy().astype(np.uint8),
            "test_labels": test_ds.targets.numpy().astype(np.int64),
        }
    except Exception as exc:
        if not allow_fake_fallback:
            raise RuntimeError(
                "Failed to load MNIST from torchvision. "
                "Enable internet access or set allow_fake_fallback=True for offline smoke runs."
            ) from exc

        train_images, train_labels = _fake_dataset_to_arrays(size=10000, seed=seed)
        test_images, test_labels = _fake_dataset_to_arrays(size=2000, seed=seed + 1)
        return {
            "source": "FakeData fallback (offline debug only)",
            "train_images": train_images,
            "train_labels": train_labels,
            "test_images": test_images,
            "test_labels": test_labels,
        }


def _stratified_indices(y: np.ndarray, n_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx_parts: List[np.ndarray] = []
    for label in range(10):
        cls_idx = np.where(y == label)[0]
        if cls_idx.shape[0] < n_per_class:
            raise ValueError(
                f"Class {label} has only {cls_idx.shape[0]} samples, required {n_per_class}."
            )
        rng.shuffle(cls_idx)
        idx_parts.append(cls_idx[:n_per_class])
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return idx


def _stratified_train_val_indices(
    y: np.ndarray, train_per_class: int, val_per_class: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []

    needed = train_per_class + val_per_class
    for label in range(10):
        cls_idx = np.where(y == label)[0]
        if cls_idx.shape[0] < needed:
            raise ValueError(
                f"Class {label} has only {cls_idx.shape[0]} samples, required {needed}."
            )
        rng.shuffle(cls_idx)
        train_parts.append(cls_idx[:train_per_class])
        val_parts.append(cls_idx[train_per_class:needed])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def load_splits(config: ExperimentConfig) -> Dict[str, np.ndarray]:
    set_seed(config.seed)
    arrays = _load_mnist_arrays(
        data_root=config.data_root,
        allow_fake_fallback=config.allow_fake_fallback,
        seed=config.seed,
    )

    train_idx, val_idx = _stratified_train_val_indices(
        arrays["train_labels"],
        train_per_class=config.train_per_class,
        val_per_class=config.val_per_class,
        seed=config.seed,
    )
    test_idx = _stratified_indices(
        arrays["test_labels"],
        n_per_class=config.test_per_class,
        seed=config.seed + 1,
    )

    return {
        "source": arrays["source"],
        "x_train_img": arrays["train_images"][train_idx],
        "y_train": arrays["train_labels"][train_idx],
        "x_val_img": arrays["train_images"][val_idx],
        "y_val": arrays["train_labels"][val_idx],
        "x_test_img": arrays["test_images"][test_idx],
        "y_test": arrays["test_labels"][test_idx],
    }


def class_count(y: np.ndarray) -> np.ndarray:
    return np.bincount(y.astype(np.int64), minlength=10)


def plot_sample_grid(images: np.ndarray, labels: np.ndarray, title: str, n: int = 20) -> None:
    n = min(n, images.shape[0])
    cols = 10
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(1.5 * cols, 1.6 * rows))
    for idx in range(n):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(images[idx], cmap="gray")
        plt.title(f"y={int(labels[idx])}", fontsize=9)
        plt.axis("off")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, title: str
) -> None:
    labels = np.arange(10)
    w = 0.25
    c_train = class_count(y_train)
    c_val = class_count(y_val)
    c_test = class_count(y_test)

    plt.figure(figsize=(10, 5))
    plt.bar(labels - w, c_train, width=w, label="Train")
    plt.bar(labels, c_val, width=w, label="Validation")
    plt.bar(labels + w, c_test, width=w, label="Test")
    plt.xticks(labels)
    plt.xlabel("Digit class")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def binarize_image(image: np.ndarray) -> np.ndarray:
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


def hu_moments_feature(image: np.ndarray) -> np.ndarray:
    binary = binarize_image(image)
    moments = cv.moments(binary.astype(np.uint8))
    hu = cv.HuMoments(moments).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu.astype(np.float64)


def squared_norm(n: int, N: int) -> float:
    return math.factorial(N + n) / ((2 * n + 1) * math.factorial(N - n - 1))


def pochhammer_symbol(a: float, k: int) -> float:
    result = 1.0
    for i in range(k):
        result *= a + i
    return result


def tchebichef_polynomial(x: int, n: int, N: int) -> float:
    t1 = pochhammer_symbol(1 - N, n)
    t2 = math.sqrt(squared_norm(n, N))

    series = 0.0
    for k in range(n + 1):
        t3 = pochhammer_symbol(-n, k)
        t4 = pochhammer_symbol(-x, k)
        t5 = pochhammer_symbol(n + 1, k)
        t6 = float(math.factorial(k) ** 2)
        t7 = pochhammer_symbol(1 - N, k)
        series += (t3 * t4 * t5) / (t6 * t7)
    return (t1 / t2) * series


def build_tchebichef_basis(size: int, order: int) -> np.ndarray:
    basis = np.zeros((order + 1, size), dtype=np.float64)
    for n in range(order + 1):
        for x in range(size):
            basis[n, x] = tchebichef_polynomial(x, n, size)
    return basis


def tchebichef_feature(image: np.ndarray, basis: np.ndarray, order: int) -> np.ndarray:
    img = image.astype(np.float64) / 255.0
    moments: List[float] = []
    for n in range(order + 1):
        for m in range(order + 1):
            if n + m <= order:
                value = basis[n].dot(img).dot(basis[m])
                moments.append(float(value))
    return np.asarray(moments, dtype=np.float64)


def extract_features(images: np.ndarray, kind: str, tchebichef_order: int = 3) -> np.ndarray:
    if kind == "raw":
        return images.reshape(images.shape[0], -1).astype(np.float64) / 255.0

    if kind == "hu":
        return np.vstack([hu_moments_feature(img) for img in images])

    if kind == "tchebichef":
        if images.shape[1] != images.shape[2]:
            raise ValueError("Tchebichef features require square images.")
        basis = build_tchebichef_basis(images.shape[1], tchebichef_order)
        return np.vstack([tchebichef_feature(img, basis, tchebichef_order) for img in images])

    raise ValueError(f"Unknown feature kind: {kind}")


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True)
        self.std_ = np.where(self.std_ < 1e-12, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fit before transform.")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


@torch.no_grad()
def knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k: int,
    chunk_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = get_device()

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    x_query_t = torch.tensor(x_query, dtype=torch.float32, device=device)

    pred_chunks: List[np.ndarray] = []
    for start in range(0, x_query_t.shape[0], chunk_size):
        end = min(start + chunk_size, x_query_t.shape[0])
        x_chunk = x_query_t[start:end]
        distances = torch.cdist(x_chunk, x_train_t, p=2)
        nn_idx = torch.topk(distances, k=k, largest=False).indices
        nn_labels = y_train_t[nn_idx]
        pred = torch.mode(nn_labels, dim=1).values
        pred_chunks.append(pred.cpu().numpy())

    return np.concatenate(pred_chunks, axis=0)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    return cm


def classification_report_from_cm(cm: np.ndarray) -> pd.DataFrame:
    rows = []
    for cls in range(cm.shape[0]):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        rows.append(
            {
                "class": cls,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(cm[cls, :].sum()),
            }
        )
    report = pd.DataFrame(rows)
    report.loc[len(report)] = {
        "class": "macro_avg",
        "precision": report["precision"].mean(),
        "recall": report["recall"].mean(),
        "f1_score": report["f1_score"].mean(),
        "support": int(report["support"].sum()),
    }
    return report


def evaluate_knn_grid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k_values: Iterable[int],
    chunk_size: int,
    device: torch.device | None = None,
) -> Dict[str, object]:
    if device is None:
        device = get_device()

    grid_rows = []
    best_k = None
    best_val_acc = -1.0

    for k in k_values:
        val_pred = knn_predict(
            x_train=x_train,
            y_train=y_train,
            x_query=x_val,
            k=k,
            chunk_size=chunk_size,
            device=device,
        )
        val_acc = accuracy(y_val, val_pred)
        grid_rows.append({"k": int(k), "val_accuracy": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = int(k)

    test_pred = knn_predict(
        x_train=x_train,
        y_train=y_train,
        x_query=x_test,
        k=best_k,
        chunk_size=chunk_size,
        device=device,
    )
    test_acc = accuracy(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred, n_classes=10)

    return {
        "best_k": best_k,
        "val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "test_pred": test_pred,
        "confusion_matrix": cm,
        "grid": pd.DataFrame(grid_rows),
    }


class DigitMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, int] = (256, 128),
        dropout: float = 0.2,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * batch_x.shape[0]
        pred = logits.argmax(dim=1)
        total_correct += int((pred == batch_y).sum().item())
        total_seen += int(batch_x.shape[0])

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = total_correct / max(total_seen, 1)
    return avg_loss, avg_acc


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: ExperimentConfig,
    device: torch.device | None = None,
) -> Tuple[DigitMLP, pd.DataFrame]:
    if device is None:
        device = get_device()

    model = DigitMLP(
        input_dim=x_train.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history_rows = []
    best_state: Dict[str, torch.Tensor] | None = None
    best_val_acc = -1.0
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = _run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
        )

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history_df = pd.DataFrame(history_rows)
    return model, history_df


@torch.no_grad()
def predict_mlp(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = get_device()
    model.eval()

    x_t = torch.tensor(x, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t), batch_size=batch_size, shuffle=False)

    preds: List[np.ndarray] = []
    for (batch_x,) in loader:
        logits = model(batch_x.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds, axis=0)


def evaluate_mlp(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device | None = None,
) -> Dict[str, object]:
    y_pred = predict_mlp(model=model, x=x_test, device=device)
    cm = confusion_matrix(y_test, y_pred, n_classes=10)
    return {
        "test_accuracy": accuracy(y_test, y_pred),
        "test_pred": y_pred,
        "confusion_matrix": cm,
    }


def plot_knn_grid(grid_df: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(grid_df["k"], grid_df["val_accuracy"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Validation accuracy")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_learning_curves(history_df: pd.DataFrame, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="Train")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.colorbar()

    threshold = cm.max() * 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame, title: str) -> None:
    x = np.arange(results_df.shape[0])
    plt.figure(figsize=(12, 5))
    bars = plt.bar(x, results_df["test_accuracy"], color="#2c7fb8")
    plt.xticks(x, results_df["model"], rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Test accuracy")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)

    for rect, value in zip(bars, results_df["test_accuracy"]):
        plt.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()
