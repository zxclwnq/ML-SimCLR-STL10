import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import umap
from sklearn.metrics import confusion_matrix
from tbparse import SummaryReader
from tqdm import tqdm
from torchvision.transforms import v2

from src.model import SimCLR
from src.dataset import get_stl10_dataloader
from src.utils import load_config
from src.eval import extract_features, get_linear_eval_model


# ==========================================
# 1. TENSORBOARD SCALARS UTILS
# ==========================================
DEFAULT_TAGS: Sequence[str] = ("Train/Loss", "Train/LR")


def _is_event_file(path: str) -> bool:
    """
    Checks if the given path is a TensorBoard event file based on its name.

    Args:
        path (str): The file path to check.

    Returns:
        bool: True if the file is an event file, False otherwise.
    """
    base = os.path.basename(path)
    return base.startswith("events.out.tfevents")


def _find_event_files(logdir: str) -> List[str]:
    """
    Finds all TensorBoard event files in the given directory recursively.

    Args:
        logdir (str): The directory to search for event files.

    Returns:
        List[str]: A list of paths to found event files.
    """
    if os.path.isfile(logdir):
        return [logdir] if _is_event_file(logdir) else []

    event_files: List[str] = []
    for root, _, files in os.walk(logdir):
        for name in files:
            if name.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, name))
    return event_files


def _pick_latest_event_file(paths: Sequence[str]) -> str:
    """
    Picks the most recently modified event file from a list of paths.

    Args:
        paths (Sequence[str]): A sequence of file paths to event files.

    Returns:
        str: The path to the most recently modified event file.
    """
    if not paths:
        raise FileNotFoundError("No TensorBoard event files found.")
    return max(paths, key=lambda p: os.path.getmtime(p))


def _sanitize_tag(tag: str) -> str:
    """
    Sanitizes a TensorBoard tag to create a safe filename by replacing non-alphanumeric characters.

    Args:
        tag (str): The original TensorBoard tag.

    Returns:
        str: A sanitized version of the tag suitable for use as a filename.
    """
    s = tag.strip().lower().replace("/", "_")
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@dataclass
class ScalarSeries:
    """
    Container for a single scalar series extracted from TensorBoard logs.
    """

    tag: str
    steps: List[int]
    values: List[float]


def _load_scalars_from_logdir(
    logdir: str, tags: Optional[Sequence[str]] = None
) -> Tuple[List[ScalarSeries], List[str]]:
    """
    Loads scalar series from TensorBoard logs in the specified directory.

    Args:
        logdir (str): The directory containing TensorBoard event files.
        tags (Optional[Sequence[str]]): An optional sequence of tags to filter. If None, all tags will be loaded.

    Returns:
        Tuple[List[ScalarSeries], List[str]]: A tuple containing a list of ScalarSeries objects and a list of available tags in the logs.
    """
    reader = SummaryReader(log_path=logdir, pivot=False)
    df = reader.scalars

    if df is None or df.empty:
        return [], []

    if "tag" not in df.columns or "step" not in df.columns or "value" not in df.columns:
        raise RuntimeError(
            f"Unexpected tbparse scalar schema. Columns={sorted(map(str, df.columns))}"
        )

    available = sorted({str(t) for t in df["tag"].dropna().unique().tolist()})
    wanted: Iterable[str] = available if not tags else tags

    out: List[ScalarSeries] = []
    for tag in wanted:
        sub = df[df["tag"] == tag]
        if sub.empty:
            continue

        sub = sub.sort_values("step")
        sub = sub.groupby("step", as_index=False).last()

        steps = [int(s) for s in sub["step"].tolist()]
        values = [float(v) for v in sub["value"].tolist()]
        out.append(ScalarSeries(tag=str(tag), steps=steps, values=values))

    return out, available


def _plot_series(
    series: ScalarSeries,
    out_path: str,
    width: float = 10.5,
    height: float = 4.2,
    dpi: int = 260,
) -> None:
    """
    Plots a single scalar series and saves it as a PNG image.

    Args:
        series (ScalarSeries): The scalar series to plot.
        out_path (str): The file path where the PNG image will be saved.
        width (float): The width of the output image in inches.
        height (float): The height of the output image in inches.
        dpi (int): The resolution of the output image in dots per inch.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not series.steps:
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        ax.set_title(f"{series.tag} (empty)")
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return

    df = pd.DataFrame({"step": series.steps, "value": series.values}).sort_values(
        "step"
    )
    df = df.groupby("step", as_index=False).last()

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.plot(
        df["step"],
        df["value"],
        linewidth=2.2,
        antialiased=True,
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    ax.set_title(series.tag, fontsize=14, pad=10)
    ax.set_xlabel("step", fontsize=11)
    ax.set_ylabel("value", fontsize=11)

    ax.grid(True, which="major", alpha=0.28, linewidth=0.8)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def export_scalars(
    logdir: str,
    outdir: str,
    tags: Sequence[str] = DEFAULT_TAGS,
    event_path: Optional[str] = None,
):
    """
    Exports scalar series from TensorBoard logs to PNG images.

    Args:
        logdir (str): The directory containing TensorBoard event files.
        outdir (str): The directory where the PNG images will be saved.
        tags (Sequence[str]): A sequence of tags to filter which scalar series to export. If empty, all tags will be exported.
        event_path (Optional[str]): An optional specific path to
    """
    if event_path is not None:
        if not os.path.isfile(event_path):
            raise FileNotFoundError(f"Event file not found: {event_path}")
        target_path = event_path
    else:
        event_files = _find_event_files(logdir)
        target_path = _pick_latest_event_file(event_files)

    run_dir = os.path.dirname(os.path.abspath(target_path))
    series_list, available = _load_scalars_from_logdir(run_dir, tags=tags)

    if not series_list:
        raise RuntimeError(
            f"No requested scalar tags were found. Requested={tags}. Available={available}"
        )

    for s in series_list:
        filename = f"{_sanitize_tag(s.tag)}.png"
        out_path = os.path.join(outdir, filename)
        _plot_series(s, out_path)
        print(f"[OK] Wrote {out_path}")


# ==========================================
# 2. UMAP VISUALIZATION UTILS
# ==========================================
def plot_umap(
    config_path: str, checkpoint_path: str, out_path: str = "assets/simclr_umap.png"
):
    """
    Extracts features from the SimCLR model using the test split of the STL10 dataset, applies UMAP for dimensionality reduction, and visualizes the results in a 2D scatter plot colored by class labels.

    Args:
        config_path (str): Path to the YAML configuration file containing dataset and model parameters.
        checkpoint_path (str): Path to the SimCLR model checkpoint file (.pth) to load the model weights from.
        out_path (str): Path where the resulting UMAP plot will be
    """
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimCLR(out_dim=config["model"]["out_dim"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
        ]
    )

    dataloader = get_stl10_dataloader(
        root_dir=config["dataset"]["root_dir"],
        split="test",
        transform=transform,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    features, labels = extract_features(model, dataloader, device)
    features = features.numpy()
    labels = labels.numpy()

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=42
    )
    embedding = reducer.fit_transform(features)

    classes = [
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    ]
    df = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "label": [classes[i] for i in labels],
        }
    )

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        palette="Spectral",
        s=15,
        alpha=0.4,
        legend=True,
    )

    for label in df["label"].unique():
        centroid = df[df["label"] == label][["x", "y"]].mean()
        plt.text(
            centroid.x,
            centroid.y,
            label,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="black",
                boxstyle="round,pad=0.3",
            ),
        )

    plt.title("UMAP with Class Centroids")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved UMAP plot to {out_path}")


# ==========================================
# 3. CONFUSION MATRIX UTILS
# ==========================================
@torch.no_grad()
def get_all_preds(model, loader, device):
    """
    Gets all predictions from a trained model on a given data loader.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): The device to run the model on.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of true labels and predicted labels.
    """
    all_preds = []
    all_labels = []
    model.eval()

    for images, labels in tqdm(loader, desc="Calculating predictions"):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_cm(
    config_path: str,
    simclr_ckpt: str,
    linear_ckpt: str,
    out_path: str = "assets/confusion_matrix.png",
):
    """
    Plots and saves a confusion matrix in PNG format.

    Args:
        config_path (str): Path to the YAML configuration file containing dataset and model parameters.
        simclr_ckpt (str): Path to the SimCLR model checkpoint file (.pth) to load the model weights from.
        linear_ckpt (str): Path to the linear evaluation model checkpoint file (.pth) to load the model weights from.
        out_path (str): Path where
    """
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = SimCLR(out_dim=config["model"]["out_dim"]).to(device)
    ckpt = torch.load(simclr_ckpt, map_location=device)
    base_model.load_state_dict(ckpt["state_dict"])

    image_size = config["dataset"]["image_size"]
    model = get_linear_eval_model(base_model, image_size=image_size, num_classes=10).to(
        device
    )

    linear_state = torch.load(linear_ckpt, map_location=device)
    model.load_state_dict(linear_state)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4467, 0.4398, 0.4066], std=[0.2603, 0.2566, 0.2713]),
        ]
    )

    loader = get_stl10_dataloader(
        root_dir=config["dataset"]["root_dir"],
        split="test",
        transform=transform,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    y_true, y_pred = get_all_preds(model, loader, device)
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    classes = [
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    ]
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm_perc,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )

    plt.title("Confusion Matrix\nValues in %", fontsize=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Matrix saved to {out_path}")
