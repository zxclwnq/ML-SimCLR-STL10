import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, Any, Tuple, Optional

# Linear Evaluation


class LinearClassifier(nn.Module):
    """
    Linear classifier for evaluating representation learning on a frozen encoder.
    """

    def __init__(
        self, encoder: nn.Module, image_size: int = 96, num_classes: int = 10
    ) -> None:
        """
        Initializes the linear classifier.

        Args:
            encoder (nn.Module): The pretrained encoder model.
            image_size (int): Expected image resolution for finding feature dimensions.
            num_classes (int): Number of target classes.
        """
        super().__init__()
        self.encoder: nn.Module = encoder

        for param in self.encoder.parameters():
            param.requires_grad = False

        # Get the dimension of the feature vector (h) with a dummy forward pass
        device: torch.device = next(encoder.parameters()).device
        dummy_input = torch.zeros(1, 3, image_size, image_size, device=device)

        with torch.no_grad():
            h, _ = self.encoder(dummy_input)
            feature_dim: int = h.shape[1]

        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Classification logits.
        """
        with torch.no_grad():
            features, _ = self.encoder(x)
        return self.fc(features)


def get_linear_eval_model(
    simclr_model: nn.Module, image_size: int = 96, num_classes: int = 10
) -> nn.Module:
    """
    Instantiates and returns the linear evaluation model.

    Args:
        simclr_model (nn.Module): Pretrained SimCLR model.
        image_size (int): Image spatial dimensions.
        num_classes (int): Number of target classes.

    Returns:
        nn.Module: The LinearClassifier instance.
    """
    return LinearClassifier(simclr_model, image_size, num_classes)


def train_linear_eval(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    save_path: str,
) -> float:
    """
    Trains and evaluates the linear classifier on top of the frozen encoder.

    Args:
        model (nn.Module): The linear classifier model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run the training on.
        config (Dict[str, Any]): Configuration dictionary.
        save_path (str): Path to save the best model weights.

    Returns:
        float: The best validation accuracy achieved.
    """
    epochs: int = config["linear_eval"]["epochs"]
    lr: float = config["linear_eval"]["lr"]
    weight_decay: float = config["linear_eval"]["weight_decay"]

    # We only train the fully connected layer
    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)  # type: ignore

    min_lr: float = config["linear_eval"].get("min_lr", 0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    criterion = nn.CrossEntropyLoss()
    best_acc: float = 0.0

    for epoch in range(epochs):
        model.train()
        model.encoder.eval()  # type: ignore
        for images, targets in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"
        ):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct: int = 0
        total: int = 0
        with torch.no_grad():
            for images, targets in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"
            ):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Step the scheduler at the end of the epoch
        scheduler.step()

        val_acc: float = 100.0 * correct / total
        current_lr: float = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}: Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"Linear Evaluation finished. BEST VAL ACC: {best_acc:.2f}%")
    return best_acc


# KNN Evaluation


@torch.no_grad()
def extract_features(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts features and labels using the encoder.

    Args:
        model (nn.Module): The SimCLR model.
        dataloader (DataLoader): DataLoader to extract features from.
        device (torch.device): Compute device.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of concatenated features and corresponding labels.
    """
    model.eval()
    features, labels = [], []
    for images, targets in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        h, _ = model(images)
        h = F.normalize(h, dim=1)
        features.append(h.cpu())
        labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)


def knn_eval(
    simclr_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    k: int = 20,
) -> float:
    """
    Evaluates the representation using a K-Nearest Neighbors classifier.

    Args:
        simclr_model (nn.Module): The pretrained SimCLR model.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Compute device.
        k (int): Number of neighbors for KNN.

    Returns:
        float: KNN accuracy percentage.
    """
    print("Extracting features for train dataset...")
    train_features, train_labels = extract_features(simclr_model, train_loader, device)

    print("Extracting features for test dataset...")
    test_features, test_labels = extract_features(simclr_model, val_loader, device)

    print(f"Learning KNN classifier (k={k})...")
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(train_features.numpy(), train_labels.numpy())

    print("KNN Evaluation...")
    accuracy: float = float(
        knn.score(test_features.numpy(), test_labels.numpy()) * 100.0
    )
    print(f"KNN Accuracy: {accuracy:.2f}%")
    return accuracy


def evaluate_model(
    method: str,
    simclr_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    save_path: Optional[str] = None,
) -> float:
    """
    Evaluation of the model according to the specified method.

    Args:
        method (str): Evaluation method: 'linear' or 'knn'.
        simclr_model (nn.Module): Pretrained SimCLR model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for testing/validation data.
        device (torch.device): Device to run evaluation on.
        config (Dict[str, Any]): Dictionary with configuration parameters.
        save_path (Optional[str]): Path to save the evaluation model (required for linear eval).

    Returns:
        float: Computed accuracy.
    """
    if method == "linear":
        if save_path is None:
            raise ValueError("save_path must be provided for linear evaluation.")
        image_size: int = config["dataset"]["image_size"]
        linear_model = get_linear_eval_model(
            simclr_model, image_size=image_size, num_classes=10
        ).to(device)
        return train_linear_eval(
            model=linear_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            save_path=save_path,
        )
    elif method == "knn":
        k: int = config.get("knn_eval", {}).get("k", 5)
        return knn_eval(
            simclr_model=simclr_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            k=k,
        )
    else:
        raise ValueError(f"Unknown evaluation method: {method}. Use 'linear' or 'knn'.")
