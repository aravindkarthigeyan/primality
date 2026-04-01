#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from black_box_prime_nn import UINT64_MAX, generate_split


BIT_SHIFTS = np.arange(63, -1, -1, dtype=np.uint64)


@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def safe_f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2.0 * precision * recall / denominator


def primes_up_to(limit: int) -> list[int]:
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for value in range(2, int(limit**0.5) + 1):
        if sieve[value]:
            step = value
            start = value * value
            sieve[start : limit + 1 : step] = [False] * len(range(start, limit + 1, step))
    return [value for value, prime in enumerate(sieve) if prime]


def parse_hidden_sizes(value: str) -> list[int]:
    try:
        values = [int(chunk.strip()) for chunk in value.split(",") if chunk.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("projection sizes must be integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("projection sizes must be positive integers")
    return values


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available on this machine.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available on this machine.")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_bits(numbers: np.ndarray) -> torch.Tensor:
    bits = ((numbers[:, None] >> BIT_SHIFTS) & np.uint64(1)).astype(np.int64)
    return torch.from_numpy(bits)


def compute_prefix_remainder_targets(numbers: np.ndarray, moduli: Iterable[int]) -> list[torch.Tensor]:
    bits = ((numbers[:, None] >> BIT_SHIFTS) & np.uint64(1)).astype(np.int64)
    targets = []
    for modulus in moduli:
        remainder = np.zeros(len(numbers), dtype=np.int64)
        prefix_targets = np.zeros((len(numbers), 64), dtype=np.int64)
        for bit_index in range(64):
            remainder = ((remainder * 2) + bits[:, bit_index]) % modulus
            prefix_targets[:, bit_index] = remainder
        targets.append(torch.from_numpy(prefix_targets))
    return targets


def compute_odd_mask(numbers: np.ndarray) -> np.ndarray:
    return (numbers % np.uint64(2)) == np.uint64(1)


def compute_hard_mask(numbers: np.ndarray, moduli: Iterable[int]) -> np.ndarray:
    hard = np.ones(len(numbers), dtype=bool)
    python_numbers = [int(number) for number in numbers.tolist()]
    for index, number in enumerate(python_numbers):
        for modulus in moduli:
            if number != modulus and number % modulus == 0:
                hard[index] = False
                break
    return hard


def exact_small_prime_prediction(number: int, moduli: Iterable[int]) -> bool | None:
    if number < 2:
        return False
    for modulus in moduli:
        if number == modulus:
            return True
        if number % modulus == 0:
            return False
    return None


class BitGRUModel(nn.Module):
    def __init__(
        self,
        *,
        moduli: list[int],
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        projection_sizes: list[int],
    ) -> None:
        super().__init__()
        self.moduli = moduli
        self.embedding = nn.Embedding(2, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        representation_size = hidden_size * 3
        layers: list[nn.Module] = [nn.LayerNorm(representation_size)]
        current_size = representation_size
        for projection_size in projection_sizes:
            layers.append(nn.Linear(current_size, projection_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_size = projection_size
        self.projection = nn.Sequential(*layers)
        self.remainder_heads = nn.ModuleList([nn.Linear(hidden_size, modulus) for modulus in moduli])
        self.prime_head = nn.Sequential(
            nn.LayerNorm(current_size + len(moduli)),
            nn.Linear(current_size + len(moduli), current_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(current_size, 1),
        )

    def forward(self, bits: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        embeddings = self.embedding(bits)
        outputs, hidden = self.gru(embeddings)
        last_hidden = hidden[-1]
        mean_pool = outputs.mean(dim=1)
        max_pool = outputs.max(dim=1).values
        representation = self.projection(torch.cat([last_hidden, mean_pool, max_pool], dim=1))
        remainder_logits = [head(outputs) for head in self.remainder_heads]
        divisibility_features = [torch.softmax(logits[:, -1, :], dim=1)[:, :1] for logits in remainder_logits]
        classifier_input = torch.cat([representation, *divisibility_features], dim=1)
        prime_logits = self.prime_head(classifier_input).squeeze(-1)
        return prime_logits, remainder_logits


def build_model(args: argparse.Namespace, moduli: list[int]) -> BitGRUModel:
    return BitGRUModel(
        moduli=moduli,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        projection_sizes=args.projection_sizes,
    )


def make_loader(
    numbers: np.ndarray,
    labels: np.ndarray,
    moduli: list[int],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, ...]]:
    bits = encode_bits(numbers)
    labels_tensor = torch.from_numpy(labels.astype(np.float32))
    remainder_targets = compute_prefix_remainder_targets(numbers, moduli)
    dataset = TensorDataset(bits, labels_tensor, *remainder_targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_metrics(probabilities: np.ndarray, labels: np.ndarray, *, threshold: float) -> Metrics:
    predictions = probabilities >= threshold
    positives = labels == 1.0
    negatives = ~positives

    tp = int(np.sum(predictions & positives))
    tn = int(np.sum((~predictions) & negatives))
    fp = int(np.sum(predictions & negatives))
    fn = int(np.sum((~predictions) & positives))

    accuracy = float((tp + tn) / len(labels))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = safe_f1(precision, recall)

    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    loss = float(-np.mean((labels * np.log(clipped)) + ((1.0 - labels) * np.log(1.0 - clipped))))
    return Metrics(loss=loss, accuracy=accuracy, precision=precision, recall=recall, f1=f1)


def compute_metrics_from_predictions(predictions: np.ndarray, labels: np.ndarray) -> Metrics:
    positives = labels == 1.0
    negatives = ~positives

    tp = int(np.sum(predictions & positives))
    tn = int(np.sum((~predictions) & negatives))
    fp = int(np.sum(predictions & negatives))
    fn = int(np.sum((~predictions) & positives))

    accuracy = float((tp + tn) / len(labels))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = safe_f1(precision, recall)
    return Metrics(loss=0.0, accuracy=accuracy, precision=precision, recall=recall, f1=f1)


def hybrid_predictions(probabilities: np.ndarray, numbers: np.ndarray, moduli: Iterable[int], *, threshold: float) -> np.ndarray:
    predictions = probabilities >= threshold
    python_numbers = [int(number) for number in numbers.tolist()]
    for index, number in enumerate(python_numbers):
        exact = exact_small_prime_prediction(number, moduli)
        if exact is not None:
            predictions[index] = exact
    return predictions


def best_threshold(probabilities: np.ndarray, labels: np.ndarray, odd_mask: np.ndarray) -> tuple[float, float]:
    best_value = 0.5
    best_score = -1.0

    for step in range(10, 91):
        threshold = step / 100.0
        overall = compute_metrics(probabilities, labels, threshold=threshold)
        odd = compute_metrics(probabilities[odd_mask], labels[odd_mask], threshold=threshold)
        score = (0.4 * overall.f1) + (0.6 * odd.f1)
        if score > best_score:
            best_score = score
            best_value = threshold

    return best_value, best_score


@torch.no_grad()
def evaluate(
    model: BitGRUModel,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        bits = batch[0].to(device)
        labels = batch[1].cpu().numpy()
        prime_logits, _ = model(bits)
        probabilities = torch.sigmoid(prime_logits).detach().cpu().numpy()
        all_probabilities.append(probabilities)
        all_labels.append(labels)

    return np.concatenate(all_probabilities), np.concatenate(all_labels)


@torch.no_grad()
def evaluate_auxiliary_accuracy(
    model: BitGRUModel,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        bits = batch[0].to(device)
        targets = [tensor.to(device) for tensor in batch[2:]]
        _, remainder_logits = model(bits)
        for logits, target in zip(remainder_logits, targets, strict=True):
            predicted = logits[:, -1, :].argmax(dim=1)
            expected = target[:, -1]
            correct += int((predicted == expected).sum().item())
            total += int(expected.numel())

    return (correct / total) if total else 0.0


def train_epoch(
    model: BitGRUModel,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    *,
    prime_weight: float,
    aux_weight: float,
    grad_clip: float,
) -> float:
    model.train()
    prime_loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    batch_count = 0
    position_weights = torch.linspace(0.15, 1.0, steps=64, device=device).view(1, 64)

    for batch in loader:
        bits = batch[0].to(device)
        labels = batch[1].to(device)
        remainder_targets = [tensor.to(device) for tensor in batch[2:]]

        optimizer.zero_grad(set_to_none=True)
        prime_logits, remainder_logits = model(bits)
        prime_loss = prime_loss_fn(prime_logits, labels)

        aux_losses = []
        for logits, targets in zip(remainder_logits, remainder_targets, strict=True):
            per_position = nn.functional.cross_entropy(
                logits.transpose(1, 2),
                targets,
                reduction="none",
            )
            weighted = (per_position * position_weights).mean()
            aux_losses.append(weighted)
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else torch.zeros((), device=device)

        loss = (prime_weight * prime_loss) + (aux_weight * aux_loss)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        batch_count += 1

    return total_loss / max(batch_count, 1)


def save_checkpoint(
    path: Path,
    model: BitGRUModel,
    *,
    args: argparse.Namespace,
    moduli: list[int],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "moduli": moduli,
            "threshold": threshold,
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "projection_sizes": args.projection_sizes,
            "seed": args.seed,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[BitGRUModel, float, dict[str, object]]:
    payload = torch.load(path, map_location=device)
    model = BitGRUModel(
        moduli=[int(value) for value in payload["moduli"]],
        embedding_dim=int(payload["embedding_dim"]),
        hidden_size=int(payload["hidden_size"]),
        num_layers=int(payload["num_layers"]),
        dropout=float(payload["dropout"]),
        projection_sizes=[int(value) for value in payload["projection_sizes"]],
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, float(payload["threshold"]), payload


def save_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = choose_device(args.device)
    moduli = primes_up_to(args.max_aux_prime)

    train_numbers, train_labels = generate_split(args.train_size, args.seed)
    val_numbers, val_labels = generate_split(args.val_size, args.seed + 1)
    test_numbers, test_labels = generate_split(args.test_size, args.seed + 2)

    train_loader = make_loader(train_numbers, train_labels, moduli, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_numbers, val_labels, moduli, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = make_loader(test_numbers, test_labels, moduli, batch_size=args.eval_batch_size, shuffle=False)

    val_odd_mask = compute_odd_mask(val_numbers)
    test_odd_mask = compute_odd_mask(test_numbers)
    test_hard_mask = compute_hard_mask(test_numbers, moduli)

    model = build_model(args, moduli).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(
        f"Training torch raw-bit sequence model on {device.type} with {len(moduli)} auxiliary moduli "
        f"(<= {args.max_aux_prime}), train={args.train_size}, val={args.val_size}, test={args.test_size}"
    )

    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    best_epoch = 0
    best_threshold_value = 0.5
    best_score = -1.0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        prime_weight = 0.0 if epoch <= args.pretrain_epochs else 1.0
        train_loss = train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            prime_weight=prime_weight,
            aux_weight=args.aux_weight,
            grad_clip=args.grad_clip,
        )

        train_probabilities, train_eval_labels = evaluate(model, train_loader, device)
        val_probabilities, val_eval_labels = evaluate(model, val_loader, device)
        train_aux_accuracy = evaluate_auxiliary_accuracy(model, train_loader, device)
        val_aux_accuracy = evaluate_auxiliary_accuracy(model, val_loader, device)
        threshold, score = best_threshold(val_probabilities, val_eval_labels, val_odd_mask)

        train_metrics = compute_metrics(train_probabilities, train_eval_labels, threshold=threshold)
        val_metrics = compute_metrics(val_probabilities, val_eval_labels, threshold=threshold)
        val_odd_metrics = compute_metrics(val_probabilities[val_odd_mask], val_eval_labels[val_odd_mask], threshold=threshold)

        print(
            f"Epoch {epoch:02d} train_loss={train_loss:.4f} threshold={threshold:.2f} prime_weight={prime_weight:.1f} "
            f"| train acc={train_metrics.accuracy:.3f} f1={train_metrics.f1:.3f} "
            f"| val acc={val_metrics.accuracy:.3f} f1={val_metrics.f1:.3f} "
            f"| val odd acc={val_odd_metrics.accuracy:.3f} f1={val_odd_metrics.f1:.3f} "
            f"| aux train={train_aux_accuracy:.3f} val={val_aux_accuracy:.3f}"
        )

        if prime_weight == 0.0:
            continue

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_threshold_value = threshold
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Stopping early after epoch {epoch} (best epoch was {best_epoch}).")
                break

    model.load_state_dict(best_state)
    model.to(device)

    train_probabilities, train_eval_labels = evaluate(model, train_loader, device)
    val_probabilities, val_eval_labels = evaluate(model, val_loader, device)
    test_probabilities, test_eval_labels = evaluate(model, test_loader, device)

    train_metrics = compute_metrics(train_probabilities, train_eval_labels, threshold=best_threshold_value)
    val_metrics = compute_metrics(val_probabilities, val_eval_labels, threshold=best_threshold_value)
    test_metrics = compute_metrics(test_probabilities, test_eval_labels, threshold=best_threshold_value)
    test_odd_metrics = compute_metrics(test_probabilities[test_odd_mask], test_eval_labels[test_odd_mask], threshold=best_threshold_value)
    test_hard_metrics = compute_metrics(test_probabilities[test_hard_mask], test_eval_labels[test_hard_mask], threshold=best_threshold_value)
    val_hybrid_predictions = hybrid_predictions(val_probabilities, val_numbers, moduli, threshold=best_threshold_value)
    test_hybrid_predictions = hybrid_predictions(test_probabilities, test_numbers, moduli, threshold=best_threshold_value)
    val_hybrid_metrics = compute_metrics_from_predictions(val_hybrid_predictions, val_eval_labels)
    test_hybrid_metrics = compute_metrics_from_predictions(test_hybrid_predictions, test_eval_labels)
    test_hybrid_odd_metrics = compute_metrics_from_predictions(test_hybrid_predictions[test_odd_mask], test_eval_labels[test_odd_mask])

    print("Best checkpoint")
    print(f"train     acc={train_metrics.accuracy:.3f} precision={train_metrics.precision:.3f} recall={train_metrics.recall:.3f} f1={train_metrics.f1:.3f}")
    print(f"val       acc={val_metrics.accuracy:.3f} precision={val_metrics.precision:.3f} recall={val_metrics.recall:.3f} f1={val_metrics.f1:.3f}")
    print(f"test      acc={test_metrics.accuracy:.3f} precision={test_metrics.precision:.3f} recall={test_metrics.recall:.3f} f1={test_metrics.f1:.3f}")
    print(f"test odd  acc={test_odd_metrics.accuracy:.3f} precision={test_odd_metrics.precision:.3f} recall={test_odd_metrics.recall:.3f} f1={test_odd_metrics.f1:.3f}")
    print(f"test hard acc={test_hard_metrics.accuracy:.3f} precision={test_hard_metrics.precision:.3f} recall={test_hard_metrics.recall:.3f} f1={test_hard_metrics.f1:.3f}")
    print(f"hybrid val  acc={val_hybrid_metrics.accuracy:.3f} precision={val_hybrid_metrics.precision:.3f} recall={val_hybrid_metrics.recall:.3f} f1={val_hybrid_metrics.f1:.3f}")
    print(f"hybrid test acc={test_hybrid_metrics.accuracy:.3f} precision={test_hybrid_metrics.precision:.3f} recall={test_hybrid_metrics.recall:.3f} f1={test_hybrid_metrics.f1:.3f}")
    print(f"hybrid odd  acc={test_hybrid_odd_metrics.accuracy:.3f} precision={test_hybrid_odd_metrics.precision:.3f} recall={test_hybrid_odd_metrics.recall:.3f} f1={test_hybrid_odd_metrics.f1:.3f}")

    metrics_payload = {
        "device": device.type,
        "best_epoch": best_epoch,
        "threshold": best_threshold_value,
        "selection_score": best_score,
        "moduli": moduli,
        "embedding_dim": args.embedding_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "projection_sizes": args.projection_sizes,
        "aux_weight": args.aux_weight,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "train": asdict(train_metrics),
        "validation": asdict(val_metrics),
        "test": asdict(test_metrics),
        "test_odd_only": asdict(test_odd_metrics),
        "test_no_small_factor": asdict(test_hard_metrics),
        "test_no_small_factor_count": int(np.sum(test_hard_mask)),
        "validation_hybrid": asdict(val_hybrid_metrics),
        "test_hybrid": asdict(test_hybrid_metrics),
        "test_hybrid_odd_only": asdict(test_hybrid_odd_metrics),
    }

    save_checkpoint(Path(args.model_path), model, args=args, moduli=moduli, threshold=best_threshold_value)
    save_metrics(Path(args.metrics_path), metrics_payload)
    print(f"Saved model to {args.model_path}")
    print(f"Saved metrics to {args.metrics_path}")

    if args.preview:
        print("Sample predictions")
        predict_numbers(model, args.preview, threshold=best_threshold_value, device=device, moduli=moduli, hybrid=True)


@torch.no_grad()
def predict_numbers(
    model: BitGRUModel,
    numbers: list[int],
    *,
    threshold: float,
    device: torch.device,
    moduli: list[int],
    hybrid: bool,
) -> None:
    for number in numbers:
        if number < 0 or number > UINT64_MAX:
            raise ValueError(f"{number} is outside the uint64 range")

    number_array = np.array(numbers, dtype=np.uint64)
    bits = encode_bits(number_array).to(device)
    probabilities = torch.sigmoid(model(bits)[0]).cpu().numpy()
    for number, probability in zip(numbers, probabilities, strict=True):
        mode = "nn"
        prediction = probability >= threshold
        if hybrid:
            exact = exact_small_prime_prediction(number, moduli)
            if exact is not None:
                prediction = exact
                mode = "exact"
        label = "prime" if prediction else "composite"
        print(f"{number:>20} probability={float(probability):.4f} guessed={label} mode={mode}")


def predict(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model, threshold, payload = load_checkpoint(Path(args.model_path), device)
    print(
        f"Loaded torch raw-bit model on {device.type} "
        f"(hidden={payload['hidden_size']}, layers={payload['num_layers']}, threshold={threshold:.2f})"
    )
    predict_numbers(model, args.numbers, threshold=threshold, device=device, moduli=[int(v) for v in payload["moduli"]], hybrid=not args.pure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch raw-bit sequence model that guesses primality for uint64 inputs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a PyTorch raw-bit sequence model")
    train_parser.add_argument("--train-size", type=int, default=24000, help="number of training examples")
    train_parser.add_argument("--val-size", type=int, default=6000, help="number of validation examples")
    train_parser.add_argument("--test-size", type=int, default=6000, help="number of test examples")
    train_parser.add_argument("--epochs", type=int, default=24, help="maximum number of epochs")
    train_parser.add_argument("--patience", type=int, default=6, help="early stopping patience")
    train_parser.add_argument("--batch-size", type=int, default=512, help="training batch size")
    train_parser.add_argument("--eval-batch-size", type=int, default=1024, help="evaluation batch size")
    train_parser.add_argument("--learning-rate", type=float, default=0.001, help="AdamW learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    train_parser.add_argument("--aux-weight", type=float, default=0.25, help="weight on remainder-prediction losses")
    train_parser.add_argument("--pretrain-epochs", type=int, default=4, help="number of auxiliary-only warmup epochs before prime classification turns on")
    train_parser.add_argument("--grad-clip", type=float, default=1.0, help="gradient clipping norm")
    train_parser.add_argument("--embedding-dim", type=int, default=48, help="bit embedding size")
    train_parser.add_argument("--hidden-size", type=int, default=256, help="GRU hidden size")
    train_parser.add_argument("--num-layers", type=int, default=3, help="number of GRU layers")
    train_parser.add_argument("--dropout", type=float, default=0.15, help="dropout applied in the GRU stack and projection head")
    train_parser.add_argument(
        "--projection-sizes",
        type=parse_hidden_sizes,
        default=[384, 192],
        help="comma-separated sizes for the projection MLP after sequence pooling",
    )
    train_parser.add_argument("--max-aux-prime", type=int, default=127, help="largest prime modulus used for auxiliary remainder targets")
    train_parser.add_argument("--seed", type=int, default=7, help="random seed")
    train_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="training device",
    )
    train_parser.add_argument(
        "--model-path",
        default="artifacts/torch_black_box_u64_model.pt",
        help="where to save the trained checkpoint",
    )
    train_parser.add_argument(
        "--metrics-path",
        default="artifacts/torch_black_box_u64_metrics.json",
        help="where to save metrics",
    )
    train_parser.add_argument(
        "--preview",
        nargs="*",
        type=int,
        default=[2, 3, 4, 5, 97, 221, 997, 10007, 18446744073709551557],
        help="numbers to score after training",
    )
    train_parser.set_defaults(handler=train)

    predict_parser = subparsers.add_parser("predict", help="predict with a saved torch checkpoint")
    predict_parser.add_argument("numbers", nargs="+", type=int, help="numbers to score")
    predict_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="device used for inference",
    )
    predict_parser.add_argument(
        "--model-path",
        default="artifacts/torch_black_box_u64_model.pt",
        help="path to the saved checkpoint",
    )
    predict_parser.add_argument(
        "--pure",
        action="store_true",
        help="use the neural network alone instead of the exact-small-prime hybrid front-end",
    )
    predict_parser.set_defaults(handler=predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
