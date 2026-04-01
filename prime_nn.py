#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Example:
    number: int
    features: list[float]
    label: int


@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def is_prime(number: int) -> bool:
    if number < 2:
        return False
    if number == 2:
        return True
    if number % 2 == 0:
        return False

    limit = math.isqrt(number)
    divisor = 3
    while divisor <= limit:
        if number % divisor == 0:
            return False
        divisor += 2
    return True


def parse_mod_bases(value: str) -> tuple[int, ...]:
    parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("at least one modulus base is required")
    try:
        bases = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("modulus bases must be integers") from exc

    if any(base <= 1 for base in bases):
        raise argparse.ArgumentTypeError("modulus bases must be greater than 1")
    return bases


def primes_up_to(limit: int) -> tuple[int, ...]:
    if limit < 2:
        return ()

    primes = []
    for number in range(2, limit + 1):
        if is_prime(number):
            primes.append(number)
    return tuple(primes)


def encode_number(number: int, *, max_number: int, bit_width: int, mod_bases: Iterable[int]) -> list[float]:
    binary_features = []
    for bit_index in range(bit_width - 1, -1, -1):
        bit = (number >> bit_index) & 1
        binary_features.append(1.0 if bit else -1.0)

    features = [
        number / max_number,
        math.log1p(number) / math.log1p(max_number),
    ]
    features.extend(binary_features)

    for base in mod_bases:
        residue = number % base
        scaled_residue = 0.0 if base == 1 else ((residue / (base - 1)) * 2.0) - 1.0
        features.append(scaled_residue)
        features.append(1.0 if residue == 0 and number != base else -1.0)

    return features


def feature_names(bit_width: int, mod_bases: Iterable[int]) -> list[str]:
    names = ["normalized_value", "log_scaled_value"]
    names.extend(f"bit_{bit_width - 1 - index}" for index in range(bit_width))
    for base in mod_bases:
        names.append(f"residue_mod_{base}")
        names.append(f"divisible_by_{base}_except_self")
    return names


def build_dataset(max_number: int, mod_bases: tuple[int, ...]) -> tuple[list[Example], int]:
    bit_width = max(2, max_number.bit_length())
    dataset = []
    for number in range(2, max_number + 1):
        dataset.append(
            Example(
                number=number,
                features=encode_number(number, max_number=max_number, bit_width=bit_width, mod_bases=mod_bases),
                label=1 if is_prime(number) else 0,
            )
        )
    return dataset, bit_width


def split_dataset(dataset: list[Example], seed: int) -> tuple[list[Example], list[Example], list[Example]]:
    shuffled = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


class MLP:
    def __init__(self, input_size: int, hidden_size: int, *, seed: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = random.Random(seed)

        limit_input = math.sqrt(6.0 / (input_size + hidden_size))
        self.weight_input_hidden = [
            [rng.uniform(-limit_input, limit_input) for _ in range(hidden_size)]
            for _ in range(input_size)
        ]
        self.bias_hidden = [0.0 for _ in range(hidden_size)]

        limit_hidden = math.sqrt(6.0 / (hidden_size + 1))
        self.weight_hidden_output = [rng.uniform(-limit_hidden, limit_hidden) for _ in range(hidden_size)]
        self.bias_output = 0.0

    def copy_state(self) -> dict[str, object]:
        return {
            "weight_input_hidden": deepcopy(self.weight_input_hidden),
            "bias_hidden": list(self.bias_hidden),
            "weight_hidden_output": list(self.weight_hidden_output),
            "bias_output": self.bias_output,
        }

    def load_state(self, state: dict[str, object]) -> None:
        self.weight_input_hidden = deepcopy(state["weight_input_hidden"])  # type: ignore[index]
        self.bias_hidden = list(state["bias_hidden"])  # type: ignore[index]
        self.weight_hidden_output = list(state["weight_hidden_output"])  # type: ignore[index]
        self.bias_output = float(state["bias_output"])  # type: ignore[arg-type]

    def forward(self, features: list[float]) -> tuple[list[float], float]:
        _, hidden, _, output = self.forward_details(features)
        return hidden, output

    def forward_details(self, features: list[float]) -> tuple[list[float], list[float], float, float]:
        hidden_pre = []
        hidden = []
        for hidden_index in range(self.hidden_size):
            total = self.bias_hidden[hidden_index]
            for input_index, feature in enumerate(features):
                total += feature * self.weight_input_hidden[input_index][hidden_index]
            hidden_pre.append(total)
            hidden.append(math.tanh(total))

        output_total = self.bias_output
        for hidden_index, hidden_value in enumerate(hidden):
            output_total += hidden_value * self.weight_hidden_output[hidden_index]
        output = sigmoid(output_total)
        return hidden_pre, hidden, output_total, output

    def predict_proba(self, features: list[float]) -> float:
        _, probability = self.forward(features)
        return probability

    def train_step(self, features: list[float], label: int, learning_rate: float, positive_weight: float) -> float:
        hidden, prediction = self.forward(features)
        sample_weight = positive_weight if label == 1 else 1.0
        output_delta = (prediction - label) * sample_weight

        hidden_deltas = []
        for hidden_index, hidden_value in enumerate(hidden):
            hidden_delta = (1.0 - (hidden_value * hidden_value)) * self.weight_hidden_output[hidden_index] * output_delta
            hidden_deltas.append(hidden_delta)

        for hidden_index, hidden_value in enumerate(hidden):
            self.weight_hidden_output[hidden_index] -= learning_rate * hidden_value * output_delta
        self.bias_output -= learning_rate * output_delta

        for input_index, feature in enumerate(features):
            row = self.weight_input_hidden[input_index]
            for hidden_index, hidden_delta in enumerate(hidden_deltas):
                row[hidden_index] -= learning_rate * feature * hidden_delta

        for hidden_index, hidden_delta in enumerate(hidden_deltas):
            self.bias_hidden[hidden_index] -= learning_rate * hidden_delta

        clipped_prediction = min(max(prediction, 1e-9), 1.0 - 1e-9)
        return -sample_weight * (
            label * math.log(clipped_prediction) + (1 - label) * math.log(1.0 - clipped_prediction)
        )


def safe_f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2.0 * precision * recall / denominator


def find_best_threshold(examples: list[Example], model: MLP) -> float:
    best_threshold = 0.5
    best_f1 = -1.0

    for step in range(5, 96):
        threshold = step / 100.0
        tp = fp = fn = 0
        for example in examples:
            prediction = 1 if model.predict_proba(example.features) >= threshold else 0
            if prediction == 1 and example.label == 1:
                tp += 1
            elif prediction == 1 and example.label == 0:
                fp += 1
            elif prediction == 0 and example.label == 1:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = safe_f1(precision, recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def evaluate(examples: list[Example], model: MLP, *, threshold: float) -> Metrics:
    tp = tn = fp = fn = 0
    loss_total = 0.0

    for example in examples:
        probability = model.predict_proba(example.features)
        clipped_probability = min(max(probability, 1e-9), 1.0 - 1e-9)
        loss_total += -(
            example.label * math.log(clipped_probability) + (1 - example.label) * math.log(1.0 - clipped_probability)
        )

        prediction = 1 if probability >= threshold else 0
        if prediction == 1 and example.label == 1:
            tp += 1
        elif prediction == 0 and example.label == 0:
            tn += 1
        elif prediction == 1 and example.label == 0:
            fp += 1
        else:
            fn += 1

    total = len(examples)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = safe_f1(precision, recall)
    mean_loss = loss_total / total if total else 0.0

    return Metrics(loss=mean_loss, accuracy=accuracy, precision=precision, recall=recall, f1=f1)


def format_metrics(label: str, metrics: Metrics) -> str:
    return (
        f"{label:<5} loss={metrics.loss:.4f} "
        f"acc={metrics.accuracy:.3f} "
        f"precision={metrics.precision:.3f} "
        f"recall={metrics.recall:.3f} "
        f"f1={metrics.f1:.3f}"
    )


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_model(args: argparse.Namespace) -> None:
    mod_bases = args.mod_bases or primes_up_to(math.isqrt(args.max_number))
    if not mod_bases:
        raise ValueError("no modulus bases available for the selected training range")

    dataset, bit_width = build_dataset(args.max_number, mod_bases)
    train_examples, val_examples, test_examples = split_dataset(dataset, args.seed)

    input_size = len(train_examples[0].features)
    model = MLP(input_size=input_size, hidden_size=args.hidden_size, seed=args.seed)

    positive_count = sum(example.label for example in train_examples)
    negative_count = len(train_examples) - positive_count
    positive_weight = (negative_count / positive_count) if positive_count else 1.0

    best_state = model.copy_state()
    best_threshold = 0.5
    best_val_f1 = -1.0
    best_epoch = 0
    patience_left = args.patience
    rng = random.Random(args.seed)

    print(
        f"Training on {len(train_examples)} examples "
        f"(val={len(val_examples)}, test={len(test_examples)}) "
        f"with {input_size} features and hidden size {args.hidden_size}."
    )
    print(f"Modulus bases: {', '.join(str(base) for base in mod_bases)}")
    print(f"Positive class weight: {positive_weight:.3f}")

    for epoch in range(1, args.epochs + 1):
        shuffled = list(train_examples)
        rng.shuffle(shuffled)
        epoch_loss = 0.0

        for example in shuffled:
            epoch_loss += model.train_step(
                example.features,
                example.label,
                learning_rate=args.learning_rate,
                positive_weight=positive_weight,
            )

        val_threshold = find_best_threshold(val_examples, model)
        train_metrics = evaluate(train_examples, model, threshold=val_threshold)
        val_metrics = evaluate(val_examples, model, threshold=val_threshold)
        average_epoch_loss = epoch_loss / len(train_examples)

        print(
            f"Epoch {epoch:02d} "
            f"train_loss={average_epoch_loss:.4f} "
            f"threshold={val_threshold:.2f} "
            f"| {format_metrics('train', train_metrics)} "
            f"| {format_metrics('val', val_metrics)}"
        )

        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            best_state = model.copy_state()
            best_threshold = val_threshold
            best_epoch = epoch
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Stopping early after epoch {epoch} (best epoch was {best_epoch}).")
                break

    model.load_state(best_state)

    train_metrics = evaluate(train_examples, model, threshold=best_threshold)
    val_metrics = evaluate(val_examples, model, threshold=best_threshold)
    test_metrics = evaluate(test_examples, model, threshold=best_threshold)

    metrics_payload = {
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "train": asdict(train_metrics),
        "validation": asdict(val_metrics),
        "test": asdict(test_metrics),
    }
    print("Best checkpoint")
    print(format_metrics("train", train_metrics))
    print(format_metrics("val", val_metrics))
    print(format_metrics("test", test_metrics))

    model_payload = {
        "max_number": args.max_number,
        "bit_width": bit_width,
        "mod_bases": list(mod_bases),
        "hidden_size": args.hidden_size,
        "threshold": best_threshold,
        "metrics": metrics_payload,
        "weights": model.copy_state(),
    }

    save_json(Path(args.model_path), model_payload)
    save_json(Path(args.metrics_path), metrics_payload)
    print(f"Saved model to {args.model_path}")
    print(f"Saved metrics to {args.metrics_path}")

    if args.preview:
        preview_model = build_model_from_payload(model_payload)
        print("Sample predictions")
        for number in args.preview:
            probability = predict_number(preview_model, model_payload, number)
            prediction = 1 if probability >= best_threshold else 0
            label = "prime" if prediction else "composite"
            print(f"{number:>6} -> {probability:.3f} ({label})")


def load_model(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_from_payload(payload: dict[str, object]) -> MLP:
    hidden_size = int(payload["hidden_size"])
    input_size = len(payload["weights"]["weight_input_hidden"])  # type: ignore[index]
    model = MLP(input_size=input_size, hidden_size=hidden_size, seed=0)
    model.load_state(payload["weights"])  # type: ignore[arg-type]
    return model


def predict_number_from_payload(payload: dict[str, object], number: int) -> float:
    model = build_model_from_payload(payload)
    return predict_number(model, payload, number)


def predict_number(model: MLP, payload: dict[str, object], number: int) -> float:
    features = encode_number(
        number,
        max_number=int(payload["max_number"]),
        bit_width=int(payload["bit_width"]),
        mod_bases=tuple(int(base) for base in payload["mod_bases"]),  # type: ignore[arg-type]
    )
    return model.predict_proba(features)


def compute_feature_ablation_importance(
    model: MLP,
    examples: list[Example],
    names: list[str],
) -> list[tuple[float, str]]:
    baseline_probabilities = [model.predict_proba(example.features) for example in examples]
    scores = []

    for feature_index, name in enumerate(names):
        total_change = 0.0
        for example, baseline_probability in zip(examples, baseline_probabilities):
            masked_features = list(example.features)
            masked_features[feature_index] = 0.0
            masked_probability = model.predict_proba(masked_features)
            total_change += abs(baseline_probability - masked_probability)
        scores.append((total_change / len(examples), name))

    scores.sort(reverse=True)
    return scores


def inspect_model(args: argparse.Namespace) -> None:
    payload = load_model(Path(args.model_path))
    model = build_model_from_payload(payload)
    max_number = int(payload["max_number"])
    bit_width = int(payload["bit_width"])
    mod_bases = tuple(int(base) for base in payload["mod_bases"])  # type: ignore[arg-type]
    names = feature_names(bit_width, mod_bases)

    dataset = [
        Example(
            number=number,
            features=encode_number(number, max_number=max_number, bit_width=bit_width, mod_bases=mod_bases),
            label=1 if is_prime(number) else 0,
        )
        for number in range(2, max_number + 1)
    ]

    print(
        f"Model summary: trained on 2..{max_number}, "
        f"{len(names)} input features, hidden size {model.hidden_size}, threshold={payload['threshold']}"
    )
    print("Top feature ablations")
    for score, name in compute_feature_ablation_importance(model, dataset, names)[: args.top]:
        print(f"  {name:<32} mean_probability_shift={score:.6f}")

    print("Hidden units")
    for hidden_index, output_weight in enumerate(model.weight_hidden_output):
        incoming = []
        for feature_index, name in enumerate(names):
            weight = model.weight_input_hidden[feature_index][hidden_index]
            incoming.append((abs(weight), weight, name))
        incoming.sort(reverse=True)
        strongest = ", ".join(f"{name}:{weight:+.2f}" for _, weight, name in incoming[: args.top_inputs])
        print(f"  hidden_{hidden_index:02d} -> output_weight={output_weight:+.2f} | {strongest}")


def explain_number(args: argparse.Namespace) -> None:
    payload = load_model(Path(args.model_path))
    model = build_model_from_payload(payload)
    max_number = int(payload["max_number"])
    bit_width = int(payload["bit_width"])
    mod_bases = tuple(int(base) for base in payload["mod_bases"])  # type: ignore[arg-type]
    names = feature_names(bit_width, mod_bases)

    for number in args.numbers:
        features = encode_number(number, max_number=max_number, bit_width=bit_width, mod_bases=mod_bases)
        hidden_pre, hidden, logit, probability = model.forward_details(features)
        prediction = "prime" if probability >= float(payload["threshold"]) else "composite"
        divisors = [base for base in mod_bases if number % base == 0 and number != base]

        print(f"Number {number}")
        print(f"  probability={probability:.6f} logit={logit:+.6f} predicted={prediction}")
        print(f"  output_bias={model.bias_output:+.6f}")
        if divisors:
            print(f"  active_divisibility_hints={', '.join(str(base) for base in divisors)}")
        else:
            print("  active_divisibility_hints=none")

        positive_contributions = []
        negative_contributions = []
        for hidden_index, hidden_value in enumerate(hidden):
            contribution = hidden_value * model.weight_hidden_output[hidden_index]
            item = (abs(contribution), contribution, hidden_index, hidden_pre[hidden_index], hidden_value)
            if contribution >= 0:
                positive_contributions.append(item)
            else:
                negative_contributions.append(item)

        positive_contributions.sort(reverse=True)
        negative_contributions.sort(reverse=True)

        print("  strongest_prime_pushes")
        for _, contribution, hidden_index, preactivation, hidden_value in positive_contributions[: args.top]:
            print(
                f"    hidden_{hidden_index:02d}: contribution={contribution:+.6f} "
                f"preactivation={preactivation:+.6f} activation={hidden_value:+.6f}"
            )

            input_contributions = []
            for feature_index, name in enumerate(names):
                delta = features[feature_index] * model.weight_input_hidden[feature_index][hidden_index]
                input_contributions.append((abs(delta), delta, name))
            input_contributions.sort(reverse=True)

            drivers = ", ".join(
                f"{name}:{delta:+.3f}" for _, delta, name in input_contributions[: args.top_inputs]
            )
            print(f"      drivers={drivers}")

        print("  strongest_composite_pushes")
        for _, contribution, hidden_index, preactivation, hidden_value in sorted(negative_contributions, reverse=True)[: args.top]:
            print(
                f"    hidden_{hidden_index:02d}: contribution={contribution:+.6f} "
                f"preactivation={preactivation:+.6f} activation={hidden_value:+.6f}"
            )

            input_contributions = []
            for feature_index, name in enumerate(names):
                delta = features[feature_index] * model.weight_input_hidden[feature_index][hidden_index]
                input_contributions.append((abs(delta), delta, name))
            input_contributions.sort(reverse=True)

            drivers = ", ".join(
                f"{name}:{delta:+.3f}" for _, delta, name in input_contributions[: args.top_inputs]
            )
            print(f"      drivers={drivers}")


def predict_numbers(args: argparse.Namespace) -> None:
    payload = load_model(Path(args.model_path))
    model = build_model_from_payload(payload)
    threshold = float(payload["threshold"])
    max_number = int(payload["max_number"])

    if any(number < 2 or number > max_number for number in args.numbers):
        print(
            f"warning: this model was trained on integers in the range 2..{max_number}; "
            "out-of-range predictions are extrapolation, not a proof."
        )

    for number in args.numbers:
        probability = predict_number(model, payload, number)
        prediction = 1 if probability >= threshold else 0
        label = "prime" if prediction else "composite"
        print(f"{number:>8} probability={probability:.3f} predicted={label}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a pure-Python neural network to classify numbers as prime.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a new model")
    train_parser.add_argument("--max-number", type=int, default=5000, help="largest integer included in training data")
    train_parser.add_argument("--epochs", type=int, default=35, help="maximum number of training epochs")
    train_parser.add_argument("--learning-rate", type=float, default=0.03, help="stochastic gradient descent step size")
    train_parser.add_argument("--hidden-size", type=int, default=24, help="hidden layer width")
    train_parser.add_argument("--patience", type=int, default=8, help="early stopping patience in epochs")
    train_parser.add_argument("--seed", type=int, default=7, help="random seed")
    train_parser.add_argument(
        "--mod-bases",
        type=parse_mod_bases,
        default=None,
        help="comma-separated modulus bases used as extra features; defaults to primes up to sqrt(max-number)",
    )
    train_parser.add_argument(
        "--model-path",
        default="artifacts/prime_mlp.json",
        help="where to save the trained model weights",
    )
    train_parser.add_argument(
        "--metrics-path",
        default="artifacts/metrics.json",
        help="where to save the final metrics report",
    )
    train_parser.add_argument(
        "--preview",
        nargs="*",
        type=int,
        default=[2, 3, 4, 5, 29, 97, 221, 997, 1024],
        help="numbers to score after training",
    )
    train_parser.set_defaults(handler=train_model)

    predict_parser = subparsers.add_parser("predict", help="score integers with a saved model")
    predict_parser.add_argument("numbers", nargs="+", type=int, help="integers to classify")
    predict_parser.add_argument(
        "--model-path",
        default="artifacts/prime_mlp.json",
        help="path to a saved model file",
    )
    predict_parser.set_defaults(handler=predict_numbers)

    inspect_parser = subparsers.add_parser("inspect", help="summarize which features and hidden units drive the model")
    inspect_parser.add_argument(
        "--model-path",
        default="artifacts/prime_mlp.json",
        help="path to a saved model file",
    )
    inspect_parser.add_argument("--top", type=int, default=12, help="how many features or units to show")
    inspect_parser.add_argument(
        "--top-inputs",
        type=int,
        default=6,
        help="how many incoming feature weights to show per hidden unit",
    )
    inspect_parser.set_defaults(handler=inspect_model)

    explain_parser = subparsers.add_parser("explain", help="trace the hidden-unit contributions for specific integers")
    explain_parser.add_argument("numbers", nargs="+", type=int, help="integers to explain")
    explain_parser.add_argument(
        "--model-path",
        default="artifacts/prime_mlp.json",
        help="path to a saved model file",
    )
    explain_parser.add_argument("--top", type=int, default=4, help="how many hidden units to show per number")
    explain_parser.add_argument(
        "--top-inputs",
        type=int,
        default=6,
        help="how many feature drivers to show for each hidden unit",
    )
    explain_parser.set_defaults(handler=explain_number)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
