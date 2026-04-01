#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "numpy is required for black_box_prime_nn.py. "
        "Create the local environment with `python3 -m venv .venv && .venv/bin/pip install numpy`."
    ) from exc


UINT64_MAX = (1 << 64) - 1
MR_BASES_64 = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
AUXILIARY_DIVISORS = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
BIT_SHIFTS = np.arange(63, -1, -1, dtype=np.uint64)


@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    positive_mask = values >= 0
    negative_mask = ~positive_mask
    output = np.empty_like(values, dtype=np.float32)
    output[positive_mask] = 1.0 / (1.0 + np.exp(-values[positive_mask]))
    exp_values = np.exp(values[negative_mask])
    output[negative_mask] = exp_values / (1.0 + exp_values)
    return output


def bce_with_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float32)
    if labels.shape != logits.shape:
        labels = labels.reshape(logits.shape)
    losses = np.maximum(logits, 0.0) - (logits * labels) + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(losses))


def safe_f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2.0 * precision * recall / denominator


def parse_hidden_sizes(value: str) -> list[int]:
    try:
        hidden_sizes = [int(chunk.strip()) for chunk in value.split(",") if chunk.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hidden sizes must be integers") from exc

    if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
        raise argparse.ArgumentTypeError("hidden sizes must be positive integers")
    return hidden_sizes


def sample_bit_length(rng: random.Random, *, minimum: int = 2) -> int:
    return rng.randint(minimum, 64)


def random_uint64_with_bit_length(rng: random.Random, bit_length: int) -> int:
    if bit_length == 64:
        return rng.getrandbits(64) | (1 << 63)
    return rng.randrange(1 << (bit_length - 1), 1 << bit_length)


def random_odd_with_bit_length(rng: random.Random, bit_length: int) -> int:
    candidate = random_uint64_with_bit_length(rng, bit_length) | 1
    upper_bound = UINT64_MAX if bit_length == 64 else (1 << bit_length) - 1
    if candidate > upper_bound:
        candidate -= 2
    return max(candidate, 3)


def is_prime_u64(number: int) -> bool:
    if number < 2:
        return False
    for prime in SMALL_PRIMES:
        if number == prime:
            return True
        if number % prime == 0:
            return False

    d = number - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for base in MR_BASES_64:
        if base % number == 0:
            continue
        witness = pow(base, d, number)
        if witness in (1, number - 1):
            continue
        for _ in range(s - 1):
            witness = pow(witness, 2, number)
            if witness == number - 1:
                break
        else:
            return False
    return True


def draw_prime_with_bit_length(rng: random.Random, bit_length: int) -> int:
    while True:
        candidate = random_odd_with_bit_length(rng, bit_length)
        upper_bound = UINT64_MAX if bit_length == 64 else (1 << bit_length) - 1
        while candidate <= upper_bound:
            if is_prime_u64(candidate):
                return candidate
            candidate += 2


def draw_prime_u64(rng: random.Random) -> int:
    if rng.random() < 0.05:
        return rng.choice((2, 3, 5, 7, 11, 13))
    bit_length = sample_bit_length(rng, minimum=2)
    return draw_prime_with_bit_length(rng, bit_length)


def draw_even_composite(rng: random.Random) -> int:
    while True:
        bit_length = sample_bit_length(rng, minimum=3)
        candidate = random_uint64_with_bit_length(rng, bit_length) & ~1
        lower_bound = 1 << (bit_length - 1)
        if candidate < lower_bound:
            candidate += 2
        if candidate > 2 and candidate <= UINT64_MAX and candidate % 2 == 0:
            return candidate


def draw_odd_composite(rng: random.Random) -> int:
    while True:
        bit_length = sample_bit_length(rng, minimum=3)
        candidate = random_odd_with_bit_length(rng, bit_length)
        if not is_prime_u64(candidate):
            return candidate


def draw_square_composite(rng: random.Random) -> int:
    while True:
        factor_bits = sample_bit_length(rng, minimum=2)
        factor = random_odd_with_bit_length(rng, factor_bits)
        square = factor * factor
        if square <= UINT64_MAX and square > 1 and not is_prime_u64(square):
            return square


def draw_semiprime(rng: random.Random) -> int:
    while True:
        first_bits = sample_bit_length(rng, minimum=2)
        second_bits = sample_bit_length(rng, minimum=2)
        first = draw_prime_with_bit_length(rng, first_bits)
        second = draw_prime_with_bit_length(rng, second_bits)
        semiprime = first * second
        if semiprime <= UINT64_MAX and not is_prime_u64(semiprime):
            return semiprime


def draw_composite_u64(rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.20:
        return draw_even_composite(rng)
    if roll < 0.70:
        return draw_odd_composite(rng)
    if roll < 0.85:
        return draw_square_composite(rng)
    return draw_semiprime(rng)


def generate_split(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    numbers = []
    labels = []

    for index in range(count):
        if index % 2 == 0:
            numbers.append(draw_prime_u64(rng))
            labels.append(1.0)
        else:
            numbers.append(draw_composite_u64(rng))
            labels.append(0.0)

    paired = list(zip(numbers, labels, strict=True))
    rng.shuffle(paired)

    number_array = np.array([number for number, _ in paired], dtype=np.uint64)
    label_array = np.array([label for _, label in paired], dtype=np.float32)
    return number_array, label_array


def encode_bits(numbers: np.ndarray) -> np.ndarray:
    bits = ((numbers[:, None] >> BIT_SHIFTS) & np.uint64(1)).astype(np.float32)
    return (bits * 2.0) - 1.0


def compute_auxiliary_labels(numbers: np.ndarray) -> np.ndarray:
    labels = np.zeros((len(numbers), len(AUXILIARY_DIVISORS)), dtype=np.float32)
    python_numbers = [int(number) for number in numbers.tolist()]
    for row_index, number in enumerate(python_numbers):
        for column_index, divisor in enumerate(AUXILIARY_DIVISORS):
            labels[row_index, column_index] = 1.0 if number % divisor == 0 else 0.0
    return labels


class AdamMLP:
    def __init__(self, layer_sizes: list[int], *, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.m_weights: list[np.ndarray] = []
        self.v_weights: list[np.ndarray] = []
        self.m_biases: list[np.ndarray] = []
        self.v_biases: list[np.ndarray] = []
        self.step = 0

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:], strict=True):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            weights = rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)
            biases = np.zeros((1, fan_out), dtype=np.float32)
            self.weights.append(weights)
            self.biases.append(biases)
            self.m_weights.append(np.zeros_like(weights))
            self.v_weights.append(np.zeros_like(weights))
            self.m_biases.append(np.zeros_like(biases))
            self.v_biases.append(np.zeros_like(biases))

    def forward(self, inputs: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        activations = [inputs]
        preactivations = []
        current = inputs

        for layer_index, (weights, biases) in enumerate(zip(self.weights, self.biases, strict=True)):
            preactivation = current @ weights + biases
            preactivations.append(preactivation)
            if layer_index == len(self.weights) - 1:
                current = preactivation
            else:
                current = np.tanh(preactivation)
            activations.append(current)

        return activations, preactivations, current

    def predict_logits(self, inputs: np.ndarray) -> np.ndarray:
        _, _, logits = self.forward(inputs)
        return logits

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        return stable_sigmoid(self.predict_logits(inputs))

    def train_batch(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        *,
        aux_labels: np.ndarray | None,
        aux_weight: float,
        learning_rate: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        weight_decay: float,
    ) -> float:
        batch_size = inputs.shape[0]
        activations, _, logits = self.forward(inputs)
        probabilities = stable_sigmoid(logits)
        delta = (probabilities - labels.reshape(-1, 1)) / batch_size
        loss = bce_with_logits(logits, labels)

        grad_weights: list[np.ndarray] = []
        grad_biases: list[np.ndarray] = []

        for layer_index in range(len(self.weights) - 1, -1, -1):
            previous_activation = activations[layer_index]
            grad_weight = previous_activation.T @ delta
            grad_bias = np.sum(delta, axis=0, keepdims=True)

            if weight_decay:
                grad_weight += weight_decay * self.weights[layer_index]

            grad_weights.append(grad_weight)
            grad_biases.append(grad_bias)

            if layer_index > 0:
                delta = (delta @ self.weights[layer_index].T) * (1.0 - np.square(activations[layer_index]))

        grad_weights.reverse()
        grad_biases.reverse()

        self.step += 1
        for layer_index in range(len(self.weights)):
            self.m_weights[layer_index] = (beta1 * self.m_weights[layer_index]) + ((1.0 - beta1) * grad_weights[layer_index])
            self.v_weights[layer_index] = (beta2 * self.v_weights[layer_index]) + (
                (1.0 - beta2) * np.square(grad_weights[layer_index])
            )
            self.m_biases[layer_index] = (beta1 * self.m_biases[layer_index]) + ((1.0 - beta1) * grad_biases[layer_index])
            self.v_biases[layer_index] = (beta2 * self.v_biases[layer_index]) + (
                (1.0 - beta2) * np.square(grad_biases[layer_index])
            )

            mhat_weights = self.m_weights[layer_index] / (1.0 - (beta1**self.step))
            vhat_weights = self.v_weights[layer_index] / (1.0 - (beta2**self.step))
            mhat_biases = self.m_biases[layer_index] / (1.0 - (beta1**self.step))
            vhat_biases = self.v_biases[layer_index] / (1.0 - (beta2**self.step))

            self.weights[layer_index] -= learning_rate * mhat_weights / (np.sqrt(vhat_weights) + epsilon)
            self.biases[layer_index] -= learning_rate * mhat_biases / (np.sqrt(vhat_biases) + epsilon)

        return loss

    def state_dict(self) -> dict[str, object]:
        return {
            "weights": [weight.astype(np.float32) for weight in self.weights],
            "biases": [bias.astype(np.float32) for bias in self.biases],
            "step": self.step,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.weights = [np.array(weight, dtype=np.float32) for weight in state["weights"]]  # type: ignore[index]
        self.biases = [np.array(bias, dtype=np.float32) for bias in state["biases"]]  # type: ignore[index]
        self.m_weights = [np.zeros_like(weight) for weight in self.weights]
        self.v_weights = [np.zeros_like(weight) for weight in self.weights]
        self.m_biases = [np.zeros_like(bias) for bias in self.biases]
        self.v_biases = [np.zeros_like(bias) for bias in self.biases]
        self.step = int(state.get("step", 0))  # type: ignore[arg-type]


class AdamBitRNN:
    def __init__(self, hidden_size: int, *, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        self.step = 0

        input_limit = math.sqrt(6.0 / (1 + hidden_size))
        recurrent_limit = math.sqrt(6.0 / (hidden_size + hidden_size))
        output_limit = math.sqrt(6.0 / (hidden_size + 1))

        self.w_x = rng.uniform(-input_limit, input_limit, size=(1, hidden_size)).astype(np.float32)
        self.w_h = rng.uniform(-recurrent_limit, recurrent_limit, size=(hidden_size, hidden_size)).astype(np.float32)
        self.b_h = np.zeros((1, hidden_size), dtype=np.float32)
        self.w_out = rng.uniform(-output_limit, output_limit, size=(hidden_size, 1)).astype(np.float32)
        self.b_out = np.zeros((1, 1), dtype=np.float32)
        self.w_aux = rng.uniform(-output_limit, output_limit, size=(hidden_size, len(AUXILIARY_DIVISORS))).astype(np.float32)
        self.b_aux = np.zeros((1, len(AUXILIARY_DIVISORS)), dtype=np.float32)

        self.m_w_x = np.zeros_like(self.w_x)
        self.v_w_x = np.zeros_like(self.w_x)
        self.m_w_h = np.zeros_like(self.w_h)
        self.v_w_h = np.zeros_like(self.w_h)
        self.m_b_h = np.zeros_like(self.b_h)
        self.v_b_h = np.zeros_like(self.b_h)
        self.m_w_out = np.zeros_like(self.w_out)
        self.v_w_out = np.zeros_like(self.w_out)
        self.m_b_out = np.zeros_like(self.b_out)
        self.v_b_out = np.zeros_like(self.b_out)
        self.m_w_aux = np.zeros_like(self.w_aux)
        self.v_w_aux = np.zeros_like(self.w_aux)
        self.m_b_aux = np.zeros_like(self.b_aux)
        self.v_b_aux = np.zeros_like(self.b_aux)

    def forward(self, inputs: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        batch_size = inputs.shape[0]
        hidden_states = [np.zeros((batch_size, self.hidden_size), dtype=np.float32)]
        preactivations = []
        current = hidden_states[0]

        for time_step in range(inputs.shape[1]):
            x_t = inputs[:, time_step : time_step + 1]
            preactivation = (x_t @ self.w_x) + (current @ self.w_h) + self.b_h
            current = np.tanh(preactivation)
            preactivations.append(preactivation)
            hidden_states.append(current)

        logits = (current @ self.w_out) + self.b_out
        auxiliary_logits = (current @ self.w_aux) + self.b_aux
        return hidden_states, preactivations, logits, auxiliary_logits

    def predict_logits(self, inputs: np.ndarray) -> np.ndarray:
        _, _, logits, _ = self.forward(inputs)
        return logits

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        return stable_sigmoid(self.predict_logits(inputs))

    def train_batch(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        *,
        aux_labels: np.ndarray | None,
        aux_weight: float,
        learning_rate: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        weight_decay: float,
    ) -> float:
        batch_size = inputs.shape[0]
        if aux_labels is None:
            raise ValueError("aux_labels are required for the recurrent black-box model")

        hidden_states, _, logits, auxiliary_logits = self.forward(inputs)
        probabilities = stable_sigmoid(logits)
        delta_out = (probabilities - labels.reshape(-1, 1)) / batch_size
        auxiliary_probabilities = stable_sigmoid(auxiliary_logits)
        delta_aux = aux_weight * (auxiliary_probabilities - aux_labels) / batch_size
        loss = bce_with_logits(logits, labels) + (aux_weight * bce_with_logits(auxiliary_logits, aux_labels))

        grad_w_out = hidden_states[-1].T @ delta_out
        grad_b_out = np.sum(delta_out, axis=0, keepdims=True)
        grad_w_aux = hidden_states[-1].T @ delta_aux
        grad_b_aux = np.sum(delta_aux, axis=0, keepdims=True)

        if weight_decay:
            grad_w_out += weight_decay * self.w_out
            grad_w_aux += weight_decay * self.w_aux

        grad_w_x = np.zeros_like(self.w_x)
        grad_w_h = np.zeros_like(self.w_h)
        grad_b_h = np.zeros_like(self.b_h)
        delta_hidden = (delta_out @ self.w_out.T) + (delta_aux @ self.w_aux.T)

        for time_step in range(inputs.shape[1] - 1, -1, -1):
            hidden = hidden_states[time_step + 1]
            previous_hidden = hidden_states[time_step]
            activation_delta = delta_hidden * (1.0 - np.square(hidden))
            x_t = inputs[:, time_step : time_step + 1]

            grad_w_x += x_t.T @ activation_delta
            grad_w_h += previous_hidden.T @ activation_delta
            grad_b_h += np.sum(activation_delta, axis=0, keepdims=True)
            delta_hidden = activation_delta @ self.w_h.T

        if weight_decay:
            grad_w_x += weight_decay * self.w_x
            grad_w_h += weight_decay * self.w_h

        self.step += 1

        self.m_w_x = (beta1 * self.m_w_x) + ((1.0 - beta1) * grad_w_x)
        self.v_w_x = (beta2 * self.v_w_x) + ((1.0 - beta2) * np.square(grad_w_x))
        self.m_w_h = (beta1 * self.m_w_h) + ((1.0 - beta1) * grad_w_h)
        self.v_w_h = (beta2 * self.v_w_h) + ((1.0 - beta2) * np.square(grad_w_h))
        self.m_b_h = (beta1 * self.m_b_h) + ((1.0 - beta1) * grad_b_h)
        self.v_b_h = (beta2 * self.v_b_h) + ((1.0 - beta2) * np.square(grad_b_h))
        self.m_w_out = (beta1 * self.m_w_out) + ((1.0 - beta1) * grad_w_out)
        self.v_w_out = (beta2 * self.v_w_out) + ((1.0 - beta2) * np.square(grad_w_out))
        self.m_b_out = (beta1 * self.m_b_out) + ((1.0 - beta1) * grad_b_out)
        self.v_b_out = (beta2 * self.v_b_out) + ((1.0 - beta2) * np.square(grad_b_out))
        self.m_w_aux = (beta1 * self.m_w_aux) + ((1.0 - beta1) * grad_w_aux)
        self.v_w_aux = (beta2 * self.v_w_aux) + ((1.0 - beta2) * np.square(grad_w_aux))
        self.m_b_aux = (beta1 * self.m_b_aux) + ((1.0 - beta1) * grad_b_aux)
        self.v_b_aux = (beta2 * self.v_b_aux) + ((1.0 - beta2) * np.square(grad_b_aux))

        correction1 = 1.0 - (beta1**self.step)
        correction2 = 1.0 - (beta2**self.step)

        self.w_x -= learning_rate * (self.m_w_x / correction1) / (np.sqrt(self.v_w_x / correction2) + epsilon)
        self.w_h -= learning_rate * (self.m_w_h / correction1) / (np.sqrt(self.v_w_h / correction2) + epsilon)
        self.b_h -= learning_rate * (self.m_b_h / correction1) / (np.sqrt(self.v_b_h / correction2) + epsilon)
        self.w_out -= learning_rate * (self.m_w_out / correction1) / (np.sqrt(self.v_w_out / correction2) + epsilon)
        self.b_out -= learning_rate * (self.m_b_out / correction1) / (np.sqrt(self.v_b_out / correction2) + epsilon)
        self.w_aux -= learning_rate * (self.m_w_aux / correction1) / (np.sqrt(self.v_w_aux / correction2) + epsilon)
        self.b_aux -= learning_rate * (self.m_b_aux / correction1) / (np.sqrt(self.v_b_aux / correction2) + epsilon)

        return loss

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "w_x": self.w_x,
            "w_h": self.w_h,
            "b_h": self.b_h,
            "w_out": self.w_out,
            "b_out": self.b_out,
            "w_aux": self.w_aux,
            "b_aux": self.b_aux,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.w_x = np.array(state["w_x"], dtype=np.float32)  # type: ignore[index]
        self.w_h = np.array(state["w_h"], dtype=np.float32)  # type: ignore[index]
        self.b_h = np.array(state["b_h"], dtype=np.float32)  # type: ignore[index]
        self.w_out = np.array(state["w_out"], dtype=np.float32)  # type: ignore[index]
        self.b_out = np.array(state["b_out"], dtype=np.float32)  # type: ignore[index]
        self.w_aux = np.array(state["w_aux"], dtype=np.float32)  # type: ignore[index]
        self.b_aux = np.array(state["b_aux"], dtype=np.float32)  # type: ignore[index]

        self.m_w_x = np.zeros_like(self.w_x)
        self.v_w_x = np.zeros_like(self.w_x)
        self.m_w_h = np.zeros_like(self.w_h)
        self.v_w_h = np.zeros_like(self.w_h)
        self.m_b_h = np.zeros_like(self.b_h)
        self.v_b_h = np.zeros_like(self.b_h)
        self.m_w_out = np.zeros_like(self.w_out)
        self.v_w_out = np.zeros_like(self.w_out)
        self.m_b_out = np.zeros_like(self.b_out)
        self.v_b_out = np.zeros_like(self.b_out)
        self.m_w_aux = np.zeros_like(self.w_aux)
        self.v_w_aux = np.zeros_like(self.w_aux)
        self.m_b_aux = np.zeros_like(self.b_aux)
        self.v_b_aux = np.zeros_like(self.b_aux)
        self.step = 0


def evaluate_split(model: AdamMLP, inputs: np.ndarray, labels: np.ndarray, *, threshold: float) -> Metrics:
    logits = model.predict_logits(inputs)
    probabilities = stable_sigmoid(logits).reshape(-1)
    loss = bce_with_logits(logits, labels)

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
    return Metrics(loss=loss, accuracy=accuracy, precision=precision, recall=recall, f1=f1)


def best_threshold(model: AdamMLP, inputs: np.ndarray, labels: np.ndarray) -> float:
    probabilities = stable_sigmoid(model.predict_logits(inputs)).reshape(-1)
    best_value = 0.5
    best_f1_score = -1.0

    for step in range(5, 96):
        threshold = step / 100.0
        predictions = probabilities >= threshold
        positives = labels == 1.0
        negatives = ~positives

        tp = int(np.sum(predictions & positives))
        fp = int(np.sum(predictions & negatives))
        fn = int(np.sum((~predictions) & positives))

        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = safe_f1(precision, recall)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_value = threshold

    return best_value


def odd_only_metrics(model: AdamMLP, inputs: np.ndarray, labels: np.ndarray, numbers: np.ndarray, threshold: float) -> Metrics:
    odd_mask = (numbers % np.uint64(2) == np.uint64(1))
    if not np.any(odd_mask):
        return Metrics(loss=0.0, accuracy=0.0, precision=0.0, recall=0.0, f1=0.0)
    return evaluate_split(model, inputs[odd_mask], labels[odd_mask], threshold=threshold)


def format_metrics(label: str, metrics: Metrics) -> str:
    return (
        f"{label:<10} loss={metrics.loss:.4f} "
        f"acc={metrics.accuracy:.3f} "
        f"precision={metrics.precision:.3f} "
        f"recall={metrics.recall:.3f} "
        f"f1={metrics.f1:.3f}"
    )


def save_model(path: Path, model: AdamMLP, *, threshold: float, hidden_sizes: list[int], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "threshold": np.array([threshold], dtype=np.float32),
        "seed": np.array([seed], dtype=np.int64),
    }
    if isinstance(model, AdamBitRNN):
        payload["model_type"] = np.array(["rnn"])
        payload["hidden_size"] = np.array([model.hidden_size], dtype=np.int64)
        payload.update(model.state_dict())
    else:
        payload["model_type"] = np.array(["mlp"])
        payload["hidden_sizes"] = np.array(hidden_sizes, dtype=np.int64)
        for index, weight in enumerate(model.weights):
            payload[f"weight_{index}"] = weight
        for index, bias in enumerate(model.biases):
            payload[f"bias_{index}"] = bias
    np.savez_compressed(path, **payload)


def load_model(path: Path) -> tuple[object, float, dict[str, object]]:
    payload = np.load(path)
    model_type = str(payload["model_type"][0])
    threshold = float(payload["threshold"][0])
    seed = int(payload["seed"][0])

    if model_type == "rnn":
        hidden_size = int(payload["hidden_size"][0])
        model = AdamBitRNN(hidden_size, seed=seed)
        model.load_state_dict(
            {
                "w_x": payload["w_x"],
                "w_h": payload["w_h"],
                "b_h": payload["b_h"],
                "w_out": payload["w_out"],
                "b_out": payload["b_out"],
                "w_aux": payload["w_aux"],
                "b_aux": payload["b_aux"],
            }
        )
        return model, threshold, {"model_type": model_type, "hidden_size": hidden_size, "seed": seed}

    hidden_sizes = [int(value) for value in payload["hidden_sizes"].tolist()]
    layer_sizes = [64, *hidden_sizes, 1]
    model = AdamMLP(layer_sizes, seed=seed)
    weights = [payload[f"weight_{index}"] for index in range(len(layer_sizes) - 1)]
    biases = [payload[f"bias_{index}"] for index in range(len(layer_sizes) - 1)]
    model.load_state_dict({"weights": weights, "biases": biases, "step": 0})
    return model, threshold, {"model_type": model_type, "hidden_sizes": hidden_sizes, "seed": seed}


def save_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    train_numbers, train_labels = generate_split(args.train_size, args.seed)
    val_numbers, val_labels = generate_split(args.val_size, args.seed + 1)
    test_numbers, test_labels = generate_split(args.test_size, args.seed + 2)

    train_inputs = encode_bits(train_numbers)
    val_inputs = encode_bits(val_numbers)
    test_inputs = encode_bits(test_numbers)
    train_aux_labels = compute_auxiliary_labels(train_numbers)

    if args.model_type == "rnn":
        model = AdamBitRNN(args.hidden_size, seed=args.seed)
        model_description = f"rnn(hidden={args.hidden_size})"
        layer_sizes = [64, args.hidden_size, 1]
    else:
        layer_sizes = [64, *args.hidden_sizes, 1]
        model = AdamMLP(layer_sizes, seed=args.seed)
        model_description = f"mlp(layers={layer_sizes})"

    print(
        f"Training black-box 64-bit model with {model_description}, "
        f"train={args.train_size}, val={args.val_size}, test={args.test_size}"
    )

    rng = np.random.default_rng(args.seed)
    best_state = model.state_dict()
    best_epoch = 0
    best_val_f1 = -1.0
    best_threshold_value = 0.5
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(args.train_size)
        shuffled_inputs = train_inputs[order]
        shuffled_labels = train_labels[order]
        shuffled_aux_labels = train_aux_labels[order]
        loss_total = 0.0
        batch_count = 0

        for start in range(0, args.train_size, args.batch_size):
            stop = min(start + args.batch_size, args.train_size)
            batch_inputs = shuffled_inputs[start:stop]
            batch_labels = shuffled_labels[start:stop]
            batch_aux_labels = shuffled_aux_labels[start:stop]
            loss_total += model.train_batch(
                batch_inputs,
                batch_labels,
                aux_labels=batch_aux_labels,
                aux_weight=args.aux_weight,
                learning_rate=args.learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                weight_decay=args.weight_decay,
            )
            batch_count += 1

        threshold = best_threshold(model, val_inputs, val_labels)
        train_metrics = evaluate_split(model, train_inputs, train_labels, threshold=threshold)
        val_metrics = evaluate_split(model, val_inputs, val_labels, threshold=threshold)
        odd_metrics = odd_only_metrics(model, val_inputs, val_labels, val_numbers, threshold)

        print(
            f"Epoch {epoch:02d} "
            f"batch_loss={loss_total / batch_count:.4f} "
            f"threshold={threshold:.2f} "
            f"| {format_metrics('train', train_metrics)} "
            f"| {format_metrics('val', val_metrics)} "
            f"| {format_metrics('val odd', odd_metrics)}"
        )

        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            best_epoch = epoch
            best_threshold_value = threshold
            best_state = model.state_dict()
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Stopping early after epoch {epoch} (best epoch was {best_epoch}).")
                break

    model.load_state_dict(best_state)

    train_metrics = evaluate_split(model, train_inputs, train_labels, threshold=best_threshold_value)
    val_metrics = evaluate_split(model, val_inputs, val_labels, threshold=best_threshold_value)
    test_metrics = evaluate_split(model, test_inputs, test_labels, threshold=best_threshold_value)
    test_odd_metrics = odd_only_metrics(model, test_inputs, test_labels, test_numbers, best_threshold_value)

    print("Best checkpoint")
    print(format_metrics("train", train_metrics))
    print(format_metrics("val", val_metrics))
    print(format_metrics("test", test_metrics))
    print(format_metrics("test odd", test_odd_metrics))

    metrics_payload = {
        "threshold": best_threshold_value,
        "best_epoch": best_epoch,
        "layers": layer_sizes,
        "model_type": args.model_type,
        "auxiliary_divisors": list(AUXILIARY_DIVISORS),
        "aux_weight": args.aux_weight,
        "train": asdict(train_metrics),
        "validation": asdict(val_metrics),
        "test": asdict(test_metrics),
        "test_odd_only": asdict(test_odd_metrics),
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
    }

    save_model(Path(args.model_path), model, threshold=best_threshold_value, hidden_sizes=args.hidden_sizes, seed=args.seed)
    save_metrics(Path(args.metrics_path), metrics_payload)
    print(f"Saved model to {args.model_path}")
    print(f"Saved metrics to {args.metrics_path}")

    if args.preview:
        print("Sample predictions")
        predict_numbers(model, best_threshold_value, args.preview)


def predict_numbers(model: AdamMLP, threshold: float, numbers: list[int]) -> None:
    valid_numbers = []
    for number in numbers:
        if number < 0 or number > UINT64_MAX:
            raise ValueError(f"{number} is outside the uint64 range")
        valid_numbers.append(number)

    inputs = encode_bits(np.array(valid_numbers, dtype=np.uint64))
    probabilities = model.predict_proba(inputs).reshape(-1)
    for number, probability in zip(valid_numbers, probabilities, strict=True):
        label = "prime" if probability >= threshold else "composite"
        print(f"{number:>20} probability={probability:.4f} guessed={label}")


def predict(args: argparse.Namespace) -> None:
    model, threshold, metadata = load_model(Path(args.model_path))
    if metadata["model_type"] == "rnn":
        print(f"Loaded black-box rnn with hidden size {metadata['hidden_size']} and threshold {threshold:.2f}")
    else:
        print(f"Loaded black-box mlp with hidden sizes {metadata['hidden_sizes']} and threshold {threshold:.2f}")
    predict_numbers(model, threshold, args.numbers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a black-box neural network that guesses primality from 64 raw input bits."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a new 64-bit black-box guesser")
    train_parser.add_argument("--train-size", type=int, default=12000, help="number of training examples")
    train_parser.add_argument("--val-size", type=int, default=3000, help="number of validation examples")
    train_parser.add_argument("--test-size", type=int, default=3000, help="number of test examples")
    train_parser.add_argument("--epochs", type=int, default=25, help="maximum number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=256, help="mini-batch size")
    train_parser.add_argument("--learning-rate", type=float, default=0.0015, help="Adam learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 regularization")
    train_parser.add_argument(
        "--aux-weight",
        type=float,
        default=0.35,
        help="extra weight on the auxiliary small-divisor supervision used by the rnn model",
    )
    train_parser.add_argument(
        "--model-type",
        choices=("rnn", "mlp"),
        default="rnn",
        help="black-box architecture to train; rnn is the default because bit sequences are order-sensitive",
    )
    train_parser.add_argument("--hidden-size", type=int, default=128, help="recurrent hidden size for rnn mode")
    train_parser.add_argument("--hidden-sizes", type=parse_hidden_sizes, default=[192, 96, 48], help="comma-separated hidden layer sizes")
    train_parser.add_argument("--patience", type=int, default=6, help="early stopping patience")
    train_parser.add_argument("--seed", type=int, default=7, help="random seed")
    train_parser.add_argument(
        "--model-path",
        default="artifacts/black_box_u64_model.npz",
        help="where to save the model weights",
    )
    train_parser.add_argument(
        "--metrics-path",
        default="artifacts/black_box_u64_metrics.json",
        help="where to save the metrics report",
    )
    train_parser.add_argument(
        "--preview",
        nargs="*",
        type=int,
        default=[2, 3, 4, 5, 97, 221, 997, 10007, 18446744073709551557],
        help="numbers to score after training",
    )
    train_parser.set_defaults(handler=train)

    predict_parser = subparsers.add_parser("predict", help="score integers with a saved black-box model")
    predict_parser.add_argument("numbers", nargs="+", type=int, help="numbers to score")
    predict_parser.add_argument(
        "--model-path",
        default="artifacts/black_box_u64_model.npz",
        help="path to a saved model file",
    )
    predict_parser.set_defaults(handler=predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
