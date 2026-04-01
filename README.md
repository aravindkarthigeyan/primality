# Primality Neural Networks

This repository now has three different primality demos:

- `prime_nn.py`: a bounded-range model with hand-engineered divisibility features.
- `black_box_prime_nn.py`: a raw 64-bit black-box guesser that sees only the input bits.
- `torch_black_box_prime_nn.py`: a stronger raw-bit PyTorch sequence model with a default hybrid front-end.

## Run it

Train a model:

```bash
python3 prime_nn.py train
```

Predict with the saved model:

```bash
python3 prime_nn.py predict 97 221 997
```

Inspect the model globally:

```bash
python3 prime_nn.py inspect
```

Explain one prediction locally:

```bash
python3 prime_nn.py explain 97 221
```

If you ask the model about values outside the range it was trained on, the CLI prints a warning because those predictions are extrapolation only.

Use a larger or smaller training range:

```bash
python3 prime_nn.py train --max-number 10000 --epochs 45 --hidden-size 32
```

## NumPy Black-Box 64-Bit Model

The black-box model uses only 64 raw input bits and exact labels from a deterministic 64-bit Miller-Rabin test. It needs `numpy`, and the project already includes a local virtualenv setup path:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Train the black-box guesser:

```bash
.venv/bin/python black_box_prime_nn.py train
```

Score a few `uint64` values:

```bash
.venv/bin/python black_box_prime_nn.py predict 97 221 997 18446744073709551557
```

The default black-box trainer uses a recurrent network over the 64-bit sequence, plus auxiliary supervision for divisibility by small primes so the model can learn arithmetic structure from raw bits more effectively.

Observed result from the current saved `uint64` checkpoint:

- Balanced test accuracy is about `0.594`.
- Balanced test F1 is about `0.709`.
- Odd-only test accuracy is about `0.553`.

That means it is a genuine black-box guesser, but not a competitive primality test.

## PyTorch Sequence Model

For a stronger version, use the PyTorch trainer:

```bash
.venv/bin/python torch_black_box_prime_nn.py train
```

Predict with the saved checkpoint:

```bash
.venv/bin/python torch_black_box_prime_nn.py predict 97 221 997 10007
```

By default, `predict` uses a hybrid path:

- exact small-prime filtering for the auxiliary prime set used during training
- the raw-bit GRU model only for numbers that survive that exact filter

You can force the pure neural residual model with:

```bash
.venv/bin/python torch_black_box_prime_nn.py predict --pure 97 221 997 10007
```

Observed result from the current saved PyTorch checkpoint:

- Pure neural test accuracy is about `0.604`.
- Pure neural odd-only test accuracy is about `0.565`.
- Hybrid test accuracy is about `0.906`.
- Hybrid odd-only test accuracy is about `0.895`.

So the PyTorch path is useful when combined with the exact small-prime front-end. The residual network then handles the harder no-small-factor cases.

## How it works

- Inputs are fixed-width binary digits plus a few modular-residue features.
- Labels are generated with an exact trial-division primality test.
- The classifier is a one-hidden-layer multilayer perceptron trained with stochastic gradient descent.
- Because primes are much rarer than composites, the loss gives extra weight to prime examples.
- By default, the modular features use prime bases up to `sqrt(max-number)`, which gives the network strong divisibility hints inside the training range.
- The `inspect` command shows which input features most shift the model's output, and `explain` traces which hidden units pushed a specific number toward prime or composite.

## Limitation

These are learned systems, not proof systems.

`prime_nn.py` performs very well on the numeric range it was trained on because the input features already encode divisibility hints.

`black_box_prime_nn.py` is much stricter on the input side, but the task becomes much harder and its guesses are only moderately useful.

`torch_black_box_prime_nn.py` gets to a much better practical result, but only because the default path uses an exact small-prime front-end before the neural residual model. Exact primality testing still requires a real algorithm.

## GRFP Release Watch

This repository also includes `fetch_grfp_awardees.py`, which polls the live NSF GRFP awardee form on Research.gov.

One-shot examples:

```bash
python3 fetch_grfp_awardees.py --award-year 2024 --award-type A --limit 5
python3 fetch_grfp_awardees.py --award-year 2026 --award-type A
```

Local watch mode:

```bash
python3 fetch_grfp_awardees.py --award-year 2026 --award-type A --watch
```

GitHub Actions watch:

- `.github/workflows/grfp-2026-watch.yml` checks `2026` every 15 minutes at `:07`, `:22`, `:37`, and `:52`.
- If Research.gov starts returning rows, the workflow uploads first-page CSV and JSON artifacts and opens one GitHub issue titled `GRFP 2026 award list is live`.
- GitHub documents that scheduled workflows can be delayed during high-load periods, especially near the top of the hour, which is why the schedule avoids `:00`.
- In public repositories, GitHub automatically disables scheduled workflows after 60 days with no repository activity. Re-enable the workflow if that happens.
