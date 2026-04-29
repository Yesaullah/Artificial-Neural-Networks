"""
Input Specific Neural Networks (ISNN-1 and ISNN-2) for the toy problems in
Jadoon et al., "Input Specific Neural Networks", Sections 2-3.1.

This submission contains:
  - Latin Hypercube generation for the two paper toy datasets.
  - Manual NumPy implementations of ISNN-1 and ISNN-2 with explicit
    backpropagation through matrix multiplication and activations.
  - PyTorch implementations of the same architectures.
  - Training/evaluation loops and PNG plots for losses and model behavior.

Use --backend manual, --backend torch, or --backend both to run the manual
NumPy implementation, the PyTorch implementation, or both.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EPS = 1e-8


def softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def toy_additive(x: np.ndarray) -> np.ndarray:
    x0, y0, t0, z0 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    return (
        np.exp(-0.5 * x0)
        + softplus_np(0.4 * y0)
        + np.tanh(t0)
        + np.sin(z0)
        - 0.4
    )[:, None]


def toy_multiplicative(x: np.ndarray) -> np.ndarray:
    x0, y0, t0, z0 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    fx = np.exp(-0.3 * x0)
    fy = (0.15 * y0) ** 2
    ft = np.tanh(0.3 * t0)
    fz = 0.2 * np.sin(0.5 * z0 + 2.0) + 0.5
    return (fx * fy * fz * ft)[:, None]


def latin_hypercube(n: int, dim: int, low: float, high: float, seed: int) -> np.ndarray:
    """Simple Latin Hypercube sampler using one random permutation per dimension."""
    rng = np.random.default_rng(seed)
    unit = np.empty((n, dim), dtype=np.float64)
    for j in range(dim):
        unit[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return low + (high - low) * unit


def make_dataset(name: str, out_dir: Path, seed: int = 123) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if name == "additive":
        f = toy_additive
        test_high = 6.0
    elif name == "multiplicative":
        f = toy_multiplicative
        test_high = 10.0
    else:
        raise ValueError(f"unknown dataset {name}")

    x_train = latin_hypercube(500, 4, 0.0, 4.0, seed)
    x_test = latin_hypercube(5000, 4, 0.0, test_high, seed + 1000)
    y_train = f(x_train)
    y_test = f(x_test)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{name}_train.npz", X=x_train, y=y_train)
    np.savez(out_dir / f"{name}_test.npz", X=x_test, y=y_test)
    write_csv(out_dir / f"{name}_train.csv", x_train, y_train)
    write_csv(out_dir / f"{name}_test.csv", x_test, y_test)
    return x_train, y_train, x_test, y_test


def write_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "t", "z", "target"])
        for row, target in zip(x, y[:, 0]):
            w.writerow([*row.tolist(), float(target)])


class Tensor:
    """Tiny tensor object for explicit reverse-mode backprop in NumPy."""

    def __init__(self, data: np.ndarray, parents: Sequence[Tuple["Tensor", object]] = ()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.parents = list(parents)

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        topo: List[Tensor] = []
        seen = set()

        def visit(t: Tensor) -> None:
            if id(t) in seen:
                return
            seen.add(id(t))
            for parent, _ in t.parents:
                visit(parent)
            topo.append(t)

        visit(self)
        self.grad += np.ones_like(self.data) if grad is None else grad
        for t in reversed(topo):
            for parent, fn in t.parents:
                parent.grad += fn(t.grad)


def add(a: Tensor, b: Tensor) -> Tensor:
    def ba(g: np.ndarray) -> np.ndarray:
        return g

    def bb(g: np.ndarray) -> np.ndarray:
        while g.ndim > b.data.ndim:
            g = g.sum(axis=0)
        for axis, size in enumerate(b.data.shape):
            if size == 1:
                g = g.sum(axis=axis, keepdims=True)
        return g

    return Tensor(a.data + b.data, [(a, ba), (b, bb)])


def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    y = Tensor(x.data @ w.data.T, [
        (x, lambda g: g @ w.data),
        (w, lambda g: g.T @ x.data),
    ])
    return add(y, b) if b is not None else y


def softplus(t: Tensor) -> Tensor:
    s = sigmoid_np(t.data)
    return Tensor(softplus_np(t.data), [(t, lambda g: g * s)])


def sigmoid(t: Tensor) -> Tensor:
    s = sigmoid_np(t.data)
    return Tensor(s, [(t, lambda g: g * s * (1.0 - s))])


def squared_norm(t: Tensor) -> Tensor:
    """Return row-wise x^T x with explicit derivative 2x."""
    return Tensor(np.sum(t.data * t.data, axis=1, keepdims=True), [(t, lambda g: 2.0 * t.data * g)])


@dataclass
class Parameter:
    name: str
    raw: Tensor
    positive: bool = False

    @property
    def value(self) -> Tensor:
        return softplus(self.raw) if self.positive else self.raw


class Adam:
    def __init__(self, params: Sequence[Parameter], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 0
        self.m = [np.zeros_like(p.raw.data) for p in self.params]
        self.v = [np.zeros_like(p.raw.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.raw.grad.fill(0.0)

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            g = np.clip(p.raw.grad, -10.0, 10.0)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            p.raw.data -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


class ManualModel:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.params: Dict[str, Parameter] = {}

    def param(self, name: str, shape: Tuple[int, ...], positive: bool = False, scale: float = 0.25) -> Parameter:
        raw = self.rng.normal(0.0, scale, size=shape)
        if positive:
            # Constrained weights are optimized in raw space and transformed
            # with softplus. Start them near small positive values instead of
            # softplus(0) ~= 0.69, which otherwise makes early activations huge.
            raw = self.rng.normal(-3.0, 0.15, size=shape)
        if name.endswith(".b"):
            raw.fill(0.0)
        p = Parameter(name, Tensor(raw), positive)
        self.params[name] = p
        return p

    def parameters(self) -> List[Parameter]:
        return list(self.params.values())

    def forward(self, x_np: np.ndarray) -> Tensor:
        raise NotImplementedError

    def predict_scaled(self, x_np: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def w(self, name: str) -> np.ndarray:
        p = self.params[name]
        return softplus_np(p.raw.data) if p.positive else p.raw.data


class ISNN1Manual(ManualModel):
    def __init__(self, seed: int, width: int = 10):
        super().__init__(seed)
        self.width = width
        for branch in ["y", "z", "t"]:
            positive = branch in {"y", "t"}
            self.param(f"{branch}0.W", (width, 1), positive=positive)
            self.param(f"{branch}0.b", (1, width))
            self.param(f"{branch}1.W", (width, width), positive=positive)
            self.param(f"{branch}1.b", (1, width))
        self.param("x0.Wxx", (width, 1), positive=False)
        self.param("x0.Wxy", (width, width), positive=True)
        self.param("x0.Wxz", (width, width), positive=False)
        self.param("x0.Wxt", (width, width), positive=True)
        self.param("x0.b", (1, width))
        self.param("x1.W", (width, width), positive=True)
        self.param("x1.b", (1, width))
        self.params["x0.b"].raw.data.fill(-2.0)
        self.params["x1.b"].raw.data.fill(-2.0)

    def branch(self, x: Tensor, name: str, act) -> Tensor:
        h = act(linear(x, self.params[f"{name}0.W"].value, self.params[f"{name}0.b"].value))
        return act(linear(h, self.params[f"{name}1.W"].value, self.params[f"{name}1.b"].value))

    def forward(self, x_np: np.ndarray) -> Tensor:
        x0 = Tensor(x_np[:, 0:1])
        y0 = Tensor(x_np[:, 1:2])
        t0 = Tensor(x_np[:, 2:3])
        z0 = Tensor(x_np[:, 3:4])
        y = self.branch(y0, "y", softplus)
        z = self.branch(z0, "z", sigmoid)
        t = self.branch(t0, "t", sigmoid)
        h = add(linear(x0, self.params["x0.Wxx"].value), linear(y, self.params["x0.Wxy"].value))
        h = add(h, linear(z, self.params["x0.Wxz"].value))
        h = add(h, linear(t, self.params["x0.Wxt"].value, self.params["x0.b"].value))
        h = softplus(h)
        h = softplus(linear(h, self.params["x1.W"].value, self.params["x1.b"].value))
        return squared_norm(h)

    def predict_scaled(self, x_np: np.ndarray) -> np.ndarray:
        x0, y0, t0, z0 = x_np[:, 0:1], x_np[:, 1:2], x_np[:, 2:3], x_np[:, 3:4]
        y = softplus_np(y0 @ self.w("y0.W").T + self.w("y0.b"))
        y = softplus_np(y @ self.w("y1.W").T + self.w("y1.b"))
        z = sigmoid_np(z0 @ self.w("z0.W").T + self.w("z0.b"))
        z = sigmoid_np(z @ self.w("z1.W").T + self.w("z1.b"))
        t = sigmoid_np(t0 @ self.w("t0.W").T + self.w("t0.b"))
        t = sigmoid_np(t @ self.w("t1.W").T + self.w("t1.b"))
        h = (
            x0 @ self.w("x0.Wxx").T
            + y @ self.w("x0.Wxy").T
            + z @ self.w("x0.Wxz").T
            + t @ self.w("x0.Wxt").T
            + self.w("x0.b")
        )
        h = softplus_np(h)
        h = softplus_np(h @ self.w("x1.W").T + self.w("x1.b"))
        return np.sum(h * h, axis=1, keepdims=True)


class ISNN2Manual(ManualModel):
    def __init__(self, seed: int, width: int = 15, depth: int = 3):
        super().__init__(seed)
        self.width = width
        self.depth = depth
        if depth < 2:
            raise ValueError("ISNN-2 depth must be at least 2.")
        for branch in ["y", "z", "t"]:
            positive = branch in {"y", "t"}
            in_dim = 1
            for layer in range(depth - 1):
                self.param(f"{branch}{layer}.W", (width, in_dim), positive=positive)
                self.param(f"{branch}{layer}.b", (1, width))
                in_dim = width
        self.param("x0.Wxx", (width, 1), positive=False)
        self.param("x0.Wxy", (width, 1), positive=True)
        self.param("x0.Wxz", (width, 1), positive=False)
        self.param("x0.Wxt", (width, 1), positive=True)
        self.param("x0.b", (1, width))
        for layer in range(1, depth):
            out_dim = 1 if layer == depth - 1 else width
            self.param(f"x{layer}.Wxx", (out_dim, width), positive=True)
            self.param(f"x{layer}.Wxx0", (out_dim, 1), positive=False)
            self.param(f"x{layer}.Wxy", (out_dim, width), positive=True)
            self.param(f"x{layer}.Wxz", (out_dim, width), positive=False)
            self.param(f"x{layer}.Wxt", (out_dim, width), positive=True)
            self.param(f"x{layer}.b", (1, out_dim))

    def branch(self, x: Tensor, name: str, act) -> List[Tensor]:
        values = []
        h = x
        for layer in range(self.depth - 1):
            h = act(linear(h, self.params[f"{name}{layer}.W"].value, self.params[f"{name}{layer}.b"].value))
            values.append(h)
        return values

    def forward(self, x_np: np.ndarray) -> Tensor:
        x0 = Tensor(x_np[:, 0:1])
        y0 = Tensor(x_np[:, 1:2])
        t0 = Tensor(x_np[:, 2:3])
        z0 = Tensor(x_np[:, 3:4])
        ys = self.branch(y0, "y", softplus)
        zs = self.branch(z0, "z", sigmoid)
        ts = self.branch(t0, "t", sigmoid)
        h = add(linear(x0, self.params["x0.Wxx"].value), linear(y0, self.params["x0.Wxy"].value))
        h = add(h, linear(z0, self.params["x0.Wxz"].value))
        h = add(h, linear(t0, self.params["x0.Wxt"].value, self.params["x0.b"].value))
        h = softplus(h)
        for layer in range(1, self.depth):
            prefix = f"x{layer}"
            next_h = add(linear(h, self.params[f"{prefix}.Wxx"].value), linear(x0, self.params[f"{prefix}.Wxx0"].value))
            next_h = add(next_h, linear(ys[layer - 1], self.params[f"{prefix}.Wxy"].value))
            next_h = add(next_h, linear(zs[layer - 1], self.params[f"{prefix}.Wxz"].value))
            next_h = add(next_h, linear(ts[layer - 1], self.params[f"{prefix}.Wxt"].value, self.params[f"{prefix}.b"].value))
            h = next_h if layer == self.depth - 1 else softplus(next_h)
        return h

    def predict_scaled(self, x_np: np.ndarray) -> np.ndarray:
        x0, y0, t0, z0 = x_np[:, 0:1], x_np[:, 1:2], x_np[:, 2:3], x_np[:, 3:4]
        ys, zs, ts = [], [], []
        y, z, t = y0, z0, t0
        for layer in range(self.depth - 1):
            y = softplus_np(y @ self.w(f"y{layer}.W").T + self.w(f"y{layer}.b"))
            z = sigmoid_np(z @ self.w(f"z{layer}.W").T + self.w(f"z{layer}.b"))
            t = sigmoid_np(t @ self.w(f"t{layer}.W").T + self.w(f"t{layer}.b"))
            ys.append(y)
            zs.append(z)
            ts.append(t)
        h = (
            x0 @ self.w("x0.Wxx").T
            + y0 @ self.w("x0.Wxy").T
            + z0 @ self.w("x0.Wxz").T
            + t0 @ self.w("x0.Wxt").T
            + self.w("x0.b")
        )
        h = softplus_np(h)
        for layer in range(1, self.depth):
            prefix = f"x{layer}"
            h_next = (
                h @ self.w(f"{prefix}.Wxx").T
                + x0 @ self.w(f"{prefix}.Wxx0").T
                + ys[layer - 1] @ self.w(f"{prefix}.Wxy").T
                + zs[layer - 1] @ self.w(f"{prefix}.Wxz").T
                + ts[layer - 1] @ self.w(f"{prefix}.Wxt").T
                + self.w(f"{prefix}.b")
            )
            h = h_next if layer == self.depth - 1 else softplus_np(h_next)
        return h


def train_manual(
    model: ManualModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    y_mean: float,
    y_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ytr = (y_train - y_mean) / y_std
    yte = (y_test - y_mean) / y_std
    opt = Adam(model.parameters(), lr=lr)
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model.forward(x_train)
        diff = pred.data - ytr
        train_losses[epoch] = float(np.mean(diff * diff))
        pred.backward(2.0 * diff / diff.size)
        opt.step()
        test_pred = model.predict_scaled(x_test)
        test_losses[epoch] = float(np.mean((test_pred - yte) ** 2))
    return train_losses, test_losses


def predict_manual(model: ManualModel, x: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    return model.predict_scaled(x) * y_std + y_mean


def make_manual_model(name: str, seed: int) -> ManualModel:
    if name == "ISNN-1":
        return ISNN1Manual(seed)
    if name == "ISNN-2":
        return ISNN2Manual(seed)
    raise ValueError(name)


def train_all_manual(
    dataset: str,
    out_dir: Path,
    epochs: int,
    seeds: int,
    lr: float,
) -> Dict[str, Dict[str, object]]:
    x_train, y_train, x_test, y_test = make_dataset(dataset, out_dir / "datasets")
    # Use raw MSE like the paper figures. This is especially important for
    # ISNN-1 because the paper's quadratic output x_H^T x_H is nonnegative.
    y_mean = 0.0
    y_std = 1.0
    results: Dict[str, Dict[str, object]] = {}
    behavior_x = np.linspace(0.0, 6.0 if dataset == "additive" else 10.0, 240)
    behavior_in = np.column_stack([behavior_x, behavior_x, behavior_x, behavior_x])
    true_behavior = toy_additive(behavior_in) if dataset == "additive" else toy_multiplicative(behavior_in)

    for model_name in ["ISNN-1", "ISNN-2"]:
        train_runs = []
        test_runs = []
        behavior_runs = []
        for i in range(seeds):
            model = make_manual_model(model_name, seed=2026 + 100 * i + (0 if model_name == "ISNN-1" else 17))
            tr, te = train_manual(model, x_train, y_train, x_test, y_test, epochs, lr, y_mean, y_std)
            train_runs.append(tr)
            test_runs.append(te)
            behavior_runs.append(predict_manual(model, behavior_in, y_mean, y_std)[:, 0])
        train_arr = np.vstack(train_runs)
        test_arr = np.vstack(test_runs)
        behavior_arr = np.vstack(behavior_runs)
        results[model_name] = {
            "train_loss_runs": train_arr,
            "test_loss_runs": test_arr,
            "behavior_runs": behavior_arr,
            "final_train_mean": float(train_arr[:, -1].mean()),
            "final_test_mean": float(test_arr[:, -1].mean()),
        }

    behavior_payload = {
        "x": behavior_x,
        "true": true_behavior[:, 0],
        "split": 4.0,
    }
    save_loss_plot(out_dir / f"{dataset}_manual_loss.png", dataset, results, epochs, backend_label="manual NumPy")
    save_behavior_plot(out_dir / f"{dataset}_manual_behavior.png", dataset, results, behavior_payload, backend_label="manual NumPy")
    save_results_json(out_dir / f"{dataset}_manual_results.json", results, epochs, seeds, lr)
    save_loss_csv(out_dir / f"{dataset}_manual_losses.csv", results)
    return results


def train_all_torch(
    dataset: str,
    out_dir: Path,
    epochs: int,
    seeds: int,
    lr: float,
    device: str,
) -> Dict[str, Dict[str, object]]:
    """Train the PyTorch ISNNs and save the same artifacts as the manual path."""
    try:
        torch_ns: Dict[str, object] = {"__name__": "isnn_pytorch_reference_embedded"}
        exec(PYTORCH_CODE, torch_ns)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError("PyTorch is not available in this Python environment.") from exc
        raise

    import torch

    x_train, y_train, x_test, y_test = make_dataset(dataset, out_dir / "datasets")
    behavior_x = np.linspace(0.0, 6.0 if dataset == "additive" else 10.0, 240)
    behavior_in = np.column_stack([behavior_x, behavior_x, behavior_x, behavior_x])
    true_behavior = toy_additive(behavior_in) if dataset == "additive" else toy_multiplicative(behavior_in)

    build_torch_model = torch_ns["build_torch_model"]
    train_torch_model = torch_ns["train_torch_model"]
    results: Dict[str, Dict[str, object]] = {}

    for model_name in ["ISNN-1", "ISNN-2"]:
        train_runs = []
        test_runs = []
        behavior_runs = []
        for i in range(seeds):
            torch.manual_seed(2026 + 100 * i + (0 if model_name == "ISNN-1" else 17))
            model = build_torch_model(model_name)
            tr, te, y_mean, y_std = train_torch_model(
                model, x_train, y_train, x_test, y_test, epochs=epochs, lr=lr, device=device
            )
            train_runs.append(np.asarray(tr, dtype=np.float64))
            test_runs.append(np.asarray(te, dtype=np.float64))
            with torch.no_grad():
                xin = torch.as_tensor(behavior_in, dtype=torch.float32, device=device)
                pred = model.to(device)(xin).detach().cpu().numpy()[:, 0] * y_std + y_mean
            behavior_runs.append(pred)

        train_arr = np.vstack(train_runs)
        test_arr = np.vstack(test_runs)
        behavior_arr = np.vstack(behavior_runs)
        results[model_name] = {
            "train_loss_runs": train_arr,
            "test_loss_runs": test_arr,
            "behavior_runs": behavior_arr,
            "final_train_mean": float(train_arr[:, -1].mean()),
            "final_test_mean": float(test_arr[:, -1].mean()),
        }

    behavior_payload = {
        "x": behavior_x,
        "true": true_behavior[:, 0],
        "split": 4.0,
    }
    save_loss_plot(out_dir / f"{dataset}_torch_loss.png", dataset, results, epochs, backend_label="PyTorch")
    save_behavior_plot(out_dir / f"{dataset}_torch_behavior.png", dataset, results, behavior_payload, backend_label="PyTorch")
    save_results_json(out_dir / f"{dataset}_torch_results.json", results, epochs, seeds, lr)
    save_loss_csv(out_dir / f"{dataset}_torch_losses.csv", results)
    return results


def finite(v: float) -> float:
    return float(v) if np.isfinite(v) else 0.0


def save_results_json(path: Path, results: Dict[str, Dict[str, object]], epochs: int, seeds: int, lr: float) -> None:
    payload = {
        "epochs": epochs,
        "seeds": seeds,
        "learning_rate": lr,
        "models": {
            name: {
                "final_train_loss_mean": finite(res["final_train_mean"]),
                "final_test_loss_mean": finite(res["final_test_mean"]),
            }
            for name, res in results.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_loss_csv(path: Path, results: Dict[str, Dict[str, object]]) -> None:
    epochs = next(iter(results.values()))["train_loss_runs"].shape[1]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "model", "train_loss_mean", "train_loss_std", "test_loss_mean", "test_loss_std"])
        for model_name, res in results.items():
            tr = res["train_loss_runs"]
            te = res["test_loss_runs"]
            for e in range(epochs):
                w.writerow([e + 1, model_name, tr[:, e].mean(), tr[:, e].std(), te[:, e].mean(), te[:, e].std()])


def draw_axes(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], title: str, xlab: str, ylab: str) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(70, 70, 70), width=1)
    draw.text((x0, y0 - 24), title, fill=(20, 20, 20))
    draw.text(((x0 + x1) // 2 - 35, y1 + 10), xlab, fill=(20, 20, 20))
    draw.text((x0 - 54, (y0 + y1) // 2), ylab, fill=(20, 20, 20))


def map_points(xs: np.ndarray, ys: np.ndarray, box: Tuple[int, int, int, int], xr: Tuple[float, float], yr: Tuple[float, float]) -> List[Tuple[int, int]]:
    x0, y0, x1, y1 = box
    px = x0 + (xs - xr[0]) / (xr[1] - xr[0] + EPS) * (x1 - x0)
    py = y1 - (ys - yr[0]) / (yr[1] - yr[0] + EPS) * (y1 - y0)
    return list(zip(px.astype(int), py.astype(int)))


def draw_series(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    xs: np.ndarray,
    ys: np.ndarray,
    xr: Tuple[float, float],
    yr: Tuple[float, float],
    color: Tuple[int, int, int],
    width: int = 3,
) -> None:
    pts = map_points(xs, ys, box, xr, yr)
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width, joint="curve")


def draw_band(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    xs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    xr: Tuple[float, float],
    yr: Tuple[float, float],
    color: Tuple[int, int, int],
) -> None:
    upper_pts = map_points(xs, upper, box, xr, yr)
    lower_pts = map_points(xs, lower, box, xr, yr)
    fill = tuple(int(0.78 * 255 + 0.22 * c) for c in color)
    draw.polygon(upper_pts + list(reversed(lower_pts)), fill=fill)


def save_loss_plot(path: Path, dataset: str, results: Dict[str, Dict[str, object]], epochs: int, backend_label: str = "ISNN") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1300, 560), "white")
    draw = ImageDraw.Draw(img)
    colors = {"ISNN-1": (32, 99, 180), "ISNN-2": (200, 82, 40)}
    xs = np.arange(1, epochs + 1)
    train_box = (90, 80, 610, 455)
    test_box = (730, 80, 1250, 455)
    draw.text((30, 22), f"{dataset.title()} toy problem: {backend_label} training", fill=(10, 10, 10))

    train_all = np.concatenate([np.log10(res["train_loss_runs"].mean(axis=0) + EPS) for res in results.values()])
    test_all = np.concatenate([np.log10(res["test_loss_runs"].mean(axis=0) + EPS) for res in results.values()])
    for box, title, all_vals, key in [
        (train_box, "Training loss (log10 MSE)", train_all, "train_loss_runs"),
        (test_box, "Testing loss (log10 MSE)", test_all, "test_loss_runs"),
    ]:
        yr = (float(all_vals.min()) - 0.1, float(all_vals.max()) + 0.1)
        draw_axes(draw, box, title, "epoch", "log MSE")
        for model_name, res in results.items():
            runs = np.log10(res[key] + EPS)
            mean = runs.mean(axis=0)
            std = runs.std(axis=0)
            if runs.shape[0] > 1:
                draw_band(draw, box, xs, mean - std, mean + std, (1, epochs), yr, colors[model_name])
            draw_series(draw, box, xs, mean, (1, epochs), yr, colors[model_name])
        for i, (model_name, c) in enumerate(colors.items()):
            draw.line((box[0] + 20, box[1] + 24 + 24 * i, box[0] + 70, box[1] + 24 + 24 * i), fill=c, width=4)
            draw.text((box[0] + 78, box[1] + 14 + 24 * i), model_name, fill=(20, 20, 20))
    img.save(path)


def save_behavior_plot(
    path: Path,
    dataset: str,
    results: Dict[str, Dict[str, object]],
    behavior: Dict[str, np.ndarray],
    backend_label: str = "ISNN",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1300, 560), "white")
    draw = ImageDraw.Draw(img)
    colors = {"true": (20, 20, 20), "ISNN-1": (32, 99, 180), "ISNN-2": (200, 82, 40)}
    x = behavior["x"]
    true = behavior["true"]
    split = float(behavior["split"])
    all_y = [true]
    for res in results.values():
        runs = res["behavior_runs"]
        all_y.append(runs.mean(axis=0))
        if runs.shape[0] > 1:
            all_y.append(runs.mean(axis=0) - runs.std(axis=0))
            all_y.append(runs.mean(axis=0) + runs.std(axis=0))
    yy = np.concatenate(all_y)
    yr = (float(yy.min()) - 0.1 * (float(yy.max() - yy.min()) + EPS), float(yy.max()) + 0.1 * (float(yy.max() - yy.min()) + EPS))
    box = (90, 80, 1250, 455)
    draw.text((30, 22), f"{dataset.title()} {backend_label} response for x = y = t = z", fill=(10, 10, 10))
    draw_axes(draw, box, "Behavioral response", "input value", "model output")
    xr = (float(x.min()), float(x.max()))
    split_px = map_points(np.array([split]), np.array([yr[0]]), box, xr, yr)[0][0]
    draw.rectangle((box[0], box[1], split_px, box[3]), fill=(235, 248, 238))
    draw.rectangle((split_px, box[1], box[2], box[3]), fill=(250, 240, 235))
    draw.rectangle(box, outline=(70, 70, 70), width=1)
    draw.text((box[0] + 20, box[1] + 12), "interpolation", fill=(35, 100, 45))
    draw.text((split_px + 20, box[1] + 12), "extrapolation", fill=(130, 70, 40))
    draw_series(draw, box, x, true, xr, yr, colors["true"], width=4)
    for model_name, res in results.items():
        runs = res["behavior_runs"]
        mean = runs.mean(axis=0)
        if runs.shape[0] > 1:
            std = runs.std(axis=0)
            draw_band(draw, box, x, mean - std, mean + std, xr, yr, colors[model_name])
        draw_series(draw, box, x, mean, xr, yr, colors[model_name], width=3)
    legend_y = box[3] + 28
    for i, name in enumerate(["true", "ISNN-1", "ISNN-2"]):
        c = colors[name]
        draw.line((box[0] + 20 + 145 * i, legend_y, box[0] + 72 + 145 * i, legend_y), fill=c, width=4)
        draw.text((box[0] + 80 + 145 * i, legend_y - 10), name, fill=(20, 20, 20))
    img.save(path)


PYTORCH_CODE = r'''
# Optional PyTorch reference implementation. Paste into a Python environment
# with torch installed, or import this file and call build_torch_model(...).
import torch
from torch import nn
import torch.nn.functional as F


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, positive=False, bias=True):
        super().__init__()
        if positive:
            init = torch.randn(out_features, in_features) * 0.15 - 3.0
        else:
            init = torch.randn(out_features, in_features) * 0.25
        self.raw_weight = nn.Parameter(init)
        self.positive = positive
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        w = F.softplus(self.raw_weight) if self.positive else self.raw_weight
        return F.linear(x, w, self.bias)


class ISNN1Torch(nn.Module):
    def __init__(self, width=10):
        super().__init__()
        self.y = nn.ModuleList([PositiveLinear(1, width, True), PositiveLinear(width, width, True)])
        self.z = nn.ModuleList([PositiveLinear(1, width, False), PositiveLinear(width, width, False)])
        self.t = nn.ModuleList([PositiveLinear(1, width, True), PositiveLinear(width, width, True)])
        self.x0_x = PositiveLinear(1, width, False, False)
        self.x0_y = PositiveLinear(width, width, True, False)
        self.x0_z = PositiveLinear(width, width, False, False)
        self.x0_t = PositiveLinear(width, width, True, True)
        self.x1 = PositiveLinear(width, width, True)
        nn.init.constant_(self.x0_t.bias, -2.0)
        nn.init.constant_(self.x1.bias, -2.0)

    def forward(self, inp):
        x0, y0, t0, z0 = inp[:, 0:1], inp[:, 1:2], inp[:, 2:3], inp[:, 3:4]
        y = F.softplus(self.y[1](F.softplus(self.y[0](y0))))
        z = torch.sigmoid(self.z[1](torch.sigmoid(self.z[0](z0))))
        t = torch.sigmoid(self.t[1](torch.sigmoid(self.t[0](t0))))
        h = F.softplus(self.x0_x(x0) + self.x0_y(y) + self.x0_z(z) + self.x0_t(t))
        h = F.softplus(self.x1(h))
        return torch.sum(h * h, dim=1, keepdim=True)


class ISNN2Torch(nn.Module):
    def __init__(self, width=15, depth=3):
        super().__init__()
        if depth < 2:
            raise ValueError("ISNN-2 depth must be at least 2.")
        self.depth = depth
        self.y = nn.ModuleList()
        self.z = nn.ModuleList()
        self.t = nn.ModuleList()
        in_dim = 1
        for _ in range(depth - 1):
            self.y.append(PositiveLinear(in_dim, width, True))
            self.z.append(PositiveLinear(in_dim, width, False))
            self.t.append(PositiveLinear(in_dim, width, True))
            in_dim = width
        self.x0_x = PositiveLinear(1, width, False, False)
        self.x0_y = PositiveLinear(1, width, True, False)
        self.x0_z = PositiveLinear(1, width, False, False)
        self.x0_t = PositiveLinear(1, width, True, True)
        self.x_layers = nn.ModuleList()
        for layer in range(1, depth):
            out_dim = 1 if layer == depth - 1 else width
            self.x_layers.append(nn.ModuleDict({
                "xx": PositiveLinear(width, out_dim, True, False),
                "xx0": PositiveLinear(1, out_dim, False, False),
                "xy": PositiveLinear(width, out_dim, True, False),
                "xz": PositiveLinear(width, out_dim, False, False),
                "xt": PositiveLinear(width, out_dim, True, True),
            }))

    def forward(self, inp):
        x0, y0, t0, z0 = inp[:, 0:1], inp[:, 1:2], inp[:, 2:3], inp[:, 3:4]
        ys, zs, ts = [], [], []
        y, z, t = y0, z0, t0
        for layer in range(self.depth - 1):
            y = F.softplus(self.y[layer](y))
            z = torch.sigmoid(self.z[layer](z))
            t = torch.sigmoid(self.t[layer](t))
            ys.append(y)
            zs.append(z)
            ts.append(t)
        h = F.softplus(self.x0_x(x0) + self.x0_y(y0) + self.x0_z(z0) + self.x0_t(t0))
        for layer, block in enumerate(self.x_layers):
            h_next = block["xx"](h) + block["xx0"](x0) + block["xy"](ys[layer]) + block["xz"](zs[layer]) + block["xt"](ts[layer])
            h = h_next if layer == len(self.x_layers) - 1 else F.softplus(h_next)
        return h


def build_torch_model(name):
    return ISNN1Torch() if name == "ISNN-1" else ISNN2Torch()


def train_torch_model(model, x_train, y_train, x_test, y_test, epochs=1000, lr=1e-3, device="cpu"):
    model = model.to(device)
    xtr = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    xte = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    yte = torch.as_tensor(y_test, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((model(xtr) - ytr) ** 2)
        loss.backward()
        opt.step()
        train_losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            test_losses.append(float(torch.mean((model(xte) - yte) ** 2).cpu()))
    return train_losses, test_losses, 0.0, 1.0


def load_npz_pair(dataset_dir, name):
    import numpy as np
    train = np.load(f"{dataset_dir}/{name}_train.npz")
    test = np.load(f"{dataset_dir}/{name}_test.npz")
    return train["X"], train["y"], test["X"], test["y"]


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="datasets")
    ap.add_argument("--dataset", choices=["additive", "multiplicative"], default="additive")
    ap.add_argument("--model", choices=["ISNN-1", "ISNN-2"], default="ISNN-1")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    xtr, ytr, xte, yte = load_npz_pair(args.dataset_dir, args.dataset)
    tr, te, y_mean, y_std = train_torch_model(
        build_torch_model(args.model), xtr, ytr, xte, yte, args.epochs, args.lr
    )
    print(json.dumps({
        "model": args.model,
        "dataset": args.dataset,
        "final_train_loss": tr[-1],
        "final_test_loss": te[-1],
        "target_mean": y_mean,
        "target_std": y_std,
    }, indent=2))
'''


def write_torch_reference(path: Path) -> None:
    path.write_text(PYTORCH_CODE, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="isnn_submission", help="output directory")
    ap.add_argument("--epochs", type=int, default=900, help="epochs per seed")
    ap.add_argument("--seeds", type=int, default=3, help="number of random initializations")
    ap.add_argument("--lr", type=float, default=0.01, help="Adam learning rate")
    ap.add_argument("--dataset", choices=["additive", "multiplicative", "both"], default="both")
    ap.add_argument("--backend", choices=["manual", "torch", "both"], default="both")
    ap.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_torch_reference(out_dir / "isnn_pytorch_reference.py")

    datasets = ["additive", "multiplicative"] if args.dataset == "both" else [args.dataset]
    summary = {}
    for ds in datasets:
        summary[ds] = {}
        if args.backend in {"manual", "both"}:
            print(f"Training manual NumPy models for {ds} ({args.seeds} seeds, {args.epochs} epochs)...")
            summary[ds]["manual_numpy"] = train_all_manual(ds, out_dir, args.epochs, args.seeds, args.lr)
        if args.backend in {"torch", "both"}:
            print(f"Training PyTorch models for {ds} ({args.seeds} seeds, {args.epochs} epochs)...")
            try:
                summary[ds]["pytorch"] = train_all_torch(ds, out_dir, args.epochs, args.seeds, args.lr, args.device)
            except RuntimeError as exc:
                if args.backend == "torch":
                    raise
                print(f"Skipping PyTorch backend: {exc}")

    compact = {}
    for ds, backend_values in summary.items():
        compact[ds] = {}
        for backend_name, vals in backend_values.items():
            compact[ds][backend_name] = {
                model: {
                    "final_train_loss_mean": float(res["final_train_mean"]),
                    "final_test_loss_mean": float(res["final_test_mean"]),
                }
                for model, res in vals.items()
            }
    (out_dir / "summary.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print(json.dumps(compact, indent=2))
    print(f"Done. Outputs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
