
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
