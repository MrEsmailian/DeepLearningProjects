from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import torch
@dataclass
class Q2Data:
    X_train: torch.Tensor; y_train: torch.Tensor
    X_val: torch.Tensor;   y_val: torch.Tensor
    X_test: torch.Tensor;  y_test: torch.Tensor

def _rot2d(theta_deg: float):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,-s],[s,c]], dtype=np.float32)

def make_q2_crescents(
    n_total=2000,
    train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
    student_run_seed=43,
    # geometry knobs (tweak if you want even longer tails/overlap)
    tail=0.35,          # how much we extend beyond a half-circle (0 = semicircle)
    dx=0.45, dy=-0.28,  # shift of bottom arc (right & down)
    k=0.75,             # vertical scale for bottom arc (squashes it a bit)
    rot_deg=10.0,       # small rotation of the whole set
    noise_sigma=0.045   # Gaussian noise
):
    assert abs(train_ratio + val_ratio - 0.9) < 1e-6 and abs(test_ratio - 0.1) < 1e-6
    gen = np.random.default_rng(123456)  # fixed internal seed

    n = n_total // 2
    # Extend parameter t beyond [0, π] to create long tails on both arcs
    t_top    = np.linspace(-tail*np.pi, (1.0+tail)*np.pi, n)
    t_bottom = np.linspace( 0.0+tail*np.pi*0.15, (1.0+tail*1.3)*np.pi, n)

    # Top arc (inverted U)
    x_top = np.stack([ np.cos(t_top) - 0.25,
                       np.sin(t_top) + 0.02 ], axis=1)

    # Bottom arc (upright U) shifted/reshaped so tails pass under the top one
    x_bot = np.stack([ np.cos(t_bottom) + 0.25,
                      -k*np.sin(t_bottom) - 0.30 ], axis=1)
    x_bot[:,0] += dx
    x_bot[:,1] += dy

    X = np.concatenate([x_top, x_bot]).astype(np.float32)
    y = np.concatenate([np.ones(n, np.int64), np.zeros(n, np.int64)])

    # Subtle global rotation to avoid accidental linear gaps
    R = _rot2d(rot_deg)
    X = (X @ R.T).astype(np.float32)

    # Add mild noise to create realistic overlap at the tails
    X += gen.normal(0.0, noise_sigma, size=X.shape).astype(np.float32)

    # Shuffle & split with student-controlled seed
    rng = np.random.default_rng(student_run_seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n_tr = int(train_ratio*len(X)); n_va = int(val_ratio*len(X))
    X_tr, X_va, X_te = X[:n_tr], X[n_tr:n_tr+n_va], X[n_tr+n_va:]
    y_tr, y_va, y_te = y[:n_tr], y[n_tr:n_tr+n_va], y[n_tr+n_va:]

    return Q2Data(
        X_train=torch.from_numpy(X_tr),
        y_train=torch.from_numpy(y_tr),
        X_val=torch.from_numpy(X_va),
        y_val=torch.from_numpy(y_va),
        X_test=torch.from_numpy(X_te),
        y_test=torch.from_numpy(y_te),
    )

def make_q2_spirals(
    n_total=2000,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    student_run_seed=43,
):
    gen = np.random.default_rng(123456)  # fixed internal seed

    # ---- Generate two interleaved spirals (non-linear) ----
    n_per_class = n_total // 2
    theta = np.sqrt(gen.random(n_per_class)) * 4 * np.pi  # spiral angle range
    r = theta
    # Spiral 1
    x1 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    # Spiral 2 (shifted by π)
    x2 = np.stack([-r * np.cos(theta), -r * np.sin(theta)], axis=1)
    X = np.concatenate([x1, x2]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class, dtype=np.int64),
                        np.ones(n_per_class, dtype=np.int64)])

    # Add Gaussian noise
    X += gen.normal(0, 0.15, X.shape).astype(np.float32)

    # ---- Shuffle and split ----
    rng = np.random.default_rng(student_run_seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_train = int(train_ratio * len(X))
    n_val = int(val_ratio * len(X))
    n_test = len(X) - n_train - n_val

    X_train, X_val, X_test = X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train + n_val], y[n_train + n_val:]

    return Q2Data(
        X_train=torch.from_numpy(X_train),
        y_train=torch.from_numpy(y_train),
        X_val=torch.from_numpy(X_val),
        y_val=torch.from_numpy(y_val),
        X_test=torch.from_numpy(X_test),
        y_test=torch.from_numpy(y_test),
    )

# data = make_q2_crescents(student_run_seed=1013)
# X = torch.cat([data.X_train,data.X_val,data.X_test])
# y = torch.cat([data.y_train,data.y_val,data.y_test])
# plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", s=20, edgecolor="k", linewidth=0.3)
# plt.title("HW1-Q2 Non-linear Dataset"); plt.xlabel("x₁"); plt.ylabel("x₂"); plt.show()
#
#
# data = make_q2_crescents(student_run_seed=1013)
# X = torch.cat([data.X_train,data.X_val,data.X_test])
# y = torch.cat([data.y_train,data.y_val,data.y_test])
# plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", s=20, edgecolor="k", linewidth=0.3)
# plt.title("HW1-Q2 Non-linear Dataset"); plt.xlabel("x₁"); plt.ylabel("x₂"); plt.show()