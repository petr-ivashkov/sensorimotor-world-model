"""Physical-quantity targets and frozen-embedding probes used in the paper."""

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# (quantity, candidate HDF5 columns, coordinate indices, encode angles as sin/cos)
TARGETS = {
    "tworoom": [("agent position", ("pos_agent", "proprio", "observation"), [0, 1], False)],
    "reacher": [("joint angles", ("qpos", "observation"), [0, 1], True)],
    "pusht": [
        ("agent position", ("state", "proprio"), [0, 1], False),
        ("block position", ("state",), [2, 3], False),
        ("block orientation", ("state",), [4], True),
    ],
    "ogbcube": [
        ("cube position", ("privileged_block_0_pos",), [0, 1, 2], False),
        ("gripper position", ("proprio_effector_pos",), [0, 1, 2], False),
        ("gripper opening", ("proprio_gripper_opening",), [0], False),
    ],
}


def load_split(directory, split):
    payload = torch.load(Path(directory) / f"{split}_embeddings.pt", map_location="cpu", weights_only=False)
    z = np.asarray(payload["embeddings"], dtype=np.float32)
    arrays = {key: np.asarray(value) for key, value in payload["arrays"].items()}
    return z, arrays


def physical_targets(env, arrays):
    targets = {}
    for quantity, candidates, dims, angles in TARGETS[env]:
        key = next((key for key in candidates if key in arrays), None)
        if key is None:
            raise KeyError(f"{env}: missing physical quantity {quantity}; expected one of {candidates}")
        values = np.asarray(arrays[key], dtype=np.float32).reshape(len(arrays[key]), -1)[:, dims]
        if angles:
            values = np.stack([np.sin(values), np.cos(values)], axis=-1).reshape(len(values), -1)
        targets[quantity] = values
    return targets


def fit_probe(z_train, y_train, z_eval, y_eval, kind="linear"):
    train_mask = np.isfinite(z_train).all(axis=1) & np.isfinite(y_train).all(axis=1)
    eval_mask = np.isfinite(z_eval).all(axis=1) & np.isfinite(y_eval).all(axis=1)
    if train_mask.sum() < 20 or eval_mask.sum() < 2:
        raise ValueError("Not enough finite train/evaluation samples for a probe.")
    x_train, y_train = z_train[train_mask], y_train[train_mask]
    x_eval, y_eval = z_eval[eval_mask], y_eval[eval_mask]
    if kind == "linear":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1e-3))
        model.fit(x_train, y_train)
        prediction = model.predict(x_eval)
    elif kind == "mlp":
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        x_train = x_scaler.fit_transform(x_train)
        x_eval = x_scaler.transform(x_eval)
        scaled_y = y_scaler.fit_transform(y_train)
        model = MLPRegressor(
            hidden_layer_sizes=(256, 128), activation="relu", alpha=1e-4,
            batch_size=512, learning_rate_init=1e-3, max_iter=200,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=12,
            random_state=0, verbose=False,
        )
        model.fit(x_train, scaled_y)
        prediction = y_scaler.inverse_transform(model.predict(x_eval).reshape(len(x_eval), -1))
    else:
        raise ValueError(f"Unknown probe: {kind}")
    return float(r2_score(y_eval, prediction, multioutput="uniform_average"))
