"""Reproduce the team 18 data, baseline, architecture, and memory analyses."""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet


TEAM = 18
TARGET = "days_left"
LIMITS = {
    "battery_v": (2.5, 4.5),
    "temp_c": (-40, 85),
    "rings_per_day": (0, 100),
    "wifi_rssi_dbm": (-100, -20),
    "age_months": (0, 120),
}
WIDTHS = (4, 8, 16, 32, 64)
EPOCHS_DEFAULT = 1500


def physical_glitches(frame):
    bad_by_feature = {
        feature: (frame[feature] < low) | (frame[feature] > high)
        for feature, (low, high) in LIMITS.items()
    }
    return bad_by_feature, pd.DataFrame(bad_by_feature).any(axis=1)


def mae(predictions, truth):
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(truth))))


def save_csv(path, rows, columns=None):
    if not rows:
        raise ValueError(f"No data to write to {path}")
    if columns is None:
        columns = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def train_network(hidden, x_train, y_train, x_val, y_val, mu, sd, epochs, normalize):
    train_x = (x_train - mu) / sd if normalize else x_train
    val_x = (x_val - mu) / sd if normalize else x_val
    torch.manual_seed(0)
    net = BatteryNet(hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    train_tensor = torch.tensor(train_x, dtype=torch.float32)
    target_tensor = torch.tensor(y_train, dtype=torch.float32)
    val_tensor = torch.tensor(val_x, dtype=torch.float32)
    start = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(len(train_tensor))
        for offset in range(0, len(train_tensor), 128):
            batch = order[offset:offset + 128]
            loss = torch.nn.functional.mse_loss(
                net(train_tensor[batch]), target_tensor[batch]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        train_predictions = net(train_tensor).numpy()
        val_predictions = net(val_tensor).numpy()
    if not np.isfinite(train_predictions).all() or not np.isfinite(val_predictions).all():
        raise RuntimeError(f"Non-finite predictions for hidden={hidden}, normalize={normalize}")
    return {
        "hidden": hidden,
        "train_mae": mae(train_predictions, y_train),
        "val_mae": mae(val_predictions, y_val),
        "params": sum(parameter.numel() for parameter in net.parameters()),
        "seconds": elapsed,
    }


def params_and_macs(width):
    params = width * width + 8 * width + 1
    macs = width * width + 6 * width
    return params, macs


def memory_table():
    rows = []
    for width in (32, 128, 256, 512, 2048):
        params, macs = params_and_macs(width)
        fp32_bytes = params * 4
        int8_bytes = params
        activation_bytes = 2 * width * 4
        rows.append({
            "hidden_width": width,
            "parameters": params,
            "fp32_weight_bytes": fp32_bytes,
            "int8_weight_bytes": int8_bytes,
            "macs_per_prediction": macs,
            "fp32_activation_bytes_batch1_estimate": activation_bytes,
            "fp32_weights_fit_300kb_sram": fp32_bytes <= 300 * 1024,
            "int8_weights_fit_300kb_sram": int8_bytes <= 300 * 1024,
            "fp32_weights_plus_activation_fit_300kb": fp32_bytes + activation_bytes <= 300 * 1024,
            "int8_weights_plus_fp32_activation_fit_300kb": int8_bytes + activation_bytes <= 300 * 1024,
            "fp32_weights_fit_8mb_psram": fp32_bytes <= 8 * 1024 * 1024,
            "int8_weights_fit_8mb_psram": int8_bytes <= 8 * 1024 * 1024,
        })
    return rows


def max_width_for_budget(byte_budget, bytes_per_parameter, include_activation=False):
    width = 0
    while True:
        next_width = width + 1
        params, _ = params_and_macs(next_width)
        required = params * bytes_per_parameter
        if include_activation:
            required += 2 * next_width * 4
        if required > byte_budget:
            return width
        width = next_width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    torch.set_num_threads(1)
    data_path = Path("data") / f"team_{TEAM:02d}_train.csv"
    model_dir = Path("model")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path)
    expected_columns = FEATURES + [TARGET]
    if list(raw.columns) != expected_columns:
        raise ValueError(f"Unexpected columns in {data_path}: {list(raw.columns)}")
    bad_by_feature, bad_rows = physical_glitches(raw)
    clean = raw.loc[~bad_rows].copy()
    clean.insert(0, "csv_row", clean.index + 2)

    glitch_counts = [
        {"range_rule": feature, "rows_out_of_range": int(mask.sum())}
        for feature, mask in bad_by_feature.items()
    ]
    glitch_counts.append({
        "range_rule": "unique_rows_removed",
        "rows_out_of_range": int(bad_rows.sum()),
    })
    save_csv(output_dir / "glitch_counts.csv", glitch_counts)

    histogram, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, column in zip(axes.flat, expected_columns):
        axis.hist(raw[column], bins=35, color="#4472C4", edgecolor="white")
        axis.set_title(column)
        axis.set_ylabel("Rows")
    axes.flat[-1].axis("off")
    histogram.suptitle(f"Team {TEAM}: raw data histograms (includes glitches)")
    histogram.tight_layout()
    histogram.savefig(output_dir / "column_histograms.png", dpi=160)
    plt.close(histogram)

    scatter, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, feature in zip(axes.flat, FEATURES):
        axis.scatter(clean[feature], clean[TARGET], s=9, alpha=0.3, color="#4472C4")
        axis.set_xlabel(feature)
        axis.set_ylabel(TARGET)
        axis.set_title(f"{feature} vs {TARGET}")
    axes.flat[-1].axis("off")
    scatter.suptitle(f"Team {TEAM}: clean readings ({len(clean)} rows)")
    scatter.tight_layout()
    scatter.savefig(output_dir / "features_vs_days_left.png", dpi=160)
    plt.close(scatter)

    rng = np.random.default_rng(0)
    permutation = rng.permutation(len(clean))
    val_count = len(clean) // 5
    val_positions, train_positions = permutation[:val_count], permutation[val_count:]
    train_frame, val_frame = clean.iloc[train_positions], clean.iloc[val_positions]
    x_train = train_frame[FEATURES].to_numpy(dtype=np.float32)
    y_train = train_frame[TARGET].to_numpy(dtype=np.float32)
    x_val = val_frame[FEATURES].to_numpy(dtype=np.float32)
    y_val = val_frame[TARGET].to_numpy(dtype=np.float32)
    mu = x_train.mean(axis=0, dtype=np.float32)
    sd = x_train.std(axis=0, dtype=np.float32)
    sd = np.where(sd < 1e-6, np.ones_like(sd), sd).astype(np.float32)

    mean_prediction = np.full_like(y_val, y_train.mean())
    linear_design = np.column_stack((x_train, np.ones(len(x_train), dtype=np.float32)))
    linear_val_design = np.column_stack((x_val, np.ones(len(x_val), dtype=np.float32)))
    linear_weights, _, _, _ = np.linalg.lstsq(linear_design, y_train, rcond=None)
    linear_prediction = linear_val_design @ linear_weights

    with (model_dir / "config.json").open() as stream:
        config = json.load(stream)
    if config["team"] != TEAM or config["features"] != FEATURES:
        raise ValueError("Existing model config does not match protected team 18/features")
    existing_net = BatteryNet(tuple(config["hidden"]))
    existing_net.load_state_dict(
        torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    existing_net.eval()
    normalized_x_train = torch.tensor((x_train - mu) / sd, dtype=torch.float32)
    normalized_x_val = torch.tensor((x_val - mu) / sd, dtype=torch.float32)
    with torch.no_grad():
        existing_train_prediction = existing_net(normalized_x_train).numpy()
        existing_val_prediction = existing_net(normalized_x_val).numpy()
    existing_train_mae = mae(existing_train_prediction, y_train)
    existing_val_mae = mae(existing_val_prediction, y_val)

    summary_rows = [
        {
            "model": "mean_baseline",
            "hidden": "",
            "params": 0,
            "train_mae_days": mae(np.full_like(y_train, y_train.mean()), y_train),
            "val_mae_days": mae(mean_prediction, y_val),
            "normalization": "not_applicable",
        },
        {
            "model": "linear_lstsq",
            "hidden": "",
            "params": len(FEATURES) + 1,
            "train_mae_days": mae(
                np.column_stack((x_train, np.ones(len(x_train)))) @ linear_weights, y_train
            ),
            "val_mae_days": mae(linear_prediction, y_val),
            "normalization": "not_required",
        },
        {
            "model": "existing_saved_network",
            "hidden": ",".join(map(str, config["hidden"])),
            "params": sum(parameter.numel() for parameter in existing_net.parameters()),
            "train_mae_days": existing_train_mae,
            "val_mae_days": existing_val_mae,
            "normalization": "training_split_statistics",
        },
    ]
    save_csv(output_dir / "mission1_baselines_and_saved_model.csv", summary_rows)

    prediction_plot, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    prediction_sets = [
        ("Mean baseline", mean_prediction),
        ("Linear least squares", linear_prediction),
        ("Existing saved network", existing_val_prediction),
    ]
    for axis, (label, predictions) in zip(axes, prediction_sets):
        axis.scatter(y_val, predictions, s=12, alpha=0.45, color="#4472C4")
        low = min(float(y_val.min()), float(predictions.min()))
        high = max(float(y_val.max()), float(predictions.max()))
        axis.plot([low, high], [low, high], "--", color="#C00000")
        axis.set_xlabel("True days_left")
        axis.set_ylabel("Predicted days_left")
        axis.set_title(f"{label}\nvalidation MAE {mae(predictions, y_val):.2f} d")
    prediction_plot.tight_layout()
    prediction_plot.savefig(output_dir / "validation_predicted_vs_true.png", dpi=160)
    plt.close(prediction_plot)

    error_rows = []
    for label, predictions in prediction_sets:
        absolute_errors = np.abs(predictions - y_val)
        largest = np.argsort(absolute_errors)[-10:][::-1]
        for rank, position in enumerate(largest, 1):
            source_row = val_frame.iloc[position]
            record = {
                "model": label,
                "rank": rank,
                "csv_row": int(source_row["csv_row"]),
                "days_left_true": float(y_val[position]),
                "days_left_predicted": float(predictions[position]),
                "absolute_error_days": float(absolute_errors[position]),
            }
            record.update({feature: float(source_row[feature]) for feature in FEATURES})
            error_rows.append(record)
    save_csv(output_dir / "largest_validation_errors.csv", error_rows)

    architecture_rows = []
    normalized_32x32 = None
    for layer_count in (1, 2):
        for width in WIDTHS:
            hidden = (width,) * layer_count
            result = train_network(
                hidden, x_train, y_train, x_val, y_val, mu, sd, args.epochs, True
            )
            row = {
                "hidden": ",".join(map(str, hidden)),
                "hidden_layers": layer_count,
                "width": width,
                "parameters": result["params"],
                "train_mae_days": result["train_mae"],
                "val_mae_days": result["val_mae"],
                "train_val_gap_days": result["val_mae"] - result["train_mae"],
                "normalized": True,
                "epochs": args.epochs,
                "training_seconds": result["seconds"],
            }
            architecture_rows.append(row)
            print(
                f"trained hidden={row['hidden']}: train MAE {row['train_mae_days']:.3f}, "
                f"val MAE {row['val_mae_days']:.3f}, params {row['parameters']}, "
                f"time {row['training_seconds']:.1f}s"
            )
            if hidden == (32, 32):
                normalized_32x32 = row
    save_csv(output_dir / "architecture_experiments.csv", architecture_rows)

    unnormalized = train_network(
        (32, 32), x_train, y_train, x_val, y_val, mu, sd, args.epochs, False
    )
    normalization_rows = [
        {
            "hidden": "32,32",
            "normalization": "training_mean_std",
            "train_mae_days": normalized_32x32["train_mae_days"],
            "val_mae_days": normalized_32x32["val_mae_days"],
            "parameters": normalized_32x32["parameters"],
            "epochs": args.epochs,
        },
        {
            "hidden": "32,32",
            "normalization": "none",
            "train_mae_days": unnormalized["train_mae"],
            "val_mae_days": unnormalized["val_mae"],
            "parameters": unnormalized["params"],
            "epochs": args.epochs,
        },
    ]
    save_csv(output_dir / "normalization_comparison.csv", normalization_rows)

    architecture_plot, axis = plt.subplots(figsize=(8, 5))
    for layer_count, marker in ((1, "o"), (2, "s")):
        rows = [row for row in architecture_rows if row["hidden_layers"] == layer_count]
        axis.plot(
            [row["parameters"] for row in rows],
            [row["val_mae_days"] for row in rows],
            marker=marker,
            label=f"{layer_count} hidden layer{'s' if layer_count > 1 else ''}",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Trainable parameters (log scale)")
    axis.set_ylabel("Validation MAE (days)")
    axis.set_title(f"Team {TEAM}: architecture validation results")
    axis.legend()
    axis.grid(True, alpha=0.25)
    architecture_plot.tight_layout()
    architecture_plot.savefig(output_dir / "validation_mae_vs_parameters.png", dpi=160)
    plt.close(architecture_plot)

    normalization_plot, axis = plt.subplots(figsize=(7, 5))
    positions = np.arange(2)
    axis.bar(positions - 0.18, [row["train_mae_days"] for row in normalization_rows],
             width=0.36, label="Train MAE")
    axis.bar(positions + 0.18, [row["val_mae_days"] for row in normalization_rows],
             width=0.36, label="Validation MAE")
    axis.set_xticks(positions, ["Training-stat\nnormalization", "No normalization"])
    axis.set_ylabel("MAE (days)")
    axis.set_title("32-32 network: normalization comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    normalization_plot.tight_layout()
    normalization_plot.savefig(output_dir / "normalization_comparison.png", dpi=160)
    plt.close(normalization_plot)

    memory_rows = memory_table()
    save_csv(output_dir / "mission4_memory_table.csv", memory_rows)
    byte_budget = 300 * 1024
    f32_max_width = max_width_for_budget(byte_budget, 4)
    int8_max_width = max_width_for_budget(byte_budget, 1)
    f32_combined_max_width = max_width_for_budget(byte_budget, 4, include_activation=True)
    int8_combined_max_width = max_width_for_budget(byte_budget, 1, include_activation=True)
    current_params, current_macs = params_and_macs(32)
    clock_hz = 240_000_000
    ideal_seconds = current_macs / clock_hz
    report_rows = [
        {"metric": "sram_budget_bytes", "value": byte_budget, "assumption": "300 KiB"},
        {"metric": "psram_budget_bytes", "value": 8 * 1024 * 1024, "assumption": "8 MiB"},
        {"metric": "max_width_fp32_weights_only", "value": f32_max_width, "assumption": "weights <= 300 KiB"},
        {"metric": "max_width_int8_weights_only", "value": int8_max_width, "assumption": "weights <= 300 KiB"},
        {"metric": "max_width_fp32_plus_two_fp32_activation_buffers", "value": f32_combined_max_width, "assumption": "weights and conservative activation estimate <= 300 KiB"},
        {"metric": "max_width_int8_weights_plus_fp32_activation_buffers", "value": int8_combined_max_width, "assumption": "INT8 weights plus conservative FP32 activations <= 300 KiB"},
        {"metric": "selected_model_parameters", "value": current_params, "assumption": "5-32-32-1 including biases"},
        {"metric": "selected_model_macs", "value": current_macs, "assumption": "5-32-32-1 per prediction"},
        {"metric": "esp32s3_clock_hz", "value": clock_hz, "assumption": "Espressif product specification"},
        {"metric": "ideal_one_mac_per_cycle_seconds", "value": ideal_seconds, "assumption": "compute-only lower bound; no runtime or memory overhead"},
        {"metric": "ideal_one_mac_per_cycle_microseconds", "value": ideal_seconds * 1_000_000, "assumption": "compute-only lower bound; not a measured board result"},
        {"metric": "selected_activation_estimate_bytes_batch1", "value": 2 * 32 * 4, "assumption": "two FP32 hidden-width activation buffers"},
    ]
    save_csv(output_dir / "mission4_calculations.csv", report_rows)

    print(f"Team {TEAM}: {len(raw)} rows; {int(bad_rows.sum())} physical-range glitches removed; "
          f"{len(clean)} clean rows")
    print(f"Split: {len(train_positions)} train / {len(val_positions)} validation")
    print(f"Mean baseline validation MAE: {summary_rows[0]['val_mae_days']:.3f} days")
    print(f"Linear lstsq validation MAE: {summary_rows[1]['val_mae_days']:.3f} days")
    print(f"Existing model train/validation MAE: {existing_train_mae:.3f} / {existing_val_mae:.3f} days")
    print(f"Normalization comparison: {normalization_rows}")
    print(f"Max 300 KiB weight-only width: FP32={f32_max_width}, INT8={int8_max_width}")
    print(f"Selected model ideal ESP32-S3 MAC floor: {ideal_seconds * 1_000_000:.3f} us "
          "(not a measured board result)")
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
