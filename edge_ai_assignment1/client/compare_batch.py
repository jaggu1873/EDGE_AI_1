"""Compare n individual predictions with one batch prediction against a running API."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import requests


def reading(rng):
    return {
        "battery_v": round(float(rng.uniform(3.4, 4.2)), 3),
        "temp_c": round(float(rng.uniform(10, 45)), 1),
        "rings_per_day": int(rng.integers(0, 30)),
        "wifi_rssi_dbm": int(rng.integers(-88, -45)),
        "age_months": int(rng.integers(0, 37)),
    }


def post(session, url, payload):
    response = session.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.n < 1 or args.n > 10000:
        parser.error("--n must be between 1 and 10000")

    base_url = args.url.rstrip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    readings = [reading(rng) for _ in range(args.n)]
    session = requests.Session()
    health_response = session.get(f"{base_url}/health", timeout=60)
    health_response.raise_for_status()
    health = health_response.json()
    if health.get("team") != 18:
        raise RuntimeError(f"Expected team 18 API, got {health}")
    print(f"health: {health}")

    single_rows = []
    single_predictions = []
    single_start = time.perf_counter()
    for index, row in enumerate(readings):
        started = time.perf_counter()
        result = post(session, f"{base_url}/predict", row)
        round_trip = (time.perf_counter() - started) * 1000
        timings = result["timings_ms"]
        single_predictions.append(result["days_left"])
        single_rows.append({
            "kind": "individual_predict",
            "index": index,
            "round_trip_ms": round_trip,
            "parse_ms": timings["parse"],
            "normalise_ms": timings["normalise"],
            "inference_ms": timings["inference"],
            "build_ms": timings["build"],
            "server_total_ms": timings["server_total"],
            "network_framework_ms": round_trip - timings["server_total"],
            "days_left": result["days_left"],
        })
    single_elapsed = (time.perf_counter() - single_start) * 1000

    batch_start = time.perf_counter()
    batch_result = post(session, f"{base_url}/predict_batch", {"rows": readings})
    batch_elapsed = (time.perf_counter() - batch_start) * 1000
    if len(batch_result["days_left"]) != args.n:
        raise RuntimeError(
            f"Batch returned {len(batch_result['days_left'])} predictions for {args.n} rows"
        )
    differences = np.abs(
        np.asarray(single_predictions, dtype=np.float64)
        - np.asarray(batch_result["days_left"], dtype=np.float64)
    )
    batch_timings = batch_result["timings_ms"]
    batch_rows = [{
        "kind": "single_batch_request",
        "index": "",
        "round_trip_ms": batch_elapsed,
        "parse_ms": batch_timings["parse"],
        "normalise_ms": batch_timings["normalise"],
        "inference_ms": batch_timings["inference"],
        "build_ms": batch_timings["build"],
        "server_total_ms": batch_timings["server_total"],
        "network_framework_ms": batch_elapsed - batch_timings["server_total"],
        "days_left": "",
    }]
    columns = [
        "kind", "index", "round_trip_ms", "parse_ms", "normalise_ms",
        "inference_ms", "build_ms", "server_total_ms", "network_framework_ms",
        "days_left",
    ]
    with (args.output_dir / "batch_comparison_requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(single_rows + batch_rows)

    single_server_total = sum(row["server_total_ms"] for row in single_rows)
    single_network_framework = single_elapsed - single_server_total
    summary = [
        {
            "mode": "individual_predict",
            "request_count": args.n,
            "elapsed_ms": single_elapsed,
            "server_total_sum_ms": single_server_total,
            "network_framework_ms": single_network_framework,
            "average_round_trip_ms": single_elapsed / args.n,
            "max_abs_prediction_difference_days": float(differences.max()),
            "mean_abs_prediction_difference_days": float(differences.mean()),
        },
        {
            "mode": "one_predict_batch",
            "request_count": 1,
            "elapsed_ms": batch_elapsed,
            "server_total_sum_ms": batch_timings["server_total"],
            "network_framework_ms": batch_elapsed - batch_timings["server_total"],
            "average_round_trip_ms": batch_elapsed,
            "max_abs_prediction_difference_days": float(differences.max()),
            "mean_abs_prediction_difference_days": float(differences.mean()),
        },
    ]
    with (args.output_dir / "batch_comparison_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"{args.n} individual requests: {single_elapsed:.3f} ms total "
          f"({single_server_total:.3f} ms reported server time, "
          f"{single_network_framework:.3f} ms client/framework remainder)")
    print(f"One batch request: {batch_elapsed:.3f} ms "
          f"({batch_timings['server_total']:.3f} ms reported server time, "
          f"{batch_elapsed - batch_timings['server_total']:.3f} ms client/framework remainder)")
    print(f"Batch speedup by elapsed time: {single_elapsed / batch_elapsed:.2f}x")
    print(f"Prediction difference: max={differences.max():.6f} days, "
          f"mean={differences.mean():.6f} days")
    print(f"Saved request timings and summary under {args.output_dir}")


if __name__ == "__main__":
    main()
