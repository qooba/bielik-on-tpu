#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.common.bench import device_name, do_bench, pick_block_size, tflops
from benchmarks.common.plotting import COLORS, plot_summary_comparison
from kernels.normalization.rms_norm import rms_norm_pallas, rms_norm_ref

EPS = 1e-6
BLOCK_ROWS = 128

PROVIDERS = [
    ("pallas", "Pallas (fused)"),
    ("jax_native", "JAX Native (jit-fused)"),
]


def _provider_fn(provider: str, x: jnp.ndarray, w: jnp.ndarray, block_rows: int):
    if provider == "pallas":
        return lambda: rms_norm_pallas(x, w, eps=EPS, block_rows=block_rows)
    else:
        jitted = jax.jit(lambda x, w: rms_norm_ref(x, w, eps=EPS))
        jitted(x, w)
        return lambda: jitted(x, w)


def _rms_norm_gbps(ms: float, n_rows: int, hidden_size: int, itemsize: int) -> float:
    total_bytes = (2 * n_rows * hidden_size + hidden_size) * itemsize
    return total_bytes / (ms * 1e6)


def _run_sweep(
    x_name: str,
    x_vals: list,
    fixed: dict,
    plot_name: str,
    save_path: str,
    xlabel: str | None = None,
):
    print(f"--- Sweep: vary {x_name} ({', '.join(f'{k}={v}' for k, v in fixed.items())}) ---")

    results = {key: {"median": [], "low": [], "high": []} for key, _ in PROVIDERS}

    for x_val in x_vals:
        n_rows = x_val if x_name == "n_rows" else fixed["n_rows"]
        hidden_size = x_val if x_name == "hidden_size" else fixed["hidden_size"]
        dtype = fixed["dtype"]

        key_x, key_w = jax.random.split(jax.random.PRNGKey(0))
        x = jax.random.normal(key_x, (n_rows, hidden_size), dtype=dtype)
        w = jax.random.normal(key_w, (hidden_size,), dtype=dtype)
        block_rows = pick_block_size(n_rows, BLOCK_ROWS)

        row = [f"{x_name}={x_val:<6}"]
        for key, label in PROVIDERS:
            fn = _provider_fn(key, x, w, block_rows)
            median, low, high = do_bench(fn)
            results[key]["median"].append(_rms_norm_gbps(median, n_rows, hidden_size, x.itemsize))
            results[key]["low"].append(_rms_norm_gbps(high, n_rows, hidden_size, x.itemsize))
            results[key]["high"].append(_rms_norm_gbps(low, n_rows, hidden_size, x.itemsize))
            row.append(f"{label}={results[key]['median'][-1]:.1f} GB/s")
        print("  " + "  ".join(row))

    if save_path:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for key, label in PROVIDERS:
            ax.plot(x_vals, results[key]["median"], label=label, color=COLORS[key], marker="o")
            ax.fill_between(
                x_vals, results[key]["low"], results[key]["high"], color=COLORS[key], alpha=0.15
            )
        ax.set_xlabel(xlabel or x_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Bandwidth (GB/s)", fontsize=12, fontweight="bold")
        ax.set_title(f"{plot_name}\nDevice: {device_name()}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        plot_file = Path(save_path) / f"{plot_name}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot to: {plot_file}")

        csv_file = Path(save_path) / f"{plot_name}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = [x_name]
            for _, label in PROVIDERS:
                header += [f"{label} median GB/s", f"{label} low GB/s", f"{label} high GB/s"]
            writer.writerow(header)
            for row_idx, x_val in enumerate(x_vals):
                row = [x_val]
                for key, _ in PROVIDERS:
                    row += [
                        results[key]["median"][row_idx],
                        results[key]["low"][row_idx],
                        results[key]["high"][row_idx],
                    ]
                writer.writerow(row)
        print(f"  Saved data to: {csv_file}")
    print()


def print_summary(hidden_size=1536, n_rows_list=None, save_path=""):
    if n_rows_list is None:
        n_rows_list = [1, 32, 128, 512]

    dtype = jnp.float32

    print()
    print("=" * 75)
    print(f"  Bielik config: hidden_size={hidden_size}, dtype=float32, device: {device_name()}")
    print("=" * 75)
    print(f"  {'n_rows':<8} {'Provider':<24} {'ms':>8} {'GB/s':>10} {'TFLOPS':>10}")
    print("-" * 75)

    data_dict = {label: [] for _, label in PROVIDERS}
    labels = [str(n) for n in n_rows_list]

    for n_rows in n_rows_list:
        key_x, key_w = jax.random.split(jax.random.PRNGKey(0))
        x = jax.random.normal(key_x, (n_rows, hidden_size), dtype=dtype)
        w = jax.random.normal(key_w, (hidden_size,), dtype=dtype)
        block_rows = pick_block_size(n_rows, BLOCK_ROWS)

        for key, label in PROVIDERS:
            fn = _provider_fn(key, x, w, block_rows)
            ms, _, _ = do_bench(fn)
            gbps_val = _rms_norm_gbps(ms, n_rows, hidden_size, x.itemsize)
            tflops_val = tflops(ms, 4 * n_rows * hidden_size)
            data_dict[label].append([gbps_val, tflops_val])
            print(f"  {n_rows:<8} {label:<24} {ms:>8.3f} {gbps_val:>10.1f} {tflops_val:>10.4f}")
        print()

    if save_path:
        plot_summary_comparison(
            data=data_dict,
            x_labels=labels,
            metrics=["GB/s", "TFLOPS"],
            title=f"RMSNorm Performance - Bielik Config (hidden_size={hidden_size})",
            xlabel="Batch Size (n_rows)",
            save_path=save_path,
            filename_prefix="rms_norm-summary-bielik-config",
            colors=[COLORS[key] for key, _ in PROVIDERS],
            device_name=device_name(),
        )
        print()


def run_benchmark(save_plots: bool = False, plot_dir: str = ".") -> None:
    print("=" * 75)
    print("RMSNorm Benchmark: Pallas vs JAX Native")
    print("=" * 75)
    print(f"Device: {device_name()}")
    print()

    if save_plots:
        plot_path = Path(plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)
        save_path = str(plot_path)
        print(f"Saving plots to: {plot_path.absolute()}/")
        print()
    else:
        save_path = ""

    _run_sweep(
        "hidden_size",
        [128, 256, 512, 768, 1024, 1536, 2048, 4096],
        {"n_rows": 512, "dtype": jnp.float32},
        "rmsnorm-bandwidth-vs-hidden-size",
        save_path,
    )

    _run_sweep(
        "n_rows",
        [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        {"hidden_size": 1536, "dtype": jnp.float32},
        "rmsnorm-bandwidth-vs-rows",
        save_path,
    )

    print_summary(save_path=save_path)

    if save_plots:
        print()
        print(f"Plots saved to {plot_path.absolute()}/")
        print("  - rmsnorm-bandwidth-vs-hidden-size.png")
        print("  - rmsnorm-bandwidth-vs-rows.png")
        print("  - rms_norm-summary-bielik-config-gb_s.png")
        print("  - rms_norm-summary-bielik-config-tflops.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMSNorm benchmark")
    parser.add_argument("--save-plots", action="store_true", help="Save plots as PNG")
    parser.add_argument("--plot-dir", default=".", help="Directory to save plots")
    args = parser.parse_args()
    run_benchmark(save_plots=args.save_plots, plot_dir=args.plot_dir)
