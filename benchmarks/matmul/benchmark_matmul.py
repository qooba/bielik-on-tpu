#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.common.bench import device_name, do_bench, pick_block_size, tflops
from benchmarks.common.plotting import COLORS, plot_summary_comparison
from kernels.matmul.autotune import autotune_matmul
from kernels.matmul.matmul import matmul_pallas, matmul_ref

BLOCK = 128

AUTOTUNE_WARMUP = 1
AUTOTUNE_REP = 3

_LAST_AUTOTUNED_CONFIG: tuple[int, int, int] | None = None


def _provider_fn(
    provider: str, a: jnp.ndarray, b: jnp.ndarray, block_m: int, block_n: int, block_k: int
):
    global _LAST_AUTOTUNED_CONFIG
    if provider == "pallas":
        return lambda: matmul_pallas(a, b, block_m=block_m, block_n=block_n, block_k=block_k)
    elif provider == "pallas_autotuned":
        M, K = a.shape
        _, N = b.shape
        bm, bn, bk = autotune_matmul(
            M, K, N, dtype=a.dtype, warmup=AUTOTUNE_WARMUP, rep=AUTOTUNE_REP
        )
        _LAST_AUTOTUNED_CONFIG = (bm, bn, bk)
        return lambda: matmul_pallas(a, b, block_m=bm, block_n=bn, block_k=bk)
    else:  # jax_native
        jitted = jax.jit(matmul_ref)
        jitted(a, b)
        return lambda: jitted(a, b)


PROVIDERS = [
    ("pallas", "Pallas (tiled, block=128)"),
    ("pallas_autotuned", "Pallas (autotuned)"),
    ("jax_native", "JAX Native (jit-fused)"),
]


def _matmul_tflops(ms: float, M: int, K: int, N: int) -> float:
    return tflops(ms, 2 * M * K * N)


def _run_size_sweep(x_vals: list, save_path: str):
    print("--- Sweep: vary size (square, M=K=N) ---")

    results = {key: {"median": [], "low": [], "high": []} for key, _ in PROVIDERS}

    for size in x_vals:
        M = K = N = size
        key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
        a = jax.random.normal(key_a, (M, K), dtype=jnp.bfloat16)
        b = jax.random.normal(key_b, (K, N), dtype=jnp.bfloat16)
        block_m, block_n, block_k = (
            pick_block_size(M, BLOCK),
            pick_block_size(N, BLOCK),
            pick_block_size(K, BLOCK),
        )

        row = [f"size={size:<6}"]
        for key_p, label in PROVIDERS:
            fn = _provider_fn(key_p, a, b, block_m, block_n, block_k)
            median, low, high = do_bench(fn)
            results[key_p]["median"].append(_matmul_tflops(median, M, K, N))
            results[key_p]["low"].append(_matmul_tflops(high, M, K, N))
            results[key_p]["high"].append(_matmul_tflops(low, M, K, N))
            suffix = f" block={_LAST_AUTOTUNED_CONFIG}" if key_p == "pallas_autotuned" else ""
            row.append(f"{label}={results[key_p]['median'][-1]:.2f} TFLOPS{suffix}")
        print("  " + "  ".join(row))

    if save_path:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for key_p, label in PROVIDERS:
            ax.plot(x_vals, results[key_p]["median"], label=label, color=COLORS[key_p], marker="o")
            ax.fill_between(
                x_vals,
                results[key_p]["low"],
                results[key_p]["high"],
                color=COLORS[key_p],
                alpha=0.15,
            )
        ax.set_xlabel("size (M = K = N)", fontsize=12, fontweight="bold")
        ax.set_ylabel("TFLOPS", fontsize=12, fontweight="bold")
        ax.set_title(
            f"matmul-tflops-vs-size\nDevice: {device_name()}", fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        plot_file = Path(save_path) / "matmul-tflops-vs-size.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot to: {plot_file}")

        csv_file = Path(save_path) / "matmul-tflops-vs-size.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["size"]
            for _, label in PROVIDERS:
                header += [f"{label} median TFLOPS", f"{label} low TFLOPS", f"{label} high TFLOPS"]
            writer.writerow(header)
            for row_idx, size in enumerate(x_vals):
                row = [size]
                for key_p, _ in PROVIDERS:
                    row += [
                        results[key_p]["median"][row_idx],
                        results[key_p]["low"][row_idx],
                        results[key_p]["high"][row_idx],
                    ]
                writer.writerow(row)
        print(f"  Saved data to: {csv_file}")
    print()


BIELIK_SHAPES = [
    ("Q proj (single)", 1, 1536, 1536),
    ("Q proj (seq=128)", 128, 1536, 1536),
    ("K proj GQA (single)", 1, 1536, 256),
    ("K proj GQA (seq=128)", 128, 1536, 256),
    ("FFN up (single)", 1, 1536, 8960),
    ("FFN up (seq=128)", 128, 1536, 8960),
    ("FFN down (single)", 1, 8960, 1536),
    ("FFN down (seq=128)", 128, 8960, 1536),
]


def run_bielik_shapes(save_path=""):
    print()
    print("=" * 90)
    print(f"  Bielik-1.5B Matmul Shapes, device: {device_name()}")
    print("=" * 90)
    print(f"  {'Shape':<22} {'MxKxN':<18} {'Provider':<24} {'ms':>8} {'TFLOPS':>10}  {'block'}")
    print("-" * 90)

    data_dict = {label: [] for _, label in PROVIDERS}
    labels = [name for name, _, _, _ in BIELIK_SHAPES]

    for name, M, K, N in BIELIK_SHAPES:
        key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
        a = jax.random.normal(key_a, (M, K), dtype=jnp.bfloat16)
        b = jax.random.normal(key_b, (K, N), dtype=jnp.bfloat16)
        block_m, block_n, block_k = (
            pick_block_size(M, BLOCK),
            pick_block_size(N, BLOCK),
            pick_block_size(K, BLOCK),
        )

        for key_p, label in PROVIDERS:
            fn = _provider_fn(key_p, a, b, block_m, block_n, block_k)
            ms, _, _ = do_bench(fn)
            tflops_val = _matmul_tflops(ms, M, K, N)
            data_dict[label].append([ms, tflops_val])
            block_info = _LAST_AUTOTUNED_CONFIG if key_p == "pallas_autotuned" else ""
            print(
                f"  {name:<22} {f'{M}x{K}x{N}':<18} {label:<24} {ms:>8.3f} {tflops_val:>10.2f}  "
                f"{block_info}"
            )
        print()

    if save_path:
        plot_summary_comparison(
            data=data_dict,
            x_labels=labels,
            metrics=["Latency (ms)", "TFLOPS"],
            title="Matmul Performance - Bielik-1.5B Shapes",
            xlabel="Operation Shape",
            save_path=save_path,
            filename_prefix="matmul-summary-bielik-shapes",
            colors=[COLORS[key] for key, _ in PROVIDERS],
            device_name=device_name(),
        )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matmul benchmark")
    parser.add_argument("--save-plots", action="store_true", help="Save plots as PNG")
    parser.add_argument("--plot-dir", default=".", help="Directory to save plots")
    args = parser.parse_args()

    print("=" * 75)
    print("Matmul Benchmark: Pallas vs Pallas (autotuned) vs JAX Native")
    print("=" * 75)
    print(f"Device: {device_name()}")
    print()

    if args.save_plots:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(plot_dir)
        print(f"Saving plots to: {plot_dir.absolute()}/")
        print()
    else:
        save_path = ""

    _run_size_sweep([512, 1024, 2048, 4096], save_path)

    run_bielik_shapes(save_path=save_path)

    if args.save_plots:
        print()
        print(f"Plots + data saved to {plot_dir.absolute()}/")
        for stem in [
            "matmul-tflops-vs-size",
            "matmul-summary-bielik-shapes-latency_(ms)",
            "matmul-summary-bielik-shapes-tflops",
        ]:
            print(f"  - {stem}.png / {stem}.csv")
