import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "pallas": "#2E86AB",
    "jax_native": "#A23B72",
    "pallas_autotuned": "#5CB85C",
}


def plot_summary_comparison(
    data: dict[str, list[float]],
    x_labels: list[str],
    metrics: list[str],
    title: str,
    xlabel: str,
    save_path: str,
    filename_prefix: str,
    colors: list[str],
    device_name: str = "",
) -> None:
    provider_labels = list(data.keys())
    assert len(colors) == len(
        provider_labels
    ), f"colors ({len(colors)}) must have one entry per provider ({len(provider_labels)})"

    data_arrays = [np.array(values) for values in data.values()]
    num_providers = len(provider_labels)
    num_cases = len(x_labels)

    for metric_idx, metric_name in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(num_cases)
        width = 0.8 / num_providers

        bars_list = []
        metric_values_per_provider = []
        for i, (values, label, color) in enumerate(
            zip(data_arrays, provider_labels, colors, strict=True)
        ):
            offset = (i - num_providers / 2 + 0.5) * width

            if values.ndim > 1:
                metric_values = values[:, metric_idx]
            elif len(values) > 0 and isinstance(values[0], (list, tuple)):
                metric_values = [v[metric_idx] for v in values]
            else:
                metric_values = values
            metric_values_per_provider.append(metric_values)

            bars = ax.bar(x + offset, metric_values, width, label=label, color=color)
            bars_list.append(bars)

        ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")

        full_title = f"{title} - {metric_name}"
        if device_name:
            full_title += f"\nDevice: {device_name}"
        ax.set_title(full_title, fontsize=14, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    if height >= 1000:
                        label_text = f"{height:.0f}"
                    elif height >= 10:
                        label_text = f"{height:.1f}"
                    else:
                        label_text = f"{height:.2f}"

                    ax.annotate(  # noqa: B023
                        label_text,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        for bars in bars_list:
            autolabel(bars)

        plt.tight_layout()

        metric_suffix = metric_name.lower().replace("/", "_").replace(" ", "_")
        plot_file = Path(save_path) / f"{filename_prefix}-{metric_suffix}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()

        csv_file = Path(save_path) / f"{filename_prefix}-{metric_suffix}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([xlabel, *provider_labels])
            for i, x_label in enumerate(x_labels):
                writer.writerow([x_label, *(values[i] for values in metric_values_per_provider)])

        print(f"  Saved {metric_name} plot to: {plot_file}")
        print(f"  Saved {metric_name} data to: {csv_file}")
