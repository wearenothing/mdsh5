#!/usr/bin/env python3
import argparse
import csv
import os
import sys

import h5py
import numpy as np

from mdsh5.read_mds import read_mds


def _load_matplotlib():
    import importlib.util

    if importlib.util.find_spec("matplotlib.pyplot") is None:
        return None
    import matplotlib.pyplot as plt

    return plt


def _iter_signals(h5):
    for shot in h5:
        for tree in h5[shot]:
            for point_name in h5[shot][tree]:
                yield shot, tree, point_name, h5[shot][tree][point_name]


def _get_dim(signal_group, dim_index=0):
    key = f"dim{dim_index}"
    if key in signal_group:
        return signal_group[key][:]
    return None


def _summarize_signal(data, dim0=None):
    summary = {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }
    if dim0 is not None and np.ndim(dim0) == 1 and len(dim0) == len(data):
        summary["integral"] = float(np.trapz(data, dim0))
    else:
        summary["integral"] = None
    return summary


def run_pipeline(config_path, out_csv, plot, plot_dir):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    read_mds(config=config_path)

    with open(config_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip().startswith("out_filename:"):
                out_filename = line.split(":", 1)[1].strip()
                break
        else:
            raise ValueError("out_filename not found in config file.")

    if not os.path.isfile(out_filename):
        raise FileNotFoundError(
            "Expected HDF5 output missing. Confirm out_filename in config."
        )

    plt = _load_matplotlib() if plot else None
    if plot and plt is None:
        raise RuntimeError(
            "Plotting requested but matplotlib is not installed. "
            "Install matplotlib or rerun without --plot."
        )

    if plot and plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    rows = []
    with h5py.File(out_filename, "r") as h5:
        for shot, tree, point_name, signal_group in _iter_signals(h5):
            if "data" not in signal_group:
                continue
            data = signal_group["data"][:]
            dim0 = _get_dim(signal_group, 0)
            summary = _summarize_signal(data, dim0)
            rows.append(
                {
                    "shot": shot,
                    "tree": tree,
                    "point_name": point_name,
                    "mean": summary["mean"],
                    "std": summary["std"],
                    "min": summary["min"],
                    "max": summary["max"],
                    "integral": summary["integral"],
                }
            )

            if plot and plt is not None:
                fig, ax = plt.subplots(figsize=(8, 4))
                x = dim0 if dim0 is not None else np.arange(len(data))
                ax.plot(x, data)
                ax.set_title(f"{shot} | {tree}:{point_name}")
                ax.set_xlabel("Time" if dim0 is not None else "Index")
                ax.set_ylabel("Signal")
                fig.tight_layout()
                if plot_dir:
                    filename = f"{shot}_{tree}_{point_name}.png"
                    fig.savefig(os.path.join(plot_dir, filename), dpi=150)
                else:
                    plt.show()
                plt.close(fig)

    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["shot", "tree", "point_name", "mean", "std", "min", "max", "integral"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return out_filename, out_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end MDSPlus read + analysis pipeline using mdsh5."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to mdsh5 YAML config file (see config_examples).",
    )
    parser.add_argument(
        "--out-csv",
        default="analysis_summary.csv",
        help="CSV file to write summary statistics.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot each signal (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Directory to save plots. If omitted, plots are shown interactively.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_h5, out_csv = run_pipeline(
        config_path=args.config,
        out_csv=args.out_csv,
        plot=args.plot,
        plot_dir=args.plot_dir,
    )
    print(f"Wrote HDF5: {out_h5}")
    print(f"Wrote CSV: {out_csv}")


if __name__ == "__main__":
    sys.exit(main())
