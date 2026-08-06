"""Render the former cumulative-sum rainfall defect beside the repaired total."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.precipitation import decode_forecast_precipitation


def open_legacy_safe(path: Path) -> xr.Dataset:
    raw = xr.open_dataset(path, decode_cf=False)
    for name in raw.variables:
        raw[name].attrs.pop("dtype", None)
    return xr.decode_cf(raw, decode_timedelta=True)


def main():
    parser = argparse.ArgumentParser(description="Compare old and repaired rainfall-map accumulation")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with open_legacy_safe(args.dataset) as dataset:
        decoded = decode_forecast_precipitation(dataset)
        dimension = next((name for name in ("step", "valid_time", "time") if name in decoded.cumulative_mm.dims), None)
        if dimension is None:
            raise RuntimeError("forecast precipitation has no lead dimension")
        old_mm = decoded.cumulative_mm.sum(dimension).load()
        repaired_mm = decoded.cumulative_mm.isel({dimension: -1}).load()

    old_inches = old_mm / 25.4
    repaired_inches = repaired_mm / 25.4
    args.image.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, values, title in (
        (axes[0], old_inches, "Old defect: sum of cumulative leads"),
        (axes[1], repaired_inches, "Repaired: final cumulative lead"),
    ):
        image = axis.imshow(values, origin="lower", cmap="Blues")
        maximum_inches = float(np.nanmax(values))
        maximum_mm = maximum_inches * 25.4
        axis.set_title(f"{title}\nmax {maximum_inches:.2f} in ({maximum_mm:.1f} mm)")
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, label="Precipitation (inches; mm shown in title)", shrink=0.8)
    figure.suptitle(f"HRRR rainfall accumulation comparison: {args.dataset.name}")
    figure.savefig(args.image, dpi=160)
    plt.close(figure)

    old_max = float(np.nanmax(old_mm))
    repaired_max = float(np.nanmax(repaired_mm))
    report = {
        "source": str(args.dataset),
        "old_calculation": "sum of cumulative forecast leads",
        "repaired_calculation": "final cumulative forecast lead",
        "old_max_mm": old_max,
        "old_max_in": old_max / 25.4,
        "repaired_max_mm": repaired_max,
        "repaired_max_in": repaired_max / 25.4,
        "maximum_overstatement_factor": old_max / repaired_max if repaired_max else None,
        "image": str(args.image),
    }
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2)); temporary.replace(args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
