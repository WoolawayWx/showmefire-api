"""Create model features through the shared causal transformer."""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.features import build_causal_features


def prepare_features():
    df = build_causal_features(pd.read_csv("data/training_set_mo.csv"))
    df.to_csv("data/ai_features_mo.csv", index=False)
    print(f"Enhanced dataset saved with {len(df)} rows.")
    sample = df[df["station_id"] == "ASLM7"].sort_values("obs_time")
    if not sample.empty:
        Path("plots").mkdir(exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(sample["obs_time"], sample["target_fm"], label="Actual FM %", color="red", marker="o")
        plt.plot(sample["obs_time"], sample["emc_baseline"], label="Physics Baseline (EMC)", linestyle="--")
        plt.legend(); plt.tight_layout(); plt.savefig("plots/station_preview.png"); plt.close()
    return df


def enhance_features_with_lags():
    # Kept as a compatibility entry point; prepare_features already creates all lags.
    df = pd.read_csv("data/ai_features_mo.csv")
    df.to_csv("data/final_training_data.csv", index=False)
    print(f"Final training data created with {len(df)} rows and causal features.")
    return df


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    prepare_features()
    enhance_features_with_lags()
