import os
import argparse
import numpy as np
import pandas as pd


HYDROPHOBIC = set("AILMFWVY")
CHARGED = set("DEKRH")
POLAR = set("STNQCY")


def pseudo_contact_map(seq, threshold_window=2, long_range_prob=0.025):
    seq = seq.strip().upper()
    n = len(seq)
    contact = np.zeros((n, n), dtype=np.float32)

    # اتصال‌های محلی بین آمینواسیدهای نزدیک در توالی
    for i in range(n):
        for j in range(max(0, i - threshold_window), min(n, i + threshold_window + 1)):
            if i != j:
                contact[i, j] = 1.0

    # اتصال‌های دوربرد شبه‌تصادفی بر اساس نوع آمینواسیدها
    for i in range(n):
        for j in range(i + 6, n):
            ai, aj = seq[i], seq[j]

            p = long_range_prob

            if ai in HYDROPHOBIC and aj in HYDROPHOBIC:
                p += 0.06

            if ai in CHARGED and aj in CHARGED:
                p += 0.025

            if ai in POLAR and aj in POLAR:
                p += 0.02

            if ai == "C" and aj == "C":
                p += 0.15

            if np.random.rand() < p:
                contact[i, j] = 1.0
                contact[j, i] = 1.0

    np.fill_diagonal(contact, 1.0)
    return contact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out_manifest", type=str, default=None)
    parser.add_argument("--map_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    df = pd.read_csv(args.manifest)

    if args.map_dir is None:
        base_dir = os.path.dirname(args.manifest)
        args.map_dir = os.path.join(base_dir, "maps")

    os.makedirs(args.map_dir, exist_ok=True)

    paths = []

    for _, row in df.iterrows():
        protein_id = str(row["protein_id"])
        seq = str(row["sequence"])

        contact = pseudo_contact_map(seq)

        out_path = os.path.join(args.map_dir, f"{protein_id}.npy")
        np.save(out_path, contact)

        paths.append(out_path)

    df["contact_map_path"] = paths

    if args.out_manifest is None:
        base_dir = os.path.dirname(args.manifest)
        args.out_manifest = os.path.join(base_dir, "manifest_with_maps.csv")

    df.to_csv(args.out_manifest, index=False)

    print(f"Saved contact maps to: {args.map_dir}")
    print(f"Saved new manifest to: {args.out_manifest}")
    print(f"Number of proteins: {len(df)}")


if __name__ == "__main__":
    main()
