# download_uniprot_manifest.py
import argparse
import requests
import pandas as pd
import random
from collections import Counter


def get_next_link(headers):
    link = headers.get("Link", "")
    if not link:
        return None

    # UniProt pagination header example:
    # <https://rest.uniprot.org/uniprotkb/search?...>; rel="next"
    # We must not split by comma because the URL itself may contain commas in fields.
    import re

    match = re.search(r'<([^>]+)>;\s*rel="next"', link)
    if match:
        return match.group(1)

    return None



def fetch_uniprot(query, max_records=1000, size=500):
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,sequence,go_id",
        "size": size,
    }

    rows = []
    url = base_url

    while url and len(rows) < max_records:
        if url == base_url:
            r = requests.get(url, params=params, timeout=60)
        else:
            r = requests.get(url, timeout=60)

        r.raise_for_status()

        lines = r.text.strip().splitlines()
        if len(lines) <= 1:
            break

        header = lines[0].split("\t")

        for line in lines[1:]:
            cols = line.split("\t")
            item = dict(zip(header, cols))

            accession = item.get("Entry", "").strip()
            sequence = item.get("Sequence", "").strip()
            go_ids = item.get("Gene Ontology IDs", "").strip()

            if not accession or not sequence or not go_ids:
                continue

            labels = [
                x.strip()
                for x in go_ids.replace(";", "|").split("|")
                if x.strip().startswith("GO:")
            ]

            if not labels:
                continue

            rows.append(
                {
                    "protein_id": accession,
                    "sequence": sequence,
                    "raw_labels": labels,
                }
            )

            if len(rows) >= max_records:
                break

        url = get_next_link(r.headers)

    return rows


def build_manifest(rows, num_labels=20, min_label_count=10, seed=42):
    random.seed(seed)

    counter = Counter()
    for row in rows:
        counter.update(row["raw_labels"])

    selected_labels = [
        lab for lab, count in counter.most_common()
        if count >= min_label_count
    ][:num_labels]

    selected_set = set(selected_labels)

    final_rows = []
    for row in rows:
        labels = [lab for lab in row["raw_labels"] if lab in selected_set]

        if not labels:
            continue

        final_rows.append(
            {
                "protein_id": row["protein_id"],
                "sequence": row["sequence"],
                "labels": "|".join(sorted(set(labels))),
                "split": "",
                "contact_map_path": "",
            }
        )

    random.shuffle(final_rows)

    n = len(final_rows)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    for i, row in enumerate(final_rows):
        if i < n_train:
            row["split"] = "train"
        elif i < n_train + n_val:
            row["split"] = "val"
        else:
            row["split"] = "test"

    return pd.DataFrame(final_rows), selected_labels, counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/mini_real/manifest_uniprot.csv")
    parser.add_argument("--max_records", type=int, default=1000)
    parser.add_argument("--num_labels", type=int, default=20)
    parser.add_argument("--min_label_count", type=int, default=10)
    parser.add_argument("--organism_id", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reviewed Swiss-Prot proteins with GO annotation
    query_parts = [
        "reviewed:true",
        "go:*",
    ]

    if args.organism_id:
        query_parts.append(f"organism_id:{args.organism_id}")

    query = " AND ".join(query_parts)

    print("Query:", query)
    print("Fetching UniProt records...")

    rows = fetch_uniprot(query=query, max_records=args.max_records)

    print(f"Fetched usable rows: {len(rows)}")

    df, selected_labels, counter = build_manifest(
        rows,
        num_labels=args.num_labels,
        min_label_count=args.min_label_count,
        seed=args.seed,
    )

    print(f"Final manifest samples: {len(df)}")
    print(f"Selected labels: {len(selected_labels)}")
    print("Top selected labels:")
    for lab in selected_labels:
        print(lab, counter[lab])

    df.to_csv(args.out, index=False)
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
