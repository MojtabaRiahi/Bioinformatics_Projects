# tawfn_repro.py
import os
import json
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA)}
UNK_IDX = len(AA)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot_sequence(seq: str, max_len: int):
    x = np.zeros((max_len, len(AA) + 1), dtype=np.float32)
    seq = seq[:max_len]
    for i, ch in enumerate(seq):
        x[i, AA_TO_IDX.get(ch, UNK_IDX)] = 1.0
    return x


def chain_adjacency(length: int, max_len: int):
    adj = np.zeros((max_len, max_len), dtype=np.float32)
    L = min(length, max_len)
    for i in range(L - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def load_contact_map(path: str, max_len: int, seq_len: int):
    if path and os.path.exists(path):
        adj = np.load(path).astype(np.float32)
        adj = adj[:max_len, :max_len]
        out = np.zeros((max_len, max_len), dtype=np.float32)
        L = min(adj.shape[0], max_len)
        out[:L, :L] = adj[:L, :L]
        return out
    return chain_adjacency(seq_len, max_len)


def parse_labels(label_str: str):
    if pd.isna(label_str):
        return []
    if isinstance(label_str, (list, tuple)):
        return list(label_str)
    for sep in ["|", ",", ";", " "]:
        if sep in str(label_str):
            return [x.strip() for x in str(label_str).split(sep) if x.strip()]
    return [str(label_str).strip()]


class ProteinManifestDataset(Dataset):
    def __init__(self, df, label_to_idx, max_len=512):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row["sequence"]).strip().upper()
        labels = parse_labels(row["labels"])
        contact_path = row["contact_map_path"] if "contact_map_path" in row and not pd.isna(row["contact_map_path"]) else None

        x = one_hot_sequence(seq, self.max_len)
        adj = load_contact_map(contact_path, self.max_len, len(seq))

        y = np.zeros((len(self.label_to_idx),), dtype=np.float32)
        for lab in labels:
            if lab in self.label_to_idx:
                y[self.label_to_idx[lab]] = 1.0

        mask = np.zeros((self.max_len,), dtype=np.float32)
        mask[: min(len(seq), self.max_len)] = 1.0

        return (
            torch.from_numpy(x),
            torch.from_numpy(adj),
            torch.from_numpy(mask),
            torch.from_numpy(y),
        )


def collate_fn(batch):
    xs, adjs, masks, ys = zip(*batch)
    return (
        torch.stack(xs, dim=0),
        torch.stack(adjs, dim=0),
        torch.stack(masks, dim=0),
        torch.stack(ys, dim=0),
    )


class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj, mask):
        # x: [B, L, D], adj: [B, L, L], mask: [B, L]
        b, l, _ = x.shape
        eye = torch.eye(l, device=x.device).unsqueeze(0).expand(b, -1, -1)
        a = adj * mask.unsqueeze(1) * mask.unsqueeze(2)
        a = a + eye * mask.unsqueeze(1)
        deg = a.sum(dim=-1).clamp(min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt.unsqueeze(2) * a * deg_inv_sqrt.unsqueeze(1)
        out = torch.bmm(norm, x)
        out = self.lin(out)
        return out


class SeqCNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: [B, L, D]
        x = x.transpose(1, 2)  # [B, D, L]
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = x.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        x = torch.max(x, dim=-1).values
        x = self.fc(x)
        return x


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, dropout=0.2):
        super().__init__()
        self.g1 = DenseGCNLayer(in_dim, hidden_dim)
        self.g2 = DenseGCNLayer(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, mask):
        x = F.relu(self.g1(x, adj, mask))
        x = self.dropout(x)
        x = F.relu(self.g2(x, adj, mask))
        x = self.dropout(x)
        x = x.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        x = torch.max(x, dim=1).values
        x = self.fc(x)
        return x


class TAWFNSimple(nn.Module):
    def __init__(self, in_dim, num_labels, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.seq_branch = SeqCNN(in_dim, hidden_dim, hidden_dim, dropout)
        self.gcn_branch = GraphEncoder(in_dim, hidden_dim, hidden_dim, dropout)

        self.seq_head = nn.Linear(hidden_dim, num_labels)
        self.gcn_head = nn.Linear(hidden_dim, num_labels)

        self.fuse = nn.Sequential(
            nn.Linear(num_labels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, x, adj, mask):
        s = self.seq_branch(x, mask)
        g = self.gcn_branch(x, adj, mask)

        logit_s = self.seq_head(s)
        logit_g = self.gcn_head(g)

        w = self.fuse(torch.cat([logit_s, logit_g], dim=-1))
        out = w[:, 0:1] * logit_s + w[:, 1:2] * logit_g
        return out, logit_s, logit_g, w


def macro_aupr(y_true, y_prob):
    scores = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        try:
            scores.append(average_precision_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            pass
    return float(np.mean(scores)) if scores else 0.0


def fmax_score(y_true, y_prob):
    best = 0.0
    thresholds = np.linspace(0.0, 1.0, 101)
    for t in thresholds:
        pred = (y_prob >= t).astype(np.float32)
        tp = (pred * y_true).sum(axis=1)
        prec = tp / np.maximum(pred.sum(axis=1), 1.0)
        rec = tp / np.maximum(y_true.sum(axis=1), 1.0)
        f = 2.0 * prec * rec / np.maximum(prec + rec, 1e-8)
        best = max(best, float(np.nanmean(f)))
    return best


def make_demo_manifest(n=120, num_labels=8, min_len=80, max_len=160, out_csv="demo_manifest.csv"):
    aas = list(AA)
    rows = []
    for i in range(n):
        L = random.randint(min_len, max_len)
        seq = "".join(random.choice(aas) for _ in range(L))

        seq_set = set(seq)
        labels = []
        if sum(seq.count(a) for a in "AILMFWVY") / L > 0.42:
            labels.append("hydrophobic")
        if sum(seq.count(a) for a in "DE") / L > 0.12:
            labels.append("acidic")
        if sum(seq.count(a) for a in "KRH") / L > 0.10:
            labels.append("basic")
        if L > (min_len + max_len) / 2:
            labels.append("long")
        if sum(seq.count(a) for a in "GP") / L > 0.15:
            labels.append("flexible")
        if "C" in seq_set:
            labels.append("cys")
        if "W" in seq_set:
            labels.append("trp")
        if not labels:
            labels.append("other")

        rows.append(
            {
                "protein_id": f"demo_{i}",
                "sequence": seq,
                "labels": "|".join(labels),
                "split": "train" if i < int(0.7 * n) else "val" if i < int(0.85 * n) else "test",
                "contact_map_path": "",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return out_csv


def split_dataframe(df):
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()
        test_df = df[df["split"] == "test"].copy()
        if len(train_df) and len(val_df) and len(test_df):
            return train_df, val_df, test_df

    idx = np.random.permutation(len(df))
    n_train = int(0.7 * len(df))
    n_val = int(0.15 * len(df))
    train_df = df.iloc[idx[:n_train]].copy()
    val_df = df.iloc[idx[n_train : n_train + n_val]].copy()
    test_df = df.iloc[idx[n_train + n_val :]].copy()
    return train_df, val_df, test_df


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true = []
    all_prob = []
    all_fusion_w = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for x, adj, mask, y in loader:
        x = x.to(device)
        adj = adj.to(device)
        mask = mask.to(device)
        y = y.to(device)

        logits, logit_s, logit_g, w = model(x, adj, mask)
        loss = criterion(logits, y)
        prob = torch.sigmoid(logits)

        total_loss += float(loss.item()) * x.size(0)
        all_true.append(y.cpu().numpy())
        all_prob.append(prob.cpu().numpy())
        all_fusion_w.append(w.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    fusion_w = np.concatenate(all_fusion_w, axis=0)

    avg_mcnn_weight = float(fusion_w[:, 0].mean())
    avg_gcn_weight = float(fusion_w[:, 1].mean())

    return {
        "loss": total_loss / len(loader.dataset),
        "aupr": macro_aupr(y_true, y_prob),
        "fmax": fmax_score(y_true, y_prob),
        "avg_mcnn_weight": avg_mcnn_weight,
        "avg_gcn_weight": avg_gcn_weight,
        "y_true": y_true,
        "y_prob": y_prob,
        "fusion_w": fusion_w,
    }


def train(args):
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    if args.demo or not os.path.exists(args.manifest):
        print("Using demo manifest.")
        args.manifest = make_demo_manifest(
            n=args.demo_size,
            out_csv=os.path.join(args.outdir, "demo_manifest.csv")
        )

    df = pd.read_csv(args.manifest)
    df["labels"] = df["labels"].fillna("")
    all_labels = sorted(set(lab for ls in df["labels"].astype(str).tolist() for lab in parse_labels(ls)))
    if args.num_labels > 0:
        all_labels = all_labels[: args.num_labels]
    label_to_idx = {lab: i for i, lab in enumerate(all_labels)}

    train_df, val_df, test_df = split_dataframe(df)

    train_ds = ProteinManifestDataset(train_df, label_to_idx, max_len=args.max_len)
    val_ds = ProteinManifestDataset(val_df, label_to_idx, max_len=args.max_len)
    test_ds = ProteinManifestDataset(test_df, label_to_idx, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    model = TAWFNSimple(
        in_dim=len(AA) + 1,
        num_labels=len(label_to_idx),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val = -1.0
    best_state = None
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_aupr": [],
        "val_fmax": [],
        "val_mcnn_weight": [],
        "val_gcn_weight": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for x, adj, mask, y in train_loader:
            x = x.to(device)
            adj = adj.to(device)
            mask = mask.to(device)
            y = y.to(device)

            optim.zero_grad()
            logits, _, _, _ = model(x, adj, mask)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

            running += float(loss.item()) * x.size(0)

        train_loss = running / len(train_loader.dataset)
        val_res = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss {train_loss:.4f} | "
            f"Val loss {val_res['loss']:.4f} | "
            f"Val AUPR {val_res['aupr']:.4f} | "
            f"Val Fmax {val_res['fmax']:.4f} | "
            f"Val MCNN weight {val_res['avg_mcnn_weight']:.4f} | "
            f"Val GCN weight {val_res['avg_gcn_weight']:.4f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_res["loss"])
        history["val_aupr"].append(val_res["aupr"])
        history["val_fmax"].append(val_res["fmax"])
        history["val_mcnn_weight"].append(val_res["avg_mcnn_weight"])
        history["val_gcn_weight"].append(val_res["avg_gcn_weight"])

        score = val_res["aupr"]
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_res = evaluate(model, test_loader, device)
    print(
        f"Test | loss {test_res['loss']:.4f} | "
        f"AUPR {test_res['aupr']:.4f} | "
        f"Fmax {test_res['fmax']:.4f} | "
        f"MCNN weight {test_res['avg_mcnn_weight']:.4f} | "
        f"GCN weight {test_res['avg_gcn_weight']:.4f}"
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": all_labels,
            "label_to_idx": label_to_idx,
            "args": vars(args),
        },
        os.path.join(args.outdir, "tawfn_simple.pt"),
    )

    with open(os.path.join(args.outdir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(args.outdir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss": test_res["loss"],
                "aupr": test_res["aupr"],
                "fmax": test_res["fmax"],
                "avg_mcnn_weight": test_res["avg_mcnn_weight"],
                "avg_gcn_weight": test_res["avg_gcn_weight"],
                "num_labels": len(all_labels),
                "num_train": len(train_ds),
                "num_val": len(val_ds),
                "num_test": len(test_ds),
            },
            f,
            indent=2,
        )

    # Plot 1: loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "loss_curve.png"), dpi=200)
    plt.close()

    # Plot 2: all training curves
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history["epoch"], history["val_aupr"], label="Val AUPR", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("AUPR")
    plt.title("Validation AUPR")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history["epoch"], history["val_fmax"], label="Val Fmax", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Fmax")
    plt.title("Validation Fmax")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(history["epoch"], history["val_mcnn_weight"], label="MCNN Weight", color="blue")
    plt.plot(history["epoch"], history["val_gcn_weight"], label="GCN Weight", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Average Fusion Weight")
    plt.title("Validation Fusion Weights")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "training_curves.png"), dpi=200)
    plt.close()

    return test_res


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default="data/mini_real/manifest_with_maps.csv")
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--demo_size", type=int, default=120)
    p.add_argument("--num_labels", type=int, default=0, help="optional cap on output labels")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
