"""
Modul Evaluasi Pengujian Model (Tugas Akhir - Subbab 4.1.1 & Pengujian K1-K5)
Memuat checkpoint model tertentu, menjalankan inference pada SATU subset testing 
FaceForensics++ C23 yang tetap, dan menyimpan metrik evaluasi ke JSON per konfigurasi.
"""

import os
import json
import yaml
import argparse
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from model import EfficientNetB4_CBAM_NonLocal
from dataset import DeepfakeFaceDataset


def load_model_for_evaluation(
    checkpoint_path: str,
    config_name: str = "k5",
    config_path: str = "src/config.yaml",
    device: torch.device = None
) -> nn.Module:
    """
    Menginisialisasi arsitektur sesuai config_name dan memuat bobot checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_lower = config_name.lower()
    if cfg_lower == "k1":
        use_cbam = False
        use_non_local = False
    elif cfg_lower == "k2":
        use_cbam = True
        use_non_local = False
    elif cfg_lower == "k3":
        use_cbam = False
        use_non_local = True
    elif cfg_lower in ("k4", "k5"):
        use_cbam = True
        use_non_local = True
    else:
        use_cbam = bool(cfg["model"].get("use_cbam", True))
        use_non_local = bool(cfg["model"].get("use_non_local", True))

    model = EfficientNetB4_CBAM_NonLocal(
        pretrained=False,
        num_classes=int(cfg["model"]["num_classes"]),
        dropout_rate=float(cfg["model"]["dropout_rate"]),
        cbam_reduction=int(cfg["model"]["cbam_reduction_ratio"]),
        use_cbam=use_cbam,
        use_non_local=use_non_local
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint model tidak ditemukan di: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Eval] Model '{config_name.upper()}' berhasil dimuat dari checkpoint: {checkpoint_path}")
    return model


def get_fixed_test_loader(
    manifest_path: str = "src/logs/processed_samples_manifest.csv",
    batch_size: int = 16,
    num_workers: int = 2,
    dry_run: bool = False
) -> DataLoader:
    """
    Memuat subset testing FF++ C23 yang tetap (split == 'test') tanpa shuffling.
    """
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if dry_run:
        print("[Eval] Mode dry-run: Menggunakan 16 sampel sintetis untuk evaluasi pengujian...")
        dummy_x = torch.randn(16, 3, 380, 380)
        dummy_y = torch.tensor([0, 1] * 8)
        dummy_ds = torch.utils.data.TensorDataset(dummy_x, dummy_y)
        return DataLoader(dummy_ds, batch_size=batch_size, shuffle=False)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"File manifest tidak ditemukan di: {manifest_path}. "
            "Pastikan run_pipeline.py sudah dijalankan untuk mengekstraksi subset testing."
        )

    df_manifest = pd.read_csv(manifest_path)
    test_df = df_manifest[df_manifest["split"] == "test"]
    if len(test_df) == 0:
        raise ValueError(f"Tidak ada sampel dengan split='test' di dalam manifest: {manifest_path}")

    print(f"[Eval] Memuat subset testing tetap: {len(test_df)} frame dari {manifest_path}")
    test_samples = list(zip(test_df["image_path"], test_df["label"]))
    test_dataset = DeepfakeFaceDataset(test_samples, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_loader


def evaluate_checkpoint(
    checkpoint_path: str,
    config_name: str = "k5",
    config_path: str = "src/config.yaml",
    manifest_path: str = "src/logs/processed_samples_manifest.csv",
    output_dir: str = "src/logs",
    batch_size: int = 16,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Menjalankan forward pass pada satu subset test FF++ C23 yang sama,
    menghitung metrik klasifikasi dan confusion matrix, lalu menyimpannya ke file JSON.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Menjalankan evaluasi pada device: {device}")

    # 1. Muat DataLoader testing tetap
    test_loader = get_fixed_test_loader(
        manifest_path=manifest_path,
        batch_size=batch_size,
        dry_run=dry_run
    )

    # 2. Muat model dari checkpoint
    model = load_model_for_evaluation(
        checkpoint_path=checkpoint_path,
        config_name=config_name,
        config_path=config_path,
        device=device
    )

    # 3. Forward Pass Pengujian
    y_true_list = []
    y_pred_list = []
    y_prob_fake_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {config_name.upper()}"):
            images, labels = batch
            images = images.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=-1)

            # Probabilitas kelas Fake (indeks 1)
            fake_probs = probs[:, 1].cpu().numpy()
            _, preds = torch.max(logits, dim=1)

            y_true_list.extend(labels.numpy().tolist())
            y_pred_list.extend(preds.cpu().numpy().tolist())
            y_prob_fake_list.extend(fake_probs.tolist())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(y_prob_fake_list)

    # 4. Perhitungan Metrik Evaluasi
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    results = {
        "config_name": config_name.lower(),
        "checkpoint": checkpoint_path,
        "num_test_samples": int(len(y_true)),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "raw_matrix": cm.tolist()
        }
    }

    # 5. Simpan ke File JSON Terpisah per Konfigurasi
    os.makedirs(output_dir, exist_ok=True)
    out_json_path = os.path.join(output_dir, f"metrics_{config_name.lower()}.json")
    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 60)
    print(f"HASIL EVALUASI KONFIGURASI: {config_name.upper()}")
    print("=" * 60)
    print(f"File Checkpoint     : {checkpoint_path}")
    print(f"Jumlah Sampel Uji   : {len(y_true)}")
    print(f"Accuracy            : {acc * 100:.2f}%")
    print(f"Precision           : {prec * 100:.2f}%")
    print(f"Recall              : {rec * 100:.2f}%")
    print(f"F1-Score            : {f1 * 100:.2f}%")
    print(f"AUC-ROC             : {auc:.4f}")
    print(f"Confusion Matrix    : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Hasil disimpan ke   : {out_json_path}")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi Model Deteksi Deepfake pada Subset Testing FF++ C23")
    parser.add_argument("--config", type=str, default="src/config.yaml", help="Path ke config.yaml")
    parser.add_argument("--config-name", type=str, default="k5", help="Nama konfigurasi ablation (k1, k2, k3, k4, k5)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path ke file checkpoint .pt/.pth")
    parser.add_argument("--manifest", type=str, default="src/logs/processed_samples_manifest.csv", help="Path manifest sampel")
    parser.add_argument("--output-dir", type=str, default="src/logs", help="Direktori penyimpanan file JSON metrik")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size evaluasi")
    parser.add_argument("--dry-run", action="store_true", help="Jalankan evaluasi menggunakan data sintetis")
    args = parser.parse_args()

    cfg_name = args.config_name.lower()
    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = os.path.join("checkpoints", f"{cfg_name}_best.pt")

    manifest = args.manifest
    if manifest == "src/logs/processed_samples_manifest.csv":
        if cfg_name in ("k1", "k2", "k3", "k4"):
            candidate = "src/logs/manifest_uniform.csv"
            if os.path.exists(candidate):
                manifest = candidate
        else:
            candidate = "src/logs/manifest_gfs.csv"
            if os.path.exists(candidate):
                manifest = candidate

    evaluate_checkpoint(
        checkpoint_path=ckpt,
        config_name=cfg_name,
        config_path=args.config,
        manifest_path=manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
