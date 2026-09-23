"""
Script Terpadu Menjalankan Pipeline Penuh K1:
1. Ekstraksi frame & preprocessing wajah dengan Uniform Sampling N=20 (K1-K4) untuk seluruh 7 kategori (5.158 train / 903 val / 939 test).
2. Otomatis melanjutkan ke Training K1 (EfficientNet-B4 baseline tanpa CBAM dan tanpa Non-Local).
3. Evaluasi akhir model K1 terbaik pada subset testing tetap (939 video / ~18.780 frame wajah).
"""

import os
import sys
import time
import argparse
import yaml

from run_pipeline import run_gfs_and_preprocessing
from train import train_pipeline
from evaluate import evaluate_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Penuh K1 Baseline")
    parser.add_argument("--resume", action="store_true", default=True, help="Otomatis resume dari latest checkpoint jika ada (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Mulai training dari epoch 1 (abaikan latest checkpoint)")
    args = parser.parse_args()

    start_total = time.time()
    print("=" * 70, flush=True)
    print("MEMULAI PIPELINE LENGKAP KONFIGURASI K1 (BASELINE UNIFORM SAMPLING)", flush=True)
    print("=" * 70, flush=True)

    # Langkah 1: Ekstraksi & Preprocessing Wajah (Uniform Sampling N=20)
    print("\n[TAHAP 1/3] Menjalankan Preprocessing Frame (Uniform Sampling N=20)...", flush=True)
    manifest_csv, stats_csv = run_gfs_and_preprocessing(
        config_path="src/config.yaml",
        use_gfs=False
    )
    print(f"[TAHAP 1 Selesai] Manifest: {manifest_csv}", flush=True)

    # Langkah 2: Training Penuh Model K1
    print("\n[TAHAP 2/3] Memulai Training Model K1 Penuh...", flush=True)
    train_results = train_pipeline(
        config_path="src/config.yaml",
        config_name="k1",
        dry_run=False,
        resume=args.resume
    )
    print(f"[TAHAP 2 Selesai] Checkpoint K1 tersimpan di checkpoints/k1_best.pt dan checkpoints/k1_latest.pt", flush=True)

    # Langkah 3: Evaluasi Model K1 pada Testing Set Tetap
    print("\n[TAHAP 3/3] Menjalankan Evaluasi Model K1 pada Testing Set...", flush=True)
    eval_results = evaluate_checkpoint(
        checkpoint_path="checkpoints/k1_best.pt",
        config_name="k1",
        config_path="src/config.yaml",
        manifest_path=manifest_csv,
        output_dir="src/logs"
    )

    total_time = (time.time() - start_total) / 3600
    print(f"\n[SELESAI] Seluruh alur K1 rampung dalam {total_time:.2f} jam!", flush=True)
