"""
Script Terpadu Pipeline K4 (Ablasi: EfficientNet-B4 + CBAM + Non-Local, Uniform Sampling):
Menggunakan data hasil Uniform Sampling N=20 (manifest_uniform.csv yang sama dengan K1-K3).
Mendukung eksekusi di Cloud (Kaggle/Google Colab) maupun lokal terminal.
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
    parser = argparse.ArgumentParser(description="Pipeline Penuh K4 (CBAM + Non-Local, Uniform Sampling)")
    parser.add_argument("--config", type=str, default="src/config.yaml", help="Path ke config.yaml")
    parser.add_argument("--resume", action="store_true", default=True, help="Otomatis resume dari checkpoint terakhir jika ada")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Mulai training dari epoch 1")
    parser.add_argument("--checkpoints-dir", type=str, default=None, help="Direktori custom checkpoint (e.g. Google Drive / Kaggle)")
    parser.add_argument("--logs-dir", type=str, default=None, help="Direktori custom log")
    parser.add_argument("--manifest", type=str, default=None, help="Path custom manifest CSV")
    parser.add_argument("--data-dir", type=str, default=None, help="Root folder citra hasil preprocessing (jika dipindah)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch")
    parser.add_argument("--resume-from", type=str, default=None, help="Path spesifik ke file checkpoint {config}_latest.pt untuk resume")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Lewati tahap preprocessing jika manifest sudah ada")
    args = parser.parse_args()

    start_total = time.time()
    print("=" * 70, flush=True)
    print("MEMULAI PIPELINE KONFIGURASI K4 (CBAM + NON-LOCAL, UNIFORM SAMPLING)", flush=True)
    print("=" * 70, flush=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Pastikan manifest uniform sudah ada
    logs_dir = args.logs_dir if args.logs_dir else cfg["paths"]["logs_dir"]
    manifest_csv = args.manifest if args.manifest else os.path.join(logs_dir, "manifest_uniform.csv")

    if not os.path.exists(manifest_csv) and not args.skip_preprocessing:
        print("\n[TAHAP 1/3] Menjalankan Preprocessing Frame (Uniform Sampling N=20)...", flush=True)
        manifest_csv, _ = run_gfs_and_preprocessing(config_path=args.config, use_gfs=False)
    else:
        print(f"\n[TAHAP 1/3] Menggunakan manifest Uniform yang sudah ada: {manifest_csv}", flush=True)

    # 2. Training K4
    print("\n[TAHAP 2/3] Memulai Training Model K4 (CBAM + Non-Local)...", flush=True)
    train_results = train_pipeline(
        config_path=args.config,
        config_name="k4",
        dry_run=False,
        custom_epochs=args.epochs,
        resume=args.resume,
        custom_checkpoints_dir=args.checkpoints_dir,
        custom_logs_dir=args.logs_dir,
        custom_manifest_path=manifest_csv,
        custom_data_dir=args.data_dir,
        custom_batch_size=args.batch_size,
        resume_checkpoint_path=args.resume_from
    )

    # 3. Evaluasi
    checkpoints_dir = args.checkpoints_dir if args.checkpoints_dir else cfg["paths"]["checkpoints_dir"]
    k4_best = os.path.join(checkpoints_dir, "k4_best.pt")
    print(f"\n[TAHAP 3/3] Menjalankan Evaluasi Model K4 pada Testing Set...", flush=True)
    eval_results = evaluate_checkpoint(
        checkpoint_path=k4_best,
        config_name="k4",
        config_path=args.config,
        manifest_path=manifest_csv,
        output_dir=logs_dir
    )

    total_time = (time.time() - start_total) / 3600
    print(f"\n[SELESAI] Seluruh alur K4 rampung dalam {total_time:.2f} jam!", flush=True)
