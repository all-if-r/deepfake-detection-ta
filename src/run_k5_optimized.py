"""
Script Pipeline Terpadu K5 Optimized (Model Usulan Penuh yang Disempurnakan):
- GFS baru: theta_diff=1.5 dengan mekanisme fallback two-tier (menjamin N=20 frame per video).
- Penyimpanan mandiri terisolasi di folder k5_optimized/:
  * k5_optimized/processed_data/         (Citra wajah hasil GFS baru)
  * k5_optimized/manifest_gfs.csv        (Manifest dataset GFS baru)
  * k5_optimized/gfs_statistics.csv      (Statistik GFS)
  * k5_optimized/training_history_k5.csv (Riwayat training baru)
  * k5_optimized/k5_best.pt              (Checkpoint model terbaik)
  * k5_optimized/k5_latest.pt            (Checkpoint model terakhir)
  * k5_optimized/metrics_k5.json         (Metrik evaluasi: default & optimal threshold)
- Seluruh file K5 versi lama di checkpoints/ dan src/logs/ tetap aman dan utuh.
- Training default dari Epoch 1 (fresh start) dengan early stopping patience = 15 dan FP32 (--no-amp).
"""

import os
import sys
import time
import argparse
import yaml

from run_pipeline import run_gfs_and_preprocessing
from train import train_pipeline
from evaluate import evaluate_checkpoint

TARGET_DIR = "k5_optimized"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Penuh K5 Optimized (GFS Baru + Training Baru + Evaluasi)")
    parser.add_argument("--config", type=str, default="src/config.yaml", help="Path ke config.yaml")
    parser.add_argument("--output-folder", type=str, default=TARGET_DIR, help="Direktori khusus output K5 baru")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training jika ada checkpoint latest di output-folder")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Lewati ekstraksi jika k5_optimized/manifest_gfs.csv sudah selesai")
    parser.add_argument("--max-videos", type=int, default=None, help="Batas video per kategori (hanya untuk testing cepat)")
    parser.add_argument("--split", type=str, default=None, help="Filter split tertentu (train/val/test)")
    parser.add_argument("--use-amp", dest="use_amp", action="store_true", default=False, help="Paksa gunakan AMP (FP16)")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", help="Jalankan presisi penuh FP32 tanpa AMP (Default & Direkomendasikan)")
    args = parser.parse_args()

    start_total = time.time()
    out_dir = args.output_folder
    os.makedirs(out_dir, exist_ok=True)
    processed_images_dir = os.path.join(out_dir, "processed_data")
    manifest_csv = os.path.join(out_dir, "manifest_gfs.csv")
    stats_csv = os.path.join(out_dir, "gfs_statistics.csv")
    k5_best_ckpt = os.path.join(out_dir, "k5_best.pt")

    print("=" * 75, flush=True)
    print("MEMULAI PIPELINE K5 OPTIMIZED (PROPOSED MODEL DENGAN GFS BARU)", flush=True)
    print(f"Folder Khusus Output   : {out_dir}/", flush=True)
    print(f"Mode Presisi Pelatihan : {'AMP (FP16)' if args.use_amp else 'FP32 (No AMP - Stabilitas Non-Local)'}", flush=True)
    print(f"Mode Training          : {'Resume dari latest' if args.resume else 'Mulai dari Epoch 1 (Fresh Start)'}", flush=True)
    print("=" * 75, flush=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # --------------------------------------------------------------------------
    # TAHAP 1: Ekstraksi Golden Frame Selection (GFS) Baru (θ_diff = 1.5, N = 20)
    # --------------------------------------------------------------------------
    if not os.path.exists(manifest_csv) and not args.skip_preprocessing:
        print(f"\n[TAHAP 1/3] Menjalankan Preprocessing GFS Baru (θ_diff=1.5, Garansi N=20)...", flush=True)
        manifest_csv, stats_csv = run_gfs_and_preprocessing(
            config_path=args.config,
            max_videos_per_cat=args.max_videos,
            target_split=args.split,
            use_gfs=True,
            custom_processed_dir=processed_images_dir,
            custom_logs_dir=out_dir,
            custom_manifest_name="manifest_gfs.csv",
            custom_stats_name="gfs_statistics.csv"
        )
    else:
        print(f"\n[TAHAP 1/3] Menggunakan manifest GFS yang sudah siap di: {manifest_csv}", flush=True)

    # --------------------------------------------------------------------------
    # TAHAP 2: Pelatihan Model K5 Baru (Mulai Epoch 1 dengan Early Stopping)
    # --------------------------------------------------------------------------
    print(f"\n[TAHAP 2/3] Memulai Pelatihan Model K5 Optimized...", flush=True)
    train_results = train_pipeline(
        config_path=args.config,
        config_name="k5",
        dry_run=False,
        custom_epochs=args.epochs,
        resume=args.resume,
        custom_checkpoints_dir=out_dir,
        custom_logs_dir=out_dir,
        custom_manifest_path=manifest_csv,
        custom_data_dir=out_dir,
        custom_batch_size=args.batch_size,
        custom_use_amp=args.use_amp
    )

    # --------------------------------------------------------------------------
    # TAHAP 3: Evaluasi Model K5 Baru (Default Threshold & Optimal Threshold)
    # --------------------------------------------------------------------------
    print(f"\n[TAHAP 3/3] Menjalankan Evaluasi Model K5 Optimized pada Subset Testing...", flush=True)
    if os.path.exists(k5_best_ckpt):
        eval_results = evaluate_checkpoint(
            checkpoint_path=k5_best_ckpt,
            config_name="k5",
            config_path=args.config,
            manifest_path=manifest_csv,
            output_dir=out_dir
        )
    else:
        print(f"[Warning] Checkpoint {k5_best_ckpt} belum ditemukan, evaluasi dilewati.")

    total_hours = (time.time() - start_total) / 3600
    print("\n" + "=" * 75, flush=True)
    print(f"[SELESAI] Pipeline K5 Optimized selesai dalam {total_hours:.2f} jam!", flush=True)
    print(f"Semua output tersimpan rapi di: {os.path.abspath(out_dir)}", flush=True)
    print("=" * 75 + "\n", flush=True)
