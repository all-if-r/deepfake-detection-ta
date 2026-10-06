"""
Script Khusus Ekstraksi & Preprocessing Wajah Golden Frame Selection (GFS) untuk K5:
Menghasilkan output di:
- Direktori citra: processed_data_gfs/
- Manifest dataset: src/logs/manifest_gfs.csv
- Statistik GFS: src/logs/gfs_statistics.csv
Dapat dijalankan secara independen di cloud atau lokal sebelum memulai training K5.
"""

import os
import sys
import time
import argparse
import yaml

from run_pipeline import run_gfs_and_preprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstraksi Golden Frame Selection (GFS) untuk Konfigurasi K5")
    parser.add_argument("--config", type=str, default="src/config.yaml", help="Path ke config.yaml")
    parser.add_argument("--output-folder", type=str, default="k5_optimized", help="Direktori target output (default: k5_optimized)")
    parser.add_argument("--max-videos", type=int, default=None, help="Batas video per kategori (opsional untuk uji coba)")
    parser.add_argument("--split", type=str, default=None, help="Filter hanya split tertentu (train/val/test)")
    args = parser.parse_args()

    start_time = time.time()
    out_dir = args.output_folder
    os.makedirs(out_dir, exist_ok=True)
    processed_images_dir = os.path.join(out_dir, "processed_data")

    print("=" * 70, flush=True)
    print("MEMULAI EKSTRAKSI FRAME MENGGUNAKAN GOLDEN FRAME SELECTION (GFS) - K5", flush=True)
    print(f"Target Output Folder: {out_dir}/", flush=True)
    print("=" * 70, flush=True)

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

    elapsed = (time.time() - start_time) / 3600
    print("=" * 70, flush=True)
    print(f"[SELESAI] Ekstraksi GFS tuntas dalam {elapsed:.2f} jam!", flush=True)
    print(f"Manifest tersimpan di: {manifest_csv}", flush=True)
    print(f"Statistik GFS di: {stats_csv}", flush=True)
    print("=" * 70, flush=True)
