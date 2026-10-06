"""
Pipeline End-to-End: GFS -> Preprocessing -> Augmentasi -> Training (Tugas 1 s.d. 6)
Script orkestrator untuk menjalankan seluruh tahapan secara otomatis atau per modul.
"""

import os
import glob
import time
import yaml
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm

from golden_frame_selection import GoldenFrameSelector, export_gfs_statistics, uniform_frame_sampling
from face_preprocessing import FacePreprocessor
from augmentation import MinorClassAugmentor
from dataset import build_video_split_metadata, get_dataloaders
from model import EfficientNetB4_CBAM_NonLocal
from train import train_one_epoch, validate_one_epoch, EarlyStopping


def run_gfs_and_preprocessing(
    config_path: str = "src/config.yaml",
    max_videos_per_cat: int = None,
    target_split: str = None,
    use_gfs: bool = None,
    custom_processed_dir: str = None,
    custom_logs_dir: str = None,
    custom_manifest_name: str = None,
    custom_stats_name: str = None
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if use_gfs is None:
        use_gfs = bool(cfg.get("gfs", {}).get("use_gfs", True))

    method_name = "Golden Frame Selection (GFS)" if use_gfs else "Uniform Sampling (N=20)"
    print(f"[Pipeline] Metode frame selection: {method_name}")

    if use_gfs:
        processed_dir = custom_processed_dir if custom_processed_dir else cfg["paths"].get("processed_dir_gfs", "processed_data_gfs")
        manifest_filename = custom_manifest_name if custom_manifest_name else "manifest_gfs.csv"
        gfs_stats_filename = custom_stats_name if custom_stats_name else "gfs_statistics.csv"
    else:
        processed_dir = custom_processed_dir if custom_processed_dir else cfg["paths"].get("processed_dir_uniform", "processed_data_uniform")
        manifest_filename = custom_manifest_name if custom_manifest_name else "manifest_uniform.csv"
        gfs_stats_filename = custom_stats_name if custom_stats_name else "uniform_statistics.csv"

    augmented_dir = cfg["paths"]["augmented_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(augmented_dir, exist_ok=True)

    print("[1/3] Membangun metadata split tingkat video...")
    split_df = build_video_split_metadata(cfg["paths"]["csv_dir"])
    print(f"Total video terdaftar di metadata: {len(split_df)}")
    print(split_df["split"].value_counts())

    if target_split:
        split_df = split_df[split_df["split"] == target_split].copy()

    # Inisialisasi Preprocessor & GFS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[2/3] Menginisialisasi YOLOv8 & MTCNN pada device: {device}...")
    preprocessor = FacePreprocessor(config_path=config_path, device=device)
    selector = GoldenFrameSelector(config_path=config_path, yolo_detector=preprocessor)
    augmentor = MinorClassAugmentor(config_path=config_path)

    # Inisialisasi daftar sampel & statistik (dengan dukungan Resume)
    logs_dir = custom_logs_dir if custom_logs_dir else cfg["paths"]["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)
    manifest_csv = os.path.join(logs_dir, manifest_filename)
    gfs_stats_csv = os.path.join(logs_dir, gfs_stats_filename)

    processed_samples = []
    processed_video_keys = set()
    if os.path.exists(manifest_csv):
        try:
            df_existing = pd.read_csv(manifest_csv)
            if not df_existing.empty and "image_path" in df_existing.columns and "category" in df_existing.columns:
                processed_samples = df_existing.to_dict("records")
                for _, r in df_existing.iterrows():
                    img_p = str(r["image_path"])
                    cat = str(r["category"])
                    b = os.path.basename(img_p)
                    prefix = f"{cat}_"
                    if b.startswith(prefix):
                        rem = b[len(prefix):]
                        parts = rem.split("_f")
                        if len(parts) >= 2:
                            vid_name = parts[0]
                            processed_video_keys.add((cat, vid_name))
                print(f"[Resume] Ditemukan {len(processed_video_keys)} video unik yang telah diproses sebelumnya ({len(processed_samples)} frame). Melanjutkan sisa...")
        except Exception as e:
            print(f"[Warning] Gagal membaca manifest yang ada ({e}), memulai dari awal.")

    all_gfs_stats = []
    if os.path.exists(gfs_stats_csv):
        try:
            df_stats_existing = pd.read_csv(gfs_stats_csv)
            all_gfs_stats = df_stats_existing.to_dict("records")
        except Exception:
            pass

    categories = split_df["category"].unique()
    videos_processed_this_session = 0
    
    for cat in categories:
        cat_df = split_df[split_df["category"] == cat]
        if max_videos_per_cat:
            cat_df = cat_df.head(max_videos_per_cat)

        print(f"\n---> Memproses kategori: {cat} ({len(cat_df)} video)")
        for _, row in tqdm(cat_df.iterrows(), total=len(cat_df), desc=f"GFS+Preproc {cat}"):
            rel_video_path = row["file_path"]
            vid_name = os.path.splitext(os.path.basename(rel_video_path))[0]
            vid_key = (str(cat), vid_name)
            if vid_key in processed_video_keys:
                continue
            full_video_path = os.path.join(cfg["paths"]["dataset_root"], rel_video_path)
            
            if not os.path.exists(full_video_path):
                # Coba cari path alternatif jika struktur folder sedikit berbeda
                full_video_path = os.path.join(cfg["paths"]["dataset_root"], rel_video_path.replace("/", os.sep))
                if not os.path.exists(full_video_path):
                    continue

            # 1. Jalankan Frame Selection (GFS atau Uniform Sampling N=20)
            try:
                if use_gfs:
                    selected_frames, stats = selector.process_video(full_video_path)
                else:
                    n_target = int(cfg.get("gfs", {}).get("n_frames", 20))
                    selected_frames, stats = uniform_frame_sampling(full_video_path, n_frames=n_target)
            except Exception as e:
                print(f"[Error Frame Selection] {full_video_path}: {e}")
                continue

            stats["category"] = cat
            stats["split"] = row["split"]
            stats["label"] = row["label"]
            all_gfs_stats.append(stats)

            if not selected_frames:
                continue

            vid_name = os.path.splitext(os.path.basename(rel_video_path))[0]
            is_train_real = (row["split"] == "train" and row["label"] == 0)

            # 2. Augmentasi Kelas Minor (HANYA untuk kelas REAL subset TRAINING)
            if is_train_real:
                for f_idx, item in enumerate(selected_frames):
                    variants = augmentor.augment_real_frame(item["bgr"])
                    for var_tag, var_img in variants:
                        face_crop = preprocessor.preprocess_frame(var_img)
                        if face_crop is not None:
                            out_fname = f"{cat}_{vid_name}_f{f_idx:02d}_{var_tag}.png"
                            out_path = os.path.join(processed_dir, out_fname)
                            cv2.imwrite(out_path, face_crop)
                            processed_samples.append({
                                "image_path": out_path,
                                "label": 0,
                                "split": "train",
                                "category": cat,
                                "video_id": row["video_id"],
                                "augmented": var_tag != "original"
                            })
            else:
                # Kelas Fake atau Val/Test: Preprocess langsung tanpa augmentasi
                faces = preprocessor.process_gfs_pool(selected_frames)
                for f_idx, face_crop in enumerate(faces):
                    out_fname = f"{cat}_{vid_name}_f{f_idx:02d}.png"
                    out_path = os.path.join(processed_dir, out_fname)
                    cv2.imwrite(out_path, face_crop)
                    processed_samples.append({
                        "image_path": out_path,
                        "label": row["label"],
                        "split": row["split"],
                        "category": cat,
                        "video_id": row["video_id"],
                        "augmented": False
                    })

            processed_video_keys.add(vid_key)
            videos_processed_this_session += 1

            # Simpan manifest berkala setiap 50 video agar tidak hilang progres jika interupsi
            if videos_processed_this_session % 50 == 0:
                pd.DataFrame(processed_samples).to_csv(manifest_csv, index=False)

    # Simpan statistik GFS / Uniform
    logs_dir = custom_logs_dir if custom_logs_dir else cfg["paths"]["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)
    gfs_stats_csv = os.path.join(logs_dir, gfs_stats_filename)
    export_gfs_statistics(all_gfs_stats, gfs_stats_csv)

    # Simpan katalog sampel yang berhasil diproses
    manifest_csv = os.path.join(logs_dir, manifest_filename)
    df_samples = pd.DataFrame(processed_samples)
    df_samples.to_csv(manifest_csv, index=False)
    # Simpan juga ke processed_samples_manifest.csv sebagai fallback jika default logs dir
    if not custom_logs_dir:
        fallback_manifest = os.path.join(logs_dir, "processed_samples_manifest.csv")
        df_samples.to_csv(fallback_manifest, index=False)
    print(f"[Preproc] Manifest sampel berhasil disimpan ke: {manifest_csv}")
    print(f"Total frame wajah tersimpan: {len(processed_samples)}")

    return manifest_csv, gfs_stats_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End GFS & Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="src/config.yaml")
    parser.add_argument("--max-videos", type=int, default=None, help="Batas video per kategori untuk uji coba")
    parser.add_argument("--split", type=str, default=None, help="Hanya jalankan split tertentu (train/val/test)")
    parser.add_argument("--use-gfs", dest="use_gfs", action="store_true", default=None, help="Gunakan Golden Frame Selection (K5)")
    parser.add_argument("--uniform", dest="use_gfs", action="store_false", help="Gunakan Uniform Sampling N=20 (K1-K4)")
    args = parser.parse_args()

    run_gfs_and_preprocessing(
        config_path=args.config,
        max_videos_per_cat=args.max_videos,
        target_split=args.split,
        use_gfs=args.use_gfs
    )
