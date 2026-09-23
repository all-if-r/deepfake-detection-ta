"""
Modul Dataset & Video-Level Split FF++ C23 (Tugas 4 & 5)
Sesuai Proposal & Subbab 4.1.1:
1. Video-level split (720 train : 140 val : 140 test) pada 1.000 video Real.
2. Replikasi proporsional pada 6.000 video Fake berdasarkan sequence ID sumber untuk mencegah data leakage.
3. PyTorch Dataset & DataLoader hemat memori untuk batch training.
"""

import os
import glob
import re
import yaml
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Dict, Tuple, Optional


import json


def load_official_ff_splits(splits_dir: str = "FaceForensics++_C23/splits") -> Dict[str, str]:
    """
    Memuat split resmi FaceForensics++ (train.json, val.json, test.json).
    Mengembalikan mapping: video_id_3digit (str) -> split_name ("train", "val", "test").
    Kedua elemen pasangan [id1, id2] digabung ke dalam himpunan split terkait.
    """
    id_to_split = {}
    for split_name in ["train", "val", "test"]:
        json_path = os.path.join(splits_dir, f"{split_name}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File split resmi tidak ditemukan: {json_path}")
        with open(json_path, "r") as f:
            pairs = json.load(f)
        for pair in pairs:
            if isinstance(pair, list):
                for vid_id in pair:
                    id_to_split[str(vid_id).zfill(3)] = split_name
            else:
                id_to_split[str(pair).zfill(3)] = split_name
    return id_to_split


def generate_or_load_dfd_actor_split(
    dfd_csv_path: str = "FaceForensics++_C23/csv/DeepFakeDetection.csv",
    save_path: str = "src/logs/dfd_actor_split.json",
    targets: Dict[str, int] = None
) -> Dict[str, str]:
    """
    Memuat atau membangun alokasi actor-disjoint untuk DeepFakeDetection menggunakan
    pendekatan greedy load balancing (target proporsi 72% train, 14% val, 14% test).
    """
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            data = json.load(f)
            if "allocation" in data:
                return data["allocation"]
            return data

    if targets is None:
        targets = {"train": 720, "val": 140, "test": 140}

    if not os.path.exists(dfd_csv_path):
        return {}

    df = pd.read_csv(dfd_csv_path)
    pos1_counts = {}
    for p in df["File Path"]:
        m = re.match(r"^DeepFakeDetection/(\d+)_", str(p))
        if m:
            actor = m.group(1)
            pos1_counts[actor] = pos1_counts.get(actor, 0) + 1

    sorted_actors = sorted(pos1_counts.items(), key=lambda x: (x[1], -int(x[0])), reverse=True)
    curr = {"train": 0, "val": 0, "test": 0}
    alloc = {}

    for actor, count in sorted_actors:
        best_subset = max(["train", "val", "test"], key=lambda s: targets[s] - curr[s])
        alloc[actor] = best_subset
        curr[best_subset] += count

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({
            "allocation": alloc,
            "summary": {
                "train_videos": curr["train"],
                "val_videos": curr["val"],
                "test_videos": curr["test"],
                "total_videos": sum(curr.values()),
                "train_pct": round(curr["train"] / 1000 * 100, 2),
                "val_pct": round(curr["val"] / 1000 * 100, 2),
                "test_pct": round(curr["test"] / 1000 * 100, 2),
                "train_dev": round((curr["train"] / 1000 * 100) - 72.0, 2),
                "val_dev": round((curr["val"] / 1000 * 100) - 14.0, 2),
                "test_dev": round((curr["test"] / 1000 * 100) - 14.0, 2)
            }
        }, f, indent=4)

    return alloc


def extract_video_id(file_path: str) -> Optional[str]:
    """
    Mengekstrak ID numerik video sumber pertama dari file path.
    - Untuk DeepFakeDetection: string 2 digit actor_id (contoh: '01_02...' -> '01')
    - Untuk kategori lain: string 3 digit video_id (contoh: '000.mp4' -> '000', '725_120.mp4' -> '725')
    """
    basename = os.path.basename(file_path)
    match = re.match(r"^(\d+)", basename)
    if match:
        raw_id = match.group(1)
        if "DeepFakeDetection" in file_path:
            return raw_id.zfill(2)
        return raw_id.zfill(3)
    return None


def get_video_split_assignment(
    video_id: str, 
    category: str = "original",
    ff_splits: Dict[str, str] = None,
    dfd_splits: Dict[str, str] = None
) -> str:
    """
    Memetakan video_id atau actor_id ke partisi resmi:
    - Jika kategori DeepFakeDetection: menggunakan dfd_actor_split.json
    - Jika kategori lain: menggunakan train.json / val.json / test.json FaceForensics++ resmi
    """
    if category == "DeepFakeDetection":
        if dfd_splits is None:
            dfd_splits = generate_or_load_dfd_actor_split()
        return dfd_splits.get(str(video_id).zfill(2), "train")
    else:
        if ff_splits is None:
            ff_splits = load_official_ff_splits()
        return ff_splits.get(str(video_id).zfill(3), "train")


def build_video_split_metadata(
    csv_dir: str = "FaceForensics++_C23/csv",
    splits_dir: str = "FaceForensics++_C23/splits",
    dfd_split_path: str = "src/logs/dfd_actor_split.json"
) -> pd.DataFrame:
    """
    Membaca seluruh katalog CSV dan membangun metadata split tingkat video
    berdasarkan partisi resmi FaceForensics++ dan actor-disjoint split DeepFakeDetection.
    """
    ff_splits = load_official_ff_splits(splits_dir)
    dfd_splits = generate_or_load_dfd_actor_split(save_path=dfd_split_path)

    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    records = []

    for fpath in sorted(csv_files):
        fname = os.path.basename(fpath)
        if "Metadata" in fname or "Mean_Data" in fname:
            continue

        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            rel_path = str(row.get("File Path", ""))
            label_str = str(row.get("Label", "")).upper()
            
            if not rel_path or pd.isna(rel_path):
                continue

            category = rel_path.split("/")[0] if "/" in rel_path else rel_path.split("\\")[0]
            vid_id = extract_video_id(rel_path)
            if vid_id is None:
                continue

            split = get_video_split_assignment(vid_id, category=category, ff_splits=ff_splits, dfd_splits=dfd_splits)
            label_binary = 0 if label_str == "REAL" else 1

            records.append({
                "file_path": rel_path,
                "category": category,
                "video_id": vid_id,
                "label_str": label_str,
                "label": label_binary,
                "split": split,
                "frame_count": row.get("Frame Count", 0)
            })

    split_df = pd.DataFrame(records)
    return split_df


class DeepfakeFaceDataset(Dataset):
    """
    PyTorch Dataset untuk membaca citra wajah hasil GFS + Preprocessing.
    """
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        """
        samples: List of (image_file_path, label)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img_bgr = cv2.imread(img_path)
        
        if img_bgr is None:
            raise FileNotFoundError(f"Citra tidak ditemukan atau rusak: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            img_tensor = self.transform(img_rgb)
        else:
            # Default PyTorch tensor transformation & ImageNet normalization
            img_tensor = transforms.functional.to_tensor(img_rgb)
            img_tensor = transforms.functional.normalize(
                img_tensor, 
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )

        return img_tensor, label


def get_dataloaders(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Membuat DataLoader untuk training dan validasi dengan augmentasi on-the-fly ringan & normalisasi ImageNet.
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DeepfakeFaceDataset(train_samples, transform=train_transform)
    val_dataset = DeepfakeFaceDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader
