"""
Modul Augmentasi Kelas Minor (Tugas 3)
Sesuai Proposal & Subbab 4.1.1:
- Diterapkan SETELAH GFS menghasilkan N=20 frame per video, SEBELUM tahap deteksi wajah YOLOv8.
- HANYA untuk frame kelas REAL pada subset TRAINING.
- Tiap frame real digandakan jadi 3 varian, sehingga total 4 versi per frame (1 asli + 3 augmentasi):
  1. Rotasi: Sudut acak pada rentang ±10° (matriks rotasi 2D standar terhadap titik pusat).
  2. Horizontal Flip: Deterministik satu kali per frame.
  3. Brightness Jitter: Pergeseran intensitas acak rentang ±15 (skala 0-255).
- Disimpan sebagai subset tetap (offline) ke disk.
"""

import os
import cv2
import yaml
import numpy as np
from typing import List, Dict, Tuple


def apply_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotasi citra dengan sudut angle_deg terhadap titik pusat citra.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image, 
        rot_mat, 
        (w, h), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REFLECT
    )
    return rotated


def apply_horizontal_flip(image: np.ndarray) -> np.ndarray:
    """
    Flip horizontal citra secara deterministik.
    """
    return cv2.flip(image, 1)


def apply_brightness_jitter(image: np.ndarray, shift: int) -> np.ndarray:
    """
    Pergeseran intensitas brightness acak pada rentang [-15, 15] dan di-clamp ke [0, 255].
    """
    img_int = image.astype(np.int16) + int(shift)
    clipped = np.clip(img_int, 0, 255).astype(np.uint8)
    return clipped


class MinorClassAugmentor:
    def __init__(self, config_path: str = "src/config.yaml", seed: int = 42):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.rot_min, self.rot_max = self.cfg["augmentation"]["rotation_range"]
        self.bright_min, self.bright_max = self.cfg["augmentation"]["brightness_range"]
        self.rng = np.random.default_rng(seed)

    def augment_real_frame(self, frame_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Menerima 1 frame BGR kelas real, menghasilkan 4 versi:
        (1 asli + 3 augmentasi: rotasi, flip, brightness jitter).
        
        Returns:
            List of (tag, image_bgr)
        """
        variants = []
        
        # 1. Citra Asli
        variants.append(("original", frame_bgr.copy()))

        # 2. Varian 1: Rotasi Acak ±10°
        angle = float(self.rng.uniform(self.rot_min, self.rot_max))
        rot_img = apply_rotation(frame_bgr, angle)
        variants.append(("rot", rot_img))

        # 3. Varian 2: Horizontal Flip Deterministik
        flip_img = apply_horizontal_flip(frame_bgr)
        variants.append(("flip", flip_img))

        # 4. Varian 3: Brightness Jitter Acak ±15
        shift = int(self.rng.integers(self.bright_min, self.bright_max + 1))
        bright_img = apply_brightness_jitter(frame_bgr, shift)
        variants.append(("bright", bright_img))

        return variants

    def augment_and_save_real_pool(
        self, 
        real_gfs_frames: List[Dict], 
        video_name: str, 
        output_dir: str
    ) -> List[str]:
        """
        Mengaugmentasi seluruh frame real terpilih dan menyimpannya ke disk (offline).
        
        Returns:
            List of saved image paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for idx, item in enumerate(real_gfs_frames):
            frame = item["bgr"]
            variants = self.augment_real_frame(frame)
            
            for tag, img in variants:
                filename = f"{video_name}_gfs{idx:02d}_{tag}.png"
                out_path = os.path.join(output_dir, filename)
                cv2.imwrite(out_path, img)
                saved_paths.append(out_path)

        return saved_paths
