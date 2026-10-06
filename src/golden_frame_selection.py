"""
Modul Golden Frame Selection (GFS) - Tugas 1
Sesuai Proposal & Subbab 4.1.1:
1. Tahap 1: Penyaringan ΔF (frame difference) threshold θ_diff = 5.0
2. Tahap 2: Penyaringan kualitas intrinsik (Sharpness Laplacian > P30 video & YOLOv8 Conf >= 0.5)
3. FQS Calculation: FQS(f) = α * S_norm(f) + (1 - α) * C_yolo(f), α = 0.5
4. N=20 frame selection dengan mekanisme Fallback terstruktur
5. Logging statistik performa per video
"""

import os
import cv2
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


def compute_frame_difference(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Menghitung ΔF_t = rata-rata selisih absolut intensitas grayscale antara frame t dan t-1.
    """
    diff = cv2.absdiff(curr_gray, prev_gray)
    return float(np.mean(diff))


def compute_laplacian_sharpness(gray_img: np.ndarray) -> float:
    """
    Menghitung Sharpness S(f) = varians Laplacian dari frame grayscale.
    """
    lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    return float(lap.var())


class GoldenFrameSelector:
    def __init__(self, config_path: str = "src/config.yaml", yolo_detector = None):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        self.theta_diff = float(self.cfg["gfs"]["theta_diff"])
        self.p30_percentile = float(self.cfg["gfs"]["sharpness_percentile"])
        self.theta_conf = float(self.cfg["gfs"]["theta_conf"])
        self.alpha = float(self.cfg["gfs"]["alpha"])
        self.n_frames = int(self.cfg["gfs"]["n_frames"])
        
        self.yolo_detector = yolo_detector

    def set_detector(self, detector):
        self.yolo_detector = detector

    def process_video(self, video_path: str) -> Tuple[List[Dict], Dict]:
        """
        Menjalankan GFS 2-tahap pada satu video file.
        
        Returns:
            selected_frames: List of dict frame terpilih beserta metadata FQS
            stats: Dict statistik GFS untuk logging
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Tidak dapat membuka file video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # -------------------------------------------------------------
        # TAHAP 1: Penyaringan ΔF (Frame Difference)
        # -------------------------------------------------------------
        t1_passed_frames = []   # Menyimpan: {'frame_idx': int, 'bgr': np.ndarray, 'gray': np.ndarray, 'delta_f': float}
        failed_t1_count = 0

        prev_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is None:
                # Frame pertama: dianggap lolos sebagai baseline awal
                delta_f = self.theta_diff + 1.0
            else:
                delta_f = compute_frame_difference(prev_gray, gray)

            prev_gray = gray

            if delta_f >= self.theta_diff:
                t1_passed_frames.append({
                    "frame_idx": frame_idx,
                    "bgr": frame,
                    "gray": gray,
                    "delta_f": delta_f
                })
            else:
                failed_t1_count += 1
            
            frame_idx += 1

        cap.release()

        # -------------------------------------------------------------
        # TAHAP 2: Penyaringan Kualitas Intrinsik (Sharpness & YOLO Confidence)
        # -------------------------------------------------------------
        t2_passed_frames = []
        t2_failed_frames = []
        failed_t2_count = 0
        passed_t2_count = 0

        if t1_passed_frames:
            for item in t1_passed_frames:
                item["sharpness"] = compute_laplacian_sharpness(item["gray"])
                if self.yolo_detector is not None:
                    conf, bbox = self.yolo_detector.get_face_confidence_and_box(item["bgr"])
                    item["yolo_conf"] = conf
                    item["bbox"] = bbox
                else:
                    item["yolo_conf"] = 1.0
                    item["bbox"] = None

            sharpness_values = [item["sharpness"] for item in t1_passed_frames]
            p30_thresh = float(np.percentile(sharpness_values, self.p30_percentile))
            s_min = float(np.min(sharpness_values))
            s_max = float(np.max(sharpness_values))

            for item in t1_passed_frames:
                if s_max > s_min:
                    s_norm = (item["sharpness"] - s_min) / (s_max - s_min)
                else:
                    s_norm = 1.0
                item["s_norm"] = s_norm
                item["fqs"] = self.alpha * s_norm + (1.0 - self.alpha) * item["yolo_conf"]

                sharp_pass = item["sharpness"] >= p30_thresh
                conf_pass = item["yolo_conf"] >= self.theta_conf
                item["t2_pass"] = bool(sharp_pass and conf_pass)

            t2_passed_frames = [item for item in t1_passed_frames if item["t2_pass"]]
            t2_failed_frames = [item for item in t1_passed_frames if not item["t2_pass"]]
            failed_t2_count = len(t2_failed_frames)
            passed_t2_count = len(t2_passed_frames)

        # -------------------------------------------------------------
        # PEMILIHAN N FRAME TERBAIK & MEKANISME FALLBACK (GUARANTEED N=20)
        # -------------------------------------------------------------
        fallback_triggered = False
        if passed_t2_count >= self.n_frames:
            # Lolos Tahap 2 mencukupi: urutkan berdasarkan FQS menurun, ambil Top N
            t2_passed_frames.sort(key=lambda x: x["fqs"], reverse=True)
            selected = t2_passed_frames[:self.n_frames]
        else:
            # Fallback Tier 1: gabungkan semua frame lolos Tahap 1 (t2_pass + t2_fail)
            fallback_triggered = True
            pool = list(t1_passed_frames)
            pool.sort(key=lambda x: x["fqs"], reverse=True)
            selected = pool[:self.n_frames]

        # Fallback Tier 2: Jika frame terpilih masih kurang dari target n_frames
        if len(selected) < self.n_frames and total_frames > 0:
            fallback_triggered = True
            needed = self.n_frames - len(selected)
            existing_indices = set(x["frame_idx"] for x in selected)
            num_probes = min(total_frames, self.n_frames * 3)
            probe_indices = np.linspace(0, total_frames - 1, num_probes, dtype=int)
            candidates = [idx for idx in probe_indices if idx not in existing_indices]

            if candidates:
                cap_fb = cv2.VideoCapture(video_path)
                added_frames = []
                for c_idx in candidates:
                    cap_fb.set(cv2.CAP_PROP_POS_FRAMES, int(c_idx))
                    ret_fb, frame_fb = cap_fb.read()
                    if ret_fb:
                        gray_fb = cv2.cvtColor(frame_fb, cv2.COLOR_BGR2GRAY)
                        sharp_fb = compute_laplacian_sharpness(gray_fb)
                        conf_fb = 1.0
                        bbox_fb = None
                        if self.yolo_detector is not None:
                            conf_fb, bbox_fb = self.yolo_detector.get_face_confidence_and_box(frame_fb)
                        added_frames.append({
                            "frame_idx": int(c_idx),
                            "bgr": frame_fb,
                            "gray": gray_fb,
                            "sharpness": sharp_fb,
                            "yolo_conf": conf_fb,
                            "bbox": bbox_fb,
                            "fqs": 0.5 * (sharp_fb / 1000.0) + 0.5 * conf_fb,
                            "s_norm": 0.5,
                            "t2_pass": False
                        })
                cap_fb.release()

                added_frames.sort(key=lambda x: x["sharpness"], reverse=True)
                selected.extend(added_frames[:needed])

        # Urutkan berdasarkan frame_idx untuk konsistensi temporal
        selected.sort(key=lambda x: x["frame_idx"])
        selected_indices = [item["frame_idx"] for item in selected]

        stats = {
            "video_path": video_path,
            "total_frames": total_frames,
            "failed_t1": failed_t1_count,
            "passed_t1": len(t1_passed_frames),
            "failed_t2": failed_t2_count,
            "passed_t2": passed_t2_count,
            "fallback_triggered": fallback_triggered,
            "selected_count": len(selected),
            "selected_indices": selected_indices
        }

        return selected, stats


def export_gfs_statistics(all_stats: List[Dict], output_csv_path: str):
    """
    Mengekspor rekapan statistik GFS seluruh video ke CSV (untuk tugas nomor 6).
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df = pd.DataFrame(all_stats)
    df.to_csv(output_csv_path, index=False)
    print(f"[GFS] Statistik berhasil disimpan ke: {output_csv_path}")


def uniform_frame_sampling(video_path: str, n_frames: int = 20) -> Tuple[List[Dict], Dict]:
    """
    Uniform frame sampling untuk konfigurasi baseline / ablation K1-K4 (tanpa GFS).
    Mengambil N frame berjarak seragam sepanjang durasi video menggunakan linspace.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Tidak dapat membuka file video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return [], {
            "video_path": video_path,
            "total_frames": 0,
            "failed_t1": 0,
            "passed_t1": 0,
            "failed_t2": 0,
            "passed_t2": 0,
            "fallback_triggered": False,
            "selected_count": 0,
            "selected_indices": [],
            "method": "uniform"
        }

    if total_frames <= n_frames:
        target_indices = set(range(total_frames))
    else:
        target_indices = set(np.linspace(0, total_frames - 1, n_frames, dtype=int))

    selected = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_indices:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            selected.append({
                "frame_idx": frame_idx,
                "bgr": frame,
                "gray": gray,
                "method": "uniform"
            })
            if len(selected) >= n_frames:
                break

        frame_idx += 1

    cap.release()

    selected_indices = [item["frame_idx"] for item in selected]
    stats = {
        "video_path": video_path,
        "total_frames": total_frames,
        "failed_t1": 0,
        "passed_t1": len(selected),
        "failed_t2": 0,
        "passed_t2": len(selected),
        "fallback_triggered": False,
        "selected_count": len(selected),
        "selected_indices": selected_indices,
        "method": "uniform"
    }
    return selected, stats

