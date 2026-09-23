"""
Modul Preprocessing Wajah - Tugas 2
Sesuai Proposal & Subbab 4.1.1:
1. Deteksi Wajah YOLOv8 (arnabdhar/YOLOv8-Face-Detection) dengan confidence >= 0.5
2. Content-Aware Padding: Margin γ = 0.20, mx = floor(γ * W_bb), my = floor(γ * H_bb), di-clamp ke tepi frame
3. Alignment MTCNN: Deteksi 5 facial landmarks & transformasi afin ke posisi kanonik
4. Crop region wajah yang telah di-align
5. Resize ke 380x380 piksel (bilinear interpolation)
"""

import os
import math
import yaml
import cv2
import numpy as np
import torch
from typing import Tuple, Optional, List, Dict
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from facenet_pytorch import MTCNN


# Koordinat landmark kanonik standar 5-titik (untuk kanonik normalisasi 112x112 / standard alignment)
# Direferensikan ke proporsi standar wajah (ArcFace/SphereFace/FF++ alignment)
STANDARD_5_LANDMARKS_380 = np.array([
    [130.4, 160.0],  # Mata kiri
    [249.6, 160.0],  # Mata kanan
    [190.0, 220.0],  # Hidung
    [142.5, 290.0],  # Sudut mulut kiri
    [237.5, 290.0],  # Sudut mulut kanan
], dtype=np.float32)


def estimate_affine_transform(source_pts: np.ndarray, target_pts: np.ndarray) -> np.ndarray:
    """
    Menghitung transformasi afin 2D dengan least-squares menggunakan OpenCV.
    """
    matrix, inliers = cv2.estimateAffinePartial2D(source_pts, target_pts)
    if matrix is None:
        matrix = cv2.getAffineTransform(source_pts[:3].astype(np.float32), target_pts[:3].astype(np.float32))
    return matrix


class FacePreprocessor:
    def __init__(self, config_path: str = "src/config.yaml", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = device
        self.gamma = float(self.cfg["preprocessing"]["gamma_padding"])
        self.target_size = tuple(self.cfg["preprocessing"]["target_size"])
        self.yolo_conf_thresh = float(self.cfg["preprocessing"]["yolo_conf_thresh"])
        
        # Load YOLOv8 Face Detection Model dari HuggingFace arnabdhar/YOLOv8-Face-Detection
        self.yolo_model = self._load_yolo_model()
        
        # Load MTCNN untuk 5-point landmark alignment
        self.mtcnn = MTCNN(
            keep_all=False,
            device=self.device,
            select_largest=True,
            post_process=False
        )

    def _load_yolo_model(self) -> YOLO:
        repo_id = self.cfg["paths"]["yolo_model_name"]
        cache_path = self.cfg["paths"].get("yolo_weights_cache", "weights/yolov8_face.pt")
        
        if os.path.exists(cache_path):
            model_file = cache_path
        else:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            print(f"[YOLO] Mengunduh model bobot YOLOv8 Face dari {repo_id}...")
            model_file = hf_hub_download(repo_id=repo_id, filename="model.pt")
        
        yolo = YOLO(model_file)
        return yolo

    def get_face_confidence_and_box(self, frame_bgr: np.ndarray) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        """
        Deteksi wajah tercepat untuk GFS (Tahap 2).
        Returns: (confidence, (x1, y1, x2, y2))
        """
        results = self.yolo_model.predict(
            source=frame_bgr,
            conf=self.yolo_conf_thresh,
            verbose=False,
            device=self.device
        )
        
        if not results or len(results[0].boxes) == 0:
            return 0.0, None

        boxes = results[0].boxes
        # Ambil deteksi dengan confidence tertinggi
        best_idx = int(torch.argmax(boxes.conf).item())
        conf = float(boxes.conf[best_idx].item())
        xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        
        return conf, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))

    def apply_content_aware_padding(
        self, 
        bbox: Tuple[int, int, int, int], 
        frame_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Tahap 2 Preprocessing: Content-aware padding (margin γ = 0.20, floor calculation).
        """
        x1, y1, x2, y2 = bbox
        w_bb = x2 - x1
        h_bb = y2 - y1

        m_x = math.floor(self.gamma * w_bb)
        m_y = math.floor(self.gamma * h_bb)

        h_frame, w_frame = frame_shape[:2]

        x1_padded = max(0, x1 - m_x)
        y1_padded = max(0, y1 - m_y)
        x2_padded = min(w_frame, x2 + m_x)
        y2_padded = min(h_frame, y2 + m_y)

        return (x1_padded, y1_padded, x2_padded, y2_padded)

    def align_and_crop_face(
        self, 
        frame_bgr: np.ndarray, 
        padded_box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Tahap 3 & 4: Deteksi 5 landmark dengan MTCNN, transformasi afin ke kanonik, dan crop.
        """
        x1, y1, x2, y2 = padded_box
        padded_crop = frame_bgr[y1:y2, x1:x2]
        
        if padded_crop.size == 0:
            return cv2.resize(frame_bgr, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Convert ke RGB untuk MTCNN
        crop_rgb = cv2.cvtColor(padded_crop, cv2.COLOR_BGR2RGB)
        
        try:
            # Deteksi landmark pada region yang sudah di-pad
            _, _, landmarks = self.mtcnn.detect(crop_rgb, landmarks=True)
            
            if landmarks is not None and len(landmarks) > 0:
                pts = landmarks[0]  # 5 landmark (x, y) relatif terhadap padded_crop
                
                # Skalakan landmark target kanonik ke dimensi target_size (380x380)
                scale_x = self.target_size[0] / 380.0
                scale_y = self.target_size[1] / 380.0
                target_pts = STANDARD_5_LANDMARKS_380.copy()
                target_pts[:, 0] *= scale_x
                target_pts[:, 1] *= scale_y

                # Hitung transformasi afin least-squares
                affine_mat = estimate_affine_transform(pts, target_pts)
                
                # Warp afin langsung ke ukuran target 380x380
                aligned_face = cv2.warpAffine(
                    padded_crop,
                    affine_mat,
                    self.target_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
                return aligned_face
        except Exception:
            pass

        # Fallback jika MTCNN landmark gagal: resize langsung padded crop ke 380x380
        aligned_face = cv2.resize(padded_crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        return aligned_face

    def preprocess_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Menjalankan 5 tahap preprocessing lengkap pada satu frame:
        1. YOLOv8 face detection
        2. Content-aware padding (γ = 0.20)
        3. Alignment MTCNN
        4. Crop
        5. Resize ke 380x380 bilinear
        """
        conf, bbox = self.get_face_confidence_and_box(frame_bgr)
        if conf < self.yolo_conf_thresh or bbox is None:
            return None

        padded_box = self.apply_content_aware_padding(bbox, frame_bgr.shape)
        aligned_face = self.align_and_crop_face(frame_bgr, padded_box)
        
        # Pastikan output tepat 380x380
        if (aligned_face.shape[1], aligned_face.shape[0]) != self.target_size:
            aligned_face = cv2.resize(aligned_face, self.target_size, interpolation=cv2.INTER_LINEAR)

        return aligned_face

    def process_gfs_pool(self, gfs_candidate_frames: List[Dict]) -> List[np.ndarray]:
        """
        Memproses pool kandidat frame GFS secara berurutan sesuai FQS tertinggi.
        Jika deteksi wajah gagal pada suatu frame, mengambil frame berikutnya dari pool.
        """
        processed_faces = []
        
        for item in gfs_candidate_frames:
            frame_bgr = item["bgr"]
            face = self.preprocess_frame(frame_bgr)
            if face is not None:
                processed_faces.append(face)
            
            if len(processed_faces) >= int(self.cfg["gfs"]["n_frames"]):
                break
                
        return processed_faces
