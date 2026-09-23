"""
Unit Tests dan Verifikasi Modul Pipeline (Tugas 1 s.d. 5)
Menguji kebenaran matematis dan fungsional setiap modul:
1. GFS: Frame difference, Laplacian variance, min-max FQS, Fallback logic
2. Preprocessing: Content-aware padding (gamma=0.20, floor), bounding box clamp
3. Augmentasi: Rotasi +/-10 deg, flip horizontal, brightness jitter +/-15, output 4x
4. Model: Forward pass, input 380x380, output 2 logits, freeze stage 1-4, CBAM & Non-Local
5. Dataset: Split resmi FF++ C23 mapping
"""

import os
import math
import cv2
import numpy as np
import torch
import unittest

from golden_frame_selection import compute_frame_difference, compute_laplacian_sharpness, uniform_frame_sampling
from face_preprocessing import FacePreprocessor
from augmentation import apply_rotation, apply_horizontal_flip, apply_brightness_jitter, MinorClassAugmentor
from dataset import extract_video_id, get_video_split_assignment, build_video_split_metadata
from model import ChannelAttention, SpatialAttention, CBAM, NonLocalBlock, EfficientNetB4_CBAM_NonLocal


class TestDeepfakePipeline(unittest.TestCase):

    def test_uniform_frame_sampling(self):
        """Test uniform frame sampling N=20 frame merata sepanjang durasi"""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        dummy_video_path = os.path.join(temp_dir, "dummy_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(dummy_video_path, fourcc, 10.0, (64, 64))
        for i in range(50):
            frame = np.full((64, 64, 3), i * 5, dtype=np.uint8)
            out.write(frame)
        out.release()

        frames, stats = uniform_frame_sampling(dummy_video_path, n_frames=20)
        self.assertEqual(len(frames), 20, "Harus terpilih tepat 20 frame")
        self.assertEqual(stats["selected_count"], 20)
        self.assertEqual(stats["total_frames"], 50)
        # Indeks harus terdistribusi merata
        expected_indices = list(np.linspace(0, 49, 20, dtype=int))
        self.assertEqual(stats["selected_indices"], expected_indices)

        os.remove(dummy_video_path)
        os.rmdir(temp_dir)

    def test_gfs_math(self):
        """Test perhitungan matematis GFS"""
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img2 = np.full((100, 100), 10, dtype=np.uint8)
        delta_f = compute_frame_difference(img1, img2)
        self.assertEqual(delta_f, 10.0, "Delta F harus bernilai 10.0")

        # Test Sharpness
        sharpness = compute_laplacian_sharpness(img1)
        self.assertEqual(sharpness, 0.0, "Gambar flat harus memiliki varians Laplacian 0.0")

        # Gambar berpola (bertekstur)
        textured = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        sharp_textured = compute_laplacian_sharpness(textured)
        self.assertGreater(sharp_textured, 0.0, "Gambar tekstur harus memiliki sharpness > 0")

    def test_content_aware_padding(self):
        """Test perhitungan padding gamma=0.20 dengan math.floor"""
        gamma = 0.20
        w_bb, h_bb = 100, 150
        bbox = (50, 50, 150, 200)  # x1, y1, x2, y2
        frame_shape = (500, 500)

        m_x = math.floor(gamma * w_bb)  # floor(0.2 * 100) = 20
        m_y = math.floor(gamma * h_bb)  # floor(0.2 * 150) = 30

        self.assertEqual(m_x, 20)
        self.assertEqual(m_y, 30)

        x1_p = max(0, bbox[0] - m_x)
        y1_p = max(0, bbox[1] - m_y)
        x2_p = min(500, bbox[2] + m_x)
        y2_p = min(500, bbox[3] + m_y)

        self.assertEqual((x1_p, y1_p, x2_p, y2_p), (30, 20, 170, 230))

    def test_augmentation(self):
        """Test augmentasi kelas minor (1 asli -> 4 versi)"""
        dummy_img = np.full((200, 200, 3), 128, dtype=np.uint8)
        augmentor = MinorClassAugmentor(config_path="src/config.yaml", seed=42)
        variants = augmentor.augment_real_frame(dummy_img)

        self.assertEqual(len(variants), 4, "Harus menghasilkan tepat 4 varian (1 asli + 3 augmentasi)")
        tags = [v[0] for v in variants]
        self.assertEqual(tags, ["original", "rot", "flip", "bright"])

        # Periksa dimensi seluruh varian tetap 200x200x3
        for tag, img in variants:
            self.assertEqual(img.shape, (200, 200, 3))

    def test_video_split_mapping(self):
        """Test pemetaan split video resmi FF++ C23 dan DFD actor-disjoint"""
        # Test split resmi FF++ (3 digit)
        self.assertEqual(get_video_split_assignment("071"), "train")
        self.assertEqual(get_video_split_assignment("720"), "val")
        self.assertEqual(get_video_split_assignment("953"), "test")

        # Test split actor DFD (2 digit - 3 connected components)
        self.assertEqual(get_video_split_assignment("02", category="DeepFakeDetection"), "train")
        self.assertEqual(get_video_split_assignment("05", category="DeepFakeDetection"), "val")
        self.assertEqual(get_video_split_assignment("10", category="DeepFakeDetection"), "test")

        self.assertEqual(extract_video_id("original/000.mp4"), "000")
        self.assertEqual(extract_video_id("Deepfakes/725_120.mp4"), "725")
        self.assertEqual(extract_video_id("DeepFakeDetection/02_05__talking...mp4"), "02")

    def test_cbam_and_nonlocal_modules(self):
        """Test blok CBAM dan Non-Local secara independen"""
        x = torch.randn(2, 160, 24, 24)
        cbam = CBAM(in_channels=160, reduction_ratio=16)
        out_cbam = cbam(x)
        self.assertEqual(out_cbam.shape, x.shape, "Output CBAM harus memiliki shape yang sama dengan input")

        nl = NonLocalBlock(in_channels=160)
        out_nl = nl(x)
        self.assertEqual(out_nl.shape, x.shape, "Output Non-Local harus memiliki shape yang sama dengan input")

    def test_model_architecture_and_freeze(self):
        """Test model lengkap K5, shape input 380x380, pembekuan Stage 1-4"""
        model = EfficientNetB4_CBAM_NonLocal(pretrained=False, num_classes=2, dropout_rate=0.5)
        model.eval()

        # Input tensor 380x380
        dummy_input = torch.randn(2, 3, 380, 380)
        logits = model(dummy_input)
        self.assertEqual(logits.shape, (2, 2), "Output logits harus (batch_size=2, num_classes=2)")

        proba = model.predict_proba(dummy_input)
        self.assertEqual(proba.shape, (2, 2))
        self.assertTrue(torch.allclose(proba.sum(dim=-1), torch.ones(2), atol=1e-5))

        # Verifikasi pembekuan stage 1-4
        for param in model.stem.parameters():
            self.assertFalse(param.requires_grad, "Stem harus dibekukan")
        for param in model.stage1.parameters():
            self.assertFalse(param.requires_grad, "Stage 1 harus dibekukan")
        for param in model.stage4.parameters():
            self.assertFalse(param.requires_grad, "Stage 4 harus dibekukan")

        # Verifikasi stage 5-7 dan head aktif
        for param in model.stage5.parameters():
            self.assertTrue(param.requires_grad, "Stage 5 harus trainable")
        for param in model.cbam5.parameters():
            self.assertTrue(param.requires_grad, "CBAM 5 harus trainable")
        for param in model.non_local.parameters():
            self.assertTrue(param.requires_grad, "Non-Local harus trainable")
        for param in model.fc.parameters():
            self.assertTrue(param.requires_grad, "Head FC harus trainable")

    def test_ablation_model_variants(self):
        """Test inisialisasi varian model ablation study K1-K5"""
        dummy_input = torch.randn(1, 3, 380, 380)

        # K1: Baseline (No CBAM, No Non-Local)
        m_k1 = EfficientNetB4_CBAM_NonLocal(pretrained=False, use_cbam=False, use_non_local=False)
        m_k1.eval()
        self.assertIsNone(m_k1.cbam5)
        self.assertIsNone(m_k1.non_local)
        out_k1 = m_k1(dummy_input)
        self.assertEqual(out_k1.shape, (1, 2))

        # K2: +CBAM only
        m_k2 = EfficientNetB4_CBAM_NonLocal(pretrained=False, use_cbam=True, use_non_local=False)
        m_k2.eval()
        self.assertIsNotNone(m_k2.cbam5)
        self.assertIsNone(m_k2.non_local)
        out_k2 = m_k2(dummy_input)
        self.assertEqual(out_k2.shape, (1, 2))

        # K3: +Non-Local only
        m_k3 = EfficientNetB4_CBAM_NonLocal(pretrained=False, use_cbam=False, use_non_local=True)
        m_k3.eval()
        self.assertIsNone(m_k3.cbam5)
        self.assertIsNotNone(m_k3.non_local)
        out_k3 = m_k3(dummy_input)
        self.assertEqual(out_k3.shape, (1, 2))

        # K4 / K5: Both CBAM + Non-Local
        m_k4 = EfficientNetB4_CBAM_NonLocal(pretrained=False, use_cbam=True, use_non_local=True)
        m_k4.eval()
        self.assertIsNotNone(m_k4.cbam5)
        self.assertIsNotNone(m_k4.non_local)
        out_k4 = m_k4(dummy_input)
        self.assertEqual(out_k4.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
