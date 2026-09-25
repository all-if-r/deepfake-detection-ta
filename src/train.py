"""
Script Training Pipeline Penuh (Tugas 4 & 5)
Sesuai Proposal & Subbab 4.1.1:
1. Skema 1-fase: Stage 1-4 dibekukan, Stage 5-7 + CBAM + Non-Local + Head dilatih sejak epoch 1
2. Differential Learning Rate: lr_backbone = 1e-5, lr_head = 1e-4
3. Mixed Precision (AMP) & Gradient Accumulation untuk VRAM 4GB
4. Optimizer Adam, Scheduler ReduceLROnPlateau, Early Stopping (patience 15)
5. Logging riwayat training per epoch ke CSV & JSON
"""

import os
import time
import yaml
import argparse
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from model import EfficientNetB4_CBAM_NonLocal
from dataset import build_video_split_metadata, get_dataloaders, DeepfakeFaceDataset


class EarlyStopping:
    def __init__(self, patience: int = 15, delta: float = 1e-4, checkpoint_path: str = "checkpoints/best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint(model)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def _save_checkpoint(self, model: nn.Module):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        print(f"[*] Checkpoint model terbaik disimpan ke: {self.checkpoint_path}")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_accum_steps: int = 4,
    use_amp: bool = True
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / grad_accum_steps
            loss.backward()

        running_loss += loss.item() * grad_accum_steps * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Gradient accumulation step
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if use_amp and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        pbar.set_postfix({
            "loss": f"{(running_loss / max(1, total)):.4f}",
            "acc": f"{(correct / max(1, total)):.4f}"
        })

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def resolve_sample_path(p: str, data_dir: Optional[str] = None) -> str:
    """
    Menormalkan path citra agar kompatibel lintas OS (Windows/Linux/Colab/Kaggle)
    dan menyelesaikan path jika data berada di folder custom (e.g. Google Drive).
    """
    norm_p = str(p).replace("\\", os.sep).replace("/", os.sep)
    if os.path.exists(norm_p):
        return norm_p
    if data_dir:
        c1 = os.path.join(data_dir, norm_p)
        if os.path.exists(c1):
            return c1
        parts = norm_p.split(os.sep)
        if len(parts) >= 2:
            c2 = os.path.join(data_dir, parts[-2], parts[-1])
            if os.path.exists(c2):
                return c2
        c3 = os.path.join(data_dir, os.path.basename(norm_p))
        if os.path.exists(c3):
            return c3
    return norm_p


def train_pipeline(
    config_path: str = "src/config.yaml", 
    config_name: str = "k5",
    dry_run: bool = False, 
    custom_epochs: int = None,
    train_loader: torch.utils.data.DataLoader = None,
    val_loader: torch.utils.data.DataLoader = None,
    resume: bool = True,
    custom_checkpoints_dir: Optional[str] = None,
    custom_logs_dir: Optional[str] = None,
    custom_manifest_path: Optional[str] = None,
    custom_data_dir: Optional[str] = None,
    custom_batch_size: Optional[int] = None,
    custom_num_workers: Optional[int] = None,
    resume_checkpoint_path: Optional[str] = None
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Seed
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pipeline] Menggunakan device: {device}")
    if device.type == "cuda":
        print(f"[Pipeline] GPU: {torch.cuda.get_device_name(0)}")

    # Pemetaan modul model berdasarkan varian konfigurasi ablation K1-K5
    cfg_lower = config_name.lower()
    if cfg_lower == "k1":
        use_cbam = False
        use_non_local = False
    elif cfg_lower == "k2":
        use_cbam = True
        use_non_local = False
    elif cfg_lower == "k3":
        use_cbam = False
        use_non_local = True
    elif cfg_lower in ("k4", "k5"):
        use_cbam = True
        use_non_local = True
    else:
        use_cbam = bool(cfg["model"].get("use_cbam", True))
        use_non_local = bool(cfg["model"].get("use_non_local", True))

    print(f"[Pipeline] Konfigurasi: {config_name.upper()} (CBAM: {use_cbam}, Non-Local: {use_non_local})")

    # Inisialisasi Model
    model = EfficientNetB4_CBAM_NonLocal(
        pretrained=bool(cfg["model"]["pretrained"]),
        num_classes=int(cfg["model"]["num_classes"]),
        dropout_rate=float(cfg["model"]["dropout_rate"]),
        cbam_reduction=int(cfg["model"]["cbam_reduction_ratio"]),
        use_cbam=use_cbam,
        use_non_local=use_non_local
    ).to(device)

    # Parameter groups untuk differential learning rate
    lr_backbone = float(cfg["train"]["lr_backbone"])
    lr_head = float(cfg["train"]["lr_head"])
    param_groups = model.get_parameter_groups(lr_backbone=lr_backbone, lr_head=lr_head)

    optimizer = torch.optim.Adam(
        param_groups,
        betas=tuple(cfg["train"]["betas"]),
        eps=float(cfg["train"]["eps"]),
        weight_decay=float(cfg["train"]["weight_decay"])
    )

    criterion = nn.CrossEntropyLoss()

    sched_cfg = cfg["train"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg["mode"],
        factor=float(sched_cfg["factor"]),
        patience=int(sched_cfg["patience"]),
        min_lr=float(sched_cfg["min_lr"])
    )

    use_amp = bool(cfg["train"]["use_amp"])
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")
    grad_accum_steps = int(cfg["train"]["grad_accum_steps"])

    max_epochs = custom_epochs if custom_epochs is not None else int(cfg["train"]["max_epochs"])
    if dry_run:
        max_epochs = 1
        print("[Pipeline] Menjalankan mode DRY-RUN (1 Epoch Uji Coba)...")

    # Checkpoint paths dinamis:
    # 1. best_model.pt -> Menyimpan model dengan val_loss terbaik
    # 2. latest.pt     -> Menyimpan state lengkap (model, optimizer, scheduler, scaler, epoch) untuk resume
    checkpoints_dir = custom_checkpoints_dir if custom_checkpoints_dir else cfg["paths"]["checkpoints_dir"]
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoints_dir, f"{cfg_lower}_best.pt")
    latest_checkpoint_path = os.path.join(checkpoints_dir, f"{cfg_lower}_latest.pt")

    early_stopping = EarlyStopping(
        patience=int(cfg["train"]["early_stopping_patience"]),
        checkpoint_path=best_checkpoint_path
    )

    logs_dir = custom_logs_dir if custom_logs_dir else cfg["paths"]["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)
    history_csv = os.path.join(logs_dir, f"training_history_{cfg_lower}.csv")

    start_epoch = 0
    history = []

    # Mekanisme Resume dari latest checkpoint jika ada
    target_resume_path = resume_checkpoint_path if (resume_checkpoint_path and os.path.exists(resume_checkpoint_path)) else latest_checkpoint_path
    if resume and os.path.exists(target_resume_path) and not dry_run:
        print(f"[Pipeline] Memuat checkpoint resumable dari: {target_resume_path}")
        try:
            ckpt = torch.load(target_resume_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if scaler and ckpt.get("scaler_state_dict"):
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
                start_epoch = int(ckpt.get("epoch", 0))
                if "early_stopping" in ckpt:
                    early_stopping.counter = ckpt["early_stopping"].get("counter", 0)
                    early_stopping.best_loss = ckpt["early_stopping"].get("best_loss", float("inf"))
                    early_stopping.early_stop = ckpt["early_stopping"].get("early_stop", False)
                if "history" in ckpt:
                    history = ckpt["history"]
                print(f"[Pipeline] Berhasil resume! Melanjutkan dari Epoch {start_epoch + 1} (Total selesai sebelumnya: {start_epoch} epoch).")
            else:
                print("[Pipeline] Checkpoint bukan format lengkap, memulai training dari epoch 1.")
        except Exception as e:
            print(f"[Warning] Gagal memuat resume checkpoint ({e}), memulai dari epoch 1.")

    # Load DataLoaders jika belum disediakan
    if train_loader is None or val_loader is None:
        if custom_manifest_path:
            manifest_csv = custom_manifest_path
        elif cfg_lower in ("k1", "k2", "k3", "k4"):
            manifest_csv = os.path.join(logs_dir, "manifest_uniform.csv")
            if not os.path.exists(manifest_csv):
                manifest_csv = os.path.join(logs_dir, "processed_samples_manifest.csv")
        else:
            manifest_csv = os.path.join(logs_dir, "manifest_gfs.csv")
            if not os.path.exists(manifest_csv):
                manifest_csv = os.path.join(logs_dir, "processed_samples_manifest.csv")

        if os.path.exists(manifest_csv):
            print(f"[Pipeline] Membaca manifest dataset dari: {manifest_csv}")
            df_manifest = pd.read_csv(manifest_csv)
            train_df = df_manifest[df_manifest["split"] == "train"]
            val_df = df_manifest[df_manifest["split"] == "val"]
            train_samples = [(resolve_sample_path(p, custom_data_dir), int(lbl)) for p, lbl in zip(train_df["image_path"], train_df["label"])]
            val_samples = [(resolve_sample_path(p, custom_data_dir), int(lbl)) for p, lbl in zip(val_df["image_path"], val_df["label"])]

            batch_size = custom_batch_size if custom_batch_size is not None else int(cfg["train"]["batch_size"])
            num_workers = custom_num_workers if custom_num_workers is not None else int(cfg["dataset"].get("num_workers", 2))
            pin_memory = bool(cfg["dataset"].get("pin_memory", True))

            train_loader, val_loader = get_dataloaders(
                train_samples, val_samples,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        elif dry_run:
            print("[Pipeline] Manifest belum ada, menggunakan synthetic dummy dataset untuk dry-run...")
            dummy_x = torch.randn(16, 3, 380, 380)
            dummy_y = torch.tensor([0, 1] * 8)
            dummy_ds = torch.utils.data.TensorDataset(dummy_x, dummy_y)
            train_loader = torch.utils.data.DataLoader(dummy_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
            val_loader = torch.utils.data.DataLoader(dummy_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
        else:
            raise FileNotFoundError(
                f"File manifest sampel tidak ditemukan di {manifest_csv}. "
                "Jalankan run_pipeline.py terlebih dahulu untuk menghasilkan sampel wajah hasil preprocessing."
            )

    print(f"\n[Pipeline] Memulai training selama {max_epochs} epoch...")
    start_time = time.time()

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        is_improved = early_stopping(val_loss, model)

        epoch_duration = time.time() - epoch_start
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "duration_sec": epoch_duration
        }
        history.append(epoch_record)

        pd.DataFrame(history).to_csv(history_csv, index=False)

        # Simpan checkpoint resumable lengkap setiap akhir epoch
        latest_state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "early_stopping": {
                "counter": early_stopping.counter,
                "best_loss": early_stopping.best_loss,
                "early_stop": early_stopping.early_stop
            },
            "history": history
        }
        torch.save(latest_state, latest_checkpoint_path)

        status_flag = " [*BEST]" if is_improved else ""
        vram_info = ""
        if device.type == "cuda":
            vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_res = torch.cuda.memory_reserved() / (1024 ** 2)
            vram_info = f" | VRAM: {vram_alloc:.0f}/{vram_res:.0f}MB"

        print(
            f"Epoch [{epoch+1:03d}/{max_epochs:03d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.2e} | Time: {epoch_duration:.1f}s{status_flag}{vram_info}",
            flush=True
        )

        if early_stopping.early_stop:
            print(f"\n[!] Early stopping terpicu pada epoch {epoch + 1} karena val_loss tidak membaik.")
            break

    total_time = time.time() - start_time
    print(f"[Pipeline] Selesai training dalam {total_time/60:.2f} menit. Riwayat disimpan di {history_csv}")

    return {
        "model": model,
        "history": history,
        "best_loss": early_stopping.best_loss,
        "history_csv": history_csv
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Pipeline Deepfake Detection EfficientNet-CBAM-NonLocal")
    parser.add_argument("--config", type=str, default="src/config.yaml", help="Path ke config.yaml")
    parser.add_argument("--config-name", type=str, default="k5", help="Nama konfigurasi ablation (k1, k2, k3, k4, k5)")
    parser.add_argument("--dry-run", action="store_true", help="Jalankan 1 epoch dry-run untuk verifikasi VRAM")
    parser.add_argument("--epochs", type=int, default=None, help="Jumlah epoch kustom")
    parser.add_argument("--resume", action="store_true", default=True, help="Otomatis resume dari checkpoint terakhir jika ada (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Mulai training dari awal (abaikan checkpoint latest)")
    parser.add_argument("--checkpoints-dir", type=str, default=None, help="Direktori custom checkpoint (e.g. Google Drive / Kaggle)")
    parser.add_argument("--logs-dir", type=str, default=None, help="Direktori custom log")
    parser.add_argument("--manifest", type=str, default=None, help="Path custom file manifest CSV")
    parser.add_argument("--data-dir", type=str, default=None, help="Root folder citra hasil preprocessing (jika dipindah)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size DataLoader")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader num_workers")
    parser.add_argument("--resume-from", type=str, default=None, help="Path spesifik ke file checkpoint {config}_latest.pt untuk resume (e.g. dari output sesi Kaggle sebelumnya)")
    args = parser.parse_args()

    train_pipeline(
        config_path=args.config, 
        config_name=args.config_name, 
        dry_run=args.dry_run, 
        custom_epochs=args.epochs,
        resume=args.resume,
        custom_checkpoints_dir=args.checkpoints_dir,
        custom_logs_dir=args.logs_dir,
        custom_manifest_path=args.manifest,
        custom_data_dir=args.data_dir,
        custom_batch_size=args.batch_size,
        custom_num_workers=args.num_workers,
        resume_checkpoint_path=args.resume_from
    )
